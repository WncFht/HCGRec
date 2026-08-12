import math
from typing import Any, Optional, Union

import torch
from accelerate.utils import gather
from trl import GRPOTrainer


class TokenPrefixGRPOTrainer(GRPOTrainer):
    """GRPO trainer variant with token-level prefix advantages.

    Design:
    - Prefix signal: token-level (matched prefix tokens only).
    - NDCG signal: sequence-level, broadcast to all valid completion tokens.
    - Rule signal: optional probe only (typically zero-weight in reward_weights).
    """

    def __init__(
        self,
        *args,
        prefix_reward_normalize: bool = True,
        token_adv_total_token_normalize: bool = False,
        token_level_ndcg_error_token_penalty: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prefix_reward_normalize = prefix_reward_normalize
        self.token_adv_total_token_normalize = token_adv_total_token_normalize
        self.token_level_ndcg_error_token_penalty = token_level_ndcg_error_token_penalty
        self._token_adv_warned = False

    @staticmethod
    def _reward_func_name(reward_func) -> str:
        if hasattr(reward_func, "__name__"):
            return reward_func.__name__
        if hasattr(reward_func, "config") and hasattr(reward_func.config, "_name_or_path"):
            return str(reward_func.config._name_or_path).split("/")[-1]
        return reward_func.__class__.__name__

    @staticmethod
    def _group_normalize(values: torch.Tensor, group_size: int, eps: float = 1e-4) -> torch.Tensor:
        grouped = values.view(-1, group_size)
        mean = grouped.mean(dim=1).repeat_interleave(group_size, dim=0)
        std = grouped.std(dim=1).repeat_interleave(group_size, dim=0)
        return (values - mean) / (std + eps)

    @staticmethod
    def _group_normalize_tokenwise(
        values: torch.Tensor, mask: torch.Tensor, group_size: int, eps: float = 1e-4
    ) -> torch.Tensor:
        """Group-normalize token rewards at each token position (mask-aware)."""
        bsz, seq_len = values.shape
        grouped_values = values.view(-1, group_size, seq_len)
        grouped_mask = mask.view(-1, group_size, seq_len).float()

        denom = grouped_mask.sum(dim=1).clamp_min(1.0)
        mean = (grouped_values * grouped_mask).sum(dim=1) / denom
        centered = (grouped_values - mean.unsqueeze(1)) * grouped_mask
        var = (centered * centered).sum(dim=1) / denom
        std = torch.sqrt(var + eps)

        normalized = ((grouped_values - mean.unsqueeze(1)) / std.unsqueeze(1)) * grouped_mask
        return normalized.view(bsz, seq_len)

    def _build_prefix_token_rewards(
        self,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        reward_models: list[dict],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build token rewards and sequence rewards from prefix matches in token space."""
        device = completion_ids.device
        bsz, comp_len = completion_ids.shape
        token_rewards = torch.zeros((bsz, comp_len), dtype=torch.float32, device=device)
        seq_rewards = torch.zeros((bsz,), dtype=torch.float32, device=device)
        eos_id = self.processing_class.eos_token_id

        for i, reward_model in enumerate(reward_models):
            gt_text = reward_model.get("ground_truth", "")
            gt_ids = self.processing_class(gt_text, add_special_tokens=False).input_ids
            if len(gt_ids) == 0:
                continue

            valid_pos = torch.nonzero(completion_mask[i] > 0, as_tuple=False).squeeze(-1)
            if valid_pos.numel() == 0:
                continue
            comp_valid_ids = completion_ids[i, valid_pos]
            # Exclude terminal EOS from prefix matching.
            if comp_valid_ids.numel() > 0 and eos_id is not None and int(comp_valid_ids[-1].item()) == int(eos_id):
                comp_valid_ids = comp_valid_ids[:-1]
                valid_pos = valid_pos[:-1]

            matched = 0
            max_cmp = min(int(comp_valid_ids.numel()), len(gt_ids))
            while matched < max_cmp and int(comp_valid_ids[matched].item()) == int(gt_ids[matched]):
                matched += 1

            if matched == 0:
                continue

            per_tok = 1.0 / float(len(gt_ids)) if self.prefix_reward_normalize else 1.0
            token_rewards[i, valid_pos[:matched]] = per_tok
            seq_rewards[i] = float(matched) * float(per_tok)

        return token_rewards, seq_rewards

    def _build_prefix_token_rewards_with_error_mask(
        self,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        reward_models: list[dict],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build prefix token rewards and token error mask (1 for wrong tokens)."""
        device = completion_ids.device
        bsz, comp_len = completion_ids.shape
        token_rewards = torch.zeros((bsz, comp_len), dtype=torch.float32, device=device)
        seq_rewards = torch.zeros((bsz,), dtype=torch.float32, device=device)
        error_token_mask = torch.zeros((bsz, comp_len), dtype=torch.float32, device=device)
        eos_id = self.processing_class.eos_token_id

        for i, reward_model in enumerate(reward_models):
            gt_text = reward_model.get("ground_truth", "")
            gt_ids = self.processing_class(gt_text, add_special_tokens=False).input_ids
            valid_pos = torch.nonzero(completion_mask[i] > 0, as_tuple=False).squeeze(-1)
            if valid_pos.numel() == 0:
                continue

            comp_valid_ids = completion_ids[i, valid_pos]
            # Exclude terminal EOS from reward/error assignment.
            if comp_valid_ids.numel() > 0 and eos_id is not None and int(comp_valid_ids[-1].item()) == int(eos_id):
                comp_valid_ids = comp_valid_ids[:-1]
                valid_pos = valid_pos[:-1]
            if valid_pos.numel() == 0:
                continue

            matched = 0
            max_cmp = min(int(comp_valid_ids.numel()), len(gt_ids))
            while matched < max_cmp and int(comp_valid_ids[matched].item()) == int(gt_ids[matched]):
                matched += 1

            if len(gt_ids) > 0 and matched > 0:
                per_tok = 1.0 / float(len(gt_ids)) if self.prefix_reward_normalize else 1.0
                token_rewards[i, valid_pos[:matched]] = per_tok
                seq_rewards[i] = float(matched) * float(per_tok)

            # Wrong tokens are tokens after the matched prefix.
            error_token_mask[i, valid_pos[matched:]] = 1.0

        return token_rewards, seq_rewards, error_token_mask

    def _build_ndcg_rank_penalties(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return per-sample NDCG-style rank penalties with shape (B,)."""
        weights = torch.tensor(
            [-1.0 / math.log2(i + 2) for i in range(self.num_generations)],
            dtype=torch.float32,
            device=device,
        )
        weights = -weights / weights.sum()  # negative values summing to -1
        global_offset = self.accelerator.process_index * batch_size
        rank_idx = (torch.arange(batch_size, device=device) + global_offset) % self.num_generations
        return weights[rank_idx]

    def _get_reward_func_indices(self) -> tuple[Optional[int], Optional[int]]:
        prefix_idx = None
        ndcg_idx = None
        for i, reward_func in enumerate(self.reward_funcs):
            name = self._reward_func_name(reward_func)
            if name == "prefix_rule_reward":
                prefix_idx = i
            elif name == "ndcg_rule_reward":
                ndcg_idx = i
        return prefix_idx, ndcg_idx

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        # Reuse all generation/logging behavior from upstream first.
        outputs = super()._generate_and_score_completions(inputs)

        prefix_idx, ndcg_idx = self._get_reward_func_indices()
        if prefix_idx is None:
            # Keep base sequence-level behavior if prefix reward func is missing.
            if not self._token_adv_warned:
                print("[WARN] TokenPrefixGRPOTrainer fallback to sequence-level advantages (missing prefix reward).")
                self._token_adv_warned = True
            return outputs

        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        completions_text = self.processing_class.batch_decode(outputs["completion_ids"], skip_special_tokens=True)
        reward_kwargs = {
            key: [example[key] for example in inputs] for key in inputs[0] if key not in ["prompt", "completion"]
        }
        reward_models = reward_kwargs.get("reward_model", [])

        # 1) Prefix token rewards (local), local prefix sequence rewards, and token error mask.
        prefix_token_rewards, prefix_seq_local, error_token_mask = self._build_prefix_token_rewards_with_error_mask(
            outputs["completion_ids"], outputs["completion_mask"], reward_models
        )

        prefix_weight = float(self.reward_weights[prefix_idx].item())
        ndcg_weight = float(self.reward_weights[ndcg_idx].item()) if ndcg_idx is not None else 0.0
        completion_mask = outputs["completion_mask"].float()
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )

        if self.token_adv_total_token_normalize:
            if ndcg_idx is None:
                ndcg_token_rewards = torch.zeros_like(prefix_token_rewards)
            elif self.token_level_ndcg_error_token_penalty:
                # NDCG token penalty: only penalize wrong tokens, and only if the group has any positive prefix.
                prefix_positive_local = (prefix_seq_local > 0).float()
                prefix_positive_global = gather(prefix_positive_local)
                group_has_prefix_global = (
                    prefix_positive_global.view(-1, self.num_generations)
                    .max(dim=1)
                    .values.repeat_interleave(self.num_generations)
                )
                group_has_prefix_local = group_has_prefix_global[process_slice].unsqueeze(1)
                ndcg_rank_local = self._build_ndcg_rank_penalties(len(prompts), device=device)
                ndcg_token_rewards = ndcg_rank_local.unsqueeze(1) * error_token_mask * group_has_prefix_local
            else:
                # 2) NDCG sequence rewards (local) from reward function, then broadcast to token level.
                ndcg_func = self.reward_funcs[ndcg_idx]
                ndcg_local = torch.tensor(
                    ndcg_func(
                        prompts=prompts,
                        completions=completions_text,
                        completion_ids=outputs["completion_ids"],
                        **reward_kwargs,
                    ),
                    dtype=torch.float32,
                    device=device,
                )
                ndcg_token_rewards = ndcg_local.unsqueeze(1) * completion_mask

            # Compose raw token rewards first, then token-wise normalize by prompt-group.
            total_token_rewards_local = prefix_weight * prefix_token_rewards + ndcg_weight * ndcg_token_rewards
            total_token_rewards_global = gather(total_token_rewards_local)
            completion_mask_global = gather(completion_mask)
            token_advantages_global = self._group_normalize_tokenwise(
                total_token_rewards_global, completion_mask_global, self.num_generations
            )
            token_advantages = token_advantages_global[process_slice]
        else:
            # 3) Group-normalize prefix signal globally.
            prefix_seq_global = gather(prefix_seq_local)
            adv_prefix_global = self._group_normalize(prefix_seq_global, self.num_generations)
            adv_prefix_local = adv_prefix_global[process_slice]

            # 4) Compose token-level advantages.

            prefix_mass = prefix_token_rewards.sum(dim=1, keepdim=True)
            prefix_dist = torch.where(
                prefix_mass > 0,
                prefix_token_rewards / (prefix_mass + 1e-8),
                torch.zeros_like(prefix_token_rewards),
            )
            token_advantages = prefix_weight * adv_prefix_local.unsqueeze(1) * prefix_dist
            if ndcg_idx is not None:
                ndcg_func = self.reward_funcs[ndcg_idx]
                ndcg_local = torch.tensor(
                    ndcg_func(
                        prompts=prompts,
                        completions=completions_text,
                        completion_ids=outputs["completion_ids"],
                        **reward_kwargs,
                    ),
                    dtype=torch.float32,
                    device=device,
                )
                ndcg_global = gather(ndcg_local)
                adv_ndcg_global = self._group_normalize(ndcg_global, self.num_generations)
                adv_ndcg_local = adv_ndcg_global[process_slice]
                token_advantages = token_advantages + ndcg_weight * adv_ndcg_local.unsqueeze(1) * completion_mask
        outputs["advantages"] = token_advantages

        mode = "eval" if self.control.should_evaluate else "train"
        nonzero_ratio = (
            (token_advantages.abs() > 0).float() * completion_mask
        ).sum() / completion_mask.sum().clamp_min(1.0)
        self._metrics[mode]["token_adv_nonzero_ratio"].append(
            self.accelerator.gather_for_metrics(nonzero_ratio).mean().item()
        )
        return outputs
