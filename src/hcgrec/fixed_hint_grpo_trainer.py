"""Fixed-hint and dynamic-hint GRPO trainers that reveal target-SID prefixes for hard samples."""

from __future__ import annotations

import inspect
import re
from typing import Any, Union

import torch
from accelerate.utils import gather_object
from trl import GRPOTrainer
from trl.data_utils import is_conversational, maybe_apply_chat_template
from trl.trainer.grpo_trainer import nanstd
from trl.trainer.utils import pad

from hcgrec.fixed_hint_utils import build_hint_text, build_prompt_with_hint, build_suffix_text


def _extract_images_from_inputs(inputs: list[dict[str, Union[torch.Tensor, Any]]]) -> list[Any] | None:
    if "images" in inputs[0]:
        images = [example.get("images") for example in inputs]
    elif "image" in inputs[0]:
        images = [[example.get("image")] if example.get("image") is not None else None for example in inputs]
    else:
        images = None
    if images is not None and all(img_list == [] for img_list in images):
        images = None
    return images


def _extract_completion_numbers(text: str) -> list[int]:
    return [int(match) for match in re.findall(r"_(\d+)", text)]


def _compute_rule_hit_rewards(
    examples: list[dict[str, Union[torch.Tensor, Any]]],
    completions_text: list[str],
) -> list[float]:
    rewards = []
    for example, completion_text in zip(examples, completions_text):
        full_completion = f"{example.get('oracle_hint_text', '')}{completion_text}"
        gt_numbers = _extract_completion_numbers(example["reward_model"]["ground_truth"])
        completion_numbers = _extract_completion_numbers(full_completion)
        rewards.append(1.0 if completion_numbers == gt_numbers else 0.0)
    return rewards


def build_prompt_target_shift_mask(prompt_lengths: list[int], target_token_counts: list[int]) -> list[list[int]]:
    if len(prompt_lengths) != len(target_token_counts):
        raise ValueError(
            f"prompt_lengths and target_token_counts must have the same length, got "
            f"{len(prompt_lengths)} and {len(target_token_counts)}."
        )

    masks: list[list[int]] = []
    for prompt_length, target_token_count in zip(prompt_lengths, target_token_counts):
        prompt_length = int(prompt_length)
        target_token_count = max(int(target_token_count), 0)
        shift_length = max(prompt_length - 1, 0)
        effective_target_token_count = min(target_token_count, shift_length)
        suffix_start = shift_length - effective_target_token_count
        masks.append([1 if index >= suffix_start else 0 for index in range(shift_length)])
    return masks


def build_prompt_hint_shift_mask(prompt_lengths: list[int], hint_token_counts: list[int]) -> list[list[int]]:
    return build_prompt_target_shift_mask(prompt_lengths, hint_token_counts)


def _selective_log_softmax(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)


def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.nn.functional.softmax(logits, dim=-1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1)


class FixedHintRuleOnlyGRPOTrainer(GRPOTrainer):
    def __init__(
        self,
        *args,
        hint_ce_loss_coef: float = 0.0,
        full_sequence_sft_loss_coef: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.hint_ce_loss_coef = float(hint_ce_loss_coef)
        self.full_sequence_sft_loss_coef = float(full_sequence_sft_loss_coef)
        if self.hint_ce_loss_coef > 0.0 and self.full_sequence_sft_loss_coef > 0.0:
            raise ValueError("hint_ce_loss_coef and full_sequence_sft_loss_coef cannot both be > 0.")
        self._cached_prompt_shift_logits: torch.Tensor | None = None

    def _build_hinted_prompts(self, inputs: list[dict[str, Union[torch.Tensor, Any]]]) -> list[str]:
        return [
            build_prompt_with_hint(
                example,
                formatter=lambda prompt: maybe_apply_chat_template({"prompt": prompt}, self.processing_class)[
                    "prompt"
                ],
            )
            for example in inputs
        ]

    def _maybe_log_prompt_completions(
        self,
        prompts_text: list[str],
        completions_text: list[str],
        images: list[Any] | None = None,
    ) -> None:
        if not getattr(self, "log_completions", False):
            return
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        if images is not None:
            self._logs["images"].extend(gather_object(images))

    def _get_hint_token_counts(self, inputs: list[dict[str, Union[torch.Tensor, Any]]]) -> list[int]:
        return [max(int(example.get("oracle_hint_depth", 0)), 0) for example in inputs]

    def _tokenize_text(self, text: str) -> list[int]:
        encoded = self.processing_class(text, add_special_tokens=False)
        input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        return list(input_ids)

    def _build_prompt_hint_ce_mask(
        self,
        prompt_ids_list: list[Any],
        hint_token_counts: list[int],
        device: torch.device,
    ) -> torch.Tensor:
        prompt_lengths = [len(ids) for ids in prompt_ids_list]
        shift_masks = build_prompt_hint_shift_mask(prompt_lengths, hint_token_counts)
        mask_tensors = [torch.tensor(mask, device=device, dtype=torch.float32) for mask in shift_masks]
        return pad(mask_tensors, padding_value=0.0, padding_side="left")

    def _build_suffix_sft_batch(
        self,
        inputs: list[dict[str, Union[torch.Tensor, Any]]],
        prompt_ids_list: list[Any],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        input_ids_rows = []
        attention_rows = []
        suffix_shift_mask_rows = []
        suffix_token_counts = []

        for example, prompt_ids in zip(inputs, prompt_ids_list):
            suffix_text = build_suffix_text(
                example["reward_model"]["ground_truth"],
                int(example.get("oracle_hint_depth", 0)),
            )
            suffix_ids = self._tokenize_text(suffix_text) if suffix_text else []
            full_ids = list(prompt_ids) + suffix_ids
            input_ids_rows.append(torch.tensor(full_ids, device=device, dtype=torch.long))
            attention_rows.append(torch.ones(len(full_ids), device=device, dtype=torch.long))
            shift_mask = build_prompt_target_shift_mask(
                prompt_lengths=[len(full_ids)],
                target_token_counts=[len(suffix_ids)],
            )[0]
            suffix_shift_mask_rows.append(torch.tensor(shift_mask, device=device, dtype=torch.float32))
            suffix_token_counts.append(min(len(suffix_ids), max(len(full_ids) - 1, 0)))

        return {
            "full_sequence_sft_input_ids": pad(
                input_ids_rows,
                padding_value=self.pad_token_id,
                padding_side="right",
            ),
            "full_sequence_sft_attention_mask": pad(
                attention_rows,
                padding_value=0,
                padding_side="right",
            ),
            "full_sequence_sft_suffix_mask": pad(
                suffix_shift_mask_rows,
                padding_value=0.0,
                padding_side="right",
            ),
            "full_sequence_sft_suffix_token_count": torch.tensor(
                suffix_token_counts,
                device=device,
                dtype=torch.float32,
            ),
        }

    def _compute_masked_ce_local_sum(
        self,
        shift_logits: torch.Tensor,
        shift_labels: torch.Tensor,
        shift_mask: torch.Tensor,
    ) -> torch.Tensor:
        token_losses = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none",
        ).view_as(shift_labels)
        return (token_losses * shift_mask).sum()

    def _compute_global_masked_ce_mean(
        self,
        local_loss_sum: torch.Tensor,
        local_token_count: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zero = local_token_count * 0.0
        gathered_token_count = self.accelerator.gather(local_token_count.detach())
        global_token_count = gathered_token_count.sum()
        if torch.isclose(global_token_count, zero):
            return zero, global_token_count
        gathered_loss_sum = self.accelerator.gather(local_loss_sum.detach())
        global_loss_mean = gathered_loss_sum.sum() / global_token_count.clamp(min=1.0)
        return global_loss_mean, global_token_count

    def _append_masked_ce_metrics(
        self,
        mode: str,
        metric_prefix: str,
        global_loss_mean: torch.Tensor,
        global_token_count: torch.Tensor,
    ) -> None:
        self._metrics[mode][f"{metric_prefix}/loss"].append(global_loss_mean.item())
        self._metrics[mode][f"{metric_prefix}/token_count"].append(global_token_count.item())

    def _compute_prompt_hint_ce_loss(self, model, inputs) -> torch.Tensor:
        prompt_hint_ce_mask = inputs.get("prompt_hint_ce_mask")
        prompt_ids = inputs["prompt_ids"]
        zero = prompt_ids.new_zeros((), dtype=torch.float32)

        if prompt_hint_ce_mask is None:
            return zero

        prompt_hint_ce_mask = prompt_hint_ce_mask.float()
        hint_token_count = prompt_hint_ce_mask.sum()
        gathered_hint_token_count = self.accelerator.gather(hint_token_count.detach())
        global_hint_token_count = gathered_hint_token_count.sum()
        mode = "train" if self.model.training else "eval"
        if torch.isclose(global_hint_token_count, zero):
            self._metrics[mode]["hint_ce/loss"].append(0.0)
            self._metrics[mode]["hint_ce/token_count"].append(0.0)
            return zero

        if torch.isclose(hint_token_count, zero):
            local_hint_ce_sum = zero
        else:
            prompt_shift_logits = inputs.get("prompt_shift_logits")
            if prompt_shift_logits is None:
                prompt_shift_logits = getattr(self, "_cached_prompt_shift_logits", None)
            if prompt_shift_logits is None:
                raise RuntimeError(
                    "Hint CE prompt logits cache is missing. This trainer expects prompt-side logits "
                    "from the main RL forward and must not run a second gradient-bearing model forward."
                )
            prompt_shift_labels = prompt_ids[:, 1:]
            local_hint_ce_sum = self._compute_masked_ce_local_sum(
                prompt_shift_logits,
                prompt_shift_labels,
                prompt_hint_ce_mask,
            )

        global_hint_ce_mean, global_hint_token_count = self._compute_global_masked_ce_mean(
            local_hint_ce_sum,
            hint_token_count,
        )
        self._append_masked_ce_metrics(mode, "hint_ce", global_hint_ce_mean, global_hint_token_count)

        if getattr(self, "loss_type", None) == "dapo":
            prompt_hint_ce_num_items_in_batch = inputs.get("prompt_hint_ce_num_items_in_batch")
            if prompt_hint_ce_num_items_in_batch is None:
                raise RuntimeError(
                    "Hint CE global token count is missing for DAPO normalization. This trainer expects "
                    "the full accumulated-step hint token count to be precomputed before splitting batches."
                )
            # DAPO normalizes the main RL loss with the total number of active completion tokens in the full
            # accumulated step. The auxiliary hint CE must share the same 'global token denominator' idea;
            # otherwise local shards with fewer hint tokens would be overweighted just because they have a
            # smaller local mean. We therefore keep the local numerator here but divide by the full-step
            # global hint-token count (scaled to this rank), not by the local hint-token count.
            normalizer = prompt_hint_ce_num_items_in_batch / self.accelerator.num_processes
            return local_hint_ce_sum / normalizer.clamp(min=1.0)

        local_hint_ce_mean = local_hint_ce_sum / hint_token_count.clamp(min=1.0)
        return local_hint_ce_mean / self.current_gradient_accumulation_steps

    def _compute_full_sequence_sft_loss(self, model, inputs) -> torch.Tensor:
        prompt_hint_ce_mask = inputs.get("prompt_hint_ce_mask")
        prompt_ids = inputs["prompt_ids"]
        zero = prompt_ids.new_zeros((), dtype=torch.float32)
        mode = "train" if self.model.training else "eval"

        if prompt_hint_ce_mask is None:
            return zero

        prompt_hint_ce_mask = prompt_hint_ce_mask.float()
        prompt_hint_token_count = prompt_hint_ce_mask.sum()
        prompt_shift_logits = inputs.get("prompt_shift_logits")
        if prompt_shift_logits is None:
            prompt_shift_logits = getattr(self, "_cached_prompt_shift_logits", None)
        if prompt_shift_logits is None:
            raise RuntimeError(
                "Full-sequence SFT prompt logits cache is missing. This trainer expects prompt-side logits "
                "from the main RL forward and must not run a second gradient-bearing prompt forward."
            )

        prompt_shift_labels = prompt_ids[:, 1:]
        if torch.isclose(prompt_hint_token_count, zero):
            local_prompt_hint_sum = zero
        else:
            local_prompt_hint_sum = self._compute_masked_ce_local_sum(
                prompt_shift_logits,
                prompt_shift_labels,
                prompt_hint_ce_mask,
            )

        suffix_input_ids = inputs.get("full_sequence_sft_input_ids")
        suffix_attention_mask = inputs.get("full_sequence_sft_attention_mask")
        suffix_shift_mask = inputs.get("full_sequence_sft_suffix_mask")
        if suffix_input_ids is None or suffix_attention_mask is None or suffix_shift_mask is None:
            raise RuntimeError("Full-sequence SFT tensors are missing from the generated batch.")

        suffix_shift_mask = suffix_shift_mask.float()
        suffix_token_count = suffix_shift_mask.sum()
        if torch.isclose(suffix_token_count, zero):
            local_suffix_sum = zero
        else:
            suffix_outputs = model(
                input_ids=suffix_input_ids,
                attention_mask=suffix_attention_mask,
                use_cache=False,
            )
            suffix_shift_logits = suffix_outputs.logits[:, :-1, :]
            suffix_shift_labels = suffix_input_ids[:, 1:]
            local_suffix_sum = self._compute_masked_ce_local_sum(
                suffix_shift_logits,
                suffix_shift_labels,
                suffix_shift_mask,
            )

        prompt_hint_mean, prompt_hint_global_count = self._compute_global_masked_ce_mean(
            local_prompt_hint_sum,
            prompt_hint_token_count,
        )
        suffix_mean, suffix_global_count = self._compute_global_masked_ce_mean(
            local_suffix_sum,
            suffix_token_count,
        )
        total_token_count = prompt_hint_token_count + suffix_token_count
        total_mean, total_global_count = self._compute_global_masked_ce_mean(
            local_prompt_hint_sum + local_suffix_sum,
            total_token_count,
        )

        self._append_masked_ce_metrics(mode, "full_sequence_sft", total_mean, total_global_count)
        self._append_masked_ce_metrics(
            mode, "full_sequence_sft/prompt_hint", prompt_hint_mean, prompt_hint_global_count
        )
        self._append_masked_ce_metrics(mode, "full_sequence_sft/suffix", suffix_mean, suffix_global_count)

        if torch.isclose(total_global_count, zero):
            return zero

        if getattr(self, "loss_type", None) == "dapo":
            full_sequence_sft_num_items_in_batch = inputs.get("full_sequence_sft_num_items_in_batch")
            if full_sequence_sft_num_items_in_batch is None:
                raise RuntimeError(
                    "Full-sequence SFT global token count is missing for DAPO normalization. This trainer "
                    "expects the full accumulated-step supervised token count to be precomputed before splitting batches."
                )
            normalizer = full_sequence_sft_num_items_in_batch / self.accelerator.num_processes
            return (local_prompt_hint_sum + local_suffix_sum) / normalizer.clamp(min=1.0)

        local_total_mean = (local_prompt_hint_sum + local_suffix_sum) / total_token_count.clamp(min=1.0)
        return local_total_mean / self.current_gradient_accumulation_steps

    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        pixel_values=None,
        image_grid_thw=None,
        num_images=None,
        pixel_attention_mask=None,
        image_sizes=None,
        token_type_ids=None,
        mm_token_type_ids=None,
        image_position_ids=None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        aux_prompt_loss_coef = max(
            getattr(self, "hint_ce_loss_coef", 0.0),
            getattr(self, "full_sequence_sft_loss_coef", 0.0),
        )
        if aux_prompt_loss_coef <= 0.0 or not compute_entropy:
            super_method = super()._get_per_token_logps_and_entropies
            supported_params = set(inspect.signature(super_method).parameters)
            kwargs = {
                "batch_size": batch_size,
                "compute_entropy": compute_entropy,
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
                "num_images": num_images,
                "pixel_attention_mask": pixel_attention_mask,
                "image_sizes": image_sizes,
                "token_type_ids": token_type_ids,
                "mm_token_type_ids": mm_token_type_ids,
                "image_position_ids": image_position_ids,
            }
            filtered_kwargs = {key: value for key, value in kwargs.items() if key in supported_params}
            return super_method(
                model,
                input_ids,
                attention_mask,
                logits_to_keep,
                **filtered_kwargs,
            )

        batch_size = batch_size or input_ids.size(0)
        prompt_width = input_ids.size(1) - logits_to_keep
        all_logps = []
        all_entropies = []
        all_prompt_shift_logits = []
        model_kwarg_keys = getattr(self, "model_kwarg_keys", set())

        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start : start + batch_size]
            attention_mask_batch = attention_mask[start : start + batch_size]

            model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}
            if image_grid_thw is not None and pixel_values is not None:
                rows_per_image = image_grid_thw.prod(dim=-1)
                rows_per_sample = torch.split(rows_per_image, num_images)
                rows_per_sample = torch.stack([s.sum() for s in rows_per_sample])
                cum_rows = torch.cat([torch.tensor([0], device=rows_per_sample.device), rows_per_sample.cumsum(0)])
                row_start, row_end = cum_rows[start].item(), cum_rows[start + batch_size].item()
                model_inputs["pixel_values"] = pixel_values[row_start:row_end]
                cum_imgs = torch.tensor([0] + num_images).cumsum(0)
                img_start, img_end = cum_imgs[start], cum_imgs[start + batch_size]
                model_inputs["image_grid_thw"] = image_grid_thw[img_start:img_end]
            elif image_position_ids is not None and pixel_values is not None:
                cum_imgs = torch.tensor([0] + num_images).cumsum(0)
                img_start, img_end = cum_imgs[start], cum_imgs[start + batch_size]
                model_inputs["pixel_values"] = pixel_values[img_start:img_end]
                model_inputs["image_position_ids"] = image_position_ids[img_start:img_end]
            elif pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values[start : start + batch_size]
            if pixel_attention_mask is not None:
                model_inputs["pixel_attention_mask"] = pixel_attention_mask[start : start + batch_size]
            if image_sizes is not None:
                model_inputs["image_sizes"] = image_sizes[start : start + batch_size]
            if token_type_ids is not None:
                model_inputs["token_type_ids"] = token_type_ids[start : start + batch_size]
            if mm_token_type_ids is not None:
                model_inputs["mm_token_type_ids"] = mm_token_type_ids[start : start + batch_size]

            # We intentionally avoid logits_to_keep here so the same forward can
            # also supply prompt-side logits for the auxiliary supervised loss.
            if "logits_to_keep" in model_kwarg_keys:
                pass

            model_inputs["use_cache"] = False
            logits = model(**model_inputs).logits
            shift_logits = logits[:, :-1, :]
            prompt_shift_logits = shift_logits[:, : prompt_width - 1, :]
            completion_logits = shift_logits[:, -logits_to_keep:, :]
            completion_logits = completion_logits / self.temperature
            completion_ids = input_ids_batch[:, -logits_to_keep:]

            all_prompt_shift_logits.append(prompt_shift_logits)
            all_logps.append(_selective_log_softmax(completion_logits, completion_ids))
            if compute_entropy:
                with torch.no_grad():
                    all_entropies.append(_entropy_from_logits(completion_logits))

        self._cached_prompt_shift_logits = torch.cat(all_prompt_shift_logits, dim=0)
        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        return logps, entropies

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        images = _extract_images_from_inputs(inputs)

        prompts: list[Any] = [example["prompt"] for example in inputs]
        prompts_text = self._build_hinted_prompts(inputs)
        (
            prompt_ids_list,
            completion_ids_list,
            num_items_in_batch,
            sampling_per_token_logps_list,
            forward_kwargs,
        ) = self._generate(prompts_text, images)

        if forward_kwargs:
            raise NotImplementedError("Fixed hint trainer currently supports text-only generation batches.")
        if sampling_per_token_logps_list is not None:
            raise NotImplementedError("Fixed hint trainer does not support sampling log-prob correction yet.")

        prompt_hint_ce_mask = None
        prompt_hint_ce_num_items_in_batch = None
        full_sequence_sft_batch = None
        full_sequence_sft_num_items_in_batch = None
        if getattr(self, "hint_ce_loss_coef", 0.0) > 0.0 or getattr(self, "full_sequence_sft_loss_coef", 0.0) > 0.0:
            hint_token_counts = self._get_hint_token_counts(inputs)
            prompt_hint_ce_mask = self._build_prompt_hint_ce_mask(prompt_ids_list, hint_token_counts, device=device)
            if getattr(self, "loss_type", None) == "dapo":
                prompt_hint_ce_num_items_in_batch = self.accelerator.gather(prompt_hint_ce_mask.sum()).sum()
        if getattr(self, "full_sequence_sft_loss_coef", 0.0) > 0.0:
            full_sequence_sft_batch = self._build_suffix_sft_batch(inputs, prompt_ids_list, device=device)
            if getattr(self, "loss_type", None) == "dapo":
                full_sequence_sft_num_items_in_batch = self.accelerator.gather(
                    prompt_hint_ce_mask.sum() + full_sequence_sft_batch["full_sequence_sft_suffix_mask"].sum()
                ).sum()

        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")

        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        with torch.no_grad():
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self.args.gradient_accumulation_steps % generate_every != 0:
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    num_images=None,
                    **forward_kwargs,
                )
            else:
                old_per_token_logps = None

            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        num_images=None,
                        **forward_kwargs,
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            num_images=None,
                            **forward_kwargs,
                        )
            else:
                ref_per_token_logps = None

        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards

        if self.scale_rewards in ["group", "none"]:
            std_rewards = rewards.view(-1, self.num_generations).std(dim=1)
            std_rewards = std_rewards.repeat_interleave(self.num_generations, dim=0)
        elif self.scale_rewards == "batch":
            std_rewards = rewards.std().expand_as(rewards)
        else:
            raise ValueError(
                f"Invalid value for scale_rewards: {self.scale_rewards}. Must be one of 'batch', 'group', or 'none'."
            )

        is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
        if self.scale_rewards != "none":
            advantages = advantages / (std_rewards + 1e-4)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()
        advantages = advantages[process_slice]

        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_func_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        self._maybe_log_prompt_completions(prompts_text, completions_text, images=images)
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
        }
        if prompt_hint_ce_mask is not None:
            output["prompt_hint_ce_mask"] = prompt_hint_ce_mask
        if prompt_hint_ce_num_items_in_batch is not None:
            output["prompt_hint_ce_num_items_in_batch"] = prompt_hint_ce_num_items_in_batch
        if full_sequence_sft_batch is not None:
            output.update(full_sequence_sft_batch)
        if full_sequence_sft_num_items_in_batch is not None:
            output["full_sequence_sft_num_items_in_batch"] = full_sequence_sft_num_items_in_batch
        return output

    def _compute_loss(self, model, inputs):
        self._cached_prompt_shift_logits = None
        loss = super()._compute_loss(model, inputs)
        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["loss/rl_base"].append(loss.item())
        try:
            total_loss = loss
            if getattr(self, "hint_ce_loss_coef", 0.0) > 0.0:
                hint_ce_loss = self._compute_prompt_hint_ce_loss(model, inputs)
                weighted_hint_ce_loss = self.hint_ce_loss_coef * hint_ce_loss
                self._metrics[mode]["loss/hint_ce_weighted"].append(weighted_hint_ce_loss.item())
                total_loss = total_loss + weighted_hint_ce_loss
            if getattr(self, "full_sequence_sft_loss_coef", 0.0) > 0.0:
                full_sequence_sft_loss = self._compute_full_sequence_sft_loss(model, inputs)
                weighted_full_sequence_sft_loss = self.full_sequence_sft_loss_coef * full_sequence_sft_loss
                self._metrics[mode]["loss/full_sequence_sft_weighted"].append(weighted_full_sequence_sft_loss.item())
                total_loss = total_loss + weighted_full_sequence_sft_loss
            return total_loss
        finally:
            self._cached_prompt_shift_logits = None


class DynamicHintRuleOnlyGRPOTrainer(FixedHintRuleOnlyGRPOTrainer):
    def __init__(
        self,
        *args,
        dynamic_hint_max_depth: int = 3,
        dynamic_hint_apply_to_eval: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dynamic_hint_max_depth = int(dynamic_hint_max_depth)
        self.dynamic_hint_apply_to_eval = bool(dynamic_hint_apply_to_eval)
        if self.dynamic_hint_max_depth < 0:
            raise ValueError("dynamic_hint_max_depth must be >= 0.")

    def _resolve_example_dynamic_hint_max_depth(
        self,
        example: dict[str, Union[torch.Tensor, Any]],
        default_max_hint_depth: int,
    ) -> int:
        override = example.get("dynamic_hint_max_depth_override")
        if override is None:
            return int(default_max_hint_depth)
        return max(min(int(override), int(default_max_hint_depth)), 0)

    def _build_runtime_hinted_example(
        self,
        example: dict[str, Union[torch.Tensor, Any]],
        hint_depth: int,
    ) -> dict[str, Union[torch.Tensor, Any]]:
        enriched = dict(example)
        hint_text = build_hint_text(example["reward_model"]["ground_truth"], hint_depth)
        enriched["oracle_hint_depth"] = int(hint_depth)
        enriched["oracle_hint_text"] = hint_text
        enriched["oracle_hint_unsolved"] = False
        return enriched

    def _generate_dynamic_stage(
        self,
        prompts_text: list[str],
        images: list[Any] | None,
    ) -> tuple[list[Any], list[Any]]:
        (
            prompt_ids_list,
            completion_ids_list,
            sampling_per_token_logps_list,
            forward_kwargs,
        ) = self._generate_single_turn(prompts_text, images)

        if forward_kwargs:
            raise NotImplementedError("Dynamic hint trainer currently supports text-only generation batches.")
        if sampling_per_token_logps_list is not None:
            raise NotImplementedError("Dynamic hint trainer does not support sampling log-prob correction yet.")

        return prompt_ids_list, completion_ids_list

    def _has_global_unresolved_groups(self, unresolved_group_count: int) -> bool:
        accelerator = getattr(self, "accelerator", None)
        if accelerator is None or not hasattr(accelerator, "device") or not hasattr(accelerator, "gather"):
            return unresolved_group_count > 0
        local_flag = torch.tensor(
            [1 if unresolved_group_count > 0 else 0],
            device=accelerator.device,
            dtype=torch.int32,
        )
        gathered_flags = accelerator.gather(local_flag)
        return bool(gathered_flags.max().item())

    def _log_selected_batch_generation_metrics(
        self,
        mode: str,
        prompt_ids_list: list[Any],
        completion_ids_list: list[Any],
    ) -> Any:
        device = self.accelerator.device
        prompt_lengths = torch.tensor([len(ids) for ids in prompt_ids_list], device=device)
        completion_lengths = torch.tensor([len(ids) for ids in completion_ids_list], device=device)
        agg_prompt_lengths = self.accelerator.gather(prompt_lengths)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        total_prompt_tokens = agg_prompt_lengths.sum()
        total_completion_tokens = agg_completion_lengths.sum()

        if mode == "train":
            self.state.num_input_tokens_seen += (total_prompt_tokens + total_completion_tokens).item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        eos_and_pad = [self.eos_token_id, self.pad_token_id]
        is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
        agg_is_truncated = self.accelerator.gather(is_truncated)
        self._metrics[mode]["completions/clipped_ratio"].append(agg_is_truncated.float().mean().item())
        term_completion_lengths = agg_completion_lengths[~agg_is_truncated]
        if len(term_completion_lengths) == 0:
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        return total_completion_tokens

    def _run_dynamic_hint_cascade(
        self,
        inputs: list[dict[str, Union[torch.Tensor, Any]]],
        images: list[Any] | None,
        max_hint_depth: int,
    ) -> dict[str, Any]:
        if len(inputs) % self.num_generations != 0:
            raise ValueError(
                f"Dynamic hint cascade expects batch size divisible by num_generations={self.num_generations}, "
                f"got len(inputs)={len(inputs)}."
            )

        prompt_groups: list[dict[str, Any]] = []
        for group_index, start in enumerate(range(0, len(inputs), self.num_generations)):
            end = start + self.num_generations
            group_inputs = inputs[start:end]
            prompt_groups.append(
                {
                    "group_index": group_index,
                    "inputs": group_inputs,
                    "images": None if images is None else images[start:end],
                    "max_hint_depth": max(
                        self._resolve_example_dynamic_hint_max_depth(example, max_hint_depth)
                        for example in group_inputs
                    ),
                }
            )

        unresolved_groups = list(prompt_groups)
        selected_group_outputs: dict[int, dict[str, Any]] = {}
        stage_stats: list[dict[str, int]] = []
        is_main_process = bool(getattr(getattr(self, "accelerator", None), "is_main_process", False))
        should_log_stages = is_main_process and max_hint_depth > 0
        dummy_group = prompt_groups[0]

        for requested_hint_depth in range(max_hint_depth + 1):
            local_unresolved_groups = list(unresolved_groups)
            stage_target_groups = local_unresolved_groups if local_unresolved_groups else [dummy_group]

            stage_inputs: list[dict[str, Union[torch.Tensor, Any]]] = []
            stage_images: list[Any] | None = [] if images is not None else None
            stage_slices: list[tuple[int, int, dict[str, Any]]] = []

            for group in stage_target_groups:
                group_stage_inputs = [
                    self._build_runtime_hinted_example(example, min(requested_hint_depth, group["max_hint_depth"]))
                    for example in group["inputs"]
                ]
                start = len(stage_inputs)
                stage_inputs.extend(group_stage_inputs)
                end = len(stage_inputs)
                if local_unresolved_groups:
                    stage_slices.append((start, end, group))
                if stage_images is not None:
                    stage_images.extend(group["images"])

            stage_prompts_text = self._build_hinted_prompts(stage_inputs)
            if should_log_stages and local_unresolved_groups:
                print(
                    "[INFO] dynamic_hint_stage "
                    f"requested_depth={requested_hint_depth} "
                    f"unresolved_groups={len(local_unresolved_groups)} "
                    f"stage_batch={len(stage_inputs)}"
                )
            stage_prompt_ids_list, stage_completion_ids_list = self._generate_dynamic_stage(
                stage_prompts_text, stage_images
            )

            stage_completions_text = self.processing_class.batch_decode(
                stage_completion_ids_list, skip_special_tokens=True
            )
            stage_rule_rewards = _compute_rule_hit_rewards(
                examples=stage_inputs,
                completions_text=stage_completions_text,
            )

            next_unresolved_groups: list[dict[str, Any]] = []
            stage_rule_hit_group_count = 0
            stage_selected_group_count = 0

            for start, end, group in stage_slices:
                group_rule_hit_any = any(reward > 0 for reward in stage_rule_rewards[start:end])
                if group_rule_hit_any:
                    stage_rule_hit_group_count += 1
                if group_rule_hit_any or requested_hint_depth == group["max_hint_depth"]:
                    selected_group_outputs[group["group_index"]] = {
                        "inputs": stage_inputs[start:end],
                        "prompt_texts": stage_prompts_text[start:end],
                        "prompt_ids_list": stage_prompt_ids_list[start:end],
                        "completion_ids_list": stage_completion_ids_list[start:end],
                        "completions_text": stage_completions_text[start:end],
                        "rule_rewards": stage_rule_rewards[start:end],
                        "hint_depth": requested_hint_depth,
                        "rule_hit_any": group_rule_hit_any,
                    }
                    stage_selected_group_count += 1
                else:
                    next_unresolved_groups.append(group)

            stage_stats.append(
                {
                    "requested_hint_depth": requested_hint_depth,
                    "evaluated_group_count": len(local_unresolved_groups),
                    "rule_hit_group_count": stage_rule_hit_group_count,
                    "selected_group_count": stage_selected_group_count,
                    "remaining_group_count": len(next_unresolved_groups),
                }
            )
            if should_log_stages and local_unresolved_groups:
                print(
                    "[INFO] dynamic_hint_stage_result "
                    f"requested_depth={requested_hint_depth} "
                    f"rule_hit_groups={stage_rule_hit_group_count} "
                    f"selected_groups={stage_selected_group_count} "
                    f"remaining_groups={len(next_unresolved_groups)}"
                )
            unresolved_groups = next_unresolved_groups
            if not self._has_global_unresolved_groups(len(unresolved_groups)):
                break

        if len(selected_group_outputs) != len(prompt_groups):
            raise RuntimeError(
                f"Dynamic hint cascade selected {len(selected_group_outputs)} groups for "
                f"{len(prompt_groups)} prompt groups."
            )

        selected_inputs: list[dict[str, Union[torch.Tensor, Any]]] = []
        selected_prompt_texts: list[str] = []
        selected_prompt_ids_list: list[Any] = []
        selected_completion_ids_list: list[Any] = []
        selected_completions_text: list[str] = []
        selected_rule_rewards: list[float] = []
        selected_group_hint_depths: list[int] = []
        selected_group_rule_hits: list[bool] = []

        for group_index in range(len(prompt_groups)):
            group_output = selected_group_outputs[group_index]
            selected_inputs.extend(group_output["inputs"])
            selected_prompt_texts.extend(group_output["prompt_texts"])
            selected_prompt_ids_list.extend(group_output["prompt_ids_list"])
            selected_completion_ids_list.extend(group_output["completion_ids_list"])
            selected_completions_text.extend(group_output["completions_text"])
            selected_rule_rewards.extend(group_output["rule_rewards"])
            selected_group_hint_depths.append(group_output["hint_depth"])
            selected_group_rule_hits.append(group_output["rule_hit_any"])

        local_num_items_in_batch = sum(len(ids) for ids in selected_completion_ids_list)

        return {
            "selected_inputs": selected_inputs,
            "selected_prompt_texts": selected_prompt_texts,
            "selected_prompt_ids_list": selected_prompt_ids_list,
            "selected_completion_ids_list": selected_completion_ids_list,
            "selected_completions_text": selected_completions_text,
            "selected_rule_rewards": selected_rule_rewards,
            "selected_group_hint_depths": selected_group_hint_depths,
            "selected_group_rule_hits": selected_group_rule_hits,
            "stage_stats": stage_stats,
            "local_num_items_in_batch": local_num_items_in_batch,
        }

    def _log_dynamic_hint_metrics(
        self,
        mode: str,
        stage_stats: list[dict[str, int]],
        selected_group_hint_depths: list[int],
        selected_group_rule_hits: list[bool],
        max_hint_depth: int,
    ) -> None:
        if mode == "eval" and not self.dynamic_hint_apply_to_eval:
            return
        if not selected_group_hint_depths:
            return

        num_groups = len(selected_group_hint_depths)
        self._metrics[mode]["dynamic_hint/selected_hint_depth_mean"].append(
            sum(selected_group_hint_depths) / float(num_groups)
        )
        for hint_depth in range(max_hint_depth + 1):
            selected_count = sum(1 for depth in selected_group_hint_depths if depth == hint_depth)
            self._metrics[mode][f"dynamic_hint/selected_depth_{hint_depth}_frac"].append(
                selected_count / float(num_groups)
            )

        max_depth_miss_count = sum(
            1
            for hint_depth, rule_hit_any in zip(selected_group_hint_depths, selected_group_rule_hits)
            if hint_depth == max_hint_depth and not rule_hit_any
        )
        self._metrics[mode]["dynamic_hint/max_depth_miss_frac"].append(max_depth_miss_count / float(num_groups))

        for stage_stat in stage_stats:
            requested_hint_depth = stage_stat["requested_hint_depth"]
            evaluated_group_count = stage_stat["evaluated_group_count"]
            denominator = float(evaluated_group_count) if evaluated_group_count else 1.0
            self._metrics[mode][f"dynamic_hint/stage_{requested_hint_depth}_rule_hit_frac"].append(
                stage_stat["rule_hit_group_count"] / denominator
            )
            self._metrics[mode][f"dynamic_hint/stage_{requested_hint_depth}_remaining_frac"].append(
                stage_stat["remaining_group_count"] / denominator
            )

        self._logs.setdefault("dynamic_hint_depth", []).extend(selected_group_hint_depths)

    def _resolve_rewards_per_func(
        self,
        inputs: list[dict[str, Union[torch.Tensor, Any]]],
        prompts: list[Any],
        completions: list[Any],
        completion_ids_list: list[Any],
        selected_rule_rewards: list[float] | None = None,
    ):
        # Reuse stage-local exact rewards only for the pure rule-only path.
        # Mixed reward modes (for example rule + ndcg) still recompute their
        # full reward vector after the cascade has selected one stage per group.
        if (
            selected_rule_rewards is not None
            and len(getattr(self, "reward_func_names", [])) == 1
            and self.reward_func_names[0] == "rule_reward"
        ):
            rewards_per_func = torch.tensor(
                [[float(reward)] for reward in selected_rule_rewards],
                dtype=torch.float32,
                device=self.accelerator.device,
            )
            return self.accelerator.gather(rewards_per_func)
        return self._calculate_rewards(inputs, prompts, completions, completion_ids_list)

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        images = _extract_images_from_inputs(inputs)
        prompts: list[Any] = [example["prompt"] for example in inputs]

        max_hint_depth = self.dynamic_hint_max_depth if (mode == "train" or self.dynamic_hint_apply_to_eval) else 0
        cascade = self._run_dynamic_hint_cascade(inputs, images=images, max_hint_depth=max_hint_depth)

        inputs = cascade["selected_inputs"]
        prompts_text = cascade["selected_prompt_texts"]
        prompt_ids_list = cascade["selected_prompt_ids_list"]
        completion_ids_list = cascade["selected_completion_ids_list"]
        completions_text = cascade["selected_completions_text"]
        selected_rule_rewards = cascade["selected_rule_rewards"]
        num_items_in_batch = self._log_selected_batch_generation_metrics(mode, prompt_ids_list, completion_ids_list)

        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")

        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        with torch.no_grad():
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self.args.gradient_accumulation_steps % generate_every != 0:
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    num_images=None,
                )
            else:
                old_per_token_logps = None

            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        num_images=None,
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            num_images=None,
                        )
            else:
                ref_per_token_logps = None

        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = self._resolve_rewards_per_func(
            inputs,
            prompts,
            completions,
            completion_ids_list,
            selected_rule_rewards=selected_rule_rewards,
        )
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards

        if self.scale_rewards in ["group", "none"]:
            std_rewards = rewards.view(-1, self.num_generations).std(dim=1)
            std_rewards = std_rewards.repeat_interleave(self.num_generations, dim=0)
        elif self.scale_rewards == "batch":
            std_rewards = rewards.std().expand_as(rewards)
        else:
            raise ValueError(
                f"Invalid value for scale_rewards: {self.scale_rewards}. Must be one of 'batch', 'group', or 'none'."
            )

        is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
        if self.scale_rewards != "none":
            advantages = advantages / (std_rewards + 1e-4)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()
        advantages = advantages[process_slice]

        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_func_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        self._log_dynamic_hint_metrics(
            mode=mode,
            stage_stats=cascade["stage_stats"],
            selected_group_hint_depths=cascade["selected_group_hint_depths"],
            selected_group_rule_hits=cascade["selected_group_rule_hits"],
            max_hint_depth=max_hint_depth,
        )

        self._maybe_log_prompt_completions(prompts_text, completions_text, images=images)
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
        }
