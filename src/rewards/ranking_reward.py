import math

from hcgrec.util import _extract_number


def _extract_reward_model_from_kwargs(reward_kwargs):
    reward_model = reward_kwargs.get("reward_model")
    if reward_model is not None:
        return reward_model
    # Backward-compatible fallback for older kwargs ordering.
    data_source, ability, reward_model, extra_info, trainer_state = reward_kwargs.values()
    return reward_model


def _extract_completion_text(completion):
    if isinstance(completion, list) and completion:
        first = completion[0]
        if isinstance(first, dict):
            return first.get("content", "")
    if isinstance(completion, dict):
        return completion.get("content", "")
    return str(completion)


def _get_full_completion_text(index: int, completion, reward_kwargs) -> str:
    completion_text = _extract_completion_text(completion)
    hint_texts = reward_kwargs.get("oracle_hint_text")
    if hint_texts is not None:
        return f"{hint_texts[index]}{completion_text}"
    return completion_text


def _prefix_match_len(pred_nums: list[int], gt_nums: list[int]) -> int:
    matched = 0
    for pred, gt in zip(pred_nums, gt_nums):
        if pred != gt:
            break
        matched += 1
    return matched


def rule_reward(prompts, completions, completion_ids, **reward_kwargs):
    reward_model = _extract_reward_model_from_kwargs(reward_kwargs)
    rewards = []
    for i, completion in enumerate(completions):
        gt_num = _extract_number(reward_model[i]["ground_truth"])
        completion_num = _extract_number(_get_full_completion_text(i, completion, reward_kwargs))
        if completion_num == gt_num:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def get_prefix_rule_reward(normalize: bool = True):
    """Prefix reward for sequence-level GRPO.

    Reward is the matched prefix length before first mismatch.
    - normalize=True: reward = matched_len / len(gt)
    - normalize=False: reward = matched_len
    """

    def prefix_rule_reward(prompts, completions, completion_ids, **reward_kwargs):
        reward_model = _extract_reward_model_from_kwargs(reward_kwargs)
        rewards = []
        for i, completion in enumerate(completions):
            gt_num = _extract_number(reward_model[i]["ground_truth"])
            completion_num = _extract_number(_get_full_completion_text(i, completion, reward_kwargs))
            if not gt_num:
                rewards.append(0.0)
                continue
            matched_len = _prefix_match_len(completion_num, gt_num)
            if normalize:
                rewards.append(float(matched_len) / float(len(gt_num)))
            else:
                rewards.append(float(matched_len))
        return rewards

    return prefix_rule_reward


def get_ndcg_rule_reward(num_beams):
    ndcg_rewards = [-1.0 / math.log2(i + 2) for i in range(num_beams)]
    ndcg_rewards = [-elm / sum(ndcg_rewards) for elm in ndcg_rewards]

    def ndcg_rule_reward(prompts, completions, completion_ids, **reward_kwargs):
        reward_model = _extract_reward_model_from_kwargs(reward_kwargs)
        repeat = num_beams
        rewards = []
        flag = False
        lis = []

        for i, completion in enumerate(completions):
            completion_num = _extract_number(_get_full_completion_text(i, completion, reward_kwargs))
            gt_num = _extract_number(reward_model[i]["ground_truth"])
            if completion_num == gt_num:
                flag = True
                lis.append(0.0)
            else:
                lis.append(ndcg_rewards[i % num_beams])
            if (i + 1) % num_beams == 0:
                if flag:
                    rewards.extend(lis)
                else:
                    rewards.extend([0.0] * repeat)
                flag = False
                lis = []
        return rewards

    return ndcg_rule_reward


def _append_rule_reward_probe(reward_funcs, reward_weights):
    if any(getattr(reward_func, "__name__", "") == "rule_reward" for reward_func in reward_funcs):
        return reward_funcs, reward_weights

    reward_funcs = list(reward_funcs) + [rule_reward]
    if reward_weights is None:
        reward_weights = [1.0] * (len(reward_funcs) - 1)
    else:
        reward_weights = list(reward_weights)
    reward_weights.append(0.0)
    return reward_funcs, reward_weights


def build_reward_setup(
    reward_mode: str,
    num_beams: int,
    prefix_reward_normalize: bool = True,
    probe_rule_with_zero_weight: bool = False,
):
    mode = reward_mode.strip().lower()
    if mode == "rule_only":
        return [rule_reward], None
    if mode == "ranking_only":
        return _append_rule_reward_probe([get_ndcg_rule_reward(num_beams)], None)
    if mode == "prefix_rule_only":
        return _append_rule_reward_probe([get_prefix_rule_reward(normalize=prefix_reward_normalize)], None)
    if mode == "prefix_only":
        prefix_func = get_prefix_rule_reward(normalize=prefix_reward_normalize)
        ndcg_func = get_ndcg_rule_reward(num_beams)
        return _append_rule_reward_probe([prefix_func, ndcg_func], None)
    if mode == "prefix_ranking":
        return [
            get_prefix_rule_reward(normalize=prefix_reward_normalize),
            rule_reward,
            get_ndcg_rule_reward(num_beams),
        ], None
    if mode == "ranking":
        return [rule_reward, get_ndcg_rule_reward(num_beams)], None
    raise ValueError(
        f"Unsupported reward_mode={reward_mode}. "
        "Use one of: ranking, rule_only, ranking_only, prefix_rule_only, prefix_only, prefix_ranking."
    )


def build_reward_funcs(reward_mode: str, num_beams: int, prefix_reward_normalize: bool = True):
    reward_funcs, _ = build_reward_setup(
        reward_mode=reward_mode,
        num_beams=num_beams,
        prefix_reward_normalize=prefix_reward_normalize,
        probe_rule_with_zero_weight=False,
    )
    return reward_funcs
