#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any


SID_TOKEN_PATTERN = re.compile(r"<[^<>]+>")
NUMBER_PATTERN = re.compile(r"_(\d+)")


def extract_sid_tokens(text: str) -> list[str]:
    return SID_TOKEN_PATTERN.findall(text or "")


def combine_hint_with_suffixes(hint_text: str, suffixes: list[str]) -> list[str]:
    return [f"{hint_text}{suffix}" for suffix in suffixes]


def build_group_pattern(match_lens: list[int]) -> str:
    counts = Counter(match_lens)
    return "|".join(f"{match_len}:{counts[match_len]}" for match_len in sorted(counts))


def _extract_numbers(text: str) -> list[int]:
    return [int(match) for match in NUMBER_PATTERN.findall(text or "")]


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _prefix_match_len(pred_nums: list[int], gt_nums: list[int]) -> int:
    matched = 0
    for pred, gt in zip(pred_nums, gt_nums):
        if pred != gt:
            break
        matched += 1
    return matched


def _rule_reward(completion: str, ground_truth: str) -> float:
    return 1.0 if _extract_numbers(completion) == _extract_numbers(ground_truth) else 0.0


def _prefix_reward(completion: str, ground_truth: str) -> float:
    gt_nums = _extract_numbers(ground_truth)
    if not gt_nums:
        return 0.0
    pred_nums = _extract_numbers(completion)
    return float(_prefix_match_len(pred_nums, gt_nums)) / float(len(gt_nums))


def _build_histogram(match_lens: list[int]) -> dict[int, int]:
    counts = Counter(match_lens)
    return {match_len: counts[match_len] for match_len in sorted(counts)}


def _summarize_group_from_match_lens(
    *,
    rule_rewards: list[float],
    full_match_lens: list[int],
    gt_len: int,
    hint_depth: int,
) -> dict[str, Any]:
    prefix_rewards = [_safe_div(match_len, gt_len) for match_len in full_match_lens]
    suffix_match_lens = [max(match_len - hint_depth, 0) for match_len in full_match_lens]
    suffix_prefix_rewards = [_safe_div(match_len, gt_len) for match_len in suffix_match_lens]

    if prefix_rewards:
        prefix_max = max(prefix_rewards)
        prefix_mean = sum(prefix_rewards) / len(prefix_rewards)
        prefix_min = min(prefix_rewards)
    else:
        prefix_max = 0.0
        prefix_mean = 0.0
        prefix_min = 0.0

    if suffix_prefix_rewards:
        suffix_prefix_max = max(suffix_prefix_rewards)
        suffix_prefix_mean = sum(suffix_prefix_rewards) / len(suffix_prefix_rewards)
        suffix_prefix_min = min(suffix_prefix_rewards)
    else:
        suffix_prefix_max = 0.0
        suffix_prefix_mean = 0.0
        suffix_prefix_min = 0.0

    return {
        "rule_rewards": rule_rewards,
        "rule_hit_count": int(sum(rule_rewards)),
        "rule_hit_any": any(reward > 0 for reward in rule_rewards),
        "gt_len": gt_len,
        "hint_depth": hint_depth,
        "prefix_rewards": prefix_rewards,
        "prefix_max": prefix_max,
        "prefix_mean": prefix_mean,
        "prefix_min": prefix_min,
        "full_prefix_match_lens": full_match_lens,
        "full_prefix_hist": _build_histogram(full_match_lens),
        "full_group_pattern": build_group_pattern(full_match_lens) if full_match_lens else "",
        "suffix_prefix_match_lens": suffix_match_lens,
        "suffix_prefix_rewards": suffix_prefix_rewards,
        "suffix_prefix_hist": _build_histogram(suffix_match_lens),
        "suffix_group_pattern": build_group_pattern(suffix_match_lens) if suffix_match_lens else "",
        "suffix_prefix_max": suffix_prefix_max,
        "suffix_prefix_mean": suffix_prefix_mean,
        "suffix_prefix_min": suffix_prefix_min,
    }


def summarize_group(completions: list[str], ground_truth: str, hint_depth: int = 0) -> dict[str, Any]:
    rule_rewards = [_rule_reward(completion, ground_truth) for completion in completions]
    gt_nums = _extract_numbers(ground_truth)
    full_match_lens = [_prefix_match_len(_extract_numbers(completion), gt_nums) for completion in completions]
    return _summarize_group_from_match_lens(
        rule_rewards=rule_rewards,
        full_match_lens=full_match_lens,
        gt_len=len(gt_nums),
        hint_depth=hint_depth,
    )


def _safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def _parse_task_names(raw_task_names: str | None) -> list[str] | None:
    if raw_task_names is None:
        return None
    task_names = [task_name.strip() for task_name in str(raw_task_names).split(",") if task_name.strip()]
    return task_names or None


def _filter_samples_by_task_names(
    samples: list[dict[str, Any]],
    raw_task_names: str | None,
) -> tuple[list[dict[str, Any]], list[str] | None, list[str]]:
    requested_task_names = _parse_task_names(raw_task_names)
    available_task_names: set[str] = set()
    sample_tasks: list[str] = []

    for index, sample in enumerate(samples):
        extra_info = sample.get("extra_info")
        if not isinstance(extra_info, dict):
            raise ValueError(f"Missing extra_info.task in train sample index {index}.")
        task_name = str(extra_info.get("task", "")).strip()
        if not task_name:
            raise ValueError(f"Missing extra_info.task in train sample index {index}.")
        available_task_names.add(task_name)
        sample_tasks.append(task_name)

    sorted_available_task_names = sorted(available_task_names)
    if not requested_task_names:
        return samples, None, sorted_available_task_names

    if not available_task_names:
        raise ValueError(f"empty filtered sample set after applying tasks {requested_task_names}; available tasks: []")

    unknown_task_names = sorted(set(requested_task_names) - available_task_names)
    if unknown_task_names:
        raise ValueError(f"unknown task names: {unknown_task_names}; available tasks: {sorted_available_task_names}")

    requested_task_name_set = set(requested_task_names)
    filtered_samples = [
        sample for sample, task_name in zip(samples, sample_tasks) if task_name in requested_task_name_set
    ]
    if not filtered_samples:
        raise ValueError(
            f"empty filtered sample set after applying tasks {requested_task_names}; "
            f"available tasks: {sorted_available_task_names}"
        )
    return filtered_samples, requested_task_names, sorted_available_task_names


def _normalize_fixed_hint_index(source_index: Any) -> str:
    try:
        return str(int(source_index))
    except (TypeError, ValueError):
        return str(source_index)


def _build_fixed_hint_sample_key(task: str, source_index: Any) -> str:
    task_name = str(task).strip()
    if not task_name:
        raise KeyError("Missing extra_info.task for fixed hint export.")
    return f"{task_name}::{_normalize_fixed_hint_index(source_index)}"


def _fixed_hint_sample_key_sort_key(sample_key: str) -> tuple[str, int | str]:
    task_name, _, source_index = sample_key.partition("::")
    try:
        normalized_index: int | str = int(source_index)
    except ValueError:
        normalized_index = source_index
    return (task_name, normalized_index)


def _resolve_fixed_hint_task(row: dict[str, Any], samples: list[dict[str, Any]] | None = None) -> str:
    if row.get("task") is not None:
        task_name = str(row["task"]).strip()
        if task_name:
            return task_name

    sample_id = row.get("sample_id")
    if samples is None or sample_id is None:
        raise KeyError("Missing task metadata for fixed hint export; provide samples or row['task'].")

    try:
        sample = samples[int(sample_id)]
    except (IndexError, TypeError, ValueError) as exc:
        raise KeyError(f"Invalid sample_id={sample_id} for fixed hint export.") from exc

    task_name = str(sample.get("extra_info", {}).get("task", "")).strip()
    if not task_name:
        raise KeyError(f"Missing extra_info.task for sample_id={sample_id} during fixed hint export.")
    return task_name


def aggregate_run_summary(base_rows: list[dict[str, Any]], hinted_rows: dict[int, dict[str, Any]]) -> dict[str, Any]:
    all_prefix_rewards = [reward for row in base_rows for reward in row["group"]["prefix_rewards"]]
    miss_rows = [row for row in base_rows if not row["group"]["rule_hit_any"]]
    hinted_miss_rows = [hinted_rows[row["sample_id"]] for row in miss_rows if row["sample_id"] in hinted_rows]

    return {
        "num_samples": len(base_rows),
        "rule_hit_sample_count": sum(1 for row in base_rows if row["group"]["rule_hit_any"]),
        "rule_hit_sample_rate": _safe_mean(float(row["group"]["rule_hit_any"]) for row in base_rows),
        "rule_hit_total_count": sum(row["group"]["rule_hit_count"] for row in base_rows),
        "rule_hit_mean_count_per_sample": _safe_mean(row["group"]["rule_hit_count"] for row in base_rows),
        "prefix_per_sample_max_avg": _safe_mean(row["group"]["prefix_max"] for row in base_rows),
        "prefix_per_sample_mean_avg": _safe_mean(row["group"]["prefix_mean"] for row in base_rows),
        "prefix_per_sample_min_avg": _safe_mean(row["group"]["prefix_min"] for row in base_rows),
        "prefix_all_beams_max": max(all_prefix_rewards) if all_prefix_rewards else 0.0,
        "prefix_all_beams_mean": _safe_mean(all_prefix_rewards),
        "prefix_all_beams_min": min(all_prefix_rewards) if all_prefix_rewards else 0.0,
        "miss_subset_size": len(miss_rows),
        "hint_evaluated_miss_size": len(hinted_miss_rows),
        "hint_rule_hit_sample_count_within_miss": sum(1 for row in hinted_miss_rows if row["group"]["rule_hit_any"]),
        "hint_rule_hit_sample_rate_within_miss": _safe_mean(
            float(row["group"]["rule_hit_any"]) for row in hinted_miss_rows
        ),
        "hint_rule_hit_total_count_within_miss": sum(row["group"]["rule_hit_count"] for row in hinted_miss_rows),
        "hint_prefix_per_sample_max_avg_within_miss": _safe_mean(
            row["group"]["prefix_max"] for row in hinted_miss_rows
        ),
        "hint_prefix_per_sample_mean_avg_within_miss": _safe_mean(
            row["group"]["prefix_mean"] for row in hinted_miss_rows
        ),
        "hint_prefix_per_sample_min_avg_within_miss": _safe_mean(
            row["group"]["prefix_min"] for row in hinted_miss_rows
        ),
    }


def normalize_cached_row(row: dict[str, Any], samples: list[dict[str, Any]], hint_depth: int) -> dict[str, Any]:
    sample = samples[row["sample_id"]]
    ground_truth = sample["reward_model"]["ground_truth"]
    gt_len = len(extract_sid_tokens(ground_truth))

    group = row.get("group", {})
    rule_rewards = list(group.get("rule_rewards", []))
    if "full_prefix_match_lens" in group:
        full_match_lens = [int(match_len) for match_len in group["full_prefix_match_lens"]]
    else:
        full_match_lens = [int(round(float(reward) * gt_len)) for reward in group.get("prefix_rewards", [])]

    normalized_group = _summarize_group_from_match_lens(
        rule_rewards=rule_rewards,
        full_match_lens=full_match_lens,
        gt_len=gt_len,
        hint_depth=hint_depth,
    )

    return {
        "sample_id": row["sample_id"],
        "source_index": row.get("source_index", sample.get("extra_info", {}).get("index")),
        "task": row.get("task", sample.get("extra_info", {}).get("task")),
        "hint_text": row.get("hint_text", ""),
        "hint_depth": hint_depth,
        "ground_truth": ground_truth,
        "group": normalized_group,
    }


def convert_legacy_details_to_stage_rows(
    details_payload: dict[str, Any], beam_size: int, samples: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    beam_key = str(beam_size)
    payload_root = details_payload.get("results", details_payload)
    beam_payload = payload_root.get(beam_key, {})

    if "stages" in beam_payload:
        stage_rows = {}
        for stage_name, stage_payload in beam_payload["stages"].items():
            stage_hint_depth = 0 if stage_name == "base" else int(stage_name.split("_")[-1])
            stage_rows[stage_name] = [
                normalize_cached_row(row, samples, hint_depth=stage_hint_depth)
                for row in stage_payload.get("rows", [])
            ]
        return stage_rows

    stage_rows: dict[str, list[dict[str, Any]]] = {}
    if "base_rows" in beam_payload:
        stage_rows["base"] = [normalize_cached_row(row, samples, hint_depth=0) for row in beam_payload["base_rows"]]
    if "hinted_rows" in beam_payload:
        hinted_rows = beam_payload["hinted_rows"]
        hinted_iter = hinted_rows.values() if isinstance(hinted_rows, dict) else hinted_rows
        stage_rows["hint_1"] = [normalize_cached_row(row, samples, hint_depth=1) for row in hinted_iter]
        stage_rows["hint_1"].sort(key=lambda row: row["sample_id"])
    return stage_rows


def aggregate_stage_summary(
    *,
    rows: list[dict[str, Any]],
    total_samples: int,
    input_subset_size: int,
    previous_cumulative_hit_count: int,
    hint_depth: int,
    beam_size: int,
) -> dict[str, Any]:
    full_hist: Counter[int] = Counter()
    suffix_hist: Counter[int] = Counter()
    full_patterns: Counter[str] = Counter()
    suffix_patterns: Counter[str] = Counter()

    stage_rule_hit_sample_count = 0
    stage_rule_hit_total_count = 0
    for row in rows:
        group = row["group"]
        stage_rule_hit_sample_count += int(group["rule_hit_any"])
        stage_rule_hit_total_count += group["rule_hit_count"]
        full_hist.update(group.get("full_prefix_match_lens", []))
        suffix_hist.update(group.get("suffix_prefix_match_lens", []))
        if group.get("full_group_pattern"):
            full_patterns[group["full_group_pattern"]] += 1
        if group.get("suffix_group_pattern"):
            suffix_patterns[group["suffix_group_pattern"]] += 1

    remaining_subset_size = max(input_subset_size - stage_rule_hit_sample_count, 0)
    cumulative_rule_hit_sample_count = previous_cumulative_hit_count + stage_rule_hit_sample_count

    return {
        "beam_size": beam_size,
        "hint_depth": hint_depth,
        "input_subset_size": input_subset_size,
        "evaluated_subset_size": len(rows),
        "stage_rule_hit_sample_count": stage_rule_hit_sample_count,
        "stage_rule_hit_sample_rate_within_input": _safe_div(stage_rule_hit_sample_count, input_subset_size),
        "stage_rule_hit_total_count": stage_rule_hit_total_count,
        "remaining_subset_size": remaining_subset_size,
        "remaining_subset_rate_within_input": _safe_div(remaining_subset_size, input_subset_size),
        "cumulative_rule_hit_sample_count": cumulative_rule_hit_sample_count,
        "cumulative_rule_hit_sample_rate": _safe_div(cumulative_rule_hit_sample_count, total_samples),
        "full_prefix_match_hist_all_beams": {match_len: full_hist[match_len] for match_len in sorted(full_hist)},
        "suffix_prefix_match_hist_all_beams": {match_len: suffix_hist[match_len] for match_len in sorted(suffix_hist)},
        "group_pattern_counts_full": {pattern: full_patterns[pattern] for pattern in sorted(full_patterns)},
        "group_pattern_counts_suffix": {pattern: suffix_patterns[pattern] for pattern in sorted(suffix_patterns)},
    }


def discover_reusable_cache(
    *,
    cache_dir: str | Path,
    model_path: str,
    data_dir: str,
    index_path: str,
    beam_sizes: list[int],
    num_samples: int,
    offset: int,
    max_samples: int | None,
) -> dict[str, Path] | None:
    if offset != 0 or max_samples is not None:
        return None

    cache_root = Path(cache_dir)
    if not cache_root.exists():
        return None

    candidates = sorted(cache_root.glob("*_summary.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for summary_path in candidates:
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if not isinstance(summary, dict):
            continue
        if summary.get("model_path") != model_path:
            continue
        if summary.get("data_dir") != data_dir:
            continue
        if summary.get("index_path") != index_path:
            continue
        if int(summary.get("num_samples", -1)) != int(num_samples):
            continue
        if [int(beam) for beam in summary.get("beam_sizes", [])] != beam_sizes:
            continue
        if int(summary.get("hint_depth", 0)) < 1:
            continue

        details_path = summary_path.with_name(summary_path.name.replace("_summary.json", "_details.json"))
        if not details_path.exists():
            continue
        return {"summary_path": summary_path, "details_path": details_path}

    return None


def build_fixed_hint_depth_map_from_details(
    details_payload: dict[str, Any],
    beam_size: int,
    unsolved_depth: int = 3,
    samples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    beam_key = str(beam_size)
    payload_root = details_payload.get("results", details_payload)
    beam_payload = payload_root.get(beam_key, {})
    if not beam_payload:
        raise KeyError(f"Missing beam_size={beam_size} in details payload.")

    stage_payload = beam_payload.get("stages")
    if stage_payload is None:
        stage_payload = {
            "base": {"rows": beam_payload.get("base_rows", [])},
            "hint_1": {
                "rows": list((beam_payload.get("hinted_rows") or {}).values())
                if isinstance(beam_payload.get("hinted_rows"), dict)
                else beam_payload.get("hinted_rows", [])
            },
        }

    hint_depth_by_sample_key: dict[str, int] = {}
    unsolved_sample_keys: list[str] = []

    for stage_name, stage_info in sorted(stage_payload.items(), key=lambda item: _stage_sort_key(item[0])):
        hint_depth = 0 if stage_name == "base" else int(stage_name.split("_")[-1])
        for row in stage_info.get("rows", []):
            source_index = row.get("source_index")
            if source_index is None:
                continue
            sample_key = _build_fixed_hint_sample_key(
                _resolve_fixed_hint_task(row, samples=samples),
                source_index,
            )
            if row.get("group", {}).get("rule_hit_any") and sample_key not in hint_depth_by_sample_key:
                hint_depth_by_sample_key[sample_key] = hint_depth

    final_stage_name = max(stage_payload, key=_stage_sort_key)
    final_stage_hint_depth = 0 if final_stage_name == "base" else int(final_stage_name.split("_")[-1])
    for row in stage_payload[final_stage_name].get("rows", []):
        source_index = row.get("source_index")
        if source_index is None:
            continue
        sample_key = _build_fixed_hint_sample_key(
            _resolve_fixed_hint_task(row, samples=samples),
            source_index,
        )
        if sample_key not in hint_depth_by_sample_key:
            hint_depth_by_sample_key[sample_key] = unsolved_depth
            unsolved_sample_keys.append(sample_key)

    return {
        "sample_key_type": "extra_info.task+index",
        "beam_size": beam_size,
        "default_unsolved_depth": unsolved_depth,
        "max_available_stage_depth": final_stage_hint_depth,
        "hint_depth_by_sample_key": dict(
            sorted(hint_depth_by_sample_key.items(), key=lambda item: _fixed_hint_sample_key_sort_key(item[0]))
        ),
        "unsolved_sample_keys": sorted(unsolved_sample_keys, key=_fixed_hint_sample_key_sort_key),
    }


def _load_train_samples(data_dir: str, max_samples: int | None = None, offset: int = 0) -> list[dict[str, Any]]:
    train_path = Path(data_dir) / "train.json"
    with train_path.open(encoding="utf-8") as file:
        samples = json.load(file)
    if offset:
        samples = samples[offset:]
    if max_samples is not None:
        samples = samples[:max_samples]
    return samples


def _load_tokenizer_and_model(
    model_path: str,
    data_dir: str,
    add_tokens_path: str | None,
    trust_remote_code: bool,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    tokens_path = Path(add_tokens_path) if add_tokens_path else Path(data_dir).parent / "new_tokens.json"
    added_tokens = 0
    if tokens_path.exists():
        with tokens_path.open(encoding="utf-8") as file:
            new_tokens = json.load(file)
        added_tokens = tokenizer.add_tokens(new_tokens)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    if added_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))
    model.eval()
    model.config.pad_token_id = model.config.eos_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    return tokenizer, model, str(tokens_path), added_tokens


def _build_processor_bundle(index_path: str, tokenizer, num_beams: int, sid_levels: int):
    from hcgrec.evaluate import build_trie_from_index, create_prefix_allowed_tokens_fn

    trie, prompt_suffix_ids, prefix_index = build_trie_from_index(
        index_path,
        tokenizer,
        prefix=None,
        sid_levels=sid_levels,
    )
    return {
        "num_beams": num_beams,
        "prefix_index": prefix_index,
        "prompt_suffix_ids": prompt_suffix_ids,
        "prefix_allowed_tokens_fn": create_prefix_allowed_tokens_fn(trie, prompt_suffix_ids),
        "eos_token_id": tokenizer.eos_token_id,
    }


def _build_logits_processor(bundle: dict[str, Any], seed_token_count: int):
    from transformers import LogitsProcessorList

    from hcgrec.logit_processor import ConstrainedLogitsProcessor

    class SeededConstrainedLogitsProcessor(ConstrainedLogitsProcessor):
        def __init__(self, *args, initial_count: int = 0, **kwargs):
            super().__init__(*args, **kwargs)
            self.initial_count = initial_count
            self.count = initial_count

        def reset(self) -> None:
            self.count = self.initial_count

    return LogitsProcessorList(
        [
            SeededConstrainedLogitsProcessor(
                prefix_allowed_tokens_fn=bundle["prefix_allowed_tokens_fn"],
                num_beams=bundle["num_beams"],
                prefix_index=bundle["prefix_index"],
                prefix_ids=bundle["prompt_suffix_ids"],
                eos_token_id=bundle["eos_token_id"],
                initial_count=seed_token_count,
            )
        ]
    )


def _format_prompt(tokenizer, prompt_messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(prompt_messages, add_generation_prompt=True, tokenize=False)


def _generate_prompt_groups(
    model,
    tokenizer,
    prompt_texts: list[str],
    logits_processor,
    num_beams: int,
    max_prompt_length: int,
    max_new_tokens: int,
    repetition_penalty: float,
):
    import torch

    encodings = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
    )
    input_ids = encodings["input_ids"].to(model.device)
    attention_mask = encodings["attention_mask"].to(model.device)
    padded_prompt_length = input_ids.shape[1]

    for processor in logits_processor:
        if hasattr(processor, "reset"):
            processor.reset()
        elif hasattr(processor, "count"):
            processor.count = 0

    with torch.no_grad():
        generated = model.generate(
            input_ids,
            attention_mask=attention_mask,
            logits_processor=logits_processor,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            repetition_penalty=repetition_penalty,
            temperature=1.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )

    grouped_outputs: list[list[str]] = []
    for sample_idx in range(len(prompt_texts)):
        sample_sequences = generated.sequences[sample_idx * num_beams : (sample_idx + 1) * num_beams]
        sample_completions = tokenizer.batch_decode(
            sample_sequences[:, padded_prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        grouped_outputs.append(sample_completions)
    return grouped_outputs


def _batched(items: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _prepare_base_records(samples: list[dict[str, Any]], tokenizer) -> list[dict[str, Any]]:
    records = []
    for sample_id, sample in enumerate(samples):
        records.append(
            {
                "sample_id": sample_id,
                "source_index": sample.get("extra_info", {}).get("index"),
                "task": sample.get("extra_info", {}).get("task"),
                "ground_truth": sample["reward_model"]["ground_truth"],
                "prompt_text": _format_prompt(tokenizer, sample["prompt"]),
                "hint_text": "",
                "seed_token_count": 0,
            }
        )
    return records


def _prepare_hint_records(
    input_rows: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    tokenizer,
    hint_depth: int,
) -> list[dict[str, Any]]:
    records = []
    for row in input_rows:
        if row["group"]["rule_hit_any"]:
            continue
        sample = samples[row["sample_id"]]
        sid_tokens = extract_sid_tokens(sample["reward_model"]["ground_truth"])
        if len(sid_tokens) < hint_depth:
            continue
        hint_text = "".join(sid_tokens[:hint_depth])
        seed_token_count = len(tokenizer(hint_text, add_special_tokens=False).input_ids)
        records.append(
            {
                "sample_id": row["sample_id"],
                "source_index": sample.get("extra_info", {}).get("index"),
                "task": sample.get("extra_info", {}).get("task"),
                "ground_truth": sample["reward_model"]["ground_truth"],
                "prompt_text": f"{_format_prompt(tokenizer, sample['prompt'])}{hint_text}",
                "hint_text": hint_text,
                "seed_token_count": seed_token_count,
            }
        )
    return records


def _run_records(
    records: list[dict[str, Any]],
    model,
    tokenizer,
    bundle: dict[str, Any],
    batch_size: int,
    max_prompt_length: int,
    max_new_tokens: int,
    repetition_penalty: float,
    progress_desc: str,
) -> list[dict[str, Any]]:
    from tqdm import tqdm

    results: list[dict[str, Any]] = []
    grouped_records: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped_records[record["seed_token_count"]].append(record)

    for seed_token_count, seed_records in grouped_records.items():
        logits_processor = _build_logits_processor(bundle, seed_token_count=seed_token_count)
        total_batches = (len(seed_records) + batch_size - 1) // batch_size
        for batch in tqdm(
            _batched(seed_records, batch_size),
            total=total_batches,
            desc=f"{progress_desc}-seed{seed_token_count}",
        ):
            prompt_texts = [record["prompt_text"] for record in batch]
            grouped_suffixes = _generate_prompt_groups(
                model=model,
                tokenizer=tokenizer,
                prompt_texts=prompt_texts,
                logits_processor=logits_processor,
                num_beams=bundle["num_beams"],
                max_prompt_length=max_prompt_length,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
            )
            for record, suffixes in zip(batch, grouped_suffixes):
                completions = combine_hint_with_suffixes(record["hint_text"], suffixes)
                actual_hint_depth = len(extract_sid_tokens(record["hint_text"]))
                results.append(
                    {
                        "sample_id": record["sample_id"],
                        "source_index": record["source_index"],
                        "task": record.get("task"),
                        "hint_text": record["hint_text"],
                        "hint_depth": actual_hint_depth,
                        "group": summarize_group(
                            completions,
                            record["ground_truth"],
                            hint_depth=actual_hint_depth,
                        ),
                    }
                )
    results.sort(key=lambda row: row["sample_id"])
    return results


def _stage_sort_key(stage_name: str) -> tuple[int, int]:
    if stage_name == "base":
        return (0, 0)
    return (1, int(stage_name.split("_")[-1]))


def _auto_max_hint_depth(samples: list[dict[str, Any]]) -> int:
    max_sid_len = 0
    for sample in samples:
        max_sid_len = max(max_sid_len, len(extract_sid_tokens(sample["reward_model"]["ground_truth"])))
    return max(max_sid_len - 1, 0)


def _load_cached_stage_rows(
    *,
    samples: list[dict[str, Any]],
    beam_sizes: list[int],
    reuse_details_path: str | None,
    reuse_summary_path: str | None,
    cache_dir: str | None,
    model_path: str,
    data_dir: str,
    index_path: str,
    num_samples: int,
    offset: int,
    max_samples: int | None,
) -> tuple[dict[int, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    cache_meta: dict[str, Any] = {"reused": False}
    details_path: Path | None = Path(reuse_details_path) if reuse_details_path else None
    summary_path: Path | None = Path(reuse_summary_path) if reuse_summary_path else None

    if details_path is None:
        cache_candidate = discover_reusable_cache(
            cache_dir=cache_dir or "temp/rl_beam_hint",
            model_path=model_path,
            data_dir=data_dir,
            index_path=index_path,
            beam_sizes=beam_sizes,
            num_samples=num_samples,
            offset=offset,
            max_samples=max_samples,
        )
        if cache_candidate is not None:
            summary_path = cache_candidate["summary_path"]
            details_path = cache_candidate["details_path"]

    if details_path is None or not details_path.exists():
        return {}, cache_meta

    details_payload = json.loads(details_path.read_text(encoding="utf-8"))
    stage_rows_by_beam = {
        beam_size: convert_legacy_details_to_stage_rows(details_payload, beam_size=beam_size, samples=samples)
        for beam_size in beam_sizes
    }
    cache_meta = {
        "reused": True,
        "details_path": str(details_path),
        "summary_path": str(summary_path) if summary_path is not None else None,
        "stages_by_beam": {
            str(beam_size): sorted(stage_rows_by_beam[beam_size], key=_stage_sort_key) for beam_size in beam_sizes
        },
    }
    return stage_rows_by_beam, cache_meta


def _run_stage_records(
    *,
    stage_name: str,
    input_rows: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    tokenizer,
    model,
    bundle: dict[str, Any],
    batch_size: int,
    max_prompt_length: int,
    max_new_tokens: int,
    repetition_penalty: float,
    hint_depth: int,
) -> list[dict[str, Any]]:
    if stage_name == "base":
        records = _prepare_base_records(samples, tokenizer)
    else:
        records = _prepare_hint_records(input_rows, samples, tokenizer, hint_depth=hint_depth)
    if not records:
        return []
    return _run_records(
        records=records,
        model=model,
        tokenizer=tokenizer,
        bundle=bundle,
        batch_size=batch_size,
        max_prompt_length=max_prompt_length,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        progress_desc=f"beam{bundle['num_beams']}-{stage_name}",
    )


def analyze_beam_size_cascade(
    samples: list[dict[str, Any]],
    model,
    tokenizer,
    index_path: str,
    beam_size: int,
    max_hint_depth: int,
    batch_size: int,
    max_prompt_length: int,
    max_new_tokens: int,
    repetition_penalty: float,
    sid_levels: int,
    cached_stage_rows: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    stage_rows = dict(cached_stage_rows or {})
    required_stage_names = ["base"] + [f"hint_{hint_depth}" for hint_depth in range(1, max_hint_depth + 1)]
    missing_stage_names = [stage_name for stage_name in required_stage_names if stage_name not in stage_rows]

    bundle = None
    if missing_stage_names:
        if tokenizer is None or model is None:
            raise ValueError("Tokenizer and model are required when cached stages are incomplete.")
        bundle = _build_processor_bundle(
            index_path=index_path,
            tokenizer=tokenizer,
            num_beams=beam_size,
            sid_levels=sid_levels,
        )

    if "base" not in stage_rows:
        stage_rows["base"] = _run_stage_records(
            stage_name="base",
            input_rows=[],
            samples=samples,
            tokenizer=tokenizer,
            model=model,
            bundle=bundle,
            batch_size=batch_size,
            max_prompt_length=max_prompt_length,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            hint_depth=0,
        )

    previous_stage_rows = stage_rows["base"]
    for hint_depth in range(1, max_hint_depth + 1):
        stage_name = f"hint_{hint_depth}"
        if stage_name in stage_rows:
            previous_stage_rows = stage_rows[stage_name]
            continue
        input_subset_size = sum(1 for row in previous_stage_rows if not row["group"]["rule_hit_any"])
        if input_subset_size == 0:
            break
        stage_rows[stage_name] = _run_stage_records(
            stage_name=stage_name,
            input_rows=previous_stage_rows,
            samples=samples,
            tokenizer=tokenizer,
            model=model,
            bundle=bundle,
            batch_size=batch_size,
            max_prompt_length=max_prompt_length,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            hint_depth=hint_depth,
        )
        previous_stage_rows = stage_rows[stage_name]

    stage_summaries: dict[str, Any] = {}
    previous_cumulative_hit_count = 0
    input_subset_size = len(samples)
    for stage_name in sorted(stage_rows, key=_stage_sort_key):
        hint_depth = 0 if stage_name == "base" else int(stage_name.split("_")[-1])
        stage_summary = aggregate_stage_summary(
            rows=stage_rows[stage_name],
            total_samples=len(samples),
            input_subset_size=input_subset_size,
            previous_cumulative_hit_count=previous_cumulative_hit_count,
            hint_depth=hint_depth,
            beam_size=beam_size,
        )
        stage_summaries[stage_name] = stage_summary
        previous_cumulative_hit_count = stage_summary["cumulative_rule_hit_sample_count"]
        input_subset_size = stage_summary["remaining_subset_size"]

    final_stage_name = max(stage_summaries, key=_stage_sort_key)
    final_stage_summary = stage_summaries[final_stage_name]
    summary = {
        "beam_size": beam_size,
        "max_hint_depth": max_hint_depth,
        "stages": stage_summaries,
        "cumulative": {
            "final_stage": final_stage_name,
            "final_rule_hit_sample_count": final_stage_summary["cumulative_rule_hit_sample_count"],
            "final_rule_hit_sample_rate": final_stage_summary["cumulative_rule_hit_sample_rate"],
            "final_remaining_subset_size": final_stage_summary["remaining_subset_size"],
            "final_remaining_subset_rate": _safe_div(final_stage_summary["remaining_subset_size"], len(samples)),
        },
    }
    return summary, stage_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze rule/prefix rewards on RL train set with and without SID hints."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--summary-path", default="temp/rl_beam_hint_summary.json")
    parser.add_argument("--details-path")
    parser.add_argument("--add-tokens-path")
    parser.add_argument("--beam-sizes", default="16")
    parser.add_argument("--hint-depth", type=int, default=1)
    parser.add_argument("--max-hint-depth", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--sid-levels", type=int, default=-1)
    parser.add_argument("--cache-dir", default="temp/rl_beam_hint")
    parser.add_argument("--reuse-summary-path")
    parser.add_argument("--reuse-details-path")
    parser.add_argument("--disable-cache-reuse", action="store_true")
    parser.add_argument("--task-names")
    parser.add_argument("--export-fixed-hint-depth-map-path")
    parser.add_argument("--export-fixed-hint-beam-size", type=int, default=16)
    parser.add_argument("--export-fixed-hint-unsolved-depth", type=int, default=3)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    beam_sizes = [int(part.strip()) for part in args.beam_sizes.split(",") if part.strip()]
    samples = _load_train_samples(args.data_dir, max_samples=args.max_samples, offset=args.offset)
    samples, requested_task_names, available_task_names = _filter_samples_by_task_names(samples, args.task_names)
    print(
        f"Task filter: requested={requested_task_names}, available={available_task_names}, filtered_sample_count={len(samples)}"
    )
    max_hint_depth = args.max_hint_depth if args.max_hint_depth is not None else args.hint_depth
    if max_hint_depth < 0:
        max_hint_depth = _auto_max_hint_depth(samples)

    cached_stage_rows_by_beam: dict[int, dict[str, list[dict[str, Any]]]] = {}
    cache_meta: dict[str, Any] = {"reused": False}
    if not args.disable_cache_reuse:
        cached_stage_rows_by_beam, cache_meta = _load_cached_stage_rows(
            samples=samples,
            beam_sizes=beam_sizes,
            reuse_details_path=args.reuse_details_path,
            reuse_summary_path=args.reuse_summary_path,
            cache_dir=args.cache_dir,
            model_path=args.model_path,
            data_dir=args.data_dir,
            index_path=args.index_path,
            num_samples=len(samples),
            offset=args.offset,
            max_samples=args.max_samples,
        )

    needs_generation = False
    for beam_size in beam_sizes:
        cached_stages = cached_stage_rows_by_beam.get(beam_size, {})
        if "base" not in cached_stages:
            needs_generation = True
            break
        for hint_depth in range(1, max_hint_depth + 1):
            if f"hint_{hint_depth}" not in cached_stages:
                needs_generation = True
                break
        if needs_generation:
            break

    tokenizer = None
    model = None
    tokens_path = str(
        Path(args.add_tokens_path) if args.add_tokens_path else Path(args.data_dir).parent / "new_tokens.json"
    )
    added_tokens = 0
    if needs_generation:
        tokenizer, model, tokens_path, added_tokens = _load_tokenizer_and_model(
            model_path=args.model_path,
            data_dir=args.data_dir,
            add_tokens_path=args.add_tokens_path,
            trust_remote_code=args.trust_remote_code,
        )

    summary = {
        "model_path": args.model_path,
        "data_dir": args.data_dir,
        "index_path": args.index_path,
        "num_samples": len(samples),
        "beam_sizes": beam_sizes,
        "hint_depth": args.hint_depth,
        "max_hint_depth": max_hint_depth,
        "task_names": requested_task_names,
        "available_task_names": available_task_names,
        "new_tokens_path": tokens_path,
        "new_tokens_added": added_tokens,
        "cache": cache_meta,
        "results": {},
    }
    detail_payload: dict[str, Any] = {
        "metadata": {
            "model_path": args.model_path,
            "data_dir": args.data_dir,
            "index_path": args.index_path,
            "num_samples": len(samples),
            "beam_sizes": beam_sizes,
            "max_hint_depth": max_hint_depth,
            "task_names": requested_task_names,
            "available_task_names": available_task_names,
            "cache": cache_meta,
        },
        "results": {},
    }

    for beam_size in beam_sizes:
        if needs_generation and (tokenizer is None or model is None):
            tokenizer, model, _, _ = _load_tokenizer_and_model(
                model_path=args.model_path,
                data_dir=args.data_dir,
                add_tokens_path=args.add_tokens_path,
                trust_remote_code=args.trust_remote_code,
            )
        run_summary, stage_rows = analyze_beam_size_cascade(
            samples=samples,
            model=model,
            tokenizer=tokenizer,
            index_path=args.index_path,
            beam_size=beam_size,
            max_hint_depth=max_hint_depth,
            batch_size=args.batch_size,
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
            sid_levels=args.sid_levels,
            cached_stage_rows=cached_stage_rows_by_beam.get(beam_size),
        )
        summary["results"][str(beam_size)] = run_summary
        if args.details_path:
            detail_payload["results"][str(beam_size)] = {
                "stages": {
                    stage_name: {
                        "rows": rows,
                        "reused": stage_name in cached_stage_rows_by_beam.get(beam_size, {}),
                    }
                    for stage_name, rows in sorted(stage_rows.items(), key=lambda item: _stage_sort_key(item[0]))
                }
            }

    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Summary saved to: {summary_path}")

    if args.details_path:
        details_path = Path(args.details_path)
        details_path.parent.mkdir(parents=True, exist_ok=True)
        details_path.write_text(json.dumps(detail_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Details saved to: {details_path}")

    if args.export_fixed_hint_depth_map_path:
        export_payload = (
            detail_payload
            if detail_payload.get("results")
            else {
                "results": {
                    str(beam_size): {"stages": {stage_name: {"rows": rows} for stage_name, rows in stage_rows.items()}}
                    for beam_size, stage_rows in (
                        (
                            beam_size,
                            analyze_beam_size_cascade(
                                samples=samples,
                                model=model,
                                tokenizer=tokenizer,
                                index_path=args.index_path,
                                beam_size=beam_size,
                                max_hint_depth=max_hint_depth,
                                batch_size=args.batch_size,
                                max_prompt_length=args.max_prompt_length,
                                max_new_tokens=args.max_new_tokens,
                                repetition_penalty=args.repetition_penalty,
                                sid_levels=args.sid_levels,
                                cached_stage_rows=cached_stage_rows_by_beam.get(beam_size),
                            )[1],
                        )
                        for beam_size in beam_sizes
                    )
                }
            }
        )
        fixed_hint_map = build_fixed_hint_depth_map_from_details(
            export_payload,
            beam_size=args.export_fixed_hint_beam_size,
            unsolved_depth=args.export_fixed_hint_unsolved_depth,
            samples=samples,
        )
        export_path = Path(args.export_fixed_hint_depth_map_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(json.dumps(fixed_hint_map, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Fixed hint depth map saved to: {export_path}")


if __name__ == "__main__":
    main()
