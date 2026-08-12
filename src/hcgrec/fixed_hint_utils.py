from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from hcgrec.analyze_rl_beam_hint import extract_sid_tokens


def split_ground_truth_by_hint_depth(ground_truth: str, hint_depth: int) -> tuple[str, str]:
    tokens = extract_sid_tokens(ground_truth)
    split_index = max(int(hint_depth), 0)
    return "".join(tokens[:split_index]), "".join(tokens[split_index:])


def build_hint_text(ground_truth: str, hint_depth: int) -> str:
    hint_text, _ = split_ground_truth_by_hint_depth(ground_truth, hint_depth)
    return hint_text


def build_suffix_text(ground_truth: str, hint_depth: int) -> str:
    _, suffix_text = split_ground_truth_by_hint_depth(ground_truth, hint_depth)
    return suffix_text


def build_prompt_with_hint(example: dict[str, Any], formatter) -> str:
    prompt_text = formatter(example["prompt"])
    return f"{prompt_text}{example.get('oracle_hint_text', '')}"


def load_fixed_hint_depth_map(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _normalize_fixed_hint_index(source_index: Any) -> str:
    try:
        return str(int(source_index))
    except (TypeError, ValueError):
        return str(source_index)


def build_fixed_hint_sample_key(task: str, source_index: Any) -> str:
    task_name = str(task).strip()
    if not task_name:
        raise KeyError("Missing extra_info.task for fixed hint lookup.")
    return f"{task_name}::{_normalize_fixed_hint_index(source_index)}"


def _get_unsolved_sample_key_set(hint_map: dict[str, Any]) -> set[str]:
    cached = hint_map.get("_unsolved_sample_keys_set")
    if cached is None:
        cached = {str(sample_key) for sample_key in hint_map.get("unsolved_sample_keys", [])}
        hint_map["_unsolved_sample_keys_set"] = cached
    return cached


def _get_unsolved_index_set(hint_map: dict[str, Any]) -> set[str]:
    cached = hint_map.get("_unsolved_indices_set")
    if cached is None:
        cached = {_normalize_fixed_hint_index(index) for index in hint_map.get("unsolved_indices", [])}
        hint_map["_unsolved_indices_set"] = cached
    return cached


def apply_fixed_hint_depth_to_example(
    example: dict[str, Any],
    hint_map: dict[str, Any],
    cap_depth: int | None = None,
    unsolved_depth: int | None = None,
) -> dict[str, Any]:
    extra_info = example.get("extra_info", {})
    source_index = extra_info.get("index")
    if source_index is None:
        raise KeyError("Missing extra_info.index for fixed hint lookup.")

    if "hint_depth_by_sample_key" in hint_map:
        task_name = extra_info.get("task")
        if task_name is None:
            raise KeyError("Missing extra_info.task for fixed hint lookup.")
        key = build_fixed_hint_sample_key(task_name, source_index)
        if key not in hint_map["hint_depth_by_sample_key"]:
            raise KeyError(f"Missing fixed hint depth for sample_key={key}.")
        mapped_depth = int(hint_map["hint_depth_by_sample_key"][key])
        oracle_hint_unsolved = key in _get_unsolved_sample_key_set(hint_map)
    else:
        key = _normalize_fixed_hint_index(source_index)
        if key not in hint_map.get("hint_depth_by_index", {}):
            raise KeyError(f"Missing fixed hint depth for index={source_index}.")
        mapped_depth = int(hint_map["hint_depth_by_index"][key])
        oracle_hint_unsolved = key in _get_unsolved_index_set(hint_map)

    effective_unsolved_depth = (
        int(unsolved_depth)
        if unsolved_depth is not None
        else int(hint_map.get("default_unsolved_depth", mapped_depth))
    )
    if oracle_hint_unsolved:
        mapped_depth = effective_unsolved_depth
    if cap_depth is not None:
        mapped_depth = min(mapped_depth, int(cap_depth))

    enriched = dict(example)
    enriched["oracle_hint_depth"] = mapped_depth
    enriched["oracle_hint_text"] = build_hint_text(example["reward_model"]["ground_truth"], mapped_depth)
    enriched["oracle_hint_unsolved"] = oracle_hint_unsolved
    return enriched


def group_examples_by_hint_depth(examples: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        grouped[int(example["oracle_hint_depth"])].append(example)
    return {hint_depth: grouped[hint_depth] for hint_depth in sorted(grouped)}


def group_generation_inputs_by_hint_depth(inputs: list[dict[str, Any]]) -> dict[int, list[tuple[int, dict[str, Any]]]]:
    grouped: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, example in enumerate(inputs):
        grouped[int(example["oracle_hint_depth"])].append((index, example))
    return {hint_depth: grouped[hint_depth] for hint_depth in sorted(grouped)}
