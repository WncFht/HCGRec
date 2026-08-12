#!/usr/bin/env python3
"""Inspect category data before running preprocess_data_sft_rl.py."""

import argparse
import glob
import json
import os
from typing import Any, Optional


def read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve_category(category_dir: str, category: Optional[str]) -> str:
    if category:
        return category
    return os.path.basename(os.path.normpath(category_dir))


def find_index_candidates(category_dir: str, category: str) -> list[str]:
    patterns = [
        f"{category}.index.json",
        f"{category}.index_*.json",
        f"{category}.index*.json",
        "*.index*.json",
    ]
    found = set()
    for p in patterns:
        for path in glob.glob(os.path.join(category_dir, p)):
            if os.path.isfile(path):
                found.add(os.path.abspath(path))
    return sorted(found)


def summarize_item(item_path: str, preview: int) -> None:
    print(f"[ITEM] {item_path}")
    if not os.path.isfile(item_path):
        print("  - missing")
        return
    item_data = read_json(item_path)
    if not isinstance(item_data, dict):
        print(f"  - invalid type: {type(item_data)}")
        return
    print(f"  - item_count: {len(item_data)}")
    shown = 0
    for item_id, feat in item_data.items():
        title = ""
        if isinstance(feat, dict):
            title = str(feat.get("title", ""))
        print(f"  - sample[{item_id}] title={title[:120]}")
        shown += 1
        if shown >= preview:
            break


def summarize_inter_json(inter_path: str, preview: int) -> None:
    print(f"[INTER_JSON] {inter_path}")
    if not os.path.isfile(inter_path):
        print("  - missing")
        return
    inter_data = read_json(inter_path)
    if not isinstance(inter_data, dict):
        print(f"  - invalid type: {type(inter_data)}")
        return

    user_count = 0
    total_events = 0
    usable_samples = 0
    lens: list[int] = []
    shown = 0

    for user_id, seq in inter_data.items():
        if not isinstance(seq, list):
            continue
        seq_len = len(seq)
        user_count += 1
        total_events += seq_len
        usable_samples += max(0, seq_len - 1)
        lens.append(seq_len)
        if shown < preview:
            head = " ".join(str(x) for x in seq[:10])
            print(f"  - sample_user[{user_id}] len={seq_len} head=[{head}]")
            shown += 1

    if not lens:
        print("  - no valid user sequence found")
        return

    avg_len = total_events / user_count
    print(f"  - user_count: {user_count}")
    print(f"  - total_events: {total_events}")
    print(f"  - seq_len min/avg/max: {min(lens)}/{avg_len:.2f}/{max(lens)}")
    print(f"  - max_possible_samples (sum(len-1)): {usable_samples}")


def summarize_inter_splits(category_dir: str, category: str) -> None:
    print("[INTER_SPLITS]")
    for split in ("train", "valid", "test"):
        path = os.path.join(category_dir, f"{category}.{split}.inter")
        if not os.path.isfile(path):
            print(f"  - {split}: missing ({path})")
            continue
        with open(path, encoding="utf-8") as f:
            line_count = sum(1 for _ in f)
        sample_count = max(0, line_count - 1)
        print(f"  - {split}: {sample_count} samples ({path})")


def summarize_index(index_path: str, preview: int) -> None:
    print(f"[INDEX] {index_path}")
    if not os.path.isfile(index_path):
        print("  - missing")
        return
    idx = read_json(index_path)
    if not isinstance(idx, dict):
        print(f"  - invalid type: {type(idx)}")
        return

    total = len(idx)
    list_like = 0
    len_ge_3 = 0
    str_like = 0
    token_set = set()

    shown = 0
    for item_id, sid in idx.items():
        if isinstance(sid, list):
            list_like += 1
            if len(sid) >= 3:
                len_ge_3 += 1
            for t in sid:
                if isinstance(t, str):
                    token_set.add(t)
        elif isinstance(sid, str):
            str_like += 1

        if shown < preview:
            print(f"  - sample[{item_id}] -> {sid}")
            shown += 1

    print(f"  - entry_count: {total}")
    print(f"  - list_value_count: {list_like}")
    print(f"  - list_len>=3_count: {len_ge_3}")
    print(f"  - str_value_count: {str_like}")
    if token_set:
        print(f"  - unique_tokens_from_list_values: {len(token_set)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect input data for preprocess_data_sft_rl.py.")
    parser.add_argument(
        "--category-dir", required=True, help="Path to raw category directory, e.g. /path/to/data/Instruments"
    )
    parser.add_argument("--category", default=None, help="Category name. Defaults to basename of category-dir.")
    parser.add_argument("--index-path", default=None, help="Specific index json to inspect.")
    parser.add_argument("--preview", type=int, default=3, help="How many sample entries to print for each file.")
    args = parser.parse_args()

    category_dir = os.path.abspath(args.category_dir)
    category = resolve_category(category_dir, args.category)
    print(f"[INFO] category_dir={category_dir}")
    print(f"[INFO] category={category}")

    item_path = os.path.join(category_dir, f"{category}.item.json")
    inter_json_path = os.path.join(category_dir, f"{category}.inter.json")
    summarize_item(item_path, args.preview)
    summarize_inter_json(inter_json_path, args.preview)
    summarize_inter_splits(category_dir, category)

    candidates = find_index_candidates(category_dir, category)
    print("[INDEX_CANDIDATES]")
    if candidates:
        for p in candidates:
            print(f"  - {p}")
    else:
        print("  - none found")

    if args.index_path:
        summarize_index(os.path.abspath(args.index_path), args.preview)
    elif os.path.isfile(os.path.join(category_dir, f"{category}.index.json")):
        summarize_index(os.path.join(category_dir, f"{category}.index.json"), args.preview)
    elif candidates:
        print("[INFO] No explicit --index-path set. Use one candidate path above with --index-path for validation.")


if __name__ == "__main__":
    main()
