import argparse
import os

import numpy as np

from index.utils import load_json


def parse_args():
    parser = argparse.ArgumentParser(description="Generate *.emb-*-td.ids.json for an existing embedding file.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--root", type=str, required=True, help="Dataset directory")
    parser.add_argument("--plm_name", type=str, required=True, help="Embedding model name")
    parser.add_argument(
        "--emb_path",
        type=str,
        default=None,
        help="Optional path to existing .npy embedding; used to validate row count.",
    )
    parser.add_argument(
        "--sort_by_id",
        action="store_true",
        default=True,
        help="Sort ids to match build_embeddings.py saving order (default: true).",
    )
    parser.add_argument(
        "--no_sort_by_id",
        action="store_false",
        dest="sort_by_id",
        help="Do not sort ids; keep the order in the .item.json file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    item_json = os.path.join(args.root, f"{args.dataset}.item.json")
    item2feature = load_json(item_json)

    ids = []
    for item in item2feature:
        try:
            item_id = int(item)
        except Exception:
            item_id = item
        ids.append(item_id)

    if args.sort_by_id:
        try:
            ids = sorted(ids)
        except TypeError:
            # Mixed types (int/str) cannot be directly compared in Python3.
            # This should not happen for typical Amazon datasets (all numeric IDs).
            ids = sorted(ids, key=lambda x: str(x))

    if args.emb_path is not None and os.path.exists(args.emb_path):
        emb = np.load(args.emb_path, mmap_mode="r")
        emb = np.squeeze(emb)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        n = int(emb.shape[0])
        if len(ids) != n:
            raise ValueError(
                f"Row count mismatch: len(ids)={len(ids)} but emb has n={n} rows. "
                "If embeddings were generated with a different ordering/filtered set, "
                "please re-run build_embeddings.py to regenerate both .npy and .ids.json."
            )

    ids_path = os.path.join(args.root, f"{args.dataset}.emb-{args.plm_name}-td.ids.json")
    os.makedirs(args.root, exist_ok=True)
    import json

    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False)

    print(f"Saved item ids to {ids_path} (n={len(ids):,})")


if __name__ == "__main__":
    main()
