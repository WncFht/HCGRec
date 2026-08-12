#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


DEFAULT_GREC_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare generate_metrics json files under index_train_runs")
    parser.add_argument(
        "--grec-root",
        type=Path,
        default=DEFAULT_GREC_ROOT,
        help="GRec project root",
    )
    parser.add_argument(
        "--metrics-root",
        type=Path,
        default=None,
        help="Metrics root directory (default: <grec-root>/index_train_runs)",
    )
    parser.add_argument(
        "--sort-by",
        choices=["collision_rate", "max_conflicts", "reencode_rounds"],
        default="collision_rate",
        help="Sort key",
    )
    parser.add_argument(
        "--desc",
        action="store_true",
        help="Sort in descending order",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="Only show top N rows (0 means show all)",
    )
    return parser.parse_args()


def infer_experiment_name(ckpt_path: str) -> str:
    p = Path(ckpt_path)
    parts = p.parts
    if "qwen3-embedding-4B" in parts:
        idx = parts.index("qwen3-embedding-4B")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    if len(parts) >= 3:
        return parts[-3]
    return "unknown"


def collect_metrics(metrics_root: Path) -> list[dict]:
    records: list[dict] = []
    for path in metrics_root.rglob("generate_metrics_*.json"):
        try:
            with path.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
        except Exception:
            continue

        ckpt_path = str(data.get("ckpt_path", ""))
        record = {
            "file": str(path),
            "exp": infer_experiment_name(ckpt_path),
            "datasets": "-".join(data.get("datasets", [])),
            "collision_rate": float(data.get("collision_rate", 0.0)),
            "max_conflicts": int(data.get("max_conflicts", 0)),
            "reencode_rounds": int(data.get("reencode_rounds", 0)),
            "cross_dataset": data.get("cross_dataset_collision_groups_round0", None),
            "created_at": str(data.get("created_at", "")),
        }
        records.append(record)

    return records


def print_table(records: list[dict]) -> None:
    headers = [
        "rank",
        "exp",
        "datasets",
        "collision_rate",
        "max_conflicts",
        "reencode_rounds",
        "cross_dataset",
        "created_at",
    ]

    rows = []
    for i, rec in enumerate(records, start=1):
        rows.append(
            [
                str(i),
                rec["exp"],
                rec["datasets"],
                f"{rec['collision_rate']:.6f}",
                str(rec["max_conflicts"]),
                str(rec["reencode_rounds"]),
                str(rec["cross_dataset"]),
                rec["created_at"],
            ]
        )

    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(cell))

    sep = " | "
    print(sep.join(h.ljust(col_widths[i]) for i, h in enumerate(headers)))
    print("-+-".join("-" * w for w in col_widths))
    for row in rows:
        print(sep.join(row[i].ljust(col_widths[i]) for i in range(len(headers))))


def main() -> None:
    args = parse_args()
    metrics_root = args.metrics_root or (args.grec_root / "index_train_runs")

    if not metrics_root.exists():
        raise SystemExit(f"metrics root not found: {metrics_root}")

    records = collect_metrics(metrics_root)
    if not records:
        raise SystemExit(f"no generate_metrics_*.json found under: {metrics_root}")

    records.sort(key=lambda x: x[args.sort_by], reverse=args.desc)
    if args.top > 0:
        records = records[: args.top]

    print(f"metrics_root: {metrics_root}")
    print(f"count: {len(records)}\n")
    print_table(records)


if __name__ == "__main__":
    main()
