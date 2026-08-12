import argparse
import json
import os


def merge(input_path: str, output_path: str, cuda_list: str, compute_metrics: bool = True):
    """Merge multi-GPU evaluation results.

    Args:
        input_path: Input dir containing 0_result.json, 1_result.json, etc.
        output_path: Output merged JSON file path
        cuda_list: GPU list, e.g. "0,1,2,3" or "0 1 2 3"
        compute_metrics: Whether to compute and print evaluation metrics
    """
    if isinstance(cuda_list, str):
        cuda_list = [x.strip() for x in cuda_list.replace(",", " ").split() if x.strip()]

    merged = []
    for gpu_id in cuda_list:
        result_file = os.path.join(input_path, f"{gpu_id}_result.json")
        if os.path.exists(result_file):
            with open(result_file, encoding="utf-8") as f:
                part = json.load(f)
            merged.extend(part)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=4, ensure_ascii=False)

    print(f"Merge done: {len(merged)} samples -> {output_path}")

    if compute_metrics and merged:
        from hcgrec.evaluate import compute_metrics as _compute_metrics

        metrics = _compute_metrics(merged)
        if metrics:
            print("\n=== Metrics ===")
            for k in [1, 3, 5, 10, 20, 50]:
                key = f"NDCG@{k}"
                if key in metrics:
                    print(f"  {key}: {metrics[key]:.4f}")
            for k in [1, 3, 5, 10, 20, 50]:
                key = f"HR@{k}"
                if key in metrics:
                    print(f"  {key}: {metrics[key]:.4f}")


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def main():
    parser = argparse.ArgumentParser(description="Merge multi-GPU evaluation result JSON files.")
    parser.add_argument("--input_path", required=True, help="Directory containing per-GPU result JSON files.")
    parser.add_argument("--output_path", required=True, help="Merged output JSON path.")
    parser.add_argument("--cuda_list", required=True, help='GPU ids, e.g. "0,1,2,3" or "0 1 2 3".')
    parser.add_argument(
        "--compute_metrics",
        default="True",
        help="Whether to compute evaluation metrics after merging.",
    )
    args = parser.parse_args()
    merge(
        input_path=args.input_path,
        output_path=args.output_path,
        cuda_list=args.cuda_list,
        compute_metrics=parse_bool(args.compute_metrics),
    )


if __name__ == "__main__":
    main()
