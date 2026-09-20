"""Analyze tokenizer vocabulary coverage and token-category distribution for a checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hcgrec.token_analysis_utils import convert_to_readable_vocab, parse_token_category


def save_json(payload: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def analyze_tokenizer(
    model_path: str, verbose: bool = False
) -> tuple[dict, dict[int, str], dict[str, dict[int, str]]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    readable_vocab = convert_to_readable_vocab(tokenizer, verbose=verbose)

    categories: dict[str, dict[int, str]] = {"Whitespace": {}, "Other": {}}
    for token_id, token_name in readable_vocab.items():
        category = parse_token_category(token_name)
        categories.setdefault(category, {})[token_id] = token_name

    stats = {
        "model_path": model_path,
        "vocab_size": tokenizer.vocab_size,
        "vocab_length": len(tokenizer),
        "added_tokens": len(getattr(tokenizer, "added_tokens_encoder", {})),
        "special_tokens": tokenizer.special_tokens_map,
        "category_distribution": {category: len(tokens) for category, tokens in categories.items() if tokens},
    }
    return stats, readable_vocab, categories


def plot_category_distribution(
    categories: dict[str, dict[int, str]], output_path: Path | None = None, log_scale: bool = True
) -> None:
    import matplotlib.pyplot as plt

    counts = {category: len(tokens) for category, tokens in categories.items()}
    counts = {key: value for key, value in sorted(counts.items(), key=lambda item: item[1], reverse=True)}

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes[0].pie(counts.values(), labels=counts.keys(), autopct="%1.1f%%", startangle=90)
    axes[0].axis("equal")
    axes[0].set_title("Vocabulary Distribution by Category")

    axes[1].bar(counts.keys(), counts.values())
    if log_scale:
        axes[1].set_yscale("log")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set_title("Vocabulary Distribution by Category")

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze tokenizer vocabulary and token categories")
    parser.add_argument("--model_path", required=True, help="Model or checkpoint directory")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to write analysis artifacts. Defaults to <model_path>/tokenizer_analysis",
    )
    parser.add_argument("--plot", action="store_true", help="Generate category distribution plot")
    parser.add_argument("--verbose", action="store_true", help="Show progress bars while decoding vocabulary")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.model_path) / "tokenizer_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    stats, readable_vocab, categories = analyze_tokenizer(args.model_path, verbose=args.verbose)
    save_json(stats, output_dir / "tokenizer_stats.json")
    save_json(readable_vocab, output_dir / "vocab_readable.json")
    save_json(categories, output_dir / "vocab_by_category.json")

    if args.plot:
        plot_category_distribution(categories, output_dir / "category_distribution.png")

    print(f"Saved tokenizer analysis to: {output_dir}")


if __name__ == "__main__":
    main()
