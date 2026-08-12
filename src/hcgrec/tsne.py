from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from hcgrec.token_analysis_utils import (
    ORIGINAL_TOKEN_PREFIX,
    build_label_distribution,
    convert_to_readable_vocab,
    get_token_display_name,
    parse_token_category,
)


if TYPE_CHECKING:
    import torch

try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


DEFAULT_EMBEDDING_KEYS = [
    "model.embed_tokens.weight",
    "model.language_model.embed_tokens.weight",
    "language_model.model.embed_tokens.weight",
    "model.model.language_model.embed_tokens.weight",
    "language_model.embed_tokens.weight",
    "embed_tokens.weight",
    "model.model.embed_tokens.weight",
    "transformer.wte.weight",
    "transformer.embed_tokens.weight",
    "embeddings.weight",
    "shared.weight",
    "lm_head.weight",
    "model.embed_tokens",
    "embed_tokens",
]

DEFAULT_ORIGINAL_TOKEN_FILTERS = ["English", "Chinese", "Japanese", "Korean"]
EXCLUDED_TOKEN_CATEGORIES = {"Other", "Invalid UTF-8", "Whitespace", "Special Token"}


@dataclass
class EmbeddingInfo:
    embeddings: np.ndarray
    token_names: list[str]
    token_ids: list[int]
    token_categories: list[str]
    tokenizer_stats: dict


def analyze_tokenizer_stats(tokenizer, model_path: str) -> dict:
    stats = {
        "model_path": model_path,
        "vocab_size": tokenizer.vocab_size,
        "added_tokens": len(getattr(tokenizer, "added_tokens_encoder", {})),
        "special_tokens": tokenizer.special_tokens_map,
        "vocab_length": len(tokenizer),
    }

    print("\n=== Tokenizer Statistics ===")
    print(f"Model: {model_path}")
    print(f"Vocabulary size: {stats['vocab_size']:,}")
    print(f"Added tokens: {stats['added_tokens']:,}")
    print(f"Special tokens: {stats['special_tokens']}")
    print(f"Total vocab length: {stats['vocab_length']:,}")

    return stats


def _maybe_cast_tensor(tensor: torch.Tensor) -> torch.Tensor:
    import torch

    return tensor.float() if tensor.dtype in {torch.bfloat16, torch.float16} else tensor


def _load_tensor_from_safetensor(file_path: str, key: str | None) -> torch.Tensor | None:
    from safetensors.torch import safe_open

    with safe_open(file_path, framework="pt") as handle:
        keys = list(handle.keys())
        if key:
            if key in keys:
                return _maybe_cast_tensor(handle.get_tensor(key))
            return None

        for candidate in DEFAULT_EMBEDDING_KEYS:
            if candidate in keys:
                print(f"Found embedding layer at: {candidate}")
                return _maybe_cast_tensor(handle.get_tensor(candidate))

        embed_keys = [name for name in keys if "embed" in name.lower() and "weight" in name.lower()]
        if embed_keys:
            print(f"Using first embedding-like layer found: {embed_keys[0]}")
            return _maybe_cast_tensor(handle.get_tensor(embed_keys[0]))
    return None


def _load_tensor_from_bin(file_path: str, key: str | None) -> torch.Tensor | None:
    import torch

    state_dict = torch.load(file_path, map_location="cpu")
    keys_to_try = [key] if key else []
    keys_to_try.extend(DEFAULT_EMBEDDING_KEYS)
    for candidate in keys_to_try:
        if candidate and candidate in state_dict:
            if candidate != key:
                print(f"Found embedding layer at: {candidate}")
            return _maybe_cast_tensor(state_dict[candidate])
    return None


def load_any_embeddings(model_dir: str, layer_name: str | None = None) -> torch.Tensor:
    model_path = Path(model_dir)
    candidate_files = sorted(model_path.glob("*.safetensors")) + sorted(model_path.glob("*.bin"))
    if not candidate_files:
        raise FileNotFoundError(f"No .safetensors or .bin files found in {model_dir}")

    for file_path in candidate_files:
        tensor = (
            _load_tensor_from_safetensor(str(file_path), layer_name)
            if file_path.suffix == ".safetensors"
            else _load_tensor_from_bin(str(file_path), layer_name)
        )
        if tensor is not None:
            print(f"Loaded embeddings from: {file_path}")
            return tensor

    target = layer_name or "auto-detected embedding layer"
    raise RuntimeError(f"Could not locate {target} in {model_dir}")


def resize_embeddings_if_needed(embeddings: torch.Tensor, tokenizer) -> torch.Tensor:
    import torch

    vocab_size = len(tokenizer)
    if embeddings.shape[0] >= vocab_size:
        return embeddings

    resized = torch.zeros((vocab_size, embeddings.shape[1]), dtype=embeddings.dtype)
    resized[: embeddings.shape[0]] = embeddings
    print(f"Resized embeddings from {embeddings.shape[0]} to {vocab_size} rows")
    return resized


def collect_original_token_ids(
    tokenizer, new_token_ids: set[int], readable_vocab: dict[int, str], filter_languages: list[str] | None
) -> list[int]:
    from tqdm import tqdm

    original_ids = [token_id for token_id in range(len(tokenizer)) if token_id not in new_token_ids]
    target_categories = filter_languages or DEFAULT_ORIGINAL_TOKEN_FILTERS

    filtered_ids = []
    for token_id in tqdm(original_ids, desc="Filtering original tokens by category"):
        category = parse_token_category(readable_vocab[token_id])
        normalized_category = category.removeprefix("Original-")
        if normalized_category in EXCLUDED_TOKEN_CATEGORIES:
            continue
        if normalized_category in target_categories:
            filtered_ids.append(token_id)
    return filtered_ids


def load_embeddings_from_model(
    model_path: str,
    layer_name: str | None = None,
    sample_original_tokens: int = 1000,
    analyze_languages: bool = True,
    filter_languages: list[str] | None = None,
    seed: int = 42,
) -> EmbeddingInfo:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    stats = analyze_tokenizer_stats(tokenizer, model_path)

    embeddings = resize_embeddings_if_needed(load_any_embeddings(model_path, layer_name), tokenizer)
    readable_vocab = convert_to_readable_vocab(tokenizer, verbose=analyze_languages)

    new_token_ids = {
        token_id
        for token_id in getattr(tokenizer, "added_tokens_decoder", {}).keys()
        if parse_token_category(readable_vocab[token_id]) not in EXCLUDED_TOKEN_CATEGORIES
    }
    filtered_original_ids = collect_original_token_ids(tokenizer, new_token_ids, readable_vocab, filter_languages)

    random.seed(seed)
    sample_count = min(sample_original_tokens, len(filtered_original_ids))
    sampled_original_ids = random.sample(filtered_original_ids, sample_count) if sample_count else []

    print(f"Selected {len(new_token_ids)} added tokens and {len(sampled_original_ids)} original tokens")
    all_token_ids = sorted(new_token_ids) + sampled_original_ids
    valid_token_ids = [token_id for token_id in all_token_ids if token_id < embeddings.shape[0]]

    token_names = []
    token_categories = []
    for token_id in valid_token_ids:
        token_name = readable_vocab.get(token_id, get_token_display_name(tokenizer, token_id))
        if token_id in sampled_original_ids:
            token_name = f"{ORIGINAL_TOKEN_PREFIX} {token_name}"
        token_names.append(token_name)
        token_categories.append(parse_token_category(token_name) if analyze_languages else "Token")

    selected_embeddings = embeddings[valid_token_ids].cpu().numpy()
    return EmbeddingInfo(selected_embeddings, token_names, valid_token_ids, token_categories, stats)


def perform_dimension_reduction(
    embeddings: np.ndarray, method: str, args: argparse.Namespace
) -> tuple[np.ndarray, str]:
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    print(f"Performing {method.upper()} dimension reduction...")
    print(f"Input shape: {embeddings.shape}")

    if method == "tsne":
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        if embeddings.shape[0] <= 1:
            raise ValueError("t-SNE requires at least 2 points")

        if embeddings.shape[0] > args.tsne_threshold:
            print(f"Dataset is large, pre-reducing with PCA to {args.pca_dim} dimensions")
            reduced = PCA(n_components=min(args.pca_dim, embeddings.shape[1]), random_state=args.seed).fit_transform(
                embeddings
            )
            tsne_input = reduced
            title = f"t-SNE Visualization (PCA pre-reduced to {tsne_input.shape[1]}D)"
        else:
            tsne_input = embeddings
            title = "t-SNE Visualization"

        perplexity = min(args.perplexity, max(1, len(tsne_input) - 1))
        tsne = TSNE(
            n_components=2,
            random_state=args.seed,
            perplexity=perplexity,
            n_iter=args.tsne_iterations,
            metric="cosine" if args.use_cosine else "euclidean",
            method="barnes_hut" if len(tsne_input) > 5000 else "exact",
        )
        return tsne.fit_transform(tsne_input), title

    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=args.seed)
    reduced = pca.fit_transform(embeddings)
    explained = pca.explained_variance_ratio_.sum()
    return reduced, f"PCA Visualization (Explained Variance: {explained:.3f})"


def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def get_category_colors(use_rgba: bool = False) -> dict[str, str]:
    colors = {
        "Category A": "#FF0000",
        "Category B": "#00FF00",
        "Category C": "#0080FF",
        "Category D": "#FFD700",
        "Category a": "#FF6666",
        "Category b": "#66FF66",
        "Category c": "#6699FF",
        "Category d": "#FFCC33",
        "Chinese": "#FF6B6B",
        "Japanese": "#4ECDC4",
        "Korean": "#45B7D1",
        "Thai": "#96CEB4",
        "Vietnamese": "#FD79A8",
        "Hindi": "#FDCB6E",
        "Bengali": "#6C5CE7",
        "Tamil": "#A29BFE",
        "Indian": "#FF7675",
        "Arabic": "#74B9FF",
        "Hebrew": "#A0E7E5",
        "English": "#81C784",
        "Russian": "#64B5F6",
        "Greek": "#BA68C8",
        "French": "#FFB74D",
        "German": "#FF8A65",
        "Spanish": "#F06292",
        "Portuguese": "#9575CD",
        "Italian": "#4FC3F7",
        "Code": "#455A64",
        "LaTeX": "#8D6E63",
        "Numeric": "#78909C",
        "Mathematical": "#5C6BC0",
        "Special Token": "#BA68C8",
        "Punctuation": "#B0BEC5",
        "Emoji": "#FFD54F",
        "Whitespace": "#ECEFF1",
        "Other": "#9E9E9E",
        "Invalid UTF-8": "#D32F2F",
        "Alphanumeric": "#7E57C2",
    }

    for key, value in list(colors.items()):
        original_key = f"Original-{key}"
        colors[original_key] = hex_to_rgba(value, 0.55) if use_rgba else f"{value}99"
    colors["Token"] = "#607D8B"
    colors["Original-Token"] = hex_to_rgba("#607D8B", 0.55) if use_rgba else "#607D8B99"
    return colors


def create_static_visualization(
    embeddings_2d: np.ndarray,
    token_names: list[str],
    token_categories: list[str],
    title: str,
    output_path: str,
    args: argparse.Namespace,
) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(args.fig_width, args.fig_height))
    color_map = get_category_colors(use_rgba=False)
    colors = [color_map.get(category, "#808080") for category in token_categories]

    plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        alpha=args.point_alpha,
        s=args.point_size,
        c=colors,
    )

    if args.show_labels:
        label_interval = max(1, len(token_names) // args.max_labels)
        for index, token_name in enumerate(token_names):
            if index % label_interval != 0:
                continue
            display_name = token_name[:20] + "..." if len(token_name) > 20 else token_name
            plt.annotate(
                display_name,
                (embeddings_2d[index, 0], embeddings_2d[index, 1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
            )

    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel("Dimension 1", fontsize=12)
    plt.ylabel("Dimension 2", fontsize=12)

    legend_categories = sorted(set(token_categories), key=lambda name: (not name.startswith("Category"), name))
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_map.get(category, "#808080"),
            markersize=10,
            label=category,
        )
        for category in legend_categories[:24]
    ]
    if legend_elements:
        plt.legend(handles=legend_elements, loc="upper right", fontsize=8, ncol=2 if len(legend_elements) > 10 else 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization to: {output_path}")


def create_interactive_visualization(
    embeddings_2d: np.ndarray,
    token_names: list[str],
    token_categories: list[str],
    title: str,
    output_path: str,
    args: argparse.Namespace,
) -> None:
    if not PLOTLY_AVAILABLE:
        print("Plotly not available, skipping interactive visualization")
        return

    color_map = get_category_colors(use_rgba=True)
    grouped_indices: dict[str, list[int]] = {}
    for index, category in enumerate(token_categories):
        grouped_indices.setdefault(category, []).append(index)

    fig = go.Figure()
    for category in sorted(grouped_indices):
        indices = grouped_indices[category]
        fig.add_trace(
            go.Scatter(
                x=[embeddings_2d[i, 0] for i in indices],
                y=[embeddings_2d[i, 1] for i in indices],
                mode="markers",
                name=category,
                marker=dict(size=8, color=color_map.get(category, "#808080"), opacity=0.7),
                text=[token_names[i] for i in indices],
                hovertemplate="<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        width=args.interactive_width,
        height=args.interactive_height,
        showlegend=True,
        legend=dict(yanchor="top", y=1, xanchor="left", x=1.02),
    )
    fig.write_html(output_path)
    print(f"Saved interactive visualization to: {output_path}")


def compute_token_similarities(embeddings: np.ndarray, token_ids: list[int], tokenizer, top_k: int = 10) -> dict:
    print(f"\nComputing top-{top_k} similar tokens")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    normalized = embeddings / norms
    similarity_matrix = np.dot(normalized, normalized.T)
    top_k = min(top_k, max(0, len(token_ids) - 1))

    similarities = {}
    for row_index, token_id in enumerate(token_ids[:100]):
        sim_scores = similarity_matrix[row_index].copy()
        sim_scores[row_index] = -1

        top_indices = np.argsort(sim_scores)[-top_k:][::-1]
        token_name = get_token_display_name(tokenizer, token_id)
        similarities[token_name] = [
            (get_token_display_name(tokenizer, token_ids[neighbor_index]), float(sim_scores[neighbor_index]))
            for neighbor_index in top_indices
        ]

    return similarities


def save_analysis_results(
    output_dir: str,
    tokenizer_stats: dict,
    category_distribution: dict[str, int],
    token_similarities: dict | None = None,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with (output_path / "tokenizer_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(tokenizer_stats, handle, indent=2, ensure_ascii=False)
    with (output_path / "category_distribution.json").open("w", encoding="utf-8") as handle:
        json.dump(category_distribution, handle, indent=2, ensure_ascii=False)
    if token_similarities:
        with (output_path / "token_similarities.json").open("w", encoding="utf-8") as handle:
            json.dump(token_similarities, handle, indent=2, ensure_ascii=False)


def list_model_layers(model_path: str) -> None:
    safetensor_files = sorted(Path(model_path).glob("*.safetensors"))
    if not safetensor_files:
        print("No safetensor files found")
        return

    from safetensors.torch import safe_open

    all_keys = set()
    for file_path in safetensor_files:
        with safe_open(str(file_path), framework="pt") as handle:
            keys = list(handle.keys())
            all_keys.update(keys)
            print(f"\n{file_path.name}: {len(keys)} layers")

    print("\n=== Potential Embedding Layers ===")
    for key in sorted(k for k in all_keys if "embed" in k.lower()):
        print(f"  {key}")

    print("\n=== All Layers (first 50) ===")
    for key in sorted(all_keys)[:50]:
        print(f"  {key}")
    if len(all_keys) > 50:
        print(f"  ... and {len(all_keys) - 50} more layers")


def main() -> None:
    parser = argparse.ArgumentParser(description="Token embedding visualization with token-category analysis")
    parser.add_argument("--model_path", required=True, help="Model or checkpoint directory")
    parser.add_argument(
        "--output_dir", default=None, help="Output directory. Defaults to <model_path>/embedding_analysis"
    )
    parser.add_argument(
        "--layer_name", default="auto", help="Embedding layer name. Use 'auto' for automatic detection."
    )
    parser.add_argument("--list_layers", action="store_true", help="List available layers and exit")
    parser.add_argument("--sample_original", type=int, default=1000, help="Number of original tokens to sample")
    parser.add_argument(
        "--filter_languages",
        nargs="+",
        default=DEFAULT_ORIGINAL_TOKEN_FILTERS,
        choices=[
            "Chinese",
            "English",
            "Japanese",
            "Korean",
            "Russian",
            "Arabic",
            "French",
            "German",
            "Spanish",
            "Italian",
            "Portuguese",
            "Thai",
            "Vietnamese",
            "Hindi",
            "Code",
            "Numeric",
            "Other",
        ],
        help="Filter original tokens by detected category before sampling",
    )
    parser.add_argument("--max_tokens", type=int, default=None, help="Maximum number of tokens to keep after sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--method", default="both", choices=["tsne", "pca", "both"], help="Dimension reduction method")
    parser.add_argument("--interactive", action="store_true", help="Generate interactive Plotly visualizations")
    parser.add_argument("--perplexity", type=int, default=30, help="t-SNE perplexity")
    parser.add_argument("--pca_dim", type=int, default=50, help="PCA dimensions used before t-SNE on large datasets")
    parser.add_argument(
        "--tsne_threshold",
        type=int,
        default=10000,
        help="Apply PCA before t-SNE when token count exceeds this threshold",
    )
    parser.add_argument("--tsne_iterations", type=int, default=400, help="Number of t-SNE iterations")
    parser.add_argument("--use_cosine", dest="use_cosine", action="store_true", help="Use cosine distance for t-SNE")
    parser.add_argument(
        "--no-use_cosine",
        dest="use_cosine",
        action="store_false",
        help="Use euclidean distance for t-SNE",
    )
    parser.set_defaults(use_cosine=True)
    parser.add_argument("--fig_width", type=int, default=20, help="Static figure width in inches")
    parser.add_argument("--fig_height", type=int, default=16, help="Static figure height in inches")
    parser.add_argument("--point_size", type=int, default=50, help="Scatter plot point size")
    parser.add_argument("--point_alpha", type=float, default=0.6, help="Scatter plot alpha")
    parser.add_argument("--dpi", type=int, default=300, help="Output image DPI")
    parser.add_argument("--show_labels", action="store_true", help="Show token labels")
    parser.add_argument("--max_labels", type=int, default=50, help="Maximum number of labels to show")
    parser.add_argument("--interactive_width", type=int, default=1400, help="Interactive plot width in pixels")
    parser.add_argument("--interactive_height", type=int, default=900, help="Interactive plot height in pixels")
    parser.add_argument("--skip_analysis", action="store_true", help="Skip token category analysis")
    parser.add_argument("--compute_similarities", action="store_true", help="Compute top-k similar tokens")
    parser.add_argument("--top_k_similar", type=int, default=10, help="Number of similar tokens to report")
    args = parser.parse_args()

    if args.list_layers:
        list_model_layers(args.model_path)
        return

    output_dir = args.output_dir or str(Path(args.model_path) / "embedding_analysis")
    os.makedirs(output_dir, exist_ok=True)
    layer_name = None if args.layer_name == "auto" else args.layer_name

    embedding_info = load_embeddings_from_model(
        model_path=args.model_path,
        layer_name=layer_name,
        sample_original_tokens=args.sample_original,
        analyze_languages=not args.skip_analysis,
        filter_languages=args.filter_languages,
        seed=args.seed,
    )

    embeddings = embedding_info.embeddings
    token_names = embedding_info.token_names
    token_ids = embedding_info.token_ids
    token_categories = embedding_info.token_categories

    if args.max_tokens and len(token_ids) > args.max_tokens:
        random.seed(args.seed)
        selected = sorted(random.sample(range(len(token_ids)), args.max_tokens))
        embeddings = embeddings[selected]
        token_names = [token_names[i] for i in selected]
        token_ids = [token_ids[i] for i in selected]
        token_categories = [token_categories[i] for i in selected]

    print(f"Processing {len(token_ids)} tokens with {embeddings.shape[1]}-dimensional embeddings")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    token_similarities = None
    if args.compute_similarities:
        token_similarities = compute_token_similarities(embeddings, token_ids, tokenizer, args.top_k_similar)

    if not args.skip_analysis:
        category_distribution = build_label_distribution(token_categories)
        save_analysis_results(output_dir, embedding_info.tokenizer_stats, category_distribution, token_similarities)

    methods = ["tsne", "pca"] if args.method == "both" else [args.method]
    for method in methods:
        print(f"\n=== {method.upper()} Visualization ===")
        embeddings_2d, title = perform_dimension_reduction(embeddings, method, args)
        title += " (Categories + Original Tokens)"

        output_path = os.path.join(output_dir, f"embeddings_{method}.png")
        create_static_visualization(embeddings_2d, token_names, token_categories, title, output_path, args)

        if args.interactive:
            interactive_path = os.path.join(output_dir, f"embeddings_{method}_interactive.html")
            create_interactive_visualization(
                embeddings_2d,
                token_names,
                token_categories,
                title,
                interactive_path,
                args,
            )

    print(f"\nSaved embedding analysis to: {output_dir}")


if __name__ == "__main__":
    main()
