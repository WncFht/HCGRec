#!/usr/bin/env python3
"""Build and resolve explicit evaluation profiles for checkpoint watchers."""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any


ASSIGNMENT_RE = re.compile(r"^\s*(?:export\s+)?([A-Z][A-Z0-9_]*)=(.+?)\s*$")
YAML_KEY_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*?)\s*$")
CHECKPOINT_RE = re.compile(r"^checkpoint-\d+$")
POSITIONAL_ARG_RE = re.compile(r"^\$(?:[0-9]+|[@*#?])$")
SUPPORTED_VARIANT_PREFIXES = (
    "Industrial_and_Scientific",
    "Instruments",
    "Games",
    "Arts",
)


def strip_inline_shell_comment(raw: str) -> str:
    quote_char = ""
    escaped = False
    for index, char in enumerate(raw):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote_char:
            if char == quote_char:
                quote_char = ""
            continue
        if char in {"'", '"'}:
            quote_char = char
            continue
        if char == "#":
            return raw[:index].rstrip()
    return raw.strip()


def strip_matching_quotes(raw: str) -> str:
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"'", '"'}:
        return raw[1:-1]
    return raw


def find_matching_brace(text: str, start_index: int) -> int:
    depth = 1
    index = start_index
    while index < len(text):
        if text.startswith("${", index):
            depth += 1
            index += 2
            continue
        if text[index] == "}":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return -1


def split_default_expression(content: str) -> tuple[str, str | None]:
    index = 0
    depth = 0
    while index < len(content):
        if content.startswith("${", index):
            depth += 1
            index += 2
            continue
        if content[index] == "}":
            depth = max(depth - 1, 0)
            index += 1
            continue
        if depth == 0 and content.startswith(":-", index):
            return (content[:index], content[index + 2 :])
        index += 1
    return (content, None)


def parse_shell_assignments(path: Path) -> dict[str, str]:
    raw_assignments: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = ASSIGNMENT_RE.match(line)
        if match is None:
            continue
        name = match.group(1)
        raw_value = strip_inline_shell_comment(match.group(2))
        if not raw_value:
            continue
        raw_value = strip_matching_quotes(raw_value)
        if name in raw_assignments and POSITIONAL_ARG_RE.match(raw_value):
            continue
        raw_assignments[name] = raw_value
    return raw_assignments


def expand_shell_text(
    text: str,
    assignments: dict[str, str],
    cache: dict[str, str],
    stack: set[str],
) -> str:
    parts: list[str] = []
    index = 0
    while index < len(text):
        if not text.startswith("${", index):
            parts.append(text[index])
            index += 1
            continue
        end_index = find_matching_brace(text, index + 2)
        if end_index < 0:
            parts.append(text[index:])
            break
        content = text[index + 2 : end_index]
        variable_name, default_value = split_default_expression(content)
        variable_name = variable_name.strip()
        if variable_name in assignments:
            resolved = resolve_shell_assignment(variable_name, assignments, cache, stack)
            if not resolved and default_value is not None:
                resolved = expand_shell_text(default_value, assignments, cache, stack)
        elif default_value is not None:
            resolved = expand_shell_text(default_value, assignments, cache, stack)
        else:
            resolved = ""
        parts.append(resolved)
        index = end_index + 1
    return "".join(parts)


def resolve_shell_assignment(
    name: str,
    assignments: dict[str, str],
    cache: dict[str, str],
    stack: set[str],
) -> str:
    if name in cache:
        return cache[name]
    if name in stack:
        return ""
    stack.add(name)
    raw_value = assignments.get(name, "")
    resolved = expand_shell_text(raw_value, assignments, cache, stack)
    cache[name] = resolved
    stack.remove(name)
    return resolved


def resolve_all_shell_assignments(path: Path) -> dict[str, str]:
    raw_assignments = parse_shell_assignments(path)
    cache: dict[str, str] = {}
    for name in raw_assignments:
        resolve_shell_assignment(name, raw_assignments, cache, set())
    return cache


def parse_simple_yaml(path: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = YAML_KEY_RE.match(line)
        if match is None:
            continue
        key = match.group(1)
        value = strip_matching_quotes(match.group(2).strip())
        parsed[key] = value
    return parsed


def load_source_config(repo_root: Path) -> dict[str, Any]:
    config_path = repo_root / "data" / "eval_profile_manifest_sources.json"
    if not config_path.is_file():
        return {}
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return payload


def variant_from_dataset_key(key: str) -> str:
    for suffix in ("_train", "_valid", "_test"):
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return key


def category_from_variant(variant: str) -> str:
    prefixes = (
        "Industrial_and_Scientific",
        "Instruments_grec_rlsidonly",
        "Instruments_grec",
        "Instruments_mimionerec",
        "Games_grec",
        "Arts_grec",
        "Instruments",
        "Games",
        "Arts",
    )
    for prefix in prefixes:
        if variant.startswith(prefix):
            return prefix
    return variant


def is_supported_variant(variant: str) -> bool:
    return any(variant.startswith(prefix) for prefix in SUPPORTED_VARIANT_PREFIXES)


def normalize_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    datasets = {name: manifest["datasets"][name] for name in sorted(manifest.get("datasets", {}).keys())}
    aliases = {
        name: {
            "dataset_variant": manifest["aliases"][name]["dataset_variant"],
            "sources": sorted(set(manifest["aliases"][name].get("sources", []))),
        }
        for name in sorted(manifest.get("aliases", {}).keys())
    }
    return {
        "version": 1,
        "datasets": datasets,
        "aliases": aliases,
    }


def manifest_file_signature(path: Path) -> tuple[bool, int, int]:
    if not path.is_file():
        return (False, 0, 0)
    stat_result = path.stat()
    return (True, stat_result.st_mtime_ns, stat_result.st_size)


def dataset_paths_for_variant(data_root: Path, variant: str) -> dict[str, Path]:
    base_dir = data_root / variant
    return {
        "base_dir": base_dir,
        "train_path": base_dir / "sft" / "train.json",
        "valid_path": base_dir / "sft" / "valid.json",
        "test_path": base_dir / "sft" / "test.json",
        "index_path": base_dir / "id2sid.json",
        "new_tokens_path": base_dir / "new_tokens.json",
        "rl_dir": base_dir / "rl",
    }


def register_dataset(manifest: dict[str, Any], variant: str) -> None:
    if not is_supported_variant(variant):
        return
    if variant in manifest.get("_disabled_variants", set()):
        return
    datasets = manifest["datasets"]
    if variant in datasets:
        return
    dataset_entry = {
        "category": category_from_variant(variant),
        "test_data_path": f"{variant}/sft/test.json",
        "index_path": f"{variant}/id2sid.json",
    }
    dataset_meta = manifest.get("_dataset_meta", {}).get(variant, {})
    for key in ("train_entry", "valid_entry", "test_entry"):
        if dataset_meta.get(key):
            dataset_entry[key] = dataset_meta[key]
    dataset_overrides = manifest.get("_dataset_overrides", {}).get(variant, {})
    for key in ("category", "test_data_path", "index_path", "train_entry", "valid_entry", "test_entry"):
        if dataset_overrides.get(key):
            dataset_entry[key] = dataset_overrides[key]
    datasets[variant] = dataset_entry


def register_alias(
    manifest: dict[str, Any],
    alias: str,
    variant: str,
    *,
    source: str,
) -> None:
    cleaned_alias = alias.strip()
    if not cleaned_alias:
        return
    if "$" in cleaned_alias:
        return
    register_dataset(manifest, variant)
    if variant not in manifest["datasets"]:
        return
    aliases = manifest["aliases"]
    existing = aliases.get(cleaned_alias)
    if existing is None:
        aliases[cleaned_alias] = {
            "dataset_variant": variant,
            "sources": [source],
        }
        return
    if existing["dataset_variant"] != variant:
        existing_sources = set(existing.get("sources", []))
        # Manual overrides are the source of truth. If an auto-discovered shell/YAML
        # alias conflicts with an explicit override, keep the explicit mapping and
        # ignore the conflicting auto-discovered one instead of failing manifest
        # rebuilds.
        if any(src.endswith("#manual_override") for src in existing_sources):
            return
        if source.endswith("#manual_override"):
            aliases[cleaned_alias] = {
                "dataset_variant": variant,
                "sources": [source],
            }
            return
        raise ValueError(f"alias {cleaned_alias!r} maps to both {existing['dataset_variant']!r} and {variant!r}")
    existing.setdefault("sources", []).append(source)


def basename_or_none(path_text: str | None) -> str | None:
    if not path_text:
        return None
    return Path(path_text).name or None


def model_root_from_model_path(path_text: str | None) -> str | None:
    if not path_text:
        return None
    path = Path(path_text)
    if CHECKPOINT_RE.match(path.name):
        return path.parent.name or None
    return path.name or None


def dataset_variant_from_test_path(test_path: str | None) -> str | None:
    if not test_path:
        return None
    parts = Path(test_path).parts
    if len(parts) < 3:
        return None
    if parts[-2:] != ("sft", "test.json"):
        return None
    return parts[-3]


def load_dataset_manifest_entries(data_root: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "datasets": {},
        "aliases": {},
        "_dataset_keys": {},
        "_dataset_meta": {},
        "_disabled_variants": set(),
        "_dataset_overrides": {},
    }
    dataset_info_path = data_root / "dataset_info.json"
    if dataset_info_path.is_file():
        payload = json.loads(dataset_info_path.read_text(encoding="utf-8"))
        for entry_name, entry in payload.items():
            file_name = str(entry.get("file_name", ""))
            parts = Path(file_name).parts
            if not parts:
                continue
            if parts[0] == "LC-Rec" and len(parts) >= 2 and is_supported_variant(parts[1]):
                variant = f"{parts[1]}_lcrec"
            else:
                variant = parts[0]
            manifest["_dataset_keys"][entry_name] = variant
            dataset_entry = manifest["_dataset_meta"].setdefault(variant, {})
            if entry_name.endswith("_train"):
                dataset_entry["train_entry"] = entry_name
            if entry_name.endswith("_valid"):
                dataset_entry["valid_entry"] = entry_name
    return manifest


def iter_scan_targets(repo_root: Path, configured_roots: Any, default_root: Path, pattern: str) -> Iterable[Path]:
    roots: list[Path]
    if isinstance(configured_roots, list) and configured_roots:
        roots = [(repo_root / str(entry)).resolve() for entry in configured_roots if isinstance(entry, str)]
    else:
        roots = [default_root]

    seen: set[Path] = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        if root.is_file():
            if root.match(pattern):
                yield root
            continue
        if not root.is_dir():
            continue
        yield from sorted(root.rglob(pattern))


def collect_yaml_aliases(repo_root: Path, manifest: dict[str, Any], source_config: dict[str, Any]) -> None:
    examples_root = repo_root / "examples" / "train_full"
    for path in iter_scan_targets(repo_root, source_config.get("yaml_roots"), examples_root, "*.yaml"):
        payload = parse_simple_yaml(path)
        dataset_name = payload.get("eval_dataset") or payload.get("dataset")
        if not dataset_name:
            continue
        variant = manifest.get("_dataset_keys", {}).get(dataset_name) or variant_from_dataset_key(dataset_name)
        if not is_supported_variant(variant):
            continue
        register_dataset(manifest, variant)
        output_dir = basename_or_none(payload.get("output_dir"))
        if output_dir:
            register_alias(
                manifest,
                output_dir,
                variant,
                source=f"{path.relative_to(repo_root)}#yaml_output_dir",
            )
        run_name = payload.get("run_name")
        if run_name:
            register_alias(
                manifest,
                run_name,
                variant,
                source=f"{path.relative_to(repo_root)}#yaml_run_name",
            )


def collect_shell_aliases(repo_root: Path, manifest: dict[str, Any], source_config: dict[str, Any]) -> None:
    shell_root = repo_root / "scripts" / "experiments"
    for path in iter_scan_targets(repo_root, source_config.get("shell_roots"), shell_root, "*.sh"):
        payload = resolve_all_shell_assignments(path)
        variant = (
            payload.get("DATA_VARIANT_DEFAULT")
            or dataset_variant_from_test_path(payload.get("TEST_DATA_PATH"))
            or dataset_variant_from_test_path(payload.get("TEST_DATA_PATH_DEFAULT"))
        )
        if not variant:
            continue
        if not is_supported_variant(variant):
            continue
        register_dataset(manifest, variant)
        source_prefix = str(path.relative_to(repo_root))
        for alias_value, alias_kind in (
            (basename_or_none(payload.get("OUTPUT_DIR_DEFAULT")), "output_dir_default"),
            (basename_or_none(payload.get("OUTPUT_DIR")), "output_dir"),
            (payload.get("RUN_NAME_DEFAULT"), "run_name_default"),
            (payload.get("RUN_NAME"), "run_name"),
            (basename_or_none(payload.get("SFT_ROOT")), "sft_root"),
            (basename_or_none(payload.get("SFT_ROOT_DEFAULT")), "sft_root_default"),
        ):
            if not alias_value:
                continue
            register_alias(
                manifest,
                alias_value,
                variant,
                source=f"{source_prefix}#{alias_kind}",
            )


def apply_overrides(overrides_path: Path, manifest: dict[str, Any]) -> None:
    if not overrides_path.is_file():
        return
    payload = json.loads(overrides_path.read_text(encoding="utf-8"))
    manifest["_disabled_variants"].update(
        variant for variant in payload.get("disabled_variants", []) if isinstance(variant, str) and variant
    )
    dataset_overrides = payload.get("dataset_overrides", {})
    if isinstance(dataset_overrides, dict):
        for variant, override in dataset_overrides.items():
            if not isinstance(variant, str) or not isinstance(override, dict):
                continue
            manifest["_dataset_overrides"][variant] = {
                key: value for key, value in override.items() if isinstance(key, str) and isinstance(value, str)
            }
    for alias, variant in payload.get("aliases", {}).items():
        if not isinstance(alias, str) or not isinstance(variant, str):
            continue
        register_alias(
            manifest,
            alias,
            variant,
            source=f"{overrides_path.name}#manual_override",
        )


def build_manifest(repo_root: Path, data_root: Path, overrides_path: Path | None = None) -> dict[str, Any]:
    manifest = load_dataset_manifest_entries(data_root)
    source_config = load_source_config(repo_root)
    if overrides_path is not None:
        apply_overrides(overrides_path, manifest)
    collect_yaml_aliases(repo_root, manifest, source_config)
    collect_shell_aliases(repo_root, manifest, source_config)
    return normalize_manifest(manifest)


def apply_runtime_overrides_to_manifest(manifest: dict[str, Any], overrides_path: Path) -> dict[str, Any]:
    if not overrides_path.is_file():
        return manifest

    payload = json.loads(overrides_path.read_text(encoding="utf-8"))
    runtime_manifest = {
        "version": manifest.get("version", 1),
        "datasets": copy.deepcopy(manifest.get("datasets", {})),
        "aliases": copy.deepcopy(manifest.get("aliases", {})),
    }

    disabled_variants = {
        variant for variant in payload.get("disabled_variants", []) if isinstance(variant, str) and variant
    }
    if disabled_variants:
        for variant in disabled_variants:
            runtime_manifest["datasets"].pop(variant, None)
        runtime_manifest["aliases"] = {
            alias: entry
            for alias, entry in runtime_manifest["aliases"].items()
            if str(entry.get("dataset_variant")) not in disabled_variants
        }

    dataset_overrides = payload.get("dataset_overrides", {})
    if isinstance(dataset_overrides, dict):
        for variant, override in dataset_overrides.items():
            if not isinstance(variant, str) or not isinstance(override, dict):
                continue
            if variant not in runtime_manifest["datasets"]:
                continue
            runtime_manifest["datasets"][variant].update(
                {key: value for key, value in override.items() if isinstance(key, str) and isinstance(value, str)}
            )

    alias_overrides = payload.get("aliases", {})
    if isinstance(alias_overrides, dict):
        for alias, variant in alias_overrides.items():
            if not isinstance(alias, str) or not isinstance(variant, str):
                continue
            if variant in disabled_variants:
                continue
            if variant not in runtime_manifest["datasets"]:
                runtime_manifest["datasets"][variant] = {
                    "category": category_from_variant(variant),
                    "test_data_path": f"{variant}/sft/test.json",
                    "index_path": f"{variant}/id2sid.json",
                }
            runtime_manifest["aliases"][alias] = {
                "dataset_variant": variant,
                "sources": [f"{overrides_path.name}#manual_override"],
            }

    return normalize_manifest(runtime_manifest)


def collect_supported_variants_from_data_root(data_root: Path) -> list[str]:
    variants: set[str] = set()
    dataset_info_path = data_root / "dataset_info.json"
    if dataset_info_path.is_file():
        payload = json.loads(dataset_info_path.read_text(encoding="utf-8"))
        for entry in payload.values():
            file_name = str(entry.get("file_name", ""))
            parts = Path(file_name).parts
            if parts and is_supported_variant(parts[0]):
                variants.add(parts[0])
    if data_root.is_dir():
        for child in data_root.iterdir():
            if child.is_dir() and is_supported_variant(child.name):
                variants.add(child.name)
    return sorted(variants)


def audit_manifest(
    repo_root: Path,
    data_root: Path,
    *,
    manifest_path: Path,
    overrides_path: Path,
) -> dict[str, Any]:
    manifest = load_manifest(repo_root, data_root, manifest_path, overrides_path)
    aliases_by_variant: dict[str, list[str]] = {}
    for alias, entry in manifest.get("aliases", {}).items():
        variant = str(entry["dataset_variant"])
        aliases_by_variant.setdefault(variant, []).append(alias)

    candidate_variants = set(manifest.get("datasets", {}).keys())
    candidate_variants.update(collect_supported_variants_from_data_root(data_root))

    datasets_report: list[dict[str, Any]] = []
    manifest_missing_eval_files: list[str] = []
    manifest_entries: dict[str, Any] = manifest.get("datasets", {})
    for variant in sorted(candidate_variants):
        paths = dataset_paths_for_variant(data_root, variant)
        in_manifest = variant in manifest_entries
        report = {
            "variant": variant,
            "in_manifest": in_manifest,
            "category": manifest_entries.get(variant, {}).get("category", category_from_variant(variant)),
            "base_dir_exists": paths["base_dir"].is_dir(),
            "train_exists": paths["train_path"].is_file(),
            "valid_exists": paths["valid_path"].is_file(),
            "test_exists": paths["test_path"].is_file(),
            "index_exists": paths["index_path"].is_file(),
            "new_tokens_exists": paths["new_tokens_path"].is_file(),
            "rl_dir_exists": paths["rl_dir"].is_dir(),
            "alias_count": len(aliases_by_variant.get(variant, [])),
            "aliases": sorted(aliases_by_variant.get(variant, [])),
        }
        if in_manifest and not (report["test_exists"] and report["index_exists"]):
            manifest_missing_eval_files.append(variant)
        datasets_report.append(report)

    return {
        "summary": {
            "manifest_dataset_count": len(manifest_entries),
            "manifest_alias_count": len(manifest.get("aliases", {})),
            "candidate_variant_count": len(candidate_variants),
            "manifest_missing_eval_files": len(manifest_missing_eval_files),
        },
        "manifest_missing_eval_files": manifest_missing_eval_files,
        "datasets": datasets_report,
    }


@lru_cache(maxsize=8)
def load_manifest_cached(
    repo_root_text: str,
    data_root_text: str,
    manifest_path_text: str,
    overrides_path_text: str,
    manifest_exists: bool,
    manifest_mtime_ns: int,
    manifest_size: int,
    overrides_exists: bool,
    overrides_mtime_ns: int,
    overrides_size: int,
) -> dict[str, Any]:
    repo_root = Path(repo_root_text)
    data_root = Path(data_root_text)
    manifest_path = Path(manifest_path_text)
    overrides_path = Path(overrides_path_text)
    if manifest_path.is_file():
        manifest = normalize_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))
        return apply_runtime_overrides_to_manifest(manifest, overrides_path)
    return build_manifest(repo_root, data_root, overrides_path)


def load_manifest(repo_root: Path, data_root: Path, manifest_path: Path, overrides_path: Path) -> dict[str, Any]:
    manifest_signature = manifest_file_signature(manifest_path)
    overrides_signature = manifest_file_signature(overrides_path)
    return load_manifest_cached(
        str(repo_root.resolve()),
        str(data_root.resolve()),
        str(manifest_path.resolve()),
        str(overrides_path.resolve()),
        *manifest_signature,
        *overrides_signature,
    )


def resolve_profile(
    repo_root: Path,
    data_root: Path,
    model_name: str,
    *,
    manifest_path: Path,
    overrides_path: Path,
) -> dict[str, Path | str] | None:
    manifest = load_manifest(repo_root, data_root, manifest_path, overrides_path)
    alias_entry = manifest.get("aliases", {}).get(model_name)
    if alias_entry is None:
        return None
    dataset_variant = str(alias_entry["dataset_variant"])
    dataset_entry = manifest["datasets"].get(dataset_variant)
    if dataset_entry is None:
        return None
    return {
        "category": str(dataset_entry["category"]),
        "test_data_path": data_root / str(dataset_entry["test_data_path"]),
        "index_path": data_root / str(dataset_entry["index_path"]),
        "data_profile": f"manifest:dataset_variant={dataset_variant};alias={model_name}",
        "cb_width": "n/a",
    }


def write_manifest(output_path: Path, manifest: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and resolve explicit evaluation profiles")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-manifest", help="Build the manifest file from repo metadata")
    build.add_argument("--repo-root", default=".")
    build.add_argument("--data-root", default="./data")
    build.add_argument("--output", default="./data/eval_profile_manifest.json")
    build.add_argument("--overrides", default="./data/eval_profile_overrides.json")

    resolve = subparsers.add_parser("resolve", help="Resolve a model/output name to an eval profile")
    resolve.add_argument("--repo-root", default=".")
    resolve.add_argument("--data-root", default="./data")
    resolve.add_argument("--manifest", default="./data/eval_profile_manifest.json")
    resolve.add_argument("--overrides", default="./data/eval_profile_overrides.json")
    resolve.add_argument("--model-name", required=True)
    resolve.add_argument("--format", choices=("json", "tsv"), default="json")

    audit = subparsers.add_parser("audit", help="Audit manifest assumptions against actual data files")
    audit.add_argument("--repo-root", default=".")
    audit.add_argument("--data-root", default="./data")
    audit.add_argument("--manifest", default="./data/eval_profile_manifest.json")
    audit.add_argument("--overrides", default="./data/eval_profile_overrides.json")
    audit.add_argument("--format", choices=("json", "tsv"), default="json")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()

    if args.command == "build-manifest":
        manifest = build_manifest(repo_root, data_root, Path(args.overrides).expanduser().resolve())
        write_manifest(Path(args.output).expanduser().resolve(), manifest)
        return 0

    if args.command == "resolve":
        profile = resolve_profile(
            repo_root,
            data_root,
            args.model_name,
            manifest_path=Path(args.manifest).expanduser().resolve(),
            overrides_path=Path(args.overrides).expanduser().resolve(),
        )
        if profile is None:
            return 1
        if args.format == "json":
            serializable = {key: str(value) if isinstance(value, Path) else value for key, value in profile.items()}
            print(json.dumps(serializable, ensure_ascii=True))
            return 0
        print(
            "\t".join(
                [
                    str(profile["category"]),
                    str(profile["test_data_path"]),
                    str(profile["index_path"]),
                    str(profile["data_profile"]),
                    str(profile["cb_width"]),
                ]
            )
        )
        return 0

    if args.command == "audit":
        report = audit_manifest(
            repo_root,
            data_root,
            manifest_path=Path(args.manifest).expanduser().resolve(),
            overrides_path=Path(args.overrides).expanduser().resolve(),
        )
        if args.format == "json":
            print(json.dumps(report, indent=2, ensure_ascii=True))
            return 0
        print("variant\tin_manifest\ttest_exists\tindex_exists\talias_count")
        for item in report["datasets"]:
            print(
                "\t".join(
                    [
                        str(item["variant"]),
                        str(item["in_manifest"]),
                        str(item["test_exists"]),
                        str(item["index_exists"]),
                        str(item["alias_count"]),
                    ]
                )
            )
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
