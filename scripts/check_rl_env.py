#!/usr/bin/env python3
import argparse
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def probe_writable_path(path: str) -> tuple[bool, str]:
    try:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        probe = target / f".genrec_write_probe_{os.getpid()}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True, f"writable: {target}"
    except OSError as exc:
        return False, f"not writable: {path} ({exc})"


def check_trl_metadata() -> tuple[bool, str]:
    try:
        return True, f"trl metadata OK: version={version('trl')}"
    except PackageNotFoundError:
        return False, "trl metadata missing: checkpoint save may fail when TRL generates a model card"


def check_path_exists(path: str, kind: str) -> tuple[bool, str]:
    target = Path(path)
    if target.exists():
        return True, f"{kind} exists: {target}"
    return False, f"{kind} missing: {target}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight checks for GenRec RL jobs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wandb-dir", default=os.environ.get("WANDB_DIR", str(Path.cwd() / "wandb")))
    parser.add_argument("--model-path")
    parser.add_argument("--data-dir")
    parser.add_argument("--index-path")
    parser.add_argument("--report-to", default="wandb")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    checks: list[tuple[str, bool, str]] = []
    checks.append(("trl_metadata", *check_trl_metadata()))
    checks.append(("output_dir", *probe_writable_path(args.output_dir)))

    if args.report_to != "none" and "wandb" in args.report_to:
        checks.append(("wandb_dir", *probe_writable_path(args.wandb_dir)))

    if args.model_path:
        checks.append(("model_path", *check_path_exists(args.model_path, "model path")))
    if args.data_dir:
        checks.append(("data_dir", *check_path_exists(args.data_dir, "data dir")))
        for split in ["train.json", "valid.json", "test.json"]:
            ok, msg = check_path_exists(str(Path(args.data_dir) / split), split)
            checks.append((split.replace(".", "_"), ok, msg))
    if args.index_path:
        checks.append(("index_path", *check_path_exists(args.index_path, "index path")))

    has_error = False
    for name, ok, msg in checks:
        level = "CHECK" if ok else "WARN"
        print(f"[{level}] {name}: {msg}")
        has_error = has_error or not ok

    if has_error:
        print("[SUMMARY] preflight finished with warnings")
        return 1 if args.strict else 0

    print("[SUMMARY] preflight passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
