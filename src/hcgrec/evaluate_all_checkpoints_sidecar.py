#!/usr/bin/env python3
"""Watch checkpoint directories and evaluate ready checkpoints serially."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Lock
from typing import Any

from hcgrec.eval_profile_manifest import resolve_profile as resolve_manifest_profile


CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
MODULE_NAME = "hcgrec.evaluate_all_checkpoints_sidecar"
_HEURISTIC_PROFILE_WARNED_MODELS: set[str] = set()
_MISSING_MANIFEST_PROFILE_WARNED_MODELS: set[str] = set()


@dataclass(frozen=True)
class WatcherConfig:
    repo_root: Path
    results_root: Path
    eval_script: Path
    python_bin: str
    cuda_list: str
    data_root: Path
    auto_data_mapping: bool
    sft_root: Path
    rl_root: Path
    include_sft: bool
    include_rl: bool
    model_filter: str
    force_reeval: bool
    allow_heuristic_fallback: bool
    stable_age_seconds: int
    stable_confirmation_polls: int
    poll_interval_seconds: int
    idle_hold_enabled: bool
    idle_hold_memory_ratio: float
    idle_hold_release_grace_seconds: int
    state_path: Path


@dataclass(frozen=True)
class CheckpointTask:
    root_kind: str
    model_root: Path
    model_name: str
    checkpoint_path: Path
    checkpoint_name: str
    checkpoint_step: int
    result_dir: Path
    category: str
    test_data_path: Path
    index_path: Path
    data_profile: str
    cb_width: str


class TorchIdleGpuHolder:
    """Keep the watch process attached to the same GPUs while idle."""

    def __init__(self, cuda_list: str, memory_ratio: float):
        self.cuda_list = cuda_list
        self.memory_ratio = memory_ratio
        self._lock = Lock()
        self._process: subprocess.Popen[str] | None = None

    def ensure_running(self) -> None:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return
            command = [
                sys.executable,
                "-m",
                MODULE_NAME,
                "idle-hold",
                "--cuda-list",
                self.cuda_list,
                "--memory-ratio",
                str(self.memory_ratio),
                "--log-level",
                os.environ.get("LOG_LEVEL", "INFO"),
            ]
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            self._process = subprocess.Popen(
                command,
                env=env,
                cwd=Path.cwd(),
                text=True,
            )
            logging.info("Idle GPU holder process started pid=%d", self._process.pid)

    def stop(self) -> None:
        with self._lock:
            process = self._process

        if process is None:
            return

        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logging.warning(
                    "Idle GPU holder process did not stop within timeout, sending SIGKILL pid=%d", process.pid
                )
                process.kill()
                process.wait(timeout=30)

        with self._lock:
            if self._process is process:
                self._process = None


def execute_idle_hold(cuda_list: str, memory_ratio: float) -> int:
    allocation_chunk_bytes = 256 * 1024 * 1024
    stop_event = Event()

    def handle_signal(signum: int, _frame: Any) -> None:
        logging.info("Idle GPU holder received signal=%d", signum)
        stop_event.set()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    gpu_ids: list[int] = []
    allocated_gib: dict[int, float] = {}
    blocks: list[list[Any]] = []
    torch = None
    try:
        import torch as torch_module
    except ImportError as exc:
        logging.warning("Idle GPU holder unavailable because torch import failed: %s", exc)
        return 0

    torch = torch_module
    if not torch.cuda.is_available():
        logging.warning("Idle GPU holder requested but CUDA is unavailable")
        return 0

    try:
        gpu_ids = parse_cuda_list(cuda_list, torch.cuda.device_count())
    except ValueError as exc:
        logging.warning("Idle GPU holder disabled due to invalid CUDA_LIST=%r: %s", cuda_list, exc)
        return 0

    if not gpu_ids:
        logging.warning("Idle GPU holder requested but no GPU ids were resolved from CUDA_LIST=%r", cuda_list)
        return 0

    try:
        for gpu_id in gpu_ids:
            if stop_event.is_set():
                return 0

            device = torch.device(f"cuda:{gpu_id}")
            total_memory = torch.cuda.get_device_properties(device).total_memory
            target_bytes = int(total_memory * memory_ratio)
            allocated_bytes = 0
            device_blocks: list[Any] = []

            while allocated_bytes < target_bytes and not stop_event.is_set():
                next_chunk = min(allocation_chunk_bytes, target_bytes - allocated_bytes)
                if next_chunk <= 0:
                    break
                try:
                    block = torch.empty(next_chunk, dtype=torch.uint8, device=device)
                except Exception as exc:
                    logging.warning(
                        "Idle GPU holder stopped allocating on cuda:%d after %.2f GiB: %s",
                        gpu_id,
                        allocated_bytes / (1024**3),
                        exc,
                    )
                    break
                device_blocks.append(block)
                allocated_bytes += block.nelement()

            blocks.append(device_blocks)
            allocated_gib[gpu_id] = allocated_bytes / (1024**3)
            logging.info(
                "Idle GPU holder reserved %.2f GiB on cuda:%d (target %.1f%%)",
                allocated_gib[gpu_id],
                gpu_id,
                memory_ratio * 100.0,
            )

        while not stop_event.wait(1):
            continue
        return 0
    finally:
        blocks.clear()
        if torch is not None:
            try:
                for gpu_id in gpu_ids:
                    with torch.cuda.device(gpu_id):
                        torch.cuda.empty_cache()
            except Exception as exc:
                logging.warning("Idle GPU holder cleanup raised: %s", exc)
        if gpu_ids:
            logging.info("Idle GPU holder released GPUs=%s", gpu_ids)


def now_utc_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def normalize_watch_state(raw: dict[str, Any]) -> dict[str, Any]:
    observations = raw.get("observations", {})
    failed_tasks = raw.get("failed_tasks", {})
    return {
        "version": 1,
        "observations": observations if isinstance(observations, dict) else {},
        "failed_tasks": failed_tasks if isinstance(failed_tasks, dict) else {},
        "current_task": raw.get("current_task"),
        "last_scan_time": raw.get("last_scan_time"),
        "last_execution_time": raw.get("last_execution_time"),
    }


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return data


def save_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)
    temp_path.replace(path)


def load_watch_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return normalize_watch_state({})
    return normalize_watch_state(load_json(path))


@contextmanager
def exclusive_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as fp:
        try:
            fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another watcher process holds lock: {lock_path}") from exc
        fp.seek(0)
        fp.truncate(0)
        fp.write(str(os.getpid()))
        fp.flush()
        try:
            yield
        finally:
            fcntl.flock(fp.fileno(), fcntl.LOCK_UN)


def stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def parse_cuda_list(raw_value: str, device_count: int) -> list[int]:
    available_gpu_ids = list(range(device_count))
    normalized = raw_value.strip()
    if not normalized or normalized in {"all", "remaining"}:
        return available_gpu_ids

    resolved_gpu_ids: list[int] = []
    seen_gpu_ids: set[int] = set()
    for token in normalized.replace(",", " ").split():
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"invalid GPU range: {token}")
            gpu_candidates = range(start, end + 1)
        else:
            gpu_candidates = [int(token)]

        for gpu_id in gpu_candidates:
            if gpu_id not in available_gpu_ids:
                raise ValueError(f"GPU {gpu_id} is unavailable")
            if gpu_id in seen_gpu_ids:
                continue
            seen_gpu_ids.add(gpu_id)
            resolved_gpu_ids.append(gpu_id)

    return resolved_gpu_ids


def parse_checkpoint_step(name: str) -> int | None:
    match = CHECKPOINT_RE.match(name)
    if not match:
        return None
    return int(match.group(1))


def extract_cb_width(model_name: str) -> str:
    match = re.search(r"cb([0-9]+)", model_name)
    if match:
        return match.group(1)
    match = re.search(r"-4-([0-9]+)(-|$)", model_name)
    if match:
        return match.group(1)
    return "n/a"


def resolve_base_category_from_model_name(model_name: str) -> str | None:
    normalized = model_name.lower()
    alias_groups = (
        (
            "Industrial_and_Scientific",
            ("industrial_and_scientific", "industrial-and-scientific", "industrial", "ind", "ias"),
        ),
        ("Instruments", ("instruments", "instrument", "ins")),
        ("Games", ("games", "game", "gam", "gms")),
        ("Arts", ("arts", "art")),
    )
    for candidate, aliases in alias_groups:
        if any(normalized.startswith(alias) for alias in aliases):
            return candidate
    return None


def matches_filter(model_name: str, raw_filter: str) -> bool:
    if not raw_filter:
        return True
    for token in raw_filter.split(","):
        normalized = token.strip()
        if normalized and normalized in model_name:
            return True
    return False


def dir_mtime_epoch(path: Path) -> int:
    return int(path.stat().st_mtime)


def pick_latest_variant_dir_by_cb(data_root: Path, variant_prefix: str, cb_width: str) -> Path | None:
    if not data_root.is_dir():
        return None
    best_dir: Path | None = None
    best_mtime = -1
    for candidate in iter_variant_dirs(data_root, variant_prefix):
        if not candidate.is_dir():
            continue
        candidate_name = candidate.name
        if (
            f"_cb{cb_width}-" in candidate_name
            or f"_cb{cb_width}_" in candidate_name
            or candidate_name.endswith(f"_cb{cb_width}")
        ):
            pass
        elif candidate_name == f"{variant_prefix}_index" and cb_width == "256":
            # Support newer compact variant names like Instruments_grec_index.
            pass
        else:
            continue
        candidate_mtime = dir_mtime_epoch(candidate)
        if best_dir is None or candidate_mtime > best_mtime:
            best_dir = candidate
            best_mtime = candidate_mtime
    return best_dir


def pick_latest_variant_dir_by_prefix(data_root: Path, variant_prefix: str) -> Path | None:
    if not data_root.is_dir():
        return None
    best_dir: Path | None = None
    best_mtime = -1
    for candidate in iter_variant_dirs(data_root, variant_prefix):
        if not candidate.is_dir():
            continue
        candidate_mtime = dir_mtime_epoch(candidate)
        if best_dir is None or candidate_mtime > best_mtime:
            best_dir = candidate
            best_mtime = candidate_mtime
    return best_dir


def iter_variant_dirs(data_root: Path, variant_prefix: str) -> list[Path]:
    patterns = [
        f"{variant_prefix}_index_emb-*",
        f"{variant_prefix}_index",
        f"{variant_prefix}_*",
    ]
    found: dict[str, Path] = {}
    for pattern in patterns:
        for candidate in data_root.glob(pattern):
            if candidate.is_dir():
                found[str(candidate.resolve())] = candidate
    return sorted(found.values(), key=lambda item: item.name)


def resolve_eval_profile_from_heuristics(config: WatcherConfig, model_name: str) -> dict[str, Path | str]:
    data_root = config.data_root
    industrial_test = data_root / "Industrial_and_Scientific" / "sft" / "test.json"
    industrial_index = data_root / "Industrial_and_Scientific" / "id2sid.json"
    instruments_test = data_root / "Instruments" / "sft" / "test.json"
    instruments_index = data_root / "Instruments" / "id2sid.json"
    games_test = data_root / "Games" / "sft" / "test.json"
    games_index = data_root / "Games" / "id2sid.json"
    arts_test = data_root / "Arts" / "sft" / "test.json"
    arts_index = data_root / "Arts" / "id2sid.json"

    instruments_grec_fallback = (
        data_root
        / "Instruments_grec_index_emb-qwen3-embedding-4B_rq4_cb256-256-256-256_dsInstruments_ridFeb-10-2026-05-40-47"
    )
    instruments_grec_compact = data_root / "Instruments_grec_index"
    instruments_grec_test = instruments_grec_fallback / "sft" / "test.json"
    instruments_grec_index = instruments_grec_fallback / "id2sid.json"
    instruments_grec_compact_test = instruments_grec_compact / "sft" / "test.json"
    instruments_grec_compact_index = instruments_grec_compact / "id2sid.json"

    cb_width = extract_cb_width(model_name)
    normalized_model_name = model_name.lower()
    base_category = resolve_base_category_from_model_name(model_name)

    def make_result(
        category: str,
        test_data_path: Path,
        index_path: Path,
        profile: str,
    ) -> dict[str, Path | str]:
        return {
            "category": category,
            "test_data_path": test_data_path,
            "index_path": index_path,
            "data_profile": profile,
            "cb_width": cb_width,
        }

    if base_category == "Industrial_and_Scientific":
        return make_result(
            "Industrial_and_Scientific",
            industrial_test,
            industrial_index,
            "fixed:industrial_default",
        )

    if normalized_model_name.startswith("ins-lc"):
        if instruments_grec_compact_test.is_file() and instruments_grec_compact_index.is_file():
            return make_result(
                "Instruments_grec",
                instruments_grec_compact_test,
                instruments_grec_compact_index,
                "fixed:legacy_ins_lc_compact_variant_dir=Instruments_grec_index",
            )
        return make_result(
            "Instruments_grec",
            instruments_grec_test,
            instruments_grec_index,
            "fallback:legacy_ins_lc_fixed_grec_cb256",
        )

    if base_category and (
        model_name.startswith(f"{base_category}-grec") or model_name.startswith(f"{base_category}_grec")
    ):
        category = f"{base_category}_grec"
        if config.auto_data_mapping and cb_width != "n/a":
            variant_dir = pick_latest_variant_dir_by_cb(data_root, category, cb_width)
            if variant_dir is not None:
                candidate_test = variant_dir / "sft" / "test.json"
                candidate_index = variant_dir / "id2sid.json"
                if candidate_test.is_file() and candidate_index.is_file():
                    return make_result(
                        category,
                        candidate_test,
                        candidate_index,
                        f"auto:variant_dir={variant_dir}",
                    )

        # Legacy Instruments-grec experiment names do not encode cb width.
        # Keep their historical fixed-cb256 fallback before scanning newer
        # variant dirs so watcher behavior matches the one-shot evaluator.
        if base_category == "Instruments" and cb_width == "n/a":
            return make_result(
                category,
                instruments_grec_test,
                instruments_grec_index,
                "fallback:fixed_grec_cb256",
            )

        latest_variant_dir = pick_latest_variant_dir_by_prefix(data_root, category)
        if latest_variant_dir is not None:
            candidate_test = latest_variant_dir / "sft" / "test.json"
            candidate_index = latest_variant_dir / "id2sid.json"
            if candidate_test.is_file() and candidate_index.is_file():
                return make_result(
                    category,
                    candidate_test,
                    candidate_index,
                    f"fallback:latest_variant_dir={latest_variant_dir}",
                )

        if base_category == "Instruments":
            return make_result(
                category,
                instruments_grec_test,
                instruments_grec_index,
                "fallback:fixed_grec_cb256",
            )

    if base_category == "Instruments":
        return make_result("Instruments", instruments_test, instruments_index, "fixed:instruments_default")
    if base_category == "Games":
        return make_result("Games", games_test, games_index, "fixed:games_default")
    if base_category == "Arts":
        return make_result("Arts", arts_test, arts_index, "fixed:arts_default")
    return make_result(
        "Industrial_and_Scientific",
        industrial_test,
        industrial_index,
        "fallback:industrial_default",
    )


def resolve_eval_profile(config: WatcherConfig, model_name: str) -> dict[str, Path | str] | None:
    manifest_profile = resolve_manifest_profile(
        config.repo_root,
        config.data_root,
        model_name,
        manifest_path=config.repo_root / "data" / "eval_profile_manifest.json",
        overrides_path=config.repo_root / "data" / "eval_profile_overrides.json",
    )
    if manifest_profile is not None:
        test_data_path = manifest_profile.get("test_data_path")
        index_path = manifest_profile.get("index_path")
        if isinstance(test_data_path, Path) and isinstance(index_path, Path):
            if test_data_path.is_file() and index_path.is_file():
                return manifest_profile
        elif test_data_path and index_path:
            test_data_path = Path(test_data_path)
            index_path = Path(index_path)
            if test_data_path.is_file() and index_path.is_file():
                manifest_profile["test_data_path"] = test_data_path
                manifest_profile["index_path"] = index_path
                return manifest_profile
        if not config.allow_heuristic_fallback:
            return manifest_profile
    if not config.allow_heuristic_fallback:
        if model_name not in _MISSING_MANIFEST_PROFILE_WARNED_MODELS:
            logging.warning(
                "Skipped model=%s because no manifest eval profile was found and heuristic fallback is disabled",
                model_name,
            )
            _MISSING_MANIFEST_PROFILE_WARNED_MODELS.add(model_name)
        return None
    heuristic_profile = resolve_eval_profile_from_heuristics(config, model_name)
    if model_name not in _HEURISTIC_PROFILE_WARNED_MODELS:
        logging.warning(
            "Using heuristic eval profile for model=%s profile=%s",
            model_name,
            heuristic_profile["data_profile"],
        )
        _HEURISTIC_PROFILE_WARNED_MODELS.add(model_name)
    return heuristic_profile


def task_key(task: CheckpointTask) -> str:
    return f"{task.root_kind}:{task.model_name}:{task.checkpoint_name}"


def checkpoint_metrics_path(config: WatcherConfig, model_name: str, checkpoint_name: str) -> Path:
    return config.results_root / model_name / checkpoint_name / "metrics.json"


def task_already_completed(config: WatcherConfig, task: CheckpointTask) -> bool:
    if config.force_reeval:
        return False
    return checkpoint_metrics_path(config, task.model_name, task.checkpoint_name).is_file()


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return load_json(path)


def index_file_has_all_shards(index_path: Path, checkpoint_dir: Path) -> bool:
    raw = read_json_if_exists(index_path)
    if raw is None:
        return False
    weight_map = raw.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        return False
    shard_names = {str(value) for value in weight_map.values() if isinstance(value, str) and value}
    if not shard_names:
        return False
    return all((checkpoint_dir / shard_name).is_file() for shard_name in shard_names)


def checkpoint_has_complete_artifacts(checkpoint_dir: Path) -> tuple[bool, str]:
    metadata_files = ("config.json", "adapter_config.json")
    has_metadata = any((checkpoint_dir / name).is_file() for name in metadata_files)
    if not has_metadata:
        return (False, "missing metadata")

    direct_weight_files = (
        "model.safetensors",
        "pytorch_model.bin",
        "adapter_model.safetensors",
        "adapter_model.bin",
    )
    if any((checkpoint_dir / name).is_file() for name in direct_weight_files):
        return (True, "direct weights")

    indexed_weight_files = (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    for index_name in indexed_weight_files:
        index_path = checkpoint_dir / index_name
        if index_file_has_all_shards(index_path, checkpoint_dir):
            return (True, f"indexed weights via {index_name}")
        if index_path.is_file():
            return (False, f"incomplete shards for {index_name}")

    return (False, "missing weights")


def top_level_file_snapshot(checkpoint_dir: Path) -> tuple[list[dict[str, Any]], int]:
    files: list[dict[str, Any]] = []
    newest_mtime = 0
    for entry in sorted(checkpoint_dir.iterdir(), key=lambda item: item.name):
        if not entry.is_file():
            continue
        stat_result = entry.stat()
        file_mtime = int(stat_result.st_mtime)
        newest_mtime = max(newest_mtime, file_mtime)
        files.append(
            {
                "name": entry.name,
                "size": stat_result.st_size,
                "mtime": file_mtime,
            }
        )
    return (files, newest_mtime)


def observe_checkpoint(
    state: dict[str, Any],
    task: CheckpointTask,
    *,
    now_epoch: int,
    stable_age_seconds: int,
    stable_confirmation_polls: int,
) -> tuple[dict[str, Any], bool]:
    files, newest_mtime = top_level_file_snapshot(task.checkpoint_path)
    fingerprint = stable_hash(files)
    observations = state["observations"]
    previous = observations.get(task_key(task), {})

    stable_poll_count = 1
    if previous.get("fingerprint") == fingerprint:
        stable_poll_count = int(previous.get("stable_poll_count", 0)) + 1

    complete, reason = checkpoint_has_complete_artifacts(task.checkpoint_path)
    ready = (
        complete
        and newest_mtime > 0
        and (now_epoch - newest_mtime) >= stable_age_seconds
        and stable_poll_count >= stable_confirmation_polls
    )

    observations[task_key(task)] = {
        "fingerprint": fingerprint,
        "stable_poll_count": stable_poll_count,
        "newest_mtime_epoch": newest_mtime,
        "artifact_status": reason,
        "complete": complete,
        "ready": ready,
        "last_seen_time": now_utc_iso(),
    }
    return (state, ready)


def collect_model_roots(base_dir: Path) -> list[Path]:
    if not base_dir.is_dir():
        return []
    return sorted((entry for entry in base_dir.iterdir() if entry.is_dir()), key=lambda item: item.name)


def iter_checkpoint_dirs(model_root: Path) -> list[tuple[int, str, Path]]:
    checkpoints: list[tuple[int, str, Path]] = []
    if not model_root.is_dir():
        return checkpoints
    for entry in model_root.iterdir():
        if not entry.is_dir():
            continue
        step = parse_checkpoint_step(entry.name)
        if step is None:
            continue
        checkpoints.append((step, entry.name, entry))

    checkpoints.sort(key=lambda item: (item[0], item[1]))
    return checkpoints


def build_candidate_tasks(config: WatcherConfig) -> list[CheckpointTask]:
    tasks: list[CheckpointTask] = []

    roots: list[tuple[str, Path]] = []
    if config.include_sft:
        roots.append(("sft", config.sft_root))
    if config.include_rl:
        roots.append(("rl", config.rl_root))

    for root_kind, root_dir in roots:
        for model_root in collect_model_roots(root_dir):
            model_name = model_root.name
            if not matches_filter(model_name, config.model_filter):
                continue
            profile = resolve_eval_profile(config, model_name)
            if profile is None:
                continue
            test_data_path = Path(str(profile["test_data_path"]))
            index_path = Path(str(profile["index_path"]))
            if not test_data_path.is_file() or not index_path.is_file():
                continue

            for checkpoint_step, checkpoint_name, checkpoint_path in iter_checkpoint_dirs(model_root):
                tasks.append(
                    CheckpointTask(
                        root_kind=root_kind,
                        model_root=model_root,
                        model_name=model_name,
                        checkpoint_path=checkpoint_path,
                        checkpoint_name=checkpoint_name,
                        checkpoint_step=checkpoint_step,
                        result_dir=config.results_root / model_name / checkpoint_name,
                        category=str(profile["category"]),
                        test_data_path=test_data_path,
                        index_path=index_path,
                        data_profile=str(profile["data_profile"]),
                        cb_width=str(profile["cb_width"]),
                    )
                )

    tasks.sort(key=lambda item: (item.model_name, item.checkpoint_step, item.checkpoint_name))
    return tasks


def scan_pending_tasks(
    config: WatcherConfig,
    state: dict[str, Any],
    *,
    now_epoch: int | None = None,
) -> tuple[dict[str, Any], list[CheckpointTask]]:
    if now_epoch is None:
        now_epoch = int(time.time())

    state = normalize_watch_state(state)
    ready_tasks: list[CheckpointTask] = []

    for task in build_candidate_tasks(config):
        if task_already_completed(config, task):
            continue
        if task_key(task) in state["failed_tasks"]:
            continue
        state, ready = observe_checkpoint(
            state,
            task,
            now_epoch=now_epoch,
            stable_age_seconds=config.stable_age_seconds,
            stable_confirmation_polls=config.stable_confirmation_polls,
        )
        if ready:
            ready_tasks.append(task)

    state["last_scan_time"] = now_utc_iso()
    return (state, ready_tasks)


def record_task_failure(
    state: dict[str, Any],
    task: CheckpointTask,
    *,
    exit_code: int,
    command: list[str],
    failed_at: str,
) -> dict[str, Any]:
    state = normalize_watch_state(state)
    state["failed_tasks"][task_key(task)] = {
        "root_kind": task.root_kind,
        "model_name": task.model_name,
        "checkpoint_name": task.checkpoint_name,
        "checkpoint_step": task.checkpoint_step,
        "command": command,
        "exit_code": exit_code,
        "failed_at": failed_at,
    }
    state["current_task"] = None
    state["last_execution_time"] = failed_at
    return state


def record_task_success(state: dict[str, Any], completed_at: str) -> dict[str, Any]:
    state = normalize_watch_state(state)
    state["current_task"] = None
    state["last_execution_time"] = completed_at
    return state


def serialize_task(task: CheckpointTask) -> dict[str, Any]:
    payload = asdict(task)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def build_idle_gpu_holder(config: WatcherConfig) -> TorchIdleGpuHolder | None:
    if not config.idle_hold_enabled:
        return None
    return TorchIdleGpuHolder(
        cuda_list=config.cuda_list,
        memory_ratio=config.idle_hold_memory_ratio,
    )


def log_ready_tasks(ready_tasks: list[CheckpointTask]) -> None:
    logging.info("Ready task count: %d", len(ready_tasks))
    for index, task in enumerate(ready_tasks, start=1):
        resolver_kind = "manifest" if str(task.data_profile).startswith("manifest:") else "heuristic"
        logging.info(
            "Ready task [%d/%d] model=%s checkpoint=%s category=%s resolver=%s profile=%s",
            index,
            len(ready_tasks),
            task.model_name,
            task.checkpoint_name,
            task.category,
            resolver_kind,
            task.data_profile,
        )


def log_planned_tasks(tasks: list[CheckpointTask]) -> None:
    logging.info("Planned task count: %d", len(tasks))
    for index, task in enumerate(tasks, start=1):
        logging.info(
            "Plan task [%d/%d] model=%s checkpoint=%s category=%s data_profile=%s test_data=%s index=%s",
            index,
            len(tasks),
            task.model_name,
            task.checkpoint_name,
            task.category,
            task.data_profile,
            task.test_data_path,
            task.index_path,
        )


def run_task(config: WatcherConfig, state: dict[str, Any], task: CheckpointTask, dry_run: bool) -> dict[str, Any]:
    command = [
        "bash",
        str(config.eval_script),
        "--checkpoint-path",
        str(task.checkpoint_path),
        "--test-data-path",
        str(task.test_data_path),
        "--index-path",
        str(task.index_path),
        "--cuda-list",
        config.cuda_list,
        "--python-bin",
        config.python_bin,
        "--output-dir",
        str(task.result_dir),
    ]

    started_at = now_utc_iso()
    state["current_task"] = {
        "task": serialize_task(task),
        "command": command,
        "started_at": started_at,
        "dry_run": dry_run,
    }
    save_json_atomic(config.state_path, state)

    logging.info(
        "Selected task model=%s checkpoint=%s category=%s profile=%s",
        task.model_name,
        task.checkpoint_name,
        task.category,
        task.data_profile,
    )
    if dry_run:
        logging.info("Dry-run command: %s", " ".join(command))
        return record_task_success(state, started_at)

    completed = subprocess.run(command, cwd=config.repo_root, check=False)
    finished_at = now_utc_iso()
    if completed.returncode != 0:
        logging.error(
            "Evaluation failed model=%s checkpoint=%s exit_code=%d",
            task.model_name,
            task.checkpoint_name,
            completed.returncode,
        )
        return record_task_failure(
            state,
            task,
            exit_code=completed.returncode,
            command=command,
            failed_at=finished_at,
        )

    logging.info("Evaluation finished model=%s checkpoint=%s", task.model_name, task.checkpoint_name)
    return record_task_success(state, finished_at)


def execute_watch_iteration(
    config: WatcherConfig,
    state: dict[str, Any],
    *,
    dry_run: bool,
    idle_holder: TorchIdleGpuHolder | Any | None,
) -> dict[str, Any]:
    state, ready_tasks = scan_pending_tasks(config, state)
    save_json_atomic(config.state_path, state)

    if ready_tasks:
        log_ready_tasks(ready_tasks)
        if idle_holder is not None:
            idle_holder.stop()
            if config.idle_hold_release_grace_seconds > 0:
                time.sleep(config.idle_hold_release_grace_seconds)
        state = run_task(config, state, ready_tasks[0], dry_run=dry_run)
        save_json_atomic(config.state_path, state)
        return state

    if idle_holder is not None:
        idle_holder.ensure_running()

    logging.info("No ready tasks. Sleeping %ss", config.poll_interval_seconds)
    time.sleep(config.poll_interval_seconds)
    return state


def execute_once(config: WatcherConfig, *, dry_run: bool) -> int:
    if dry_run:
        state = load_watch_state(config.state_path)
        planned_tasks = [
            task
            for task in build_candidate_tasks(config)
            if not task_already_completed(config, task) and task_key(task) not in state["failed_tasks"]
        ]
        log_planned_tasks(planned_tasks)
        return 0

    state = load_watch_state(config.state_path)
    state, ready_tasks = scan_pending_tasks(config, state)
    save_json_atomic(config.state_path, state)

    log_ready_tasks(ready_tasks)
    if not ready_tasks:
        return 0

    state = run_task(config, state, ready_tasks[0], dry_run=dry_run)
    save_json_atomic(config.state_path, state)
    failed_key = task_key(ready_tasks[0])
    if failed_key in state["failed_tasks"]:
        return 1
    return 0


def execute_watch(config: WatcherConfig, *, dry_run: bool) -> int:
    lock_path = config.state_path.parent / ".watch.lock"
    with exclusive_lock(lock_path):
        idle_holder = build_idle_gpu_holder(config)
        try:
            while True:
                state = load_watch_state(config.state_path)
                execute_watch_iteration(
                    config,
                    state,
                    dry_run=dry_run,
                    idle_holder=idle_holder,
                )
        except KeyboardInterrupt:
            logging.info("Watcher interrupted")
            return 130
        finally:
            if idle_holder is not None:
                idle_holder.stop()


def resolve_path(path_str: str, base: Path) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def build_config_from_args(args: argparse.Namespace, cwd: Path) -> WatcherConfig:
    repo_root = resolve_path(args.repo_root, cwd)
    if not 0 < args.idle_hold_memory_ratio <= 1:
        raise ValueError("--idle-hold-memory-ratio must be within (0, 1]")
    if args.idle_hold_release_grace_seconds < 0:
        raise ValueError("--idle-hold-release-grace-seconds must be >= 0")
    return WatcherConfig(
        repo_root=repo_root,
        results_root=resolve_path(args.results_root, cwd),
        eval_script=resolve_path(args.eval_script, cwd),
        python_bin=args.python_bin,
        cuda_list=args.cuda_list,
        data_root=resolve_path(args.data_root, cwd),
        auto_data_mapping=args.auto_data_mapping == "1",
        sft_root=resolve_path(args.sft_root, cwd),
        rl_root=resolve_path(args.rl_root, cwd),
        include_sft=args.include_sft == "1",
        include_rl=args.include_rl == "1",
        model_filter=args.model_filter,
        force_reeval=args.force_reeval == "1",
        allow_heuristic_fallback=args.allow_heuristic_fallback == "1",
        stable_age_seconds=args.stable_age_seconds,
        stable_confirmation_polls=args.stable_confirmation_polls,
        poll_interval_seconds=args.poll_interval_seconds,
        idle_hold_enabled=args.idle_hold_enabled == "1",
        idle_hold_memory_ratio=args.idle_hold_memory_ratio,
        idle_hold_release_grace_seconds=args.idle_hold_release_grace_seconds,
        state_path=resolve_path(args.state_path, cwd),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Checkpoint evaluation watcher")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_flags(target: argparse.ArgumentParser) -> None:
        target.add_argument("--repo-root", default=".")
        target.add_argument("--results-root", default="./results")
        target.add_argument("--eval-script", default="./scripts/eval/evaluate_checkpoint.sh")
        target.add_argument("--python-bin", default="python")
        target.add_argument("--cuda-list", default="0 1 2 3")
        target.add_argument("--data-root", default="./data")
        target.add_argument("--auto-data-mapping", default="1")
        target.add_argument("--sft-root", default="./saves/qwen2.5-3b/full")
        target.add_argument("--rl-root", default="./rl_outputs")
        target.add_argument("--include-sft", default="1")
        target.add_argument("--include-rl", default="1")
        target.add_argument("--model-filter", default="")
        target.add_argument("--force-reeval", default="0")
        target.add_argument(
            "--allow-heuristic-fallback",
            default="1",
        )
        target.add_argument("--stable-age-seconds", type=int, default=180)
        target.add_argument(
            "--stable-confirmation-polls",
            type=int,
            default=2,
        )
        target.add_argument(
            "--poll-interval-seconds",
            type=int,
            default=60,
        )
        target.add_argument(
            "--idle-hold-enabled",
            default="1",
        )
        target.add_argument(
            "--idle-hold-memory-ratio",
            type=float,
            default=0.95,
        )
        target.add_argument(
            "--idle-hold-release-grace-seconds",
            type=int,
            default=5,
        )
        target.add_argument(
            "--state-path",
            default="state/evaluate_all_checkpoints/watch_state.json",
        )
        target.add_argument("--dry-run", action="store_true")
        target.add_argument("--log-level", default="INFO")

    once = subparsers.add_parser("once", help="Run one scan, and if ready, execute one checkpoint")
    add_common_flags(once)

    watch = subparsers.add_parser("watch", help="Run the watcher loop")
    add_common_flags(watch)

    idle_hold = subparsers.add_parser("idle-hold", help=argparse.SUPPRESS)
    idle_hold.add_argument("--cuda-list", required=True)
    idle_hold.add_argument("--memory-ratio", type=float, required=True)
    idle_hold.add_argument("--log-level", default="INFO")

    return parser


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)

    if args.command == "idle-hold":
        return execute_idle_hold(args.cuda_list, args.memory_ratio)

    config = build_config_from_args(args, Path.cwd())

    if not config.eval_script.is_file():
        raise FileNotFoundError(f"eval script not found: {config.eval_script}")

    if args.command == "once":
        return execute_once(config, dry_run=args.dry_run)
    if args.command == "watch":
        return execute_watch(config, dry_run=args.dry_run)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
