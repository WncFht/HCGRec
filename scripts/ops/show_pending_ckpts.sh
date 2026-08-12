#!/usr/bin/env bash
set -eo pipefail

# Local ops helper for the original remote environment.
# Not part of the public reproduction workflow.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
INSTANCE="remote_eval_plan"
LOG_DIR="${REPO_ROOT}/log/evaluate_all_checkpoints"
LOG_FILE="${LOG_DIR}/${INSTANCE}.log"
CUDA_VISIBLE_DEVICES="0,1,2,3"
CUDA_LIST="0 1 2 3"
ALLOW_HEURISTIC_FALLBACK="0"

cd "$REPO_ROOT"

mkdir -p "$LOG_DIR"

echo "[INFO] repo_root=$REPO_ROOT"
echo "[INFO] instance=$INSTANCE"
echo "[INFO] log_file=$LOG_FILE"
echo "[INFO] launch_mode=foreground-dry-run-once"

export CUDA_VISIBLE_DEVICES
exec > >(tee -a "$LOG_FILE") 2>&1
exec bash "$REPO_ROOT/scripts/eval/evaluate_all_checkpoints.sh" once \
  --dry-run \
  --cuda-list "$CUDA_LIST" \
  --allow-heuristic-fallback "$ALLOW_HEURISTIC_FALLBACK"
