#!/usr/bin/env bash
set -eo pipefail

# Local ops helper for the original remote environment.
# Not part of the public reproduction workflow.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
INSTANCE="remote_eval"
LOG_DIR="${REPO_ROOT}/log/evaluate_all_checkpoints"
LOG_FILE="${LOG_DIR}/${INSTANCE}.log"
CUDA_VISIBLE_DEVICES="0,1,2,3"
CUDA_LIST="0 1 2 3"
ALLOW_HEURISTIC_FALLBACK="0"
IDLE_HOLD_ENABLED="1"
IDLE_HOLD_MEMORY_RATIO="0.90"
IDLE_HOLD_RELEASE_GRACE_SECONDS="10"
POLL_INTERVAL_SECONDS="60"
STABLE_AGE_SECONDS="180"
STABLE_CONFIRMATION_POLLS="2"

cd "$REPO_ROOT"

mkdir -p "$LOG_DIR"

echo "[INFO] repo_root=$REPO_ROOT"
echo "[INFO] instance=$INSTANCE"
echo "[INFO] log_file=$LOG_FILE"
echo "[INFO] launch_mode=foreground-once"

export CUDA_VISIBLE_DEVICES
exec > >(tee -a "$LOG_FILE") 2>&1
exec bash "$REPO_ROOT/scripts/eval/evaluate_all_checkpoints.sh" once \
  --cuda-list "$CUDA_LIST" \
  --allow-heuristic-fallback "$ALLOW_HEURISTIC_FALLBACK" \
  --idle-hold-enabled "$IDLE_HOLD_ENABLED" \
  --idle-hold-memory-ratio "$IDLE_HOLD_MEMORY_RATIO" \
  --idle-hold-release-grace-seconds "$IDLE_HOLD_RELEASE_GRACE_SECONDS" \
  --poll-interval-seconds "$POLL_INTERVAL_SECONDS" \
  --stable-age-seconds "$STABLE_AGE_SECONDS" \
  --stable-confirmation-polls "$STABLE_CONFIRMATION_POLLS"
