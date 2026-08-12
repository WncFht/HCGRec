#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

DATA_VARIANT_DEFAULT="${DATA_VARIANT_DEFAULT:-Games_grec_index}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/saves/qwen2.5-3b/full/Games-grec-genrec-aligned-sft-qwen4B-4-256-dsz3-8gpu}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/rl_outputs/Games-grec-genrec-ndcg-from-sft}"
RUN_NAME="${RUN_NAME:-games_grec_genrec_ndcg_from_sft}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
MAIN_PORT="${MAIN_PORT:-29517}"
GRAD_ACC="${GRAD_ACC:-2}"
DEFAULT_CUDA_VISIBLE_DEVICES="${DEFAULT_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

LAUNCH_ARGS=(
  --repo-root "$REPO_ROOT"
  --data-variant "$DATA_VARIANT_DEFAULT"
  --model-path "$MODEL_PATH"
  --output-dir "$OUTPUT_DIR"
  --run-name "$RUN_NAME"
  --num-processes "$NUM_PROCESSES"
  --main-port "$MAIN_PORT"
  --grad-acc "$GRAD_ACC"
  --default-cuda-visible-devices "$DEFAULT_CUDA_VISIBLE_DEVICES"
  --reward-mode ranking
)

exec bash "${REPO_ROOT}/scripts/experiments/_canonical_rl_launcher.sh" \
  "${LAUNCH_ARGS[@]}" \
  "$@"
