#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

DATA_VARIANT_DEFAULT="${DATA_VARIANT_DEFAULT:-Instruments_grec_index}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/saves/qwen2.5-3b/full/Instruments-grec-genrec-aligned-sft-qwen4B-4-256-dsz3-8gpu/checkpoint-2751}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/rl_outputs/Instruments-grec-genrec-fixed-from-sft}"
RUN_NAME="${RUN_NAME:-instruments_grec_genrec_fixed_from_sft}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
MAIN_PORT="${MAIN_PORT:-29516}"
GRAD_ACC="${GRAD_ACC:-2}"
DEFAULT_CUDA_VISIBLE_DEVICES="${DEFAULT_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
HINT_CE_LOSS_COEF="${HINT_CE_LOSS_COEF:-0.0}"

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
  --reward-mode rule_only
)

FIXED_HINT_ARGS=(
  --fixed-hint
  --hint-ce-loss-coef "$HINT_CE_LOSS_COEF"
)

exec bash "${REPO_ROOT}/scripts/experiments/_canonical_rl_launcher.sh" \
  "${LAUNCH_ARGS[@]}" \
  "${FIXED_HINT_ARGS[@]}" \
  "$@"
