#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

YAML_PATH="${YAML_PATH:-${REPO_ROOT}/examples/train_full/Arts/arts_rec_full_sft_3b_dsz3_qwen4b_4_256_grec_genrec_aligned_8gpu.yaml}"
RUN_NAME="${RUN_NAME:-Arts-grec-genrec-aligned-qwen2.5-3b-sft-qwen4B-4-256-dsz3-8gpu}"
DEFAULT_CUDA_VISIBLE_DEVICES="${DEFAULT_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

LAUNCH_ARGS=(
  --repo-root "$REPO_ROOT"
  --yaml-path "$YAML_PATH"
  --run-name "$RUN_NAME"
  --default-cuda-visible-devices "$DEFAULT_CUDA_VISIBLE_DEVICES"
)

exec bash "${REPO_ROOT}/scripts/experiments/_canonical_sft_launcher.sh" \
  "${LAUNCH_ARGS[@]}" \
  "$@"
