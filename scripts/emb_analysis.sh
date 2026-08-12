#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/saves/qwen2.5-3b/full/Instruments-grec-sft-qwen4B-4-256-dsz0/checkpoint-495}"
OUTPUT_DIR="${OUTPUT_DIR:-${MODEL_PATH}/embedding_analysis}"

cd "${REPO_ROOT}"

python3 -m hcgrec.tsne \
  --model_path "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --interactive \
  --sample_original 2000 \
  --filter_languages English Chinese Japanese Korean \
  --use_cosine \
  --method tsne
