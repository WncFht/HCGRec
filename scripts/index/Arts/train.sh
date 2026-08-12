#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_GREC_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
: "${GREC_ROOT:=$DEFAULT_GREC_ROOT}"

if [[ ! -d "$GREC_ROOT" ]]; then
  echo "Error: GREC_ROOT does not exist: $GREC_ROOT" >&2
  exit 1
fi

cd "$GREC_ROOT" || exit 1

# Single dataset: Arts
: "${ROOT_DIR:=${GREC_ROOT}}"
: "${MODEL_NAME:=qwen3-embedding-4B}"
: "${USE_MULTI_DATASETS:=false}"
: "${DATASET:=Arts}"

# RQ config: 4 layers, 256 codebook
: "${INDEX_N_LAYERS:=4}"
: "${INDEX_CODEBOOK_SIZE:=256}"
: "${INDEX_LAST_SK_EPSILON:=0.003}"
: "${INDEX_RUN_SCRIPT_DIR:=Arts-qwen3-embedding-4B-rq4_cb256-256-256-256_sk0.0-0.0-0.0-0.003}"

export ROOT_DIR MODEL_NAME USE_MULTI_DATASETS DATASET INDEX_RUN_SCRIPT_DIR
export INDEX_N_LAYERS INDEX_CODEBOOK_SIZE INDEX_LAST_SK_EPSILON

# Optional overrides (examples):
#   NPROC_PER_NODE=4 BATCH_SIZE=256 EPOCHS=500
bash "$GREC_ROOT/scripts/index/base/train.sh"
