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

: "${ROOT_DIR:=${GREC_ROOT}}"
: "${DATASET:=Instruments}"
: "${MODEL_NAME:=qwen3-embedding-4B}"
: "${DATA_PATH:=${ROOT_DIR}/data/${DATASET}/${DATASET}.emb-${MODEL_NAME}-td.npy}"

: "${NPROC_PER_NODE:=1}"
: "${EPOCHS:=10000}"
: "${BATCH_SIZE:=2048}"
: "${LR:=1e-4}"
: "${USE_MULTI_DATASETS:=false}"

: "${INDEX_N_LAYERS:=4}"
: "${INDEX_CODEBOOK_SIZE:=256}"
: "${INDEX_LAST_SK_EPSILON:=0.003}"
: "${INDEX_KMEANS_ITERS:=100}"

: "${KMEANS_INIT_ARG:=true}"
: "${LARGE_SCALE_KMEANS_ARG:=true}"

: "${USE_WANDB:=False}"
: "${WANDB_PROJECT:=grec_index}"
: "${WANDB_RUN_NAME:=${DATASET}-${MODEL_NAME}-nohup}"

mkdir -p ./log/index
: "${LOG_FILE:=./log/index/index_$(date +%Y%m%d%H%M%S).log}"

export ROOT_DIR DATASET MODEL_NAME DATA_PATH
export NPROC_PER_NODE EPOCHS BATCH_SIZE LR USE_MULTI_DATASETS
export INDEX_N_LAYERS INDEX_CODEBOOK_SIZE INDEX_LAST_SK_EPSILON INDEX_KMEANS_ITERS
export KMEANS_INIT_ARG LARGE_SCALE_KMEANS_ARG
export USE_WANDB WANDB_PROJECT WANDB_RUN_NAME LOG_FILE

nohup bash "$GREC_ROOT/scripts/index/base/train.sh" >/dev/null 2>&1 &
PID=$!

echo "Index training launched in background. PID=${PID}"
echo "Log file: ${LOG_FILE}"
echo "W&B Run Name: ${WANDB_RUN_NAME}"
