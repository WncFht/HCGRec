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

# K-Means ablation mode:
# - large: kmeans_init=true,  large_scale_kmeans=true
# - small: kmeans_init=true,  large_scale_kmeans=false
# - none:  kmeans_init=false, large_scale_kmeans=false
: "${KMEANS_MODE:=none}"

case "${KMEANS_MODE}" in
  large)
    KMEANS_INIT_ARG=true
    LARGE_SCALE_KMEANS_ARG=true
    WANDB_SUFFIX="-LargeKMeans"
    ;;
  small)
    KMEANS_INIT_ARG=true
    LARGE_SCALE_KMEANS_ARG=false
    WANDB_SUFFIX="-SmallKMeans"
    ;;
  none)
    KMEANS_INIT_ARG=false
    LARGE_SCALE_KMEANS_ARG=false
    WANDB_SUFFIX="-NoKMeans"
    ;;
  *)
    echo "Error: invalid KMEANS_MODE='${KMEANS_MODE}', expected one of: large|small|none"
    exit 1
    ;;
esac

: "${ROOT_DIR:=${GREC_ROOT}}"
: "${DATASET:=Instruments}"
: "${MODEL_NAME:=qwen3-embedding-4B}"
: "${DATA_PATH:=${ROOT_DIR}/data/${DATASET}/${DATASET}.emb-${MODEL_NAME}-td.npy}"

: "${USE_MULTI_DATASETS:=false}"
: "${NPROC_PER_NODE:=1}"
: "${EPOCHS:=10000}"
: "${BATCH_SIZE:=2048}"
: "${LR:=1e-3}"

: "${INDEX_N_LAYERS:=4}"
: "${INDEX_CODEBOOK_SIZE:=256}"
: "${INDEX_LAST_SK_EPSILON:=0.003}"
: "${INDEX_KMEANS_ITERS:=100}"

: "${USE_WANDB:=False}"
: "${WANDB_PROJECT:=grec_index}"
: "${WANDB_RUN_NAME:=${DATASET}-${MODEL_NAME}${WANDB_SUFFIX}}"

mkdir -p ./log
: "${LOG_FILE:=./log/index_kmeans_${KMEANS_MODE}_$(date +%Y%m%d%H%M%S).log}"

export ROOT_DIR DATASET MODEL_NAME DATA_PATH
export USE_MULTI_DATASETS NPROC_PER_NODE EPOCHS BATCH_SIZE LR
export INDEX_N_LAYERS INDEX_CODEBOOK_SIZE INDEX_LAST_SK_EPSILON INDEX_KMEANS_ITERS
export KMEANS_INIT_ARG LARGE_SCALE_KMEANS_ARG
export USE_WANDB WANDB_PROJECT WANDB_RUN_NAME LOG_FILE

nohup bash "$GREC_ROOT/scripts/index/base/train.sh" >/dev/null 2>&1 &
PID=$!

echo "K-Means ablation launched in background. PID=${PID}"
echo "Mode: ${KMEANS_MODE}"
echo "Log file: ${LOG_FILE}"
echo "W&B Run Name: ${WANDB_RUN_NAME}"
