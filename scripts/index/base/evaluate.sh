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

: "${INDEX_TRAIN_ROOT:=./index_train_runs}"

: "${DATASET:=Instruments}"
: "${MODEL_NAME:=qwen7B}"
: "${MODEL_FILE:=best_collision_model.pth}"
: "${TIMESTAMP:=}"
: "${CKPT_BASE_DIR:=}"
: "${CKPT_PATH:=}"

if [[ -z "$CKPT_PATH" ]]; then
  if [[ -z "$CKPT_BASE_DIR" ]]; then
    if [[ -z "$TIMESTAMP" ]]; then
      echo "Error: please set one of CKPT_PATH or CKPT_BASE_DIR (or provide TIMESTAMP to auto-build CKPT_BASE_DIR)."
      exit 1
    fi
    CKPT_BASE_DIR="${INDEX_TRAIN_ROOT}/${DATASET}/index/${MODEL_NAME}/${TIMESTAMP}"
  fi
  CKPT_PATH="${CKPT_BASE_DIR}/${MODEL_FILE}"
fi

: "${DEVICE:=cuda:0}"
: "${BATCH_SIZE:=2048}"

if [ ! -f "$CKPT_PATH" ]; then
  echo "Error: Checkpoint file not found at '$CKPT_PATH'"
  exit 1
fi

echo "======================================================"
echo "Starting evaluation for checkpoint: $CKPT_PATH"
echo "Device: $DEVICE, Batch Size: $BATCH_SIZE"
echo "======================================================"

python3 -m index.evaluate_index \
  --ckpt_path "$CKPT_PATH" \
  --device "$DEVICE" \
  --batch_size "$BATCH_SIZE"

echo "======================================================"
echo "Evaluation finished."
echo "======================================================"
