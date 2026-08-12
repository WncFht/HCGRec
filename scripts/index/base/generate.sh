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

: "${DATA_ROOT:=${GREC_ROOT}/data}"

: "${MODEL_NAME:=qwen3-embedding-4B}"
: "${CKPT_PATH:=}"
if [[ -z "$CKPT_PATH" ]]; then
  echo "Error: CKPT_PATH is empty. Please export CKPT_PATH to a trained checkpoint path."
  exit 1
fi

if [[ "$CKPT_PATH" == *"/run_time/"* ]]; then
  ckpt_prefix="${CKPT_PATH%%/run_time/*}"
  ckpt_suffix="${CKPT_PATH#*/run_time/}"

  if [[ ! -d "$ckpt_prefix" ]]; then
    echo "Error: checkpoint root does not exist: $ckpt_prefix" >&2
    exit 1
  fi

  latest_run_dir="$(ls -1dt "$ckpt_prefix"/*/ 2>/dev/null | head -n 1 || true)"
  latest_run_dir="${latest_run_dir%/}"
  if [[ -z "$latest_run_dir" || ! -d "$latest_run_dir" ]]; then
    echo "Error: cannot find any run_time directory under: $ckpt_prefix" >&2
    exit 1
  fi

  CKPT_PATH="${latest_run_dir}/${ckpt_suffix}"
  echo "Resolved CKPT_PATH from run_time to latest: $CKPT_PATH"
fi

if [[ ! -f "$CKPT_PATH" ]]; then
  echo "Error: checkpoint file not found: $CKPT_PATH" >&2
  exit 1
fi

: "${USE_MULTI_DATASETS:=true}"
: "${DATASET:=Instruments}"
: "${DATA_PATH:=${DATA_ROOT}/${DATASET}/${DATASET}.emb-${MODEL_NAME}-td.npy}"
: "${DATASETS:=Arts Automotive Cell Games Pet Sports Tools Toys Instruments}"
: "${OUTPUT_SUFFIX:=}" # 留空时自动命名，包含 emb/rq/cb/ds/rid
: "${DEVICE:=cuda:0}"
: "${BATCH_SIZE:=64}"

read -r -a DATASET_LIST <<< "$DATASETS"
DATA_PATHS=(${DATA_PATHS:-})

echo "Generate config: USE_MULTI_DATASETS=${USE_MULTI_DATASETS}, MODEL_NAME=${MODEL_NAME}"
echo "CKPT_PATH=${CKPT_PATH}"
if [[ -n "$OUTPUT_SUFFIX" ]]; then
  echo "OUTPUT_SUFFIX=${OUTPUT_SUFFIX}"
else
  echo "OUTPUT_SUFFIX=<auto>"
fi

gen_one() {
  local dataset="$1"
  local data_path="$2"
  local output_dir="${DATA_ROOT}/${dataset}/"

  cmd=(
    python3 -m index.generate_indices
    --dataset "$dataset"
    --ckpt_path "$CKPT_PATH"
    --data_path "$data_path"
    --output_dir "$output_dir"
    --device "$DEVICE"
    --batch_size "$BATCH_SIZE"
  )

  if [[ -n "$OUTPUT_SUFFIX" ]]; then
    cmd+=(--output_suffix "$OUTPUT_SUFFIX")
  fi

  "${cmd[@]}"
}

if [ "${USE_MULTI_DATASETS,,}" = "true" ]; then
  if [ ${#DATASET_LIST[@]} -eq 0 ]; then
    echo "Error: USE_MULTI_DATASETS=true but DATASETS is empty."
    exit 1
  fi

  if [ ${#DATA_PATHS[@]} -eq 0 ]; then
    for dataset_name in "${DATASET_LIST[@]}"; do
      DATA_PATHS+=("${DATA_ROOT}/${dataset_name}/${dataset_name}.emb-${MODEL_NAME}-td.npy")
    done
  fi

  if [ ${#DATA_PATHS[@]} -ne ${#DATASET_LIST[@]} ]; then
    echo "Error: DATASETS and DATA_PATHS must have the same length."
    exit 1
  fi

  cmd=(
    python3 -m index.generate_indices
    --datasets "${DATASET_LIST[@]}"
    --ckpt_path "$CKPT_PATH"
    --data_paths "${DATA_PATHS[@]}"
    --output_dir "$DATA_ROOT"
    --device "$DEVICE"
    --batch_size "$BATCH_SIZE"
  )

  if [[ -n "$OUTPUT_SUFFIX" ]]; then
    cmd+=(--output_suffix "$OUTPUT_SUFFIX")
  fi

  "${cmd[@]}"
else
  gen_one "$DATASET" "$DATA_PATH"
fi
