#!/bin/bash
set -eo pipefail

# Limit BLAS threads to avoid OpenBLAS "too many memory regions" on high-core machines.
: "${INDEX_BLAS_NUM_THREADS:=32}"
export INDEX_BLAS_NUM_THREADS
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$INDEX_BLAS_NUM_THREADS}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$INDEX_BLAS_NUM_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$INDEX_BLAS_NUM_THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$INDEX_BLAS_NUM_THREADS}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_GREC_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
: "${GREC_ROOT:=$DEFAULT_GREC_ROOT}"

if [[ ! -d "$GREC_ROOT" ]]; then
  echo "Error: GREC_ROOT does not exist: $GREC_ROOT" >&2
  exit 1
fi

cd "$GREC_ROOT" || exit 1

: "${INDEX_TRAIN_ROOT:=./index_train_runs}"

# =========================
# Index quantizer config
# =========================
: "${INDEX_N_LAYERS:=4}"
: "${INDEX_CODEBOOK_SIZE:=512}"
: "${INDEX_LAST_SK_EPSILON:=0.003}"
: "${INDEX_KMEANS_ITERS:=100}"

NUM_EMB_LIST=()
SK_EPSILONS=()
for ((i = 0; i < INDEX_N_LAYERS; i++)); do
  NUM_EMB_LIST+=("$INDEX_CODEBOOK_SIZE")
  if [ "$i" -eq $((INDEX_N_LAYERS - 1)) ]; then
    SK_EPSILONS+=("$INDEX_LAST_SK_EPSILON")
  else
    SK_EPSILONS+=("0.0")
  fi
done

# =========================
# Data config
# =========================
: "${ROOT_DIR:=${GREC_ROOT}}"
: "${DATASET:=Instruments}"
: "${MODEL_NAME:=qwen3-embedding-4B}"
: "${DATA_PATH:=${ROOT_DIR}/data/$DATASET/${DATASET}.emb-${MODEL_NAME}-td.npy}"

# Multi-dataset mode
DATASETS=(${DATASETS:-Instruments})
DATA_PATHS=(${DATA_PATHS:-})
TRAIN_DATASET=$(IFS=-; echo "${DATASETS[*]}")
: "${USE_MULTI_DATASETS:=true}"   # true: use DATASETS; false: use DATA_PATH

# =========================
# Train config
# =========================
: "${DEVICE:=cuda:0}"
: "${NPROC_PER_NODE:=4}"          # >1 uses torchrun DDP
: "${MASTER_PORT:=29600}"
: "${EPOCHS:=500}"
: "${BATCH_SIZE:=256}"            # per-GPU batch size
: "${AUTO_LR:=true}"              # true: scale LR by global batch when LR not set
: "${BASE_LR:=1e-3}"
: "${BASE_BATCH_SIZE:=1024}"
: "${BASE_NPROC_PER_NODE:=1}"
: "${WEIGHT_DECAY:=1e-4}"
: "${LR_SCHEDULER_TYPE:=linear}"
: "${DROPOUT_PROB:=0.0}"
: "${BN:=False}"
: "${E_DIM:=32}"
: "${QUANT_LOSS_WEIGHT:=1.0}"
: "${BETA:=0.25}"
: "${LAYERS:=2048 1024 512 256 128 64}"

: "${KMEANS_INIT_ARG:=true}"
: "${LARGE_SCALE_KMEANS_ARG:=true}"
: "${KMEANS_ITERS:=$INDEX_KMEANS_ITERS}"

: "${USE_WANDB:=False}"
: "${WANDB_PROJECT:=grec_index}"

BASE_GLOBAL_BATCH=$((BASE_BATCH_SIZE * BASE_NPROC_PER_NODE))
GLOBAL_BATCH=$((BATCH_SIZE * NPROC_PER_NODE))

if [ "${AUTO_LR,,}" = "true" ] && [ -z "${LR:-}" ]; then
  LR="$(python3 - <<PY
base_lr=float("${BASE_LR}")
gb=int("${GLOBAL_BATCH}")
base_gb=int("${BASE_GLOBAL_BATCH}")
print(f"{base_lr * gb / base_gb:.10g}")
PY
)"
fi
: "${LR:=$BASE_LR}"

mkdir -p ./log
LOG_FILE="${LOG_FILE:-./log/index_train_$(date +%Y%m%d%H%M%S).log}"

DATA_ARGS=()
CKPT_ROOT=""
RUN_NAME=""

if [ "${USE_MULTI_DATASETS,,}" = "true" ]; then
  if [ ${#DATASETS[@]} -eq 0 ]; then
    echo "Error: USE_MULTI_DATASETS=true but DATASETS is empty."
    exit 1
  fi
  if [ ${#DATA_PATHS[@]} -eq 0 ]; then
    for dataset_name in "${DATASETS[@]}"; do
      DATA_PATHS+=("${ROOT_DIR}/data/$dataset_name/${dataset_name}.emb-${MODEL_NAME}-td.npy")
    done
  fi
  DATA_ARGS=(--data_paths "${DATA_PATHS[@]}")
  CKPT_ROOT="${INDEX_TRAIN_ROOT}/$TRAIN_DATASET/index/$MODEL_NAME/"
  RUN_NAME="${TRAIN_DATASET}-${MODEL_NAME}"
else
  DATA_ARGS=(--data_path "$DATA_PATH")
  CKPT_ROOT="${INDEX_TRAIN_ROOT}/$DATASET/index/$MODEL_NAME/"
  RUN_NAME="${DATASET}-${MODEL_NAME}"
fi

VQ_TAG="$(IFS=-; echo "${NUM_EMB_LIST[*]}")"
SK_TAG="$(IFS=-; echo "${SK_EPSILONS[*]}")"
KM_TAG="km${KMEANS_INIT_ARG}-lkm${LARGE_SCALE_KMEANS_ARG}-kmi${KMEANS_ITERS}"
CKPT_TAG_DEFAULT="rq${#NUM_EMB_LIST[@]}_cb${VQ_TAG}_sk${SK_TAG}_${KM_TAG}"
: "${CKPT_TAG:=$CKPT_TAG_DEFAULT}"

: "${CKPT_DIR:=${CKPT_ROOT}${CKPT_TAG}/}"
RUN_NAME="${RUN_NAME}-${CKPT_TAG}"

RUN_SCRIPT_DIR_NAME="${INDEX_RUN_SCRIPT_DIR:-base}"
WANDB_RUN_NAME_DEFAULT="${RUN_SCRIPT_DIR_NAME}-bs${BATCH_SIZE}-lr${LR}-ep${EPOCHS}-np${NPROC_PER_NODE}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-$WANDB_RUN_NAME_DEFAULT}"

echo "Train config: NPROC_PER_NODE=${NPROC_PER_NODE}, BATCH_SIZE=${BATCH_SIZE} (global=${GLOBAL_BATCH}), LR=${LR}, AUTO_LR=${AUTO_LR}"
echo "Train data mode: USE_MULTI_DATASETS=${USE_MULTI_DATASETS}, MODEL_NAME=${MODEL_NAME}"
echo "Index checkpoint root: INDEX_TRAIN_ROOT=${INDEX_TRAIN_ROOT}"
echo "NUM_EMB_LIST=${NUM_EMB_LIST[*]}"
echo "SK_EPSILONS=${SK_EPSILONS[*]}"
echo "CKPT_DIR=${CKPT_DIR}"

LAUNCH_CMD=(python3 -u -m index.train_index)
if [ "${NPROC_PER_NODE}" -gt 1 ]; then
  if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    CUDA_VISIBLE_DEVICES=""
    for ((i = 0; i < NPROC_PER_NODE; i++)); do
      if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES},${i}"
      else
        CUDA_VISIBLE_DEVICES="${i}"
      fi
    done
    export CUDA_VISIBLE_DEVICES
  fi
  TORCHRUN_ARGS=(--nproc_per_node "${NPROC_PER_NODE}" --master_port "${MASTER_PORT}")
  if [ -z "${MASTER_ADDR:-}" ]; then
    TORCHRUN_ARGS=(--standalone "${TORCHRUN_ARGS[@]}")
  fi
  LAUNCH_CMD=(torchrun "${TORCHRUN_ARGS[@]}" -m index.train_index)
fi

${LAUNCH_CMD[@]} \
  --lr "$LR" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --weight_decay "$WEIGHT_DECAY" \
  --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
  --dropout_prob "$DROPOUT_PROB" \
  --bn "$BN" \
  --e_dim "$E_DIM" \
  --quant_loss_weight "$QUANT_LOSS_WEIGHT" \
  --beta "$BETA" \
  --num_emb_list "${NUM_EMB_LIST[@]}" \
  --sk_epsilons "${SK_EPSILONS[@]}" \
  --layers $LAYERS \
  --kmeans_init "$KMEANS_INIT_ARG" \
  --large_scale_kmeans "$LARGE_SCALE_KMEANS_ARG" \
  --kmeans_iters "$KMEANS_ITERS" \
  --device "$DEVICE" \
  "${DATA_ARGS[@]}" \
  --ckpt_dir "$CKPT_DIR" \
  --run_name "$RUN_NAME" \
  --use_wandb "$USE_WANDB" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_name "$WANDB_RUN_NAME" \
  > >(tee "$LOG_FILE") 2>&1

echo "Index training started. Log file: $LOG_FILE"
echo "W&B Run Name: $WANDB_RUN_NAME"
