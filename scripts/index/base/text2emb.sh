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

: "${CUDA_VISIBLE_DEVICES:=0,1,2,3}"
export CUDA_VISIBLE_DEVICES

: "${HOME_DIR:=${GREC_ROOT}}"
: "${MODEL_PATH:=meta-llama/Llama-3.1-8B-Instruct}"
: "${PLM_NAME:=Llama-3.1-8B-Instruct}"
: "${NUM_PROCESSES:=4}"
: "${BATCH_SIZE:=256}"   # will auto-reduce on OOM
: "${MAX_SENT_LEN:=2048}"
: "${FORCE_REBUILD:=0}"  # set to 1 to overwrite existing embeddings
: "${TMP_DIR:=${GREC_ROOT}/temp}"         # optional, e.g. /tmp

# Space-separated dataset names.
: "${DATASETS:=Arts Automotive Cell Games Pet Sports Tools Toys Instruments}"
read -r -a DATASET_LIST <<< "$DATASETS"
if [ ${#DATASET_LIST[@]} -eq 0 ]; then
  echo "Error: DATASETS is empty."
  exit 1
fi

# Helps avoid CUDA memory fragmentation on long runs.
: "${PYTORCH_CUDA_ALLOC_CONF:=expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF

echo "Embedding extraction config:"
echo "  PLM_NAME=${PLM_NAME}"
echo "  MODEL_PATH=${MODEL_PATH}"
echo "  NUM_PROCESSES=${NUM_PROCESSES}, BATCH_SIZE=${BATCH_SIZE}, MAX_SENT_LEN=${MAX_SENT_LEN}"
echo "  DATASETS=${DATASET_LIST[*]}"

for dataset in "${DATASET_LIST[@]}"; do
    DATASET_DIR="${GREC_ROOT}/data/${dataset}"
    echo "========== Start ${dataset} =========="
    echo "ROOT: ${DATASET_DIR}"

    OUT_EMB="${DATASET_DIR}/${dataset}.emb-${PLM_NAME}-td.npy"
    OUT_IDS="${DATASET_DIR}/${dataset}.emb-${PLM_NAME}-td.ids.json"

    echo "OUT_EMB=${OUT_EMB}"
    echo "OUT_IDS=${OUT_IDS}"
    if [[ "$FORCE_REBUILD" != "1" && -f "$OUT_EMB" ]]; then
        if [[ -f "$OUT_IDS" ]]; then
            echo "Skip ${dataset} (embedding exists): ${OUT_EMB}"
            continue
        fi
        echo "Embedding exists but ids missing, generating ids only: ${OUT_IDS}"
        python3 -m index.build_embedding_ids \
            --dataset "$dataset" \
            --root "$DATASET_DIR" \
            --plm_name "$PLM_NAME" \
            --emb_path "$OUT_EMB"
        echo "========== Done ${dataset} (ids only) =========="
        continue
    fi

    RUN_ID="$(date +%s%N)"
    TMP_ARGS=()
    if [[ -n "$TMP_DIR" ]]; then
        TMP_ARGS+=(--tmp_dir "$TMP_DIR")
    fi
    accelerate launch --num_processes "$NUM_PROCESSES" -m index.build_embeddings \
        --dataset "$dataset" \
        --root "$DATASET_DIR" \
        --plm_name "$PLM_NAME" \
        --run_id "$RUN_ID" \
        --batch_size "$BATCH_SIZE" \
        --max_sent_len "$MAX_SENT_LEN" \
        "${TMP_ARGS[@]}" \
        --plm_checkpoint "$MODEL_PATH"

    echo "========== Done ${dataset} =========="
done
