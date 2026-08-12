#!/usr/bin/env bash
set -eo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/experiments/Arts/prepare_arts_grec_lcrec_aligned_data.sh [check|build|build-data] [--dry-run]

Modes:
  check       Inspect resolved Arts raw inputs and target output paths
  build       Build GRec-style Arts data aligned to current preprocess settings
  build-data  Alias of build

Important defaults:
  - split strategy defaults to GRec
  - history_max defaults to 20 to match LC-Rec
  - train row order defaults to forward to match LC-Rec
  - seq sample defaults to 10000
  - task3 sample defaults to -1, meaning keep all fusion train samples

Common env overrides:
  REPO_ROOT GENREC_ROOT DATA_ROOT CATEGORY INDEX_PATH
  DATA_VARIANT DATA_VARIANT_TAG OUTPUT_DIR DATASET_SUBDIR DATASET_KEY_PREFIX
  SEQ_SAMPLE TASK3_SAMPLE SEED SID_LEVELS HISTORY_MAX TRAIN_ROW_ORDER
  RL_ONLY_TASK1 RL_ONLY_TASK4 RL_ONLY_TASK5
  PYTHON_BIN
EOF
}

run_cmd() {
  printf '[CMD] '
  printf '%q ' "$@"
  echo
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../_launcher_common.sh"

REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
GENREC_ROOT="${GENREC_ROOT:-${REPO_ROOT}}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
CATEGORY="${CATEGORY:-Arts}"
INDEX_PATH="${INDEX_PATH:-${DATA_ROOT}/${CATEGORY}/Arts.index.json}"

MODE="check"
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    check|build|build-data)
      MODE="$1"
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

PYTHON_BIN="${PYTHON_BIN:-python3}"
SEQ_SAMPLE="${SEQ_SAMPLE:-10000}"
TASK3_SAMPLE="${TASK3_SAMPLE:--1}"
SEED="${SEED:-42}"
SID_LEVELS="${SID_LEVELS:--1}"
HISTORY_MAX="${HISTORY_MAX:-20}"
TRAIN_ROW_ORDER="${TRAIN_ROW_ORDER:-forward}"
SPLIT_STRATEGY="${SPLIT_STRATEGY:-grec}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
VALID_RATIO="${VALID_RATIO:-0.1}"
RL_ONLY_TASK1="${RL_ONLY_TASK1:-false}"
RL_ONLY_TASK4="${RL_ONLY_TASK4:-false}"
RL_ONLY_TASK5="${RL_ONLY_TASK5:-false}"

INDEX_FILE_NAME="$(basename "${INDEX_PATH}")"
INDEX_STEM="${INDEX_FILE_NAME%.json}"
if [[ "${INDEX_STEM}" == "${CATEGORY}."* ]]; then
  INDEX_STEM="${INDEX_STEM#${CATEGORY}.}"
fi

DATA_VARIANT_TAG="${DATA_VARIANT_TAG:-}"
if [[ -n "${DATA_VARIANT:-}" ]]; then
  RESOLVED_DATA_VARIANT="${DATA_VARIANT}"
elif [[ -n "${DATA_VARIANT_TAG}" ]]; then
  RESOLVED_DATA_VARIANT="${CATEGORY}_${SPLIT_STRATEGY}_${DATA_VARIANT_TAG}_${INDEX_STEM}"
else
  RESOLVED_DATA_VARIANT="${CATEGORY}_${SPLIT_STRATEGY}_${INDEX_STEM}"
fi

OUTPUT_DIR="${OUTPUT_DIR:-${GENREC_ROOT}/data/${RESOLVED_DATA_VARIANT}}"
DATASET_SUBDIR="${DATASET_SUBDIR:-${RESOLVED_DATA_VARIANT}}"
DATASET_KEY_PREFIX="${DATASET_KEY_PREFIX:-$(printf '%s' "${RESOLVED_DATA_VARIANT}" | sed -E 's/[^A-Za-z0-9_]+/_/g; s/_+/_/g; s/^_+//; s/_+$//')}"
DATASET_INFO_PATH="${DATASET_INFO_PATH:-${GENREC_ROOT}/data/dataset_info.json}"

print_config() {
  cat <<EOF
[INFO] MODE=${MODE}
[INFO] REPO_ROOT=${REPO_ROOT}
[INFO] GENREC_ROOT=${GENREC_ROOT}
[INFO] DATA_ROOT=${DATA_ROOT}
[INFO] CATEGORY=${CATEGORY}
[INFO] INDEX_PATH=${INDEX_PATH}
[INFO] INDEX_STEM=${INDEX_STEM}
[INFO] SPLIT_STRATEGY=${SPLIT_STRATEGY}
[INFO] RESOLVED_DATA_VARIANT=${RESOLVED_DATA_VARIANT}
[INFO] OUTPUT_DIR=${OUTPUT_DIR}
[INFO] DATASET_SUBDIR=${DATASET_SUBDIR}
[INFO] DATASET_KEY_PREFIX=${DATASET_KEY_PREFIX}
[INFO] DATASET_INFO_PATH=${DATASET_INFO_PATH}
[INFO] PYTHON_BIN=${PYTHON_BIN}
[INFO] SEQ_SAMPLE=${SEQ_SAMPLE}
[INFO] TASK3_SAMPLE=${TASK3_SAMPLE}
[INFO] SEED=${SEED}
[INFO] SID_LEVELS=${SID_LEVELS}
[INFO] HISTORY_MAX=${HISTORY_MAX}
[INFO] TRAIN_ROW_ORDER=${TRAIN_ROW_ORDER}
[INFO] TRAIN_RATIO=${TRAIN_RATIO}
[INFO] VALID_RATIO=${VALID_RATIO}
[INFO] RL_ONLY_TASK1=${RL_ONLY_TASK1}
[INFO] RL_ONLY_TASK4=${RL_ONLY_TASK4}
[INFO] RL_ONLY_TASK5=${RL_ONLY_TASK5}
[INFO] DRY_RUN=${DRY_RUN}
EOF
}

check_inputs() {
  [[ -d "${GENREC_ROOT}" ]] || { echo "[ERROR] Missing GENREC_ROOT: ${GENREC_ROOT}"; exit 1; }
  [[ -f "${INDEX_PATH}" ]] || { echo "[ERROR] Missing INDEX_PATH: ${INDEX_PATH}"; exit 1; }
  [[ -f "${DATA_ROOT}/${CATEGORY}/${CATEGORY}.item.json" ]] || { echo "[ERROR] Missing item json"; exit 1; }
  [[ -f "${DATA_ROOT}/${CATEGORY}/${CATEGORY}.inter.json" ]] || { echo "[ERROR] Missing inter json"; exit 1; }
  [[ -f "${GENREC_ROOT}/scripts/prepare_category_from_inter_json.py" ]] || { echo "[ERROR] Missing prepare_category_from_inter_json.py"; exit 1; }
}

step_check() {
  check_inputs
  run_cmd "${PYTHON_BIN}" "${GENREC_ROOT}/scripts/inspect_preprocess_category.py" \
    --category-dir "${DATA_ROOT}/${CATEGORY}" \
    --category "${CATEGORY}" \
    --index-path "${INDEX_PATH}"
}

step_build() {
  check_inputs
  PREPARE_CMD=(
    "${PYTHON_BIN}" "${GENREC_ROOT}/scripts/prepare_category_from_inter_json.py"
    --genrec-root "${GENREC_ROOT}"
    --category-dir "${DATA_ROOT}/${CATEGORY}"
    --category "${CATEGORY}"
    --index-path "${INDEX_PATH}"
    --output-dir "${OUTPUT_DIR}"
    --dataset-subdir "${DATASET_SUBDIR}"
    --dataset-key-prefix "${DATASET_KEY_PREFIX}"
    --dataset-info-path "${DATASET_INFO_PATH}"
    --split-strategy "${SPLIT_STRATEGY}"
    --train-ratio "${TRAIN_RATIO}"
    --valid-ratio "${VALID_RATIO}"
    --seq-sample "${SEQ_SAMPLE}"
    --task3-sample "${TASK3_SAMPLE}"
    --seed "${SEED}"
    --sid-levels "${SID_LEVELS}"
    --history-max "${HISTORY_MAX}"
    --train-row-order "${TRAIN_ROW_ORDER}"
    --python-bin "${PYTHON_BIN}"
  )
  if is_true "${RL_ONLY_TASK1}"; then
    PREPARE_CMD+=(--rl-only-task1)
  fi
  if is_true "${RL_ONLY_TASK4}"; then
    PREPARE_CMD+=(--rl-only-task4)
  fi
  if is_true "${RL_ONLY_TASK5}"; then
    PREPARE_CMD+=(--rl-only-task5)
  fi
  run_cmd "${PREPARE_CMD[@]}"
}

print_config

case "${MODE}" in
  check) step_check ;;
  build|build-data) step_build ;;
esac
