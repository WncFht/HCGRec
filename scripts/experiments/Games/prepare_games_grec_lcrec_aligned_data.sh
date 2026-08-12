#!/usr/bin/env bash
set -eo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/experiments/Games/prepare_games_grec_lcrec_aligned_data.sh [check|build|build-data] [--dry-run]

Modes:
  check       Inspect resolved Games paths and stable preprocess settings
  build       Build GRec-style Games data aligned to current preprocess settings
  build-data  Alias of build

Important defaults:
  - split strategy defaults to GRec
  - history_max defaults to 20 to match LC-Rec
  - train row order defaults to forward to match LC-Rec
  - seq sample defaults to 10000, consistent with current Games preprocess setup
  - task3 sample defaults to -1, meaning keep all fusion train samples
  - RL task filters remain disabled; this is a plain LCRecAligned SFT data build

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
CATEGORY="${CATEGORY:-Games}"
INDEX_PATH="${INDEX_PATH:-${DATA_ROOT}/${CATEGORY}/Games.index.json}"

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
DATA_VARIANT="${DATA_VARIANT:-Games_grec_index}"
DATASET_SUBDIR="${DATASET_SUBDIR:-${DATA_VARIANT}}"
DATASET_KEY_PREFIX="${DATASET_KEY_PREFIX:-Games_grec_index}"
DATASET_INFO_PATH="${DATASET_INFO_PATH:-${GENREC_ROOT}/data/dataset_info.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${GENREC_ROOT}/data/${DATA_VARIANT}}"

print_config() {
  cat <<EOF
[INFO] MODE=${MODE}
[INFO] REPO_ROOT=${REPO_ROOT}
[INFO] GENREC_ROOT=${GENREC_ROOT}
[INFO] DATA_ROOT=${DATA_ROOT}
[INFO] CATEGORY=${CATEGORY}
[INFO] INDEX_PATH=${INDEX_PATH}
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
[INFO] SPLIT_STRATEGY=${SPLIT_STRATEGY}
[INFO] TRAIN_RATIO=${TRAIN_RATIO}
[INFO] VALID_RATIO=${VALID_RATIO}
[INFO] DRY_RUN=${DRY_RUN}
EOF
}

check_inputs() {
  [[ -d "${GENREC_ROOT}" ]] || { echo "[ERROR] Missing GENREC_ROOT: ${GENREC_ROOT}"; exit 1; }
  [[ -f "${INDEX_PATH}" ]] || { echo "[ERROR] Missing INDEX_PATH: ${INDEX_PATH}"; exit 1; }
  [[ -f "${DATA_ROOT}/${CATEGORY}/${CATEGORY}.item.json" ]] || { echo "[ERROR] Missing item json"; exit 1; }
  [[ -f "${DATA_ROOT}/${CATEGORY}/${CATEGORY}.inter.json" ]] || { echo "[ERROR] Missing inter json"; exit 1; }
  [[ -f "${GENREC_ROOT}/scripts/inspect_preprocess_category.py" ]] || { echo "[ERROR] Missing inspect_preprocess_category.py"; exit 1; }
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
  if is_true "${RL_ONLY_TASK1:-false}"; then
    PREPARE_CMD+=(--rl-only-task1)
  fi
  if is_true "${RL_ONLY_TASK4:-false}"; then
    PREPARE_CMD+=(--rl-only-task4)
  fi
  if is_true "${RL_ONLY_TASK5:-false}"; then
    PREPARE_CMD+=(--rl-only-task5)
  fi
  run_cmd "${PREPARE_CMD[@]}"
}

print_config

case "${MODE}" in
  check) step_check ;;
  build|build-data) step_build ;;
esac
