#!/usr/bin/env bash
set -eo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/experiments/Instruments/prepare_instruments_grec_sft_fixed_hint_data.sh [check|build-data|export-fixed-hint|all] [--dry-run]

Modes:
  check              Inspect raw Instruments inputs and print resolved paths
  build-data         Build GRec-style SFT/RL data under data/<DATA_VARIANT>/
  export-fixed-hint  Export fixed_hint_depth_map.json from an SFT checkpoint
  all                build-data + export-fixed-hint

Important defaults:
  - split strategy defaults to GRec
  - history_max defaults to 20 to match LC-Rec
  - GRec train rows default to forward (left-to-right) to match LC-Rec
  - SFT fusion train samples default to 20000 to match LC-Rec Instruments run.sh
  - fixed-hint export analyzes rl/train.json and writes a task-aware map keyed by extra_info.task::extra_info.index
  - if RL_ONLY_TASK1=true and DATA_VARIANT/DATA_VARIANT_TAG are unset, DATA_VARIANT_TAG defaults to rlsidonly

Common env overrides:
  REPO_ROOT GENREC_ROOT DATA_ROOT CATEGORY INDEX_PATH
  DATA_VARIANT DATA_VARIANT_TAG OUTPUT_DIR DATASET_SUBDIR DATASET_KEY_PREFIX
  SEQ_SAMPLE TASK3_SAMPLE SEED SID_LEVELS HISTORY_MAX TRAIN_ROW_ORDER
  RL_ONLY_TASK1 RL_ONLY_TASK4 RL_ONLY_TASK5
  MODEL_PATH BEAM_SIZE MAX_HINT_DEPTH UNSOLVED_DEPTH TASK_NAMES
  ANALYSIS_SUMMARY_PATH ANALYSIS_DETAILS_PATH FIXED_HINT_MAP_PATH
  BATCH_SIZE MAX_PROMPT_LENGTH MAX_NEW_TOKENS TRUST_REMOTE_CODE PYTHON_BIN

Examples:
  bash scripts/experiments/Instruments/prepare_instruments_grec_sft_fixed_hint_data.sh check
  bash scripts/experiments/Instruments/prepare_instruments_grec_sft_fixed_hint_data.sh build-data
  MODEL_PATH=/path/to/checkpoint-495 \
    bash scripts/experiments/Instruments/prepare_instruments_grec_sft_fixed_hint_data.sh export-fixed-hint
  RL_ONLY_TASK1=true MODEL_PATH=/path/to/checkpoint-495 \
    bash scripts/experiments/Instruments/prepare_instruments_grec_sft_fixed_hint_data.sh all
  TASK_NAMES=task1_sid_sft MODEL_PATH=/path/to/checkpoint-495 \
    bash scripts/experiments/Instruments/prepare_instruments_grec_sft_fixed_hint_data.sh export-fixed-hint
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
CATEGORY="${CATEGORY:-Instruments}"
INDEX_PATH="${INDEX_PATH:-${DATA_ROOT}/${CATEGORY}/${CATEGORY}.index.json}"

MODE="check"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    check|build-data|export-fixed-hint|all)
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
TASK3_SAMPLE="${TASK3_SAMPLE:-20000}"
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
if [[ -z "${DATA_VARIANT:-}" && -z "${DATA_VARIANT_TAG}" ]]; then
  if is_true "${RL_ONLY_TASK1}" && ! is_true "${RL_ONLY_TASK4}" && ! is_true "${RL_ONLY_TASK5}"; then
    DATA_VARIANT_TAG="rlsidonly"
  fi
fi

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

DATA_DIR="${DATA_DIR:-${OUTPUT_DIR}/rl}"
SFT_DIR="${SFT_DIR:-${OUTPUT_DIR}/sft}"
PROCESSED_INDEX_PATH="${PROCESSED_INDEX_PATH:-${OUTPUT_DIR}/id2sid.json}"
ADD_TOKENS_PATH="${ADD_TOKENS_PATH:-${OUTPUT_DIR}/new_tokens.json}"

MODEL_PATH="${MODEL_PATH:-${GENREC_ROOT}/saves/qwen2.5-3b/full/Instruments-grec-sft-qwen4B-4-256-dsz0/checkpoint-495}"
BEAM_SIZE="${BEAM_SIZE:-16}"
MAX_HINT_DEPTH="${MAX_HINT_DEPTH:-3}"
UNSOLVED_DEPTH="${UNSOLVED_DEPTH:-3}"
TASK_NAMES="${TASK_NAMES:-}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-false}"

ANALYSIS_DIR="${ANALYSIS_DIR:-${REPO_ROOT}/temp/rl_beam_hint}"
ANALYSIS_PREFIX="${ANALYSIS_PREFIX:-$(printf '%s' "${RESOLVED_DATA_VARIANT}" | sed -E 's/[^A-Za-z0-9_]+/_/g')_beam${BEAM_SIZE}}"
ANALYSIS_SUMMARY_PATH="${ANALYSIS_SUMMARY_PATH:-${ANALYSIS_DIR}/${ANALYSIS_PREFIX}_summary.json}"
ANALYSIS_DETAILS_PATH="${ANALYSIS_DETAILS_PATH:-${ANALYSIS_DIR}/${ANALYSIS_PREFIX}_details.json}"
FIXED_HINT_MAP_PATH="${FIXED_HINT_MAP_PATH:-${ANALYSIS_DIR}/${ANALYSIS_PREFIX}_fixed_hint_map.json}"
REUSE_SUMMARY_PATH="${REUSE_SUMMARY_PATH:-}"
REUSE_DETAILS_PATH="${REUSE_DETAILS_PATH:-}"

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
[INFO] DATA_DIR=${DATA_DIR}
[INFO] SFT_DIR=${SFT_DIR}
[INFO] DATASET_SUBDIR=${DATASET_SUBDIR}
[INFO] DATASET_KEY_PREFIX=${DATASET_KEY_PREFIX}
[INFO] DATASET_INFO_PATH=${DATASET_INFO_PATH}
[INFO] SEQ_SAMPLE=${SEQ_SAMPLE}
[INFO] TASK3_SAMPLE=${TASK3_SAMPLE}
[INFO] SEED=${SEED}
[INFO] SID_LEVELS=${SID_LEVELS}
[INFO] HISTORY_MAX=${HISTORY_MAX}
[INFO] TRAIN_ROW_ORDER=${TRAIN_ROW_ORDER}
[INFO] RL_ONLY_TASK1=${RL_ONLY_TASK1}
[INFO] RL_ONLY_TASK4=${RL_ONLY_TASK4}
[INFO] RL_ONLY_TASK5=${RL_ONLY_TASK5}
[INFO] MODEL_PATH=${MODEL_PATH}
[INFO] BEAM_SIZE=${BEAM_SIZE}
[INFO] MAX_HINT_DEPTH=${MAX_HINT_DEPTH}
[INFO] UNSOLVED_DEPTH=${UNSOLVED_DEPTH}
[INFO] TASK_NAMES=${TASK_NAMES:-<all tasks>}
[INFO] ANALYSIS_SUMMARY_PATH=${ANALYSIS_SUMMARY_PATH}
[INFO] ANALYSIS_DETAILS_PATH=${ANALYSIS_DETAILS_PATH}
[INFO] FIXED_HINT_MAP_PATH=${FIXED_HINT_MAP_PATH}
[INFO] DRY_RUN=${DRY_RUN}
EOF
}

check_raw_inputs() {
  require_dir "${REPO_ROOT}" "REPO_ROOT"
  require_dir "${DATA_ROOT}/${CATEGORY}" "raw category dir"
  require_file "${INDEX_PATH}" "raw index json"
  require_file "${DATA_ROOT}/${CATEGORY}/${CATEGORY}.item.json" "raw item json"
  require_file "${DATA_ROOT}/${CATEGORY}/${CATEGORY}.inter.json" "raw inter json"
  require_file "${REPO_ROOT}/scripts/inspect_preprocess_category.py" "inspect script"
  require_file "${REPO_ROOT}/scripts/prepare_category_from_inter_json.py" "prepare script"
  require_file "${REPO_ROOT}/src/hcgrec/analyze_rl_beam_hint.py" "fixed-hint analyzer"
}

step_check() {
  check_raw_inputs
  run_cmd "${PYTHON_BIN}" "${REPO_ROOT}/scripts/inspect_preprocess_category.py" \
    --category-dir "${DATA_ROOT}/${CATEGORY}" \
    --category "${CATEGORY}" \
    --index-path "${INDEX_PATH}"
  if [[ -d "${OUTPUT_DIR}" ]]; then
    echo "[INFO] Existing output dir: ${OUTPUT_DIR}"
    find "${OUTPUT_DIR}" -maxdepth 2 -type f | sort
  else
    echo "[INFO] Output dir does not exist yet: ${OUTPUT_DIR}"
  fi
}

step_build_data() {
  check_raw_inputs
  PREPARE_CMD=(
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/prepare_category_from_inter_json.py"
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

step_export_fixed_hint() {
  if [[ "$DRY_RUN" -eq 0 ]]; then
    require_file "${DATA_DIR}/train.json" "RL train json"
    require_file "${DATA_DIR}/valid.json" "RL valid json"
    require_file "${DATA_DIR}/test.json" "RL test json"
    require_file "${PROCESSED_INDEX_PATH}" "processed id2sid json"
    require_file "${ADD_TOKENS_PATH}" "new_tokens json"
    require_path "${MODEL_PATH}" "SFT checkpoint/model path"
  fi
  mkdir -p "${ANALYSIS_DIR}"

  ANALYZE_CMD=(
    "${PYTHON_BIN}"
    -m
    hcgrec.analyze_rl_beam_hint
    --model-path "${MODEL_PATH}"
    --data-dir "${DATA_DIR}"
    --index-path "${PROCESSED_INDEX_PATH}"
    --add-tokens-path "${ADD_TOKENS_PATH}"
    --summary-path "${ANALYSIS_SUMMARY_PATH}"
    --details-path "${ANALYSIS_DETAILS_PATH}"
    --beam-sizes "${BEAM_SIZE}"
    --hint-depth 1
    --max-hint-depth "${MAX_HINT_DEPTH}"
    --batch-size "${BATCH_SIZE}"
    --max-prompt-length "${MAX_PROMPT_LENGTH}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --sid-levels "${SID_LEVELS}"
    --export-fixed-hint-depth-map-path "${FIXED_HINT_MAP_PATH}"
    --export-fixed-hint-beam-size "${BEAM_SIZE}"
    --export-fixed-hint-unsolved-depth "${UNSOLVED_DEPTH}"
  )

  if [[ -n "${TASK_NAMES}" ]]; then
    ANALYZE_CMD+=(--task-names "${TASK_NAMES}")
  fi
  if [[ -n "${REUSE_SUMMARY_PATH}" ]]; then
    ANALYZE_CMD+=(--reuse-summary-path "${REUSE_SUMMARY_PATH}")
  fi
  if [[ -n "${REUSE_DETAILS_PATH}" ]]; then
    ANALYZE_CMD+=(--reuse-details-path "${REUSE_DETAILS_PATH}")
  fi
  if is_true "${TRUST_REMOTE_CODE}"; then
    ANALYZE_CMD+=(--trust-remote-code)
  fi

  run_cmd "${ANALYZE_CMD[@]}"
}

print_config

case "${MODE}" in
  check)
    step_check
    ;;
  build-data)
    step_build_data
    ;;
  export-fixed-hint)
    step_export_fixed_hint
    ;;
  all)
    step_build_data
    step_export_fixed_hint
    ;;
esac
