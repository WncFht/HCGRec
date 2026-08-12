#!/usr/bin/env bash
set -eo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/experiments/_canonical_rl_launcher.sh [options] [--dry-run]

This launcher is called by thin wrappers under scripts/experiments/*.
Wrappers should pass explicit CLI arguments instead of exporting a large
environment surface.
EOF
}

require_transformers_payload() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "[ERROR] Missing model directory: $path"
    exit 1
  fi
  if [[ ! -f "$path/config.json" ]]; then
    echo "[ERROR] Missing config.json under resolved model directory: $path"
    exit 1
  fi
  if [[ ! -f "$path/tokenizer.json" && ! -f "$path/tokenizer_config.json" && ! -f "$path/vocab.txt" ]]; then
    echo "[ERROR] Missing tokenizer files under resolved model directory: $path"
    exit 1
  fi
}

resolve_model_payload_dir() {
  local path="$1"
  local checkpoint_best="$path/checkpoint-best"
  local latest_checkpoint=""

  if [[ -d "$path" && -f "$path/config.json" ]]; then
    echo "$path"
    return 0
  fi
  if [[ -d "$checkpoint_best" && -f "$checkpoint_best/config.json" ]]; then
    echo "$checkpoint_best"
    return 0
  fi
  if [[ -d "$path" ]]; then
    latest_checkpoint="$(
      find "$path" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1
    )"
    if [[ -n "$latest_checkpoint" && -f "$latest_checkpoint/config.json" ]]; then
      echo "$latest_checkpoint"
      return 0
    fi
  fi
  echo "$path"
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_launcher_common.sh"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_launcher_runtime.sh"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_fixed_hint_artifacts.sh"

REPO_ROOT="$DEFAULT_REPO_ROOT"
PYTHON_BIN="python"
DATA_VARIANT_DEFAULT=""
MODEL_PATH=""
OUTPUT_DIR=""
RUN_NAME=""
DATA_DIR=""
INDEX_PATH=""
ADD_TOKENS_PATH=""
DS_CONFIG=""
DEFAULT_CUDA_VISIBLE_DEVICES="0,1,2,3"

NUM_PROCESSES="4"
MAIN_PORT="29516"
NUM_BEAMS="16"
SID_LEVELS="-1"
PER_DEVICE_TRAIN_BSZ="64"
PER_DEVICE_EVAL_BSZ="64"
GRAD_ACC="4"
NUM_EPOCHS="2"
LEARNING_RATE="1e-5"
EVAL_STEP="100"
EVAL_ON_START="true"
MAX_COMPLETION_LENGTH="128"
BETA="1e-3"
TEMPERATURE="1.0"
SAVE_TOTAL_LIMIT="10"
REPORT_TO="wandb"
RESUME_FROM_CHECKPOINT="auto"
REWARD_MODE="rule_only"

FIXED_HINT_ENABLED=0
FIXED_HINT_APPLY_TO_EVAL="false"
FIXED_HINT_TASK_NAMES=""
HINT_CE_LOSS_COEF="0.0"
FULL_SEQUENCE_SFT_LOSS_COEF="0.0"

DYNAMIC_HINT_MAX_DEPTH=""
DYNAMIC_HINT_APPLY_TO_EVAL="false"
DYNAMIC_HINT_TASK_NAMES=""

TRAIN_TASK_NAMES=""
EVAL_TASK_NAMES=""
ANALYSIS_TASK_NAMES=""
FORCE_REANALYZE="false"
BEAM_SIZE="16"
UNSOLVED_DEPTH="3"
CAP_DEPTH=""
ANALYZE_HINT_DEPTH="1"
ANALYZE_MAX_HINT_DEPTH="3"
ANALYZE_BATCH_SIZE="8"
ANALYZE_MAX_PROMPT_LENGTH="512"
ANALYZE_MAX_NEW_TOKENS="128"
ANALYZE_REPETITION_PENALTY="1.0"
ANALYSIS_DIR_DEFAULT=""
ANALYSIS_ARTIFACT_DIR=""
ANALYSIS_SUMMARY_PATH=""
ANALYSIS_DETAILS_PATH=""
FIXED_HINT_MAP_PATH=""
ANALYSIS_DATASET_ID=""
ANALYSIS_SCOPE_ID=""
ANALYSIS_MODEL_ID=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root)
      REPO_ROOT="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --data-variant)
      DATA_VARIANT_DEFAULT="$2"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --index-path)
      INDEX_PATH="$2"
      shift 2
      ;;
    --add-tokens-path)
      ADD_TOKENS_PATH="$2"
      shift 2
      ;;
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --ds-config)
      DS_CONFIG="$2"
      shift 2
      ;;
    --default-cuda-visible-devices)
      DEFAULT_CUDA_VISIBLE_DEVICES="$2"
      shift 2
      ;;
    --num-processes)
      NUM_PROCESSES="$2"
      shift 2
      ;;
    --main-port)
      MAIN_PORT="$2"
      shift 2
      ;;
    --num-beams)
      NUM_BEAMS="$2"
      shift 2
      ;;
    --sid-levels)
      SID_LEVELS="$2"
      shift 2
      ;;
    --per-device-train-bsz)
      PER_DEVICE_TRAIN_BSZ="$2"
      shift 2
      ;;
    --per-device-eval-bsz)
      PER_DEVICE_EVAL_BSZ="$2"
      shift 2
      ;;
    --grad-acc)
      GRAD_ACC="$2"
      shift 2
      ;;
    --num-epochs)
      NUM_EPOCHS="$2"
      shift 2
      ;;
    --learning-rate)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --eval-step)
      EVAL_STEP="$2"
      shift 2
      ;;
    --eval-on-start)
      EVAL_ON_START="$2"
      shift 2
      ;;
    --max-completion-length)
      MAX_COMPLETION_LENGTH="$2"
      shift 2
      ;;
    --beta)
      BETA="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --save-total-limit)
      SAVE_TOTAL_LIMIT="$2"
      shift 2
      ;;
    --report-to)
      REPORT_TO="$2"
      shift 2
      ;;
    --resume-from-checkpoint)
      RESUME_FROM_CHECKPOINT="$2"
      shift 2
      ;;
    --reward-mode)
      REWARD_MODE="$2"
      shift 2
      ;;
    --fixed-hint)
      FIXED_HINT_ENABLED=1
      shift
      ;;
    --fixed-hint-apply-to-eval)
      FIXED_HINT_APPLY_TO_EVAL="$2"
      shift 2
      ;;
    --fixed-hint-task-names)
      FIXED_HINT_TASK_NAMES="$2"
      shift 2
      ;;
    --hint-ce-loss-coef)
      HINT_CE_LOSS_COEF="$2"
      shift 2
      ;;
    --full-sequence-sft-loss-coef)
      FULL_SEQUENCE_SFT_LOSS_COEF="$2"
      shift 2
      ;;
    --dynamic-hint-max-depth)
      DYNAMIC_HINT_MAX_DEPTH="$2"
      shift 2
      ;;
    --dynamic-hint-apply-to-eval)
      DYNAMIC_HINT_APPLY_TO_EVAL="$2"
      shift 2
      ;;
    --dynamic-hint-task-names)
      DYNAMIC_HINT_TASK_NAMES="$2"
      shift 2
      ;;
    --train-task-names)
      TRAIN_TASK_NAMES="$2"
      shift 2
      ;;
    --eval-task-names)
      EVAL_TASK_NAMES="$2"
      shift 2
      ;;
    --analysis-task-names)
      ANALYSIS_TASK_NAMES="$2"
      shift 2
      ;;
    --force-reanalyze)
      FORCE_REANALYZE="true"
      shift
      ;;
    --beam-size)
      BEAM_SIZE="$2"
      shift 2
      ;;
    --unsolved-depth)
      UNSOLVED_DEPTH="$2"
      shift 2
      ;;
    --cap-depth)
      CAP_DEPTH="$2"
      shift 2
      ;;
    --analyze-hint-depth)
      ANALYZE_HINT_DEPTH="$2"
      shift 2
      ;;
    --analyze-max-hint-depth)
      ANALYZE_MAX_HINT_DEPTH="$2"
      shift 2
      ;;
    --analyze-batch-size)
      ANALYZE_BATCH_SIZE="$2"
      shift 2
      ;;
    --analyze-max-prompt-length)
      ANALYZE_MAX_PROMPT_LENGTH="$2"
      shift 2
      ;;
    --analyze-max-new-tokens)
      ANALYZE_MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --analyze-repetition-penalty)
      ANALYZE_REPETITION_PENALTY="$2"
      shift 2
      ;;
    --analysis-dir)
      ANALYSIS_DIR_DEFAULT="$2"
      shift 2
      ;;
    --analysis-artifact-dir)
      ANALYSIS_ARTIFACT_DIR="$2"
      shift 2
      ;;
    --analysis-summary-path)
      ANALYSIS_SUMMARY_PATH="$2"
      shift 2
      ;;
    --analysis-details-path)
      ANALYSIS_DETAILS_PATH="$2"
      shift 2
      ;;
    --fixed-hint-map-path)
      FIXED_HINT_MAP_PATH="$2"
      shift 2
      ;;
    --analysis-dataset-id)
      ANALYSIS_DATASET_ID="$2"
      shift 2
      ;;
    --analysis-scope-id)
      ANALYSIS_SCOPE_ID="$2"
      shift 2
      ;;
    --analysis-model-id)
      ANALYSIS_MODEL_ID="$2"
      shift 2
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

if [[ -z "$DATA_VARIANT_DEFAULT" ]]; then
  echo "[ERROR] --data-variant is required"
  exit 1
fi
if [[ -z "$MODEL_PATH" ]]; then
  echo "[ERROR] --model-path is required"
  exit 1
fi
if [[ -z "$OUTPUT_DIR" ]]; then
  echo "[ERROR] --output-dir is required"
  exit 1
fi
if [[ -z "$RUN_NAME" ]]; then
  echo "[ERROR] --run-name is required"
  exit 1
fi

if [[ -z "$DS_CONFIG" ]]; then
  DS_CONFIG="${REPO_ROOT}/config/zero3.yaml"
fi
if [[ -z "$ANALYSIS_DIR_DEFAULT" ]]; then
  ANALYSIS_DIR_DEFAULT="${REPO_ROOT}/temp/rl_beam_hint/artifacts"
fi

DYNAMIC_HINT_ENABLED=0
if [[ -n "$DYNAMIC_HINT_MAX_DEPTH" ]]; then
  if [[ ! "$DYNAMIC_HINT_MAX_DEPTH" =~ ^-?[0-9]+$ ]]; then
    echo "[ERROR] DYNAMIC_HINT_MAX_DEPTH must be an integer: $DYNAMIC_HINT_MAX_DEPTH"
    exit 1
  fi
  if (( DYNAMIC_HINT_MAX_DEPTH > 0 )); then
    DYNAMIC_HINT_ENABLED=1
  fi
fi
if (( FIXED_HINT_ENABLED )) && (( DYNAMIC_HINT_ENABLED )); then
  echo "[ERROR] fixed hint mode and dynamic hint mode cannot be enabled at the same time."
  exit 1
fi

DATA_VARIANT_DIR="$(resolve_data_variant_dir "$REPO_ROOT" "$DATA_VARIANT_DEFAULT")"
if [[ -z "$DATA_DIR" ]]; then
  DATA_DIR="${DATA_VARIANT_DIR}/rl"
fi
if [[ -z "$INDEX_PATH" ]]; then
  INDEX_PATH="${DATA_VARIANT_DIR}/id2sid.json"
fi
if [[ -z "$ADD_TOKENS_PATH" ]]; then
  ADD_TOKENS_PATH="${DATA_VARIANT_DIR}/new_tokens.json"
fi

RESOLVED_MODEL_PATH="$(resolve_model_payload_dir "$MODEL_PATH")"
if [[ -z "$ANALYSIS_DATASET_ID" ]]; then
  ANALYSIS_DATASET_ID="$(default_fixed_hint_dataset_id "$DATA_DIR")"
fi
DEFAULT_ANALYSIS_SCOPE_RAW="${ANALYSIS_TASK_NAMES:-${TRAIN_TASK_NAMES:-}}"
if [[ -z "$ANALYSIS_SCOPE_ID" ]]; then
  ANALYSIS_SCOPE_ID="$(default_fixed_hint_scope_id "$DEFAULT_ANALYSIS_SCOPE_RAW")"
fi
if [[ -z "$ANALYSIS_MODEL_ID" ]]; then
  ANALYSIS_MODEL_ID="$(default_fixed_hint_model_id "$RESOLVED_MODEL_PATH")"
fi

init_fixed_hint_artifact_paths \
  "$ANALYSIS_DIR_DEFAULT" \
  "$ANALYSIS_DATASET_ID" \
  "$ANALYSIS_SCOPE_ID" \
  "$ANALYSIS_MODEL_ID" \
  "$BEAM_SIZE" \
  "$ANALYZE_MAX_HINT_DEPTH" \
  "$SID_LEVELS" \
  "$UNSOLVED_DEPTH"

LAUNCH_ARGS=(
  accelerate launch
  --config_file "$DS_CONFIG"
  --num_processes "$NUM_PROCESSES"
  --main_process_port "$MAIN_PORT"
)

MODEL_ARGS=(
  -m hcgrec.trl_trainer
  --model "$RESOLVED_MODEL_PATH"
  --data_dir "$DATA_DIR"
  --index_path "$INDEX_PATH"
  --output_dir "$OUTPUT_DIR"
)

GENERATION_ARGS=(
  --num_beams "$NUM_BEAMS"
  --sid_levels "$SID_LEVELS"
  --max_completion_length "$MAX_COMPLETION_LENGTH"
  --temperature "$TEMPERATURE"
)

TRAINING_ARGS=(
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BSZ"
  --per_device_eval_batch_size "$PER_DEVICE_EVAL_BSZ"
  --gradient_accumulation_steps "$GRAD_ACC"
  --num_train_epochs "$NUM_EPOCHS"
  --learning_rate "$LEARNING_RATE"
  --eval_step "$EVAL_STEP"
  --eval_on_start "$EVAL_ON_START"
  --save_total_limit "$SAVE_TOTAL_LIMIT"
  --save_only_model true
)

REWARD_ARGS=(
  --beta "$BETA"
  --reward_mode "$REWARD_MODE"
  --prefix_reward_normalize true
  --probe_rule_with_zero_weight false
  --token_level_prefix_advantage false
)

RUNTIME_ARGS=(
  --report_to "$REPORT_TO"
  --run_name "$RUN_NAME"
  --resume_from_checkpoint "$RESUME_FROM_CHECKPOINT"
)

TRAIN_CMD=(
  "${LAUNCH_ARGS[@]}"
  "${MODEL_ARGS[@]}"
  "${GENERATION_ARGS[@]}"
  "${TRAINING_ARGS[@]}"
  "${REWARD_ARGS[@]}"
)
if [[ -n "$TRAIN_TASK_NAMES" ]]; then
  TRAIN_CMD+=(--train_task_names "$TRAIN_TASK_NAMES")
fi
if [[ -n "$EVAL_TASK_NAMES" ]]; then
  TRAIN_CMD+=(--eval_task_names "$EVAL_TASK_NAMES")
fi

ANALYZE_CMD=()
ANALYSIS_STATUS="disabled"
ANALYSIS_HAVE_SUMMARY=0
ANALYSIS_HAVE_DETAILS=0
ANALYSIS_HAVE_MAP=0
if (( FIXED_HINT_ENABLED )); then
  if path_exists "$ANALYSIS_SUMMARY_PATH"; then
    ANALYSIS_HAVE_SUMMARY=1
  fi
  if path_exists "$ANALYSIS_DETAILS_PATH"; then
    ANALYSIS_HAVE_DETAILS=1
  fi
  if path_exists "$FIXED_HINT_MAP_PATH"; then
    ANALYSIS_HAVE_MAP=1
  fi

  if ! is_true "$FORCE_REANALYZE" && [[ "$ANALYSIS_HAVE_SUMMARY" == "1" && "$ANALYSIS_HAVE_DETAILS" == "1" && "$ANALYSIS_HAVE_MAP" == "1" ]]; then
    ANALYSIS_STATUS="skip-existing"
  else
    ANALYSIS_STATUS="run"
  fi

  ANALYZE_CMD=(
    "$PYTHON_BIN"
    -m hcgrec.analyze_rl_beam_hint
    --model-path "$RESOLVED_MODEL_PATH"
    --data-dir "$DATA_DIR"
    --index-path "$INDEX_PATH"
    --add-tokens-path "$ADD_TOKENS_PATH"
    --summary-path "$ANALYSIS_SUMMARY_PATH"
    --details-path "$ANALYSIS_DETAILS_PATH"
    --beam-sizes "$BEAM_SIZE"
    --hint-depth "$ANALYZE_HINT_DEPTH"
    --max-hint-depth "$ANALYZE_MAX_HINT_DEPTH"
    --batch-size "$ANALYZE_BATCH_SIZE"
    --max-prompt-length "$ANALYZE_MAX_PROMPT_LENGTH"
    --max-new-tokens "$ANALYZE_MAX_NEW_TOKENS"
    --repetition-penalty "$ANALYZE_REPETITION_PENALTY"
    --sid-levels "$SID_LEVELS"
    --cache-dir "$ANALYSIS_DIR_DEFAULT"
    --export-fixed-hint-depth-map-path "$FIXED_HINT_MAP_PATH"
    --export-fixed-hint-beam-size "$BEAM_SIZE"
    --export-fixed-hint-unsolved-depth "$UNSOLVED_DEPTH"
  )
  if [[ -n "$ANALYSIS_TASK_NAMES" ]]; then
    ANALYZE_CMD+=(--task-names "$ANALYSIS_TASK_NAMES")
  fi
  if [[ "$ANALYSIS_HAVE_SUMMARY" == "1" ]]; then
    ANALYZE_CMD+=(--reuse-summary-path "$ANALYSIS_SUMMARY_PATH")
  fi
  if [[ "$ANALYSIS_HAVE_DETAILS" == "1" ]]; then
    ANALYZE_CMD+=(--reuse-details-path "$ANALYSIS_DETAILS_PATH")
  fi

  FIXED_HINT_ARGS=(
    --fixed_hint_depth_map_path "$FIXED_HINT_MAP_PATH"
    --fixed_hint_unsolved_depth "$UNSOLVED_DEPTH"
    --fixed_hint_apply_to_eval "$FIXED_HINT_APPLY_TO_EVAL"
    --hint_ce_loss_coef "$HINT_CE_LOSS_COEF"
    --full_sequence_sft_loss_coef "$FULL_SEQUENCE_SFT_LOSS_COEF"
  )
  if [[ -n "$FIXED_HINT_TASK_NAMES" ]]; then
    FIXED_HINT_ARGS+=(--fixed_hint_task_names "$FIXED_HINT_TASK_NAMES")
  fi
  TRAIN_CMD+=("${FIXED_HINT_ARGS[@]}")
  if [[ -n "$CAP_DEPTH" ]]; then
    TRAIN_CMD+=(--fixed_hint_depth_cap "$CAP_DEPTH")
  fi
elif (( DYNAMIC_HINT_ENABLED )); then
  DYNAMIC_HINT_ARGS=(
    --dynamic_hint_max_depth "$DYNAMIC_HINT_MAX_DEPTH"
    --dynamic_hint_apply_to_eval "$DYNAMIC_HINT_APPLY_TO_EVAL"
    --hint_ce_loss_coef "$HINT_CE_LOSS_COEF"
  )
  if [[ -n "$DYNAMIC_HINT_TASK_NAMES" ]]; then
    DYNAMIC_HINT_ARGS+=(--dynamic_hint_task_names "$DYNAMIC_HINT_TASK_NAMES")
  fi
  TRAIN_CMD+=("${DYNAMIC_HINT_ARGS[@]}")
fi

TRAIN_CMD+=("${RUNTIME_ARGS[@]}")

if [[ "$DRY_RUN" == "1" ]]; then
  if [[ "${#ANALYZE_CMD[@]}" -gt 0 && "$ANALYSIS_STATUS" == "run" ]]; then
    print_cmd "${ANALYZE_CMD[@]}"
  fi
  print_cmd "${TRAIN_CMD[@]}"
  exit 0
fi

require_exists "$REPO_ROOT" "REPO_ROOT"
require_exists "$MODEL_PATH" "MODEL_PATH"
require_transformers_payload "$RESOLVED_MODEL_PATH"
require_file "${DATA_DIR}/train.json" "RL train dataset"
require_file "${DATA_DIR}/valid.json" "RL valid dataset"
require_file "${DATA_DIR}/test.json" "RL test dataset"
require_file "$INDEX_PATH" "id2sid index file"
require_file "$DS_CONFIG" "DeepSpeed config"
require_file "${REPO_ROOT}/src/hcgrec/trl_trainer.py" "src/hcgrec/trl_trainer.py"

setup_genrec_runtime_env "$REPO_ROOT" "$DEFAULT_CUDA_VISIBLE_DEVICES"
require_accelerate

export WANDB_PROJECT="${WANDB_PROJECT:-MIMIGenRec-GRPO}"
export WANDB_MODE="${WANDB_MODE:-offline}"
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  export WANDB_API_KEY
fi
export WANDB_RUN_NAME="$RUN_NAME"

if [[ "${#ANALYZE_CMD[@]}" -gt 0 ]]; then
  require_file "$ADD_TOKENS_PATH" "new tokens file"
  require_file "${REPO_ROOT}/src/hcgrec/analyze_rl_beam_hint.py" "src/hcgrec/analyze_rl_beam_hint.py"
  mkdir -p "$(dirname -- "$ANALYSIS_SUMMARY_PATH")"
  mkdir -p "$(dirname -- "$ANALYSIS_DETAILS_PATH")"
  mkdir -p "$(dirname -- "$FIXED_HINT_MAP_PATH")"
fi

cd "$REPO_ROOT"

echo "[INFO] DATA_VARIANT_DEFAULT=${DATA_VARIANT_DEFAULT}"
echo "[INFO] DATA_VARIANT_DIR=${DATA_VARIANT_DIR}"
echo "[INFO] MODEL_PATH=${MODEL_PATH}"
echo "[INFO] RESOLVED_MODEL_PATH=${RESOLVED_MODEL_PATH}"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] RUN_NAME=${RUN_NAME}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] ANALYSIS_STATUS=${ANALYSIS_STATUS}"
if (( FIXED_HINT_ENABLED )); then
  echo "[INFO] ANALYSIS_HAVE_SUMMARY=${ANALYSIS_HAVE_SUMMARY}"
  echo "[INFO] ANALYSIS_HAVE_DETAILS=${ANALYSIS_HAVE_DETAILS}"
  echo "[INFO] ANALYSIS_HAVE_MAP=${ANALYSIS_HAVE_MAP}"
  echo "[INFO] FORCE_REANALYZE=${FORCE_REANALYZE}"
  echo "[INFO] TRAIN_TASK_NAMES=${TRAIN_TASK_NAMES}"
  echo "[INFO] EVAL_TASK_NAMES=${EVAL_TASK_NAMES}"
  echo "[INFO] ANALYSIS_TASK_NAMES=${ANALYSIS_TASK_NAMES}"
  echo "[INFO] FIXED_HINT_TASK_NAMES=${FIXED_HINT_TASK_NAMES}"
elif (( DYNAMIC_HINT_ENABLED )); then
  echo "[INFO] DYNAMIC_HINT_MAX_DEPTH=${DYNAMIC_HINT_MAX_DEPTH}"
  echo "[INFO] DYNAMIC_HINT_APPLY_TO_EVAL=${DYNAMIC_HINT_APPLY_TO_EVAL}"
  echo "[INFO] DYNAMIC_HINT_TASK_NAMES=${DYNAMIC_HINT_TASK_NAMES}"
  echo "[INFO] HINT_CE_LOSS_COEF=${HINT_CE_LOSS_COEF}"
fi

if [[ "${#ANALYZE_CMD[@]}" -gt 0 && "$ANALYSIS_STATUS" == "run" ]]; then
  set -x
  "${ANALYZE_CMD[@]}"
  set +x

  require_file "$ANALYSIS_SUMMARY_PATH" "analysis summary"
  require_file "$ANALYSIS_DETAILS_PATH" "analysis details"
  require_file "$FIXED_HINT_MAP_PATH" "fixed hint map"
elif (( FIXED_HINT_ENABLED )); then
  require_file "$ANALYSIS_SUMMARY_PATH" "analysis summary"
  require_file "$ANALYSIS_DETAILS_PATH" "analysis details"
  require_file "$FIXED_HINT_MAP_PATH" "fixed hint map"
fi

set -x
"${TRAIN_CMD[@]}"
