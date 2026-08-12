#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

CHECKPOINT_PATH=""
TEST_DATA_PATH=""
INDEX_PATH=""
CUDA_LIST="0 1 2 3"
PYTHON_BIN="python"
RESULTS_ROOT="${REPO_ROOT}/results"
OUTPUT_DIR=""
TEMP_ROOT="${REPO_ROOT}/temp"

BATCH_SIZE="8"
MAX_NEW_TOKENS="256"
NUM_BEAMS="50"
TEMPERATURE="1.0"
DO_SAMPLE="False"
LENGTH_PENALTY="0.0"
SID_LEVELS="-1"

SPLIT_JSON_MODULE="hcgrec.split_json"
MERGE_JSON_MODULE="hcgrec.merge_json"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/eval/evaluate_checkpoint.sh --checkpoint-path PATH --test-data-path PATH --index-path PATH [options]
  bash scripts/eval/evaluate_checkpoint.sh PATH --test-data-path PATH --index-path PATH [options]

Required:
  --checkpoint-path PATH   Checkpoint directory to evaluate
  --test-data-path PATH    Evaluation dataset JSON
  --index-path PATH        Index JSON used by constrained decoding

Optional:
  --cuda-list "0 1 2 3"    Space/comma separated GPU ids
  --python-bin python      Python executable
  --results-root PATH      Base results directory
  --output-dir PATH        Explicit output directory
  --temp-root PATH         Temporary directory for multi-GPU shards
  --batch-size 8
  --max-new-tokens 256
  --num-beams 50
  --temperature 1.0
  --do-sample False
  --length-penalty 0.0
  --sid-levels -1
EOF
}

normalize_cuda_list() {
  local raw="$1"
  local normalized=""
  local token

  for token in ${raw//,/ }; do
    [[ -z "$token" ]] && continue
    normalized="${normalized}${normalized:+ }${token}"
  done

  printf '%s\n' "$normalized"
}

require_dir() {
  local path="$1"
  local desc="$2"
  if [[ ! -d "$path" ]]; then
    echo "[ERROR] Missing ${desc}: $path"
    exit 1
  fi
}

require_file() {
  local path="$1"
  local desc="$2"
  if [[ ! -f "$path" ]]; then
    echo "[ERROR] Missing ${desc}: $path"
    exit 1
  fi
}

run_single_gpu_eval() {
  local gpu_id="$1"

  CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" -u -m hcgrec.evaluate \
    --model_name_or_path "$CHECKPOINT_PATH" \
    --test_data_path "$TEST_DATA_PATH" \
    --result_json_path "$OUTPUT_DIR/final_result.json" \
    --index_path "$INDEX_PATH" \
    --batch_size "$BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --num_beams "$NUM_BEAMS" \
    --temperature "$TEMPERATURE" \
    --do_sample "$DO_SAMPLE" \
    --length_penalty "$LENGTH_PENALTY" \
    --sid_levels "$SID_LEVELS" \
    --compute_metrics_flag True
}

run_multi_gpu_eval() {
  local temp_dir="$1"
  local gpu_id=""
  local pids=()
  local launched_gpu_ids=()

  rm -rf "$temp_dir"
  mkdir -p "$temp_dir"

  "$PYTHON_BIN" -m "$SPLIT_JSON_MODULE" \
    --input_path "$TEST_DATA_PATH" \
    --output_path "$temp_dir" \
    --cuda_list "$CUDA_LIST"

  for gpu_id in "${GPU_ARR[@]}"; do
    [[ ! -f "$temp_dir/${gpu_id}.json" ]] && continue

    CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" -u -m hcgrec.evaluate \
      --model_name_or_path "$CHECKPOINT_PATH" \
      --test_data_path "$temp_dir/${gpu_id}.json" \
      --result_json_path "$temp_dir/${gpu_id}_result.json" \
      --index_path "$INDEX_PATH" \
      --batch_size "$BATCH_SIZE" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --num_beams "$NUM_BEAMS" \
      --temperature "$TEMPERATURE" \
      --do_sample "$DO_SAMPLE" \
      --length_penalty "$LENGTH_PENALTY" \
      --sid_levels "$SID_LEVELS" \
      --compute_metrics_flag False &
    pids+=("$!")
    launched_gpu_ids+=("$gpu_id")
  done

  if [[ ${#pids[@]} -eq 0 ]]; then
    echo "[ERROR] No shard files were created under: $temp_dir"
    rm -rf "$temp_dir"
    exit 1
  fi

  local wait_failed="0"
  local pid_index=""
  local exit_code=""
  for pid_index in "${!pids[@]}"; do
    if wait "${pids[$pid_index]}"; then
      continue
    fi
    exit_code="$?"
    echo "[ERROR] GPU ${launched_gpu_ids[$pid_index]} evaluation failed with exit code ${exit_code}"
    wait_failed="1"
  done

  if [[ "$wait_failed" == "1" ]]; then
    rm -rf "$temp_dir"
    exit 1
  fi

  for gpu_id in "${launched_gpu_ids[@]}"; do
    if [[ ! -f "$temp_dir/${gpu_id}_result.json" ]]; then
      echo "[ERROR] Missing per-GPU result file: $temp_dir/${gpu_id}_result.json"
      rm -rf "$temp_dir"
      exit 1
    fi
  done

  "$PYTHON_BIN" -m "$MERGE_JSON_MODULE" \
    --input_path "$temp_dir" \
    --output_path "$OUTPUT_DIR/final_result.json" \
    --cuda_list "$CUDA_LIST" \
    --compute_metrics False

  "$PYTHON_BIN" -u -m hcgrec.evaluate \
    --result_json_path "$OUTPUT_DIR/final_result.json" \
    --metrics_only True

  rm -rf "$temp_dir"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint-path)
      CHECKPOINT_PATH="$2"
      shift 2
      ;;
    --test-data-path)
      TEST_DATA_PATH="$2"
      shift 2
      ;;
    --index-path)
      INDEX_PATH="$2"
      shift 2
      ;;
    --cuda-list)
      CUDA_LIST="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --results-root)
      RESULTS_ROOT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --temp-root)
      TEMP_ROOT="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --max-new-tokens)
      MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --num-beams)
      NUM_BEAMS="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --do-sample)
      DO_SAMPLE="$2"
      shift 2
      ;;
    --length-penalty)
      LENGTH_PENALTY="$2"
      shift 2
      ;;
    --sid-levels)
      SID_LEVELS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "[ERROR] Unknown option: $1"
      usage
      exit 1
      ;;
    *)
      if [[ -z "$CHECKPOINT_PATH" ]]; then
        CHECKPOINT_PATH="$1"
        shift
        continue
      fi
      echo "[ERROR] Unexpected argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$CHECKPOINT_PATH" || -z "$TEST_DATA_PATH" || -z "$INDEX_PATH" ]]; then
  echo "[ERROR] --checkpoint-path, --test-data-path, and --index-path are required."
  usage
  exit 1
fi

require_dir "$CHECKPOINT_PATH" "checkpoint directory"
require_file "$TEST_DATA_PATH" "test data"
require_file "$INDEX_PATH" "index file"

CUDA_LIST="$(normalize_cuda_list "$CUDA_LIST")"
read -r -a GPU_ARR <<< "$CUDA_LIST"
if [[ ${#GPU_ARR[@]} -eq 0 ]]; then
  echo "[ERROR] --cuda-list resolved to an empty GPU list."
  exit 1
fi

MODEL_NAME="$(basename "$(dirname "$CHECKPOINT_PATH")")"
CHECKPOINT_NAME="$(basename "$CHECKPOINT_PATH")"

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="${RESULTS_ROOT}/${MODEL_NAME}/${CHECKPOINT_NAME}"
fi

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Single Checkpoint Evaluation"
echo "checkpoint_path=$CHECKPOINT_PATH"
echo "test_data_path=$TEST_DATA_PATH"
echo "index_path=$INDEX_PATH"
echo "cuda_list=$CUDA_LIST"
echo "output_dir=$OUTPUT_DIR"
echo "=========================================="

if [[ ${#GPU_ARR[@]} -eq 1 ]]; then
  echo "Single GPU evaluation on GPU ${GPU_ARR[0]}"
  run_single_gpu_eval "${GPU_ARR[0]}"
else
  TEMP_DIR="${TEMP_ROOT}/eval-${MODEL_NAME}-${CHECKPOINT_NAME}"
  echo "Multi GPU evaluation on GPUs: $CUDA_LIST"
  run_multi_gpu_eval "$TEMP_DIR"
fi

echo "Done. Results: $OUTPUT_DIR"
