#!/usr/bin/env bash
set -eo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/experiments/_canonical_sft_launcher.sh [options] [--dry-run]

Wrappers should pass explicit CLI arguments instead of exporting launcher
configuration into the environment.
EOF
}

sanitize_log_name() {
  local value="$1"
  value="${value//\//_}"
  value="${value// /_}"
  echo "$value"
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_launcher_common.sh"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_launcher_runtime.sh"

REPO_ROOT="$DEFAULT_REPO_ROOT"
YAML_PATH=""
DATASET_INFO_PATH=""
RUN_NAME=""
DEFAULT_CUDA_VISIBLE_DEVICES="0,1,2,3"
LOG_DIR=""
LOG_FILE=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root)
      REPO_ROOT="$2"
      shift 2
      ;;
    --yaml-path)
      YAML_PATH="$2"
      shift 2
      ;;
    --dataset-info-path)
      DATASET_INFO_PATH="$2"
      shift 2
      ;;
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --default-cuda-visible-devices)
      DEFAULT_CUDA_VISIBLE_DEVICES="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --log-file)
      LOG_FILE="$2"
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

if [[ -z "$YAML_PATH" ]]; then
  echo "[ERROR] --yaml-path is required"
  exit 1
fi
if [[ -z "$DATASET_INFO_PATH" ]]; then
  DATASET_INFO_PATH="${REPO_ROOT}/data/dataset_info.json"
fi
if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="$(basename "${YAML_PATH%.yaml}")"
fi
if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="${REPO_ROOT}/log"
fi
if [[ -z "$LOG_FILE" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  LOG_FILE="${LOG_DIR}/$(sanitize_log_name "${RUN_NAME}")_${TS}.log"
fi

TRAIN_CMD=(
  llamafactory-cli train "$YAML_PATH"
)

if [[ "$DRY_RUN" == "1" ]]; then
  print_cmd "${TRAIN_CMD[@]}"
  exit 0
fi

require_dir "$REPO_ROOT" "REPO_ROOT"
require_file "$YAML_PATH" "YAML config"
require_file "$DATASET_INFO_PATH" "dataset_info.json"

setup_genrec_runtime_env "$REPO_ROOT" "$DEFAULT_CUDA_VISIBLE_DEVICES"

export WANDB_PROJECT="${WANDB_PROJECT:-MIMIGenRec-SFT}"
export WANDB_MODE="${WANDB_MODE:-offline}"
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  export WANDB_API_KEY
fi

if ! command -v llamafactory-cli >/dev/null 2>&1; then
  echo "[ERROR] llamafactory-cli not found in PATH"
  exit 1
fi

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

echo "[INFO] REPO_ROOT=${REPO_ROOT}"
echo "[INFO] YAML_PATH=${YAML_PATH}"
echo "[INFO] DATASET_INFO_PATH=${DATASET_INFO_PATH}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] LOG_FILE=${LOG_FILE}"

set -x
"${TRAIN_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
