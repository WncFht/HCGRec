#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_GREC_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
: "${GREC_ROOT:=$DEFAULT_GREC_ROOT}"
: "${CONTINUE_ON_ERROR:=false}"
: "${SKIP_IF_CKPT_EXISTS:=true}"
: "${COOLDOWN_SECONDS:=30}"

TARGET_DATASETS=""
POSITIONAL_PRESETS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets)
      TARGET_DATASETS="${2:-}"
      if [[ -z "$TARGET_DATASETS" ]]; then
        echo "Error: --datasets requires a value, e.g. --datasets Instruments,Instruments-Arts-Games" >&2
        exit 1
      fi
      shift 2
      ;;
    --no-skip-existing)
      SKIP_IF_CKPT_EXISTS=false
      shift
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        POSITIONAL_PRESETS+=("$1")
        shift
      done
      ;;
    *)
      POSITIONAL_PRESETS+=("$1")
      shift
      ;;
  esac
done

if [[ ! -d "$GREC_ROOT" ]]; then
  echo "Error: GREC_ROOT does not exist: $GREC_ROOT" >&2
  exit 1
fi

cd "$GREC_ROOT" || exit 1

collect_scripts() {
  if [[ $# -gt 0 ]]; then
    for preset in "$@"; do
      if [[ -f "$preset" ]]; then
        echo "$preset"
      elif [[ -f "scripts/index/$preset/train.sh" ]]; then
        echo "scripts/index/$preset/train.sh"
      else
        echo "Error: cannot find preset train script for '$preset'" >&2
        exit 1
      fi
    done
  else
    find scripts/index -mindepth 2 -maxdepth 2 -name train.sh \
      ! -path "scripts/index/base/*" | sort
  fi
}

extract_dataset_from_script() {
  local script="$1"
  local preset_name
  preset_name="$(basename "$(dirname "$script")")"
  echo "${preset_name%-qwen3-embedding-4B-*}"
}

extract_tag_from_script() {
  local script="$1"
  local preset_name
  preset_name="$(basename "$(dirname "$script")")"
  echo "${preset_name#*-qwen3-embedding-4B-}"
}

should_skip_existing_ckpt() {
  local script="$1"
  local dataset tag ckpt_root

  dataset="$(extract_dataset_from_script "$script")"
  tag="$(extract_tag_from_script "$script")"
  ckpt_root="index_train_runs/${dataset}/index/qwen3-embedding-4B/${tag}_kmtrue-lkmtrue-kmi100"

  shopt -s nullglob
  local existing=("${ckpt_root}"/*/best_collision_model.pth)
  shopt -u nullglob

  if [[ ${#existing[@]} -gt 0 ]]; then
    echo "[SKIP] existing ckpt found for ${script}"
    echo "       -> ${existing[0]}"
    return 0
  fi
  return 1
}

TRAIN_SCRIPTS=()
if [[ ${#POSITIONAL_PRESETS[@]} -gt 0 ]]; then
  while IFS= read -r script_path; do
    TRAIN_SCRIPTS+=("$script_path")
  done < <(collect_scripts "${POSITIONAL_PRESETS[@]}")
else
  while IFS= read -r script_path; do
    TRAIN_SCRIPTS+=("$script_path")
  done < <(collect_scripts)
fi

if [[ ${#TRAIN_SCRIPTS[@]} -eq 0 ]]; then
  echo "No train scripts found." >&2
  exit 1
fi

if [[ -n "$TARGET_DATASETS" ]]; then
  IFS=',' read -r -a DATASET_FILTERS <<< "$TARGET_DATASETS"
  FILTERED=()
  for script in "${TRAIN_SCRIPTS[@]}"; do
    ds="$(extract_dataset_from_script "$script")"
    for wanted in "${DATASET_FILTERS[@]}"; do
      wanted="${wanted// /}"
      if [[ -n "$wanted" && "$ds" == "$wanted" ]]; then
        FILTERED+=("$script")
        break
      fi
    done
  done

  if [[ ${#FILTERED[@]} -gt 0 ]]; then
    TRAIN_SCRIPTS=("${FILTERED[@]}")
  else
    TRAIN_SCRIPTS=()
  fi
fi

if [[ ${#TRAIN_SCRIPTS[@]} -eq 0 ]]; then
  echo "No train scripts matched filters." >&2
  exit 1
fi

echo "Batch index training start"
echo "GREC_ROOT=$GREC_ROOT"
echo "CONTINUE_ON_ERROR=$CONTINUE_ON_ERROR"
echo "SKIP_IF_CKPT_EXISTS=$SKIP_IF_CKPT_EXISTS"
echo "COOLDOWN_SECONDS=$COOLDOWN_SECONDS"
echo "TARGET_DATASETS=${TARGET_DATASETS:-<all>}"
echo "Total jobs: ${#TRAIN_SCRIPTS[@]}"
printf ' - %s\n' "${TRAIN_SCRIPTS[@]}"

success=0
failed=0
failed_list=()
total_jobs=${#TRAIN_SCRIPTS[@]}

for ((job_idx = 0; job_idx < total_jobs; job_idx++)); do
  script="${TRAIN_SCRIPTS[$job_idx]}"

  if [[ "${SKIP_IF_CKPT_EXISTS,,}" == "true" ]]; then
    if should_skip_existing_ckpt "$script"; then
      continue
    fi
  fi

  echo
  echo "======================================================"
  echo "[START] $script"
  echo "======================================================"

  if bash "$script"; then
    echo "[OK] $script"
    success=$((success + 1))
  else
    echo "[FAIL] $script" >&2
    failed=$((failed + 1))
    failed_list+=("$script")

    if [[ "${CONTINUE_ON_ERROR,,}" != "true" ]]; then
      echo "Stopped because CONTINUE_ON_ERROR=false" >&2
      break
    fi
  fi

  if [[ $COOLDOWN_SECONDS -gt 0 && $job_idx -lt $((total_jobs - 1)) ]]; then
    echo "[COOLDOWN] sleeping ${COOLDOWN_SECONDS}s before next job..."
    sleep "$COOLDOWN_SECONDS"
  fi
done

echo
echo "================ Batch Summary ================"
echo "success: $success"
echo "failed:  $failed"
if [[ $failed -gt 0 ]]; then
  printf 'failed scripts:\n'
  printf ' - %s\n' "${failed_list[@]}"
  exit 1
fi

echo "All jobs finished successfully."
