#!/usr/bin/env bash

sanitize_name() {
  local value="$1"
  value="${value//\//_}"
  value="${value// /_}"
  value="${value//,/+}"
  echo "$value"
}

default_fixed_hint_scope_id() {
  local raw_task_names="$1"
  if [[ -z "$raw_task_names" ]]; then
    echo "all"
    return
  fi
  case "$raw_task_names" in
    "task1_sid_sft")
      echo "sid-only"
      ;;
    "task1_sid_sft,task5_title_desc2sid")
      echo "sid-title-desc"
      ;;
    *)
      echo "$(sanitize_name "$raw_task_names")"
      ;;
  esac
}

default_fixed_hint_model_id() {
  local model_path="$1"
  local base_name
  base_name="$(basename "$model_path")"
  if [[ "$base_name" == checkpoint-* ]]; then
    echo "$base_name"
    return
  fi
  echo "$(sanitize_name "$base_name")"
}

default_fixed_hint_dataset_id() {
  local data_dir="$1"
  local parent_dir
  parent_dir="$(basename "$(dirname "$data_dir")")"
  echo "$(sanitize_name "$parent_dir")"
}

init_fixed_hint_artifact_paths() {
  local artifact_root="$1"
  local dataset_id="$2"
  local scope_id="$3"
  local model_id="$4"
  local beam_size="$5"
  local max_hint_depth="$6"
  local sid_levels="$7"
  local unsolved_depth="$8"

  CANONICAL_ANALYSIS_ARTIFACT_DIR="${artifact_root}/$(sanitize_name "$dataset_id")/$(sanitize_name "$scope_id")/$(sanitize_name "$model_id")/beam${beam_size}_hint${max_hint_depth}_sid${sid_levels}"
  CANONICAL_ANALYSIS_SUMMARY_PATH="${CANONICAL_ANALYSIS_ARTIFACT_DIR}/summary.json"
  CANONICAL_ANALYSIS_DETAILS_PATH="${CANONICAL_ANALYSIS_ARTIFACT_DIR}/details.json"
  CANONICAL_FIXED_HINT_MAP_PATH="${CANONICAL_ANALYSIS_ARTIFACT_DIR}/fixed_hint_map.unsolved${unsolved_depth}.json"
  CANONICAL_ANALYSIS_META_PATH="${CANONICAL_ANALYSIS_ARTIFACT_DIR}/meta.json"

  ANALYSIS_ARTIFACT_DIR="${ANALYSIS_ARTIFACT_DIR:-$CANONICAL_ANALYSIS_ARTIFACT_DIR}"
  ANALYSIS_SUMMARY_PATH="${ANALYSIS_SUMMARY_PATH:-$CANONICAL_ANALYSIS_SUMMARY_PATH}"
  ANALYSIS_DETAILS_PATH="${ANALYSIS_DETAILS_PATH:-$CANONICAL_ANALYSIS_DETAILS_PATH}"
  FIXED_HINT_MAP_PATH="${FIXED_HINT_MAP_PATH:-$CANONICAL_FIXED_HINT_MAP_PATH}"
  ANALYSIS_META_PATH="${ANALYSIS_META_PATH:-$CANONICAL_ANALYSIS_META_PATH}"
}
