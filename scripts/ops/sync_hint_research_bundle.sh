#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_GENREC_REPO_DIR_DEFAULT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REMOTE_HOME_ROOT_DEFAULT="${HOME:-}"

LOCAL_GENREC_REPO_DIR="${LOCAL_GENREC_REPO_DIR:-$LOCAL_GENREC_REPO_DIR_DEFAULT}"
BUNDLE_REPO_ROOT="${BUNDLE_REPO_ROOT:-$REMOTE_HOME_ROOT_DEFAULT/GenRec}"
BUNDLE_RAW_DATA_ROOT="${BUNDLE_RAW_DATA_ROOT:-$REMOTE_HOME_ROOT_DEFAULT/data}"

CATEGORY="${CATEGORY:-Instruments}"
DATA_VARIANT="${DATA_VARIANT:-Instruments_grec_index_emb-qwen3-embedding-4B_rq4_cb256-256-256-256_dsInstruments_ridFeb-10-2026-05-40-47}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BUNDLE_NAME="${BUNDLE_NAME:-instruments_grec_hint_research_bundle}"
CHUNK_SIZE="${CHUNK_SIZE:-10485760}"  # 10MB

DIFFICULTY_TABLE_REL="${DIFFICULTY_TABLE_REL:-output/jupyter-notebook/genrec-hint-cascade-artifacts/instruments_grec_beam16_hint_difficulty_table.csv}"
SUMMARY_REL="${SUMMARY_REL:-temp/rl_beam_hint/instruments_grec_beam_hint_cascade_20260314_summary.json}"
DETAILS_REL="${DETAILS_REL:-temp/rl_beam_hint/instruments_grec_beam_hint_cascade_20260314_details.json}"
TRAIN_JSON_REL="${TRAIN_JSON_REL:-data/${DATA_VARIANT}/rl/train.json}"
INDEX_JSON_REL="${INDEX_JSON_REL:-data/${DATA_VARIANT}/id2sid.json}"
NEW_TOKENS_REL="${NEW_TOKENS_REL:-data/${DATA_VARIANT}/new_tokens.json}"
NOTEBOOK_REL="${NOTEBOOK_REL:-docs/hint_research/genrec-hint-cascade-analysis-2.ipynb}"

RAW_INTER_REL="${RAW_INTER_REL:-${CATEGORY}/${CATEGORY}.inter.json}"
RAW_ITEM_REL="${RAW_ITEM_REL:-${CATEGORY}/${CATEGORY}.item.json}"

DEFAULT_UNPACK_ROOT="${DEFAULT_UNPACK_ROOT:-$LOCAL_GENREC_REPO_DIR/output/local-research-bundles}"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/ops/sync_hint_research_bundle.sh pack [archive_path]
  bash scripts/ops/sync_hint_research_bundle.sh unpack [archive_path] [dest_root]

Description:
  - pack:
      Create a compact research bundle for GenRec hint analysis.
      The archive includes the beam16 difficulty table, cascade summary/details,
      train/index/new_tokens files, raw inter/item files, and the current analysis notebook.
  - unpack:
      Extract the archive into a local destination directory.

Environment overrides:
  BUNDLE_REPO_ROOT
  BUNDLE_RAW_DATA_ROOT
  CATEGORY
  DATA_VARIANT
  PYTHON_BIN
  BUNDLE_NAME
  CHUNK_SIZE
  DIFFICULTY_TABLE_REL
  SUMMARY_REL
  DETAILS_REL
  TRAIN_JSON_REL
  INDEX_JSON_REL
  NEW_TOKENS_REL
  NOTEBOOK_REL
  RAW_INTER_REL
  RAW_ITEM_REL
  DEFAULT_UNPACK_ROOT
USAGE
}

find_latest_bundle_archive() {
  local search_dir="$1"
  local latest_archive
  latest_archive="$(ls -1t "$search_dir"/*.tar.gz "$search_dir"/*.tar.gz.part.000 2>/dev/null | head -n 1 || true)"
  if [[ -z "$latest_archive" ]]; then
    return 1
  fi
  printf '%s\n' "$latest_archive"
}

normalize_archive_path() {
  local archive_arg="$1"
  local archive_path
  if [[ -z "$archive_arg" ]]; then
    archive_path="$(pwd)/${BUNDLE_NAME}.tar.gz"
  elif [[ "$archive_arg" =~ \.part\.[0-9]{3}$ ]]; then
    archive_path="$archive_arg"
  elif [[ "$archive_arg" = /* ]]; then
    archive_path="$archive_arg"
  else
    archive_path="$(pwd)/$archive_arg"
  fi
  if [[ ! "$archive_path" =~ \.part\.[0-9]{3}$ && "$archive_path" != *.tar.gz ]]; then
    archive_path="${archive_path}.tar.gz"
  fi
  printf '%s\n' "$archive_path"
}

archive_part_prefix() {
  local archive_path="$1"
  printf '%s.part.' "$archive_path"
}

split_archive_if_needed() {
  local archive_path="$1"
  local chunk_size="$2"
  local archive_size
  archive_size=$(wc -c < "$archive_path" | tr -d '[:space:]')
  if (( archive_size <= chunk_size )); then
    return
  fi

  local part_prefix
  part_prefix="$(archive_part_prefix "$archive_path")"
  rm -f "${part_prefix}"[0-9][0-9][0-9]
  split -b "$chunk_size" -d -a 3 "$archive_path" "$part_prefix"
  rm -f "$archive_path"

  echo "Archive exceeded CHUNK_SIZE=${chunk_size} bytes and was split into parts:"
  ls -1 "${part_prefix}"[0-9][0-9][0-9]
}

rebuild_archive_from_parts() {
  local input_path="$1"
  local output_path="$2"
  local part_prefix
  local -a part_files=()

  if [[ "$input_path" =~ \.part\.[0-9]{3}$ ]]; then
    part_prefix="${input_path%.part.[0-9][0-9][0-9]}.part."
  else
    part_prefix="$(archive_part_prefix "$input_path")"
  fi

  shopt -s nullglob
  part_files=("${part_prefix}"[0-9][0-9][0-9])
  shopt -u nullglob

  if [[ ${#part_files[@]} -eq 0 ]]; then
    echo "Error: no archive parts found for: $input_path" >&2
    exit 1
  fi

  cat "${part_files[@]}" > "$output_path"
}

copy_with_parents() {
  local src="$1"
  local dest="$2"
  mkdir -p "$(dirname "$dest")"
  cp "$src" "$dest"
}

write_manifest() {
  local manifest_path="$1"
  local bundle_root="$2"
  shift 2

  "$PYTHON_BIN" - "$manifest_path" "$bundle_root" "$@" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
bundle_root = Path(sys.argv[2])
files = sys.argv[3:]

payload = {
    "bundle_name": bundle_root.name,
    "files": sorted(files),
}

manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

pack_bundle() {
  if [[ $# -gt 1 ]]; then
    usage
    exit 1
  fi

  local archive_path
  archive_path="$(normalize_archive_path "${1:-}")"
  mkdir -p "$(dirname "$archive_path")"

  local -a repo_files=(
    "$DIFFICULTY_TABLE_REL"
    "$SUMMARY_REL"
    "$DETAILS_REL"
    "$TRAIN_JSON_REL"
    "$INDEX_JSON_REL"
    "$NEW_TOKENS_REL"
    "$NOTEBOOK_REL"
  )
  local -a raw_files=(
    "$RAW_INTER_REL"
    "$RAW_ITEM_REL"
  )

  local -a missing_files=()
  local rel src
  for rel in "${repo_files[@]}"; do
    src="$BUNDLE_REPO_ROOT/$rel"
    if [[ ! -f "$src" ]]; then
      missing_files+=("$src")
    fi
  done
  for rel in "${raw_files[@]}"; do
    src="$BUNDLE_RAW_DATA_ROOT/$rel"
    if [[ ! -f "$src" ]]; then
      missing_files+=("$src")
    fi
  done

  if [[ ${#missing_files[@]} -gt 0 ]]; then
    printf 'Missing required files:\n' >&2
    printf '  %s\n' "${missing_files[@]}" >&2
    exit 1
  fi

  local stage_root
  stage_root="$(mktemp -d)"
  trap "rm -rf '$stage_root'" RETURN

  local bundle_root="$stage_root/$BUNDLE_NAME"
  mkdir -p "$bundle_root"

  local -a manifest_files=()
  for rel in "${repo_files[@]}"; do
    src="$BUNDLE_REPO_ROOT/$rel"
    copy_with_parents "$src" "$bundle_root/GenRec/$rel"
    manifest_files+=("GenRec/$rel")
  done
  for rel in "${raw_files[@]}"; do
    src="$BUNDLE_RAW_DATA_ROOT/$rel"
    copy_with_parents "$src" "$bundle_root/raw_data/$rel"
    manifest_files+=("raw_data/$rel")
  done

  write_manifest "$bundle_root/bundle_manifest.json" "$bundle_root" "${manifest_files[@]}"

  rm -f "$archive_path"
  tar -czf "$archive_path" -C "$stage_root" "$BUNDLE_NAME"
  split_archive_if_needed "$archive_path" "$CHUNK_SIZE"

  echo "Created research bundle: $archive_path"
  echo "Bundle root inside archive: $BUNDLE_NAME"
  echo "Included files:"
  printf '  %s\n' "${manifest_files[@]}"
}

unpack_bundle() {
  if [[ $# -gt 2 ]]; then
    usage
    exit 1
  fi

  local archive_arg="${1:-}"
  local dest_root="${2:-$DEFAULT_UNPACK_ROOT}"
  local archive_path
  local tmp_archive_path=""

  if [[ -n "$archive_arg" ]]; then
    archive_path="$(normalize_archive_path "$archive_arg")"
  else
    archive_path="$(find_latest_bundle_archive "$(pwd)")"
  fi

  if [[ ! -f "$archive_path" ]]; then
    if [[ "$archive_path" =~ \.part\.[0-9]{3}$ || -f "$(archive_part_prefix "$archive_path")000" ]]; then
      :
    else
      echo "Error: archive not found: $archive_path" >&2
      exit 1
    fi
  fi

  mkdir -p "$dest_root"

  if [[ "$archive_path" =~ \.part\.[0-9]{3}$ || ! -f "$archive_path" ]]; then
    tmp_archive_path="$(mktemp "${TMPDIR:-/tmp}/hint_research_bundle_XXXXXX.tar.gz")"
    trap "rm -f '$tmp_archive_path'" RETURN
    rebuild_archive_from_parts "$archive_path" "$tmp_archive_path"
    archive_path="$tmp_archive_path"
  fi

  local first_entry
  local bundle_dir_name
  first_entry="$(tar -tzf "$archive_path" | sed -n '1p')"
  bundle_dir_name="${first_entry%%/*}"
  if [[ -z "$bundle_dir_name" ]]; then
    echo "Error: failed to detect bundle directory in archive: $archive_path" >&2
    exit 1
  fi
  if [[ -e "$dest_root/$bundle_dir_name" ]]; then
    echo "Error: destination already exists: $dest_root/$bundle_dir_name" >&2
    exit 1
  fi

  tar -xzf "$archive_path" -C "$dest_root"

  local manifest_path="$dest_root/$bundle_dir_name/bundle_manifest.json"
  if [[ ! -f "$manifest_path" ]]; then
    echo "Error: bundle manifest missing after unpack: $manifest_path" >&2
    exit 1
  fi

  echo "Extracted bundle to: $dest_root/$bundle_dir_name"
  echo "Manifest: $manifest_path"
}

main() {
  local cmd="${1:-}"
  case "$cmd" in
    pack)
      shift
      pack_bundle "$@"
      ;;
    unpack)
      shift
      unpack_bundle "$@"
      ;;
    -h|--help|help|'')
      usage
      ;;
    *)
      echo "Error: unknown command: $cmd" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"
