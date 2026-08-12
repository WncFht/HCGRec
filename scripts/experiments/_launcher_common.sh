#!/usr/bin/env bash

is_true() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

print_cmd() {
  printf '%q ' "$@"
  echo
}

require_exists() {
  local path="$1"
  local desc="$2"
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] Missing ${desc}: $path"
    exit 1
  fi
}

require_path() {
  local path="$1"
  local desc="$2"
  require_exists "$path" "$desc"
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

path_exists() {
  local path="$1"
  [[ -n "$path" && -e "$path" ]]
}
