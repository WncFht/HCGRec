#!/usr/bin/env bash

setup_genrec_runtime_env() {
  local repo_root="$1"
  local default_cuda_visible_devices="$2"

  export DISABLE_VERSION_CHECK=1
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$default_cuda_visible_devices}"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
  unset HF_ENDPOINT || true
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
  export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
  export ACCELERATE_LOG_LEVEL="${ACCELERATE_LOG_LEVEL:-warning}"
  export PYTHONPATH="${repo_root}/src:${PYTHONPATH:-}"
}

require_accelerate() {
  if ! command -v accelerate >/dev/null 2>&1; then
    echo "[ERROR] accelerate not found in PATH"
    exit 1
  fi
}

resolve_data_variant_dir() {
  local repo_root="$1"
  local data_variant="$2"
  local primary_dir="${repo_root}/data/${data_variant}"
  local lcrec_dir="${repo_root}/data/LC-Rec/${data_variant}"
  local lcrec_suffix="_lcrec"
  local lcrec_variant_base=""

  if [[ "$data_variant" == *"${lcrec_suffix}" ]]; then
    lcrec_variant_base="${data_variant%${lcrec_suffix}}"
    lcrec_dir="${repo_root}/data/LC-Rec/${lcrec_variant_base}"
  fi

  if [[ -d "$primary_dir" ]]; then
    echo "$primary_dir"
    return 0
  fi
  if [[ -d "$lcrec_dir" ]]; then
    echo "$lcrec_dir"
    return 0
  fi

  echo "$primary_dir"
}
