<div align="center">

# HCGRec

**Hint-Conditioned Generative Recommendation with Semantic IDs**

</div>

HCGRec is a research codebase for semantic-ID generative recommendation. The repository covers the full workflow used in the project: raw Amazon-style filtering, semantic ID construction, supervised fine-tuning with LLaMA-Factory, hint-conditioned GRPO training, and constrained-decoding evaluation.

## Overview

HCGRec focuses on a training failure mode in recommendation-oriented RL. In many rollout groups, none of the sampled continuations reaches the target item branch, so every sample receives the same reward and the update carries little or no useful learning signal. This repository adds hint-aware training on top of a standard SID-based generative recommendation stack.

- Reachability diagnosis checks whether the current model can reach the target SID branch under the rollout budget.
- Training-only hints expose a short target prefix for hard samples during RL.
- Hint-aware optimization separates hinted prefix tokens from sampled suffix tokens during the policy update.
- Constrained decoding keeps generation inside the valid SID set during evaluation.

## Repository Overview

| Path                            | Description                                                                            |
| ------------------------------- | -------------------------------------------------------------------------------------- |
| `src/hcgrec/`                   | HCGRec training, hint analysis, constrained evaluation, and utility modules            |
| `src/index/`                    | embedding extraction, RQ-VAE index training, SID export, and index evaluation          |
| `src/rewards/`                  | reward implementations for RL training                                                 |
| `scripts/experiments/<Domain>/` | paper-facing wrappers for data prep, SFT, and RL on `Instruments`, `Games`, and `Arts` |
| `scripts/index/base/`           | base SID pipeline: `text2emb`, `train`, `generate`, and `evaluate`                     |
| `scripts/eval/`                 | single-checkpoint and multi-checkpoint evaluation helpers                              |
| `scripts/data/`                 | generic dataset conversion helpers                                                     |
| `examples/train_full/`          | LLaMA-Factory YAML configs used by SFT launchers                                       |
| `config/`                       | DeepSpeed configs from `zero0` to `zero3_offload`                                      |

## Quickstart

### 1. Environment setup

The supported install path is `uv`. This repo is a normal `src/` layout package, and the installable Python package name is `hcgrec`.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

`.env.example` is only a template. Nothing in the current repo auto-loads it. If you want reusable local defaults, copy it to something like `.env` and source it yourself before launching the scripts.

### 2. Build a paper-facing dataset variant

If you already have raw category files and an index under `data/Instruments/`, the shortest path is the per-domain wrapper:

```bash
bash scripts/experiments/Instruments/prepare.sh check
bash scripts/experiments/Instruments/prepare.sh build-data
```

By default this writes `data/Instruments_grec_index/` with:

- `sft/train.json`, `sft/valid.json`, `sft/test.json`
- `rl/train.json`, `rl/valid.json`, `rl/test.json`
- `id2sid.json`
- `new_tokens.json`

Use the corresponding wrappers under `scripts/experiments/Games/` and `scripts/experiments/Arts/` for those domains.

### 3. Launch SFT

```bash
bash scripts/experiments/Instruments/sft.sh --dry-run
bash scripts/experiments/Instruments/sft.sh
```

The default Instruments launcher points at `examples/train_full/Instruments/instruments_rec_full_sft_3b_dsz3_qwen4b_4_256_grec_genrec_aligned_8gpu.yaml`. To switch configs, override `YAML_PATH`.

### 4. Launch HCGRec RL

Run the preflight first:

```bash
python scripts/check_rl_env.py \
  --model-path /path/to/sft_checkpoint \
  --data-dir data/Instruments_grec_index/rl \
  --index-path data/Instruments_grec_index/id2sid.json \
  --output-dir rl_outputs/instruments_hcgrec_smoke
```

Then launch RL:

```bash
MODEL_PATH=/path/to/sft_checkpoint \
bash scripts/experiments/Instruments/rl_rule.sh --dry-run

MODEL_PATH=/path/to/sft_checkpoint \
bash scripts/experiments/Instruments/rl_rule.sh
```

Other RL variants live beside it, such as `rl_dynamic.sh`, `rl_fixed.sh`, `rl_fixed_ce.sh`, and `rl_ndcg.sh`.

### 5. Evaluate a checkpoint

```bash
bash scripts/eval/evaluate_checkpoint.sh \
  --checkpoint-path /path/to/checkpoint \
  --test-data-path data/Instruments_grec_index/sft/test.json \
  --index-path data/Instruments_grec_index/id2sid.json
```

For direct Python evaluation:

```bash
python -m hcgrec.evaluate \
  --model_name_or_path /path/to/checkpoint \
  --test_data_path data/Instruments_grec_index/sft/test.json \
  --index_path data/Instruments_grec_index/id2sid.json \
  --result_json_path temp/eval/instruments_result.json
```

## Full Pipeline Walk-through

### 0. Prerequisites

- Python `3.11` to `3.13`
- One or more CUDA GPUs for training
- A causal LLM checkpoint for SFT and RL
- An embedding model checkpoint for SID construction

### 1. Environment Setup

Use `uv` for the full workspace setup:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

Useful environment variables:

- `HF_ENDPOINT` if you use a Hugging Face mirror
- `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_MODE` for experiment tracking
- `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1` for offline runs
- `CUDA_VISIBLE_DEVICES`, `CUDA_LIST`, `RESULTS_ROOT`, and `TEMP_ROOT` for launcher control

If you only want the external training CLI outside this repo, the equivalent standalone install is:

```bash
uv pip install "llamafactory==0.9.5"
```

### 2. Raw Data Filtering

The raw Amazon18-style preprocessing entrypoint is `data/amazon18_data_process.py`.

If your raw files follow the script defaults, run it inside `data/`:

```bash
cd data
python amazon18_data_process.py \
  --dataset Instruments \
  --user_k 5 \
  --item_k 5 \
  --st_year 1996 \
  --st_month 10 \
  --ed_year 2018 \
  --ed_month 10 \
  --output_path .
```

This mode expects files such as `meta_Instruments.json` and `Instruments_5.json` in the current directory.

If your raw filenames or locations differ, call the Python entrypoint from the repo root and pass them explicitly:

```bash
python data/amazon18_data_process.py \
  --dataset Instruments \
  --metadata_file /path/to/meta_Instruments.json \
  --reviews_file /path/to/Instruments_5.json \
  --output_path data
```

The script writes files such as:

- `data/Instruments/Instruments.train.inter`
- `data/Instruments/Instruments.valid.inter`
- `data/Instruments/Instruments.test.inter`
- `data/Instruments/Instruments.item.json`
- `data/Instruments/Instruments.review.json`
- `data/Instruments/Instruments.inter.json`

### 3. SID Construction

The base SID pipeline under `scripts/index/base/` is:

1. `text2emb.sh`
2. `train.sh`
3. `generate.sh`
4. `evaluate.sh`

Extract item embeddings:

```bash
DATASETS="Instruments" \
PLM_NAME="qwen3-embedding-4B" \
MODEL_PATH="/path/to/embedding-model" \
NUM_PROCESSES=1 \
BATCH_SIZE=256 \
bash scripts/index/base/text2emb.sh
```

Train the index model:

```bash
USE_MULTI_DATASETS=false \
DATASET=Instruments \
MODEL_NAME=qwen3-embedding-4B \
NPROC_PER_NODE=1 \
BATCH_SIZE=2048 \
bash scripts/index/base/train.sh
```

Generate SID files from a trained checkpoint:

```bash
USE_MULTI_DATASETS=false \
DATASET=Instruments \
MODEL_NAME=qwen3-embedding-4B \
CKPT_PATH=/path/to/best_collision_model.pth \
bash scripts/index/base/generate.sh
```

Evaluate the index checkpoint:

```bash
CKPT_PATH=/path/to/best_collision_model.pth \
DEVICE=cuda:0 \
BATCH_SIZE=2048 \
bash scripts/index/base/evaluate.sh
```

### 4. Convert Raw Category Data into SFT and RL Datasets

For the current paper workflow, prefer the domain wrappers:

```bash
bash scripts/experiments/Instruments/prepare.sh check
bash scripts/experiments/Instruments/prepare.sh build-data
```

Equivalent wrappers exist for:

- `scripts/experiments/Games/prepare.sh`
- `scripts/experiments/Arts/prepare.sh`

These wrappers validate the raw category directory, pick the resolved index file, build the SFT and RL JSON files, and update `data/dataset_info.json` for LLaMA-Factory.

### 5. Supervised Fine-Tuning

The SFT wrappers are thin shells around `scripts/experiments/_canonical_sft_launcher.sh`.

```bash
bash scripts/experiments/Instruments/sft.sh --dry-run
bash scripts/experiments/Instruments/sft.sh
```

To switch domains:

- `bash scripts/experiments/Games/sft.sh`
- `bash scripts/experiments/Arts/sft.sh`

To switch the exact YAML, override `YAML_PATH` with a file from `examples/train_full/`.

### 6. Hint-Conditioned RL

The RL wrappers are thin shells around `scripts/experiments/_canonical_rl_launcher.sh`.

Start with a preflight:

```bash
python scripts/check_rl_env.py \
  --model-path /path/to/sft_checkpoint \
  --data-dir data/Instruments_grec_index/rl \
  --index-path data/Instruments_grec_index/id2sid.json \
  --output-dir rl_outputs/instruments_hcgrec_smoke
```

Then launch the paper-style rule-only run:

```bash
MODEL_PATH=/path/to/sft_checkpoint \
bash scripts/experiments/Instruments/rl_rule.sh --dry-run

MODEL_PATH=/path/to/sft_checkpoint \
bash scripts/experiments/Instruments/rl_rule.sh
```

Other wrappers expose different hinting or reward settings:

- `rl_dynamic.sh`
- `rl_fixed.sh`
- `rl_fixed_ce.sh`
- `rl_fixed_full_sequence_sft.sh`
- `rl_ndcg.sh`

### 7. Evaluation

Use the Python module for direct constrained decoding:

```bash
python -m hcgrec.evaluate \
  --model_name_or_path /path/to/checkpoint \
  --test_data_path data/Instruments_grec_index/sft/test.json \
  --index_path data/Instruments_grec_index/id2sid.json \
  --result_json_path temp/eval/instruments_result.json
```

Use the shell wrapper for checkpoint-oriented evaluation:

```bash
bash scripts/eval/evaluate_checkpoint.sh \
  --checkpoint-path /path/to/checkpoint \
  --test-data-path data/Instruments_grec_index/sft/test.json \
  --index-path data/Instruments_grec_index/id2sid.json
```

For multi-checkpoint or watcher workflows, see:

- `scripts/eval/evaluate_all_checkpoints.sh`
- `python -m hcgrec.evaluate_all_checkpoints_sidecar`

### 8. Optional Ops Helpers

`scripts/ops/` contains repo-local helpers for syncing results, uploader state, and evaluation maintenance. They are optional and not required for the main training pipeline.

## Acknowledgements

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for the SFT training stack
- [TRL](https://github.com/huggingface/trl) for GRPO and reward-model training utilities
- Open-source generative recommendation work that helped shape the SID-based training and evaluation workflow
