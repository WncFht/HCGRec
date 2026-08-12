# Index Scripts README

This document is a quick reference for `scripts/index/base/`. It should help you answer three questions fast:

1. Which step of the index pipeline am I running right now?
2. What are the main parameters for this script?
3. Where will the configuration and outputs for this run be written?

---

## 1. Index Pipeline at a Glance

Recommended order for the SID-based `index/` pipeline:

1. `text2emb.sh`: extract embeddings from `*.item.json` and generate `*.ids.json`
2. `train.sh`: train the RQVAE index model and write checkpoints plus `run_meta.json`
3. `generate.sh`: map items into discrete token sequences and export `Dataset.index_*.json`
4. `evaluate.sh`: evaluate a checkpoint with collision / utilization style metrics

There are also two training wrappers:

- `train_nohup.sh`: launch training in the background
- `train_kmeans_ablation.sh`: run KMeans ablations with `large`, `small`, or `none`

---

## 2. Script List

- `train.sh`: main training entrypoint; use this by default
- `train_nohup.sh`: inject common env vars and launch `train.sh` in the background
- `train_kmeans_ablation.sh`: switch KMeans settings and launch `train.sh`
- `text2emb.sh`: build embeddings for one or multiple datasets
- `generate.sh`: export index JSON for one or multiple datasets
- `evaluate.sh`: evaluate a checkpoint

---

## 3. Training Core: `train.sh`

### 3.1 Key Environment Variables

Data:

- `ROOT_DIR`: parent directory for dataset roots; defaults to the server-side path
- `MODEL_NAME`: embedding model name; used to build default data paths
- `USE_MULTI_DATASETS`: `true` or `false`
- `DATASET`, `DATA_PATH`: single-dataset mode
- `DATASETS`, `DATA_PATHS`: multi-dataset mode, separated by spaces

Quantization:

- `INDEX_N_LAYERS`: number of RQVAE layers
- `INDEX_CODEBOOK_SIZE`: codebook size per layer
- `INDEX_LAST_SK_EPSILON`: `sk_epsilon` for the final layer
- `INDEX_KMEANS_ITERS`: KMeans iteration count
- `KMEANS_INIT_ARG`, `LARGE_SCALE_KMEANS_ARG`: KMeans initialization strategy

Training:

- `NPROC_PER_NODE`, `MASTER_PORT`
- `BATCH_SIZE`, `EPOCHS`
- `AUTO_LR`, `BASE_LR`
- `LR`: overrides automatic LR scaling when set manually
- `USE_WANDB`, `WANDB_PROJECT`, `WANDB_RUN_NAME`

Outputs:

- `INDEX_TRAIN_ROOT`: root directory for index training artifacts; defaults to `./index_train_runs`
- `CKPT_TAG`: run name tag; defaults to an auto-generated string
- `CKPT_DIR`: checkpoint directory; defaults to `${INDEX_TRAIN_ROOT}/<dataset>/index/<model>/<ckpt_tag>/`
- `LOG_FILE`: training log path

### 3.2 Common Examples

Single dataset:

```bash
USE_MULTI_DATASETS=false \
DATASET=Instruments \
MODEL_NAME=qwen3-embedding-4B \
NPROC_PER_NODE=1 \
BATCH_SIZE=2048 \
bash scripts/index/base/train.sh
```

Multiple datasets:

```bash
USE_MULTI_DATASETS=true \
DATASETS="Arts Automotive Cell Games Pet Sports Tools Toys Instruments" \
MODEL_NAME=qwen3-embedding-4B \
INDEX_CODEBOOK_SIZE=1024 \
INDEX_N_LAYERS=4 \
INDEX_LAST_SK_EPSILON=0.003 \
NPROC_PER_NODE=4 \
BATCH_SIZE=256 \
bash scripts/index/base/train.sh
```

Custom run tag:

```bash
CKPT_TAG="rq4_cb1024_sk0-0-0-0.003_expA" \
WANDB_RUN_NAME="rq4-expA" \
bash scripts/index/base/train.sh
```

Put all index checkpoints under a dedicated root:

```bash
INDEX_TRAIN_ROOT=./index_train_runs \
bash scripts/index/base/train.sh
```

---

## 4. Background Training Wrappers

### 4.1 `train_nohup.sh`

Use this when you want to start training in the background without changing `train.sh`.

```bash
DATASET=Instruments \
MODEL_NAME=qwen3-embedding-4B \
BATCH_SIZE=2048 \
LR=1e-4 \
bash scripts/index/base/train_nohup.sh
```

### 4.2 `train_kmeans_ablation.sh`

Use this for quick KMeans ablations.

- `KMEANS_MODE=large`: `kmeans_init=true`, `large_scale_kmeans=true`
- `KMEANS_MODE=small`: `kmeans_init=true`, `large_scale_kmeans=false`
- `KMEANS_MODE=none`: `kmeans_init=false`, `large_scale_kmeans=false`

```bash
KMEANS_MODE=none \
DATASET=Instruments \
MODEL_NAME=qwen3-embedding-4B \
bash scripts/index/base/train_kmeans_ablation.sh
```

---

## 5. Embedding Extraction: `text2emb.sh`

### 5.1 Inputs and Outputs

Input: `data/<DATASET>/<DATASET>.item.json`

Outputs:

- `data/<DATASET>/<DATASET>.emb-<PLM_NAME>-td.npy`
- `data/<DATASET>/<DATASET>.emb-<PLM_NAME>-td.ids.json`

### 5.2 Common Example

```bash
DATASETS="Instruments Toys" \
PLM_NAME="Llama-3.1-8B-Instruct" \
MODEL_PATH="/path/to/your/plm" \
NUM_PROCESSES=4 \
BATCH_SIZE=256 \
FORCE_REBUILD=0 \
bash scripts/index/base/text2emb.sh
```

Notes:

- If `FORCE_REBUILD=0` and the embedding file already exists, the script skips rebuilding it. If only the ids file is missing, it will regenerate just that file.
- `TMP_DIR` can be used for temporary files. For large datasets, place it on a fast disk.

---

## 6. Export Index JSON: `generate.sh`

### 6.1 Key Points

- Required: `CKPT_PATH`
- Supports both single-dataset and multi-dataset modes
- Default output suffix is auto-generated and includes emb / rq / cb / ds / rid
- Naming template: `.index_emb-<emb>_rq<layers>_cb<cb-list>_ds<train-datasets>_rid<train-id>.json`
- If you want to force a custom suffix, set `OUTPUT_SUFFIX` explicitly

Single dataset:

```bash
USE_MULTI_DATASETS=false \
DATASET=Instruments \
MODEL_NAME=qwen3-embedding-4B \
CKPT_PATH=/path/to/best_collision_model.pth \
bash scripts/index/base/generate.sh
```

If `OUTPUT_SUFFIX` is not provided, the script generates one automatically from the training metadata. Downstream SFT or RL jobs only need to reuse that suffix as `INDEX_FILE`.

Multiple datasets:

```bash
USE_MULTI_DATASETS=true \
DATASETS="Arts Automotive Cell Games Pet Sports Tools Toys Instruments" \
MODEL_NAME=qwen3-embedding-4B \
CKPT_PATH=/path/to/best_collision_model.pth \
bash scripts/index/base/generate.sh
```

Manual override:

```bash
OUTPUT_SUFFIX=.index_my_exp_tag.json \
CKPT_PATH=/path/to/best_collision_model.pth \
bash scripts/index/base/generate.sh
```

---

## 7. Evaluation: `evaluate.sh`

The clearest pattern is to pass `CKPT_PATH` directly:

```bash
CKPT_PATH=/path/to/best_collision_model.pth \
DEVICE=cuda:0 \
BATCH_SIZE=2048 \
bash scripts/index/base/evaluate.sh
```

It also supports `CKPT_BASE_DIR + MODEL_FILE`, or automatic path construction from `TIMESTAMP` using the default layout `INDEX_TRAIN_ROOT/<dataset>/index/<model>/<timestamp>`.

---

## 8. Experiment Management Tips

- Set `CKPT_TAG` or `WANDB_RUN_NAME` explicitly for every experiment.
- After training, read `run_meta.json` under the timestamped output directory to recover the real configuration.
- Change only one primary factor per experiment, such as codebook size, `sk_epsilon`, dataset mix, or embedding model.
- Keep the order of `DATASETS` fixed across multi-dataset runs so directory names and result files remain aligned.

---

## 9. Common Pitfalls

- `DATASETS` and `DATA_PATHS` are space-separated, not comma-separated.
- `generate.sh` exits immediately if `CKPT_PATH` is missing.
- The default `MODEL_NAME=qwen7B` in `evaluate.sh` is only a fallback; it is safer to override it explicitly or pass `CKPT_PATH` directly.
- `run_meta.json` is the source of truth for a run. Do not rely only on directory timestamps.
