# Experiment launchers

This directory holds the paper-facing entry points for data preparation, SFT, and RL.
Everything user-facing lives in a per-domain subdirectory (`Instruments/`, `Games/`,
`Arts/`); files prefixed with `_` at this level are shared internals that wrappers call
into — they are not meant to be invoked directly.

## Wrapper convention

Every wrapper is a thin shell that follows the same pattern:

```bash
MODEL_PATH="${MODEL_PATH:-<default>}"   # every knob is an env var with a default
LAUNCH_ARGS=(--model-path "$MODEL_PATH" ...)
exec bash "${REPO_ROOT}/scripts/experiments/_canonical_rl_launcher.sh" "${LAUNCH_ARGS[@]}" "$@"
```

So you can either edit the defaults in the script or override them from the environment:

```bash
MODEL_PATH=/path/to/sft_checkpoint NUM_PROCESSES=4 \
bash scripts/experiments/Instruments/rl_fixed_ce.sh
```

All RL and SFT launchers accept `--dry-run`, which prints the fully resolved command
lines (analysis + training) without executing them. Always dry-run first.

## Entry points per domain

| Script                    | What it does                                                                 |
| ------------------------- | ---------------------------------------------------------------------------- |
| `prepare.sh check`        | validate that the raw category dir + index file are in place                 |
| `prepare.sh build-data`   | build `sft/` + `rl/` JSON splits, `id2sid.json`, `new_tokens.json` into `data/<Domain>_grec_index/` and refresh `data/dataset_info.json` |
| `sft.sh`                  | launch LLaMA-Factory SFT via `_canonical_sft_launcher.sh`                    |
| `rl_rule.sh`, `rl_*.sh`   | launch RL via `_canonical_rl_launcher.sh` (variant table in the repo README) |

`prepare.sh` forwards into the one substantive script per domain
(`prepare_<domain>_*.sh`), which is where the actual conversion logic lives.
The RL/SFT wrappers contain no logic beyond defaults — read them to see which
flags each variant sets.

## Canonical launchers and helpers

| File                          | Role                                                                        |
| ----------------------------- | --------------------------------------------------------------------------- |
| `_canonical_sft_launcher.sh`  | resolves the YAML, sets runtime env, calls `llamafactory-cli train`          |
| `_canonical_rl_launcher.sh`   | parses all RL flags, optionally runs hint analysis, then `accelerate launch -m hcgrec.trl_trainer` |
| `_launcher_common.sh`         | `require_file`/`require_dir` guards, `is_true`, `print_cmd`                  |
| `_launcher_runtime.sh`        | `setup_genrec_runtime_env`: CUDA devices, offline HF flags, `PYTHONPATH=src` |
| `_fixed_hint_artifacts.sh`    | canonical path scheme for diagnosis artifacts                                |

## RL execution flow

```
rl_*.sh  →  _canonical_rl_launcher.sh
             ├─ (fixed-hint variants only) hcgrec.analyze_rl_beam_hint
             │     beam-search diagnosis → per-sample hint-depth map
             └─ accelerate launch -m hcgrec.trl_trainer
                    selects GRPOTrainer / FixedHintRuleOnlyGRPOTrainer /
                    DynamicHintRuleOnlyGRPOTrainer / TokenPrefixGRPOTrainer
                    based on the flags the wrapper passed
```

## Common environment overrides

| Variable                    | Default (Instruments)          | Meaning                                   |
| --------------------------- | ------------------------------ | ----------------------------------------- |
| `MODEL_PATH`                | `saves/.../checkpoint-2751`    | SFT checkpoint to start RL from           |
| `OUTPUT_DIR` / `RUN_NAME`   | `rl_outputs/...`               | where checkpoints + logs go               |
| `DATA_VARIANT_DEFAULT`      | `Instruments_grec_index`       | dataset dir under `data/`                 |
| `NUM_PROCESSES` / `MAIN_PORT` | `8` / `29516`                | accelerate launch shape                   |
| `DEFAULT_CUDA_VISIBLE_DEVICES` | `0,1,...,7`                 | used when `CUDA_VISIBLE_DEVICES` is unset |
| `PER_DEVICE_TRAIN_BSZ` / `PER_DEVICE_EVAL_BSZ` / `GRAD_ACC` | `64` / `64` / `2` | batch shape                        |
| `YAML_PATH` (sft only)      | `examples/train_full/...`      | LLaMA-Factory config                      |
| `HINT_CE_LOSS_COEF`         | `0.0` (`0.005` in `*_ce*`)     | weight of prefix CE loss on hinted tokens |
| `FULL_SEQUENCE_SFT_LOSS_COEF` | `0.0`                      | weight of full-sequence SFT regularizer   |
| `DYNAMIC_HINT_MAX_DEPTH`    | `3` (dynamic only)             | max hint depth for dynamic-hint mode      |
| `TRAIN_TASK_NAMES` / `EVAL_TASK_NAMES` / `ANALYSIS_TASK_NAMES` | per-variant | restrict which RL tasks run / are analyzed |

## Fixed-hint diagnosis artifacts

Fixed-hint variants cache the beam-search diagnosis under
`temp/rl_beam_hint/artifacts/<dataset>/<task-scope>/<model>/beam<B>_hint<D>_sid<L>/`
(`summary.json`, `details.json`, `fixed_hint_map.unsolved<N>.json`). If all three
exist, the analysis step is skipped on subsequent runs; set `FORCE_REANALYZE=true`
to force a rebuild. The map file is what the trainer consumes via
`--fixed_hint_depth_map_path`.
