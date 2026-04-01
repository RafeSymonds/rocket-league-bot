# AGENTS.md

## Purpose

This repository trains and packages a Rocket League bot built on `rlgym`, `rlgym-ppo`, and `rlbot`.

The main code paths are:

- `train.py`: PPO training entry point.
- `watch.py`: local rollout/viewer script for a saved checkpoint.
- `rocket_league_bot_src/`: environment, curriculum, reward, observation, and mutator code.
- `BotBoi_v1/src/bot.py`: RLBot runtime bot that loads the exported policy weights.
- `bin/train`, `bin/stats`, `bin/progress_report`, `bin/metrics_report`: convenience scripts.

## Working Rules

- Prefer small, targeted edits. This repo is simple and does not need framework-style abstractions.
- Do not modify or delete checkpoint/model artifacts unless the user explicitly asks.
- Treat `data/checkpoints/`, `old_runs/`, and `BotBoi_v1/src/PPO_POLICY.pt` as large generated/runtime assets, not normal source files.
- Keep training code and deployed bot code aligned. If observation layout, action lookup behavior, or model dimensions change, verify whether `BotBoi_v1/src/bot.py` must change too.
- Preserve ASCII unless a file already requires otherwise.
- Avoid adding new dependencies unless necessary.

## Setup

This repo currently exposes Python dependencies only via `requirements.txt`.

Typical setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:

- `requirements.txt` includes GPU-oriented PyTorch/NVIDIA packages and a git dependency on `rlgym-ppo`.
- Network access may be required to install dependencies.

## Common Commands

Run training with the repo defaults:

```bash
bin/train
```

Run training directly with custom args:

```bash
python3 train.py --n-proc 1 --resume-latest
```

Watch a saved checkpoint locally:

```bash
python3 watch.py
```

Inspect training progress:

```bash
bin/progress_report data/checkpoints
bin/metrics_report data/training_metrics.csv
```

Open the combined stats helper:

```bash
bin/stats
```

## Repo Map

`rocket_league_bot_src/config.py`

- Defines curriculum stages and reward weight presets.

`rocket_league_bot_src/env.py`

- Builds the environment.
- Writes `data/training_metrics.csv`.
- Contains per-process iteration logging and curriculum reporting.

`rocket_league_bot_src/curriculum.py`

- Manages stage progression and curriculum state.

`rocket_league_bot_src/rewards.py`

- Central reward shaping logic.

`rocket_league_bot_src/obs.py`

- Observation construction. Changes here can break compatibility with saved/exported policies.

`rocket_league_bot_src/mutators.py`

- State reset and match setup behavior.

`BotBoi_v1/src/bot.py`

- Standalone inference/runtime bot for RLBot.
- Assumes a specific observation size (`OBS_DIM = 44`) and model structure.

## Change Guidance

- For training behavior changes, start with `config.py`, `rewards.py`, `obs.py`, `mutators.py`, and `env.py`.
- For checkpoint resume/save behavior, inspect `train.py`.
- For inference/export compatibility issues, inspect both `watch.py` and `BotBoi_v1/src/bot.py`.
- If you change observation features, action dimensions, or policy architecture, call that out explicitly because existing weights may become unusable.
- If you add scripts, keep them in `bin/` when they are operator-facing helpers.

## Validation

There is no formal test suite in the repo at the moment. Use lightweight validation appropriate to the change:

- Syntax check targeted files with `python3 -m py_compile ...`.
- For training-path changes, run a short local smoke test rather than a long training job unless the user asks for more.
- For reporting-script changes, run the script directly against existing `data/` artifacts if available.
- For RLBot runtime changes, verify assumptions in `BotBoi_v1/src/bot.py` against the training-side observation/action code.

## Safety Notes

- `watch.py` currently hardcodes a checkpoint path. If it fails, update the path instead of assuming the script auto-discovers the latest run.
- `bin/stats` launches TensorBoard against `runs`. Confirm that path still matches actual logging output before changing monitoring workflows.
- `train.py` defaults to `--resume-latest`, so be careful not to accidentally continue from an incompatible checkpoint after architecture changes.
