# AGENTS.md

## Purpose

This repository trains and packages a Rocket League bot built on `rlgym`, `rlgym-ppo`, and `rlbot`.

The main code paths are:

- `train.py`: PPO training entry point.
- `watch.py`: local rollout/viewer script for a saved checkpoint.
- `rocket_league_bot_src/`: environment, curriculum, reward, observation, and mutator code.
- `BotBoi_v1/src/bot.py`: RLBot runtime bot that loads the exported policy weights.
- `bin/manage_training`: unattended background training manager.
- `bin/progress_dashboard`, `bin/render_training_report`: live progress views.
- `bin/export_rlbot`, `bin/validate_rlbot_package`: checkpoint export and RLBot package validation.

## Working Rules

- Prefer small, targeted edits. This repo is simple and does not need framework-style abstractions.
- Do not modify or delete checkpoint/model artifacts unless the user explicitly asks.
- Treat `data/checkpoints/`, `old_runs/`, `data/logs/`, `data/training_report.html`, and `BotBoi_v1/src/PPO_POLICY.pt` as generated/runtime assets, not normal source files.
- Keep training code and deployed bot code aligned. If observation layout, action lookup behavior, or model dimensions change, verify whether `BotBoi_v1/src/bot.py` must change too.
- Preserve unattended workflows. New training changes should keep resume behavior, progress reporting, and RLBot export working end to end.
- Preserve context across turns by updating local docs and `AGENTS.md` whenever the workflow or training architecture changes materially.
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
- Full-match `1v1` training uses `rlgym-tools`.
- Network access may be required to install dependencies.

## Common Commands

Run training with the repo defaults:

```bash
bin/train
```

Run training with stronger multi-process defaults and resume the latest compatible checkpoint:

```bash
bin/train_tuned
```

Run the same tuned setup without resuming:

```bash
bin/train_tuned_fresh
```

Run unattended background training:

```bash
python3 bin/manage_training start
python3 bin/manage_training status
python3 bin/manage_training logs -f
python3 bin/manage_training stop
```

Run training directly with custom args:

```bash
python3 train.py --n-proc 1 --resume-latest
```

Self-play and defend stages can now run blue against a frozen opponent checkpoint instead of the current policy on both sides.
Use `--opponent-checkpoint <dir>` to force a specific opponent, or `--opponent-gap-ts 4000000` to keep the opponent a few million timesteps behind the current resumed checkpoint.

Watch a saved checkpoint locally:

```bash
python3 watch.py
```

Inspect training progress:

```bash
python3 bin/progress_report data/checkpoints
python3 bin/metrics_report data/training_metrics.csv
python3 bin/progress_dashboard --watch 5
python3 bin/render_training_report
```

Export and validate the RLBot package:

```bash
python3 bin/export_rlbot
python3 bin/validate_rlbot_package
```

## Repo Map

`rocket_league_bot_src/config.py`

- Defines curriculum stages, reward weights, action repeat, observation size, and reset presets.

`rocket_league_bot_src/env.py`

- Builds the environment.
- Writes `data/training_metrics.csv`.
- Regenerates `data/training_report.html`.
- Contains per-process iteration logging, curriculum reporting, and auto-export of fresh checkpoints to the RLBot package.

`rocket_league_bot_src/curriculum.py`

- Manages stage progression and curriculum state.

`rocket_league_bot_src/rewards.py`

- Central reward shaping logic.

`rocket_league_bot_src/obs.py`

- Observation construction. Changes here can break compatibility with saved/exported policies.

`rocket_league_bot_src/mutators.py`

- State reset and match setup behavior.
- The final stage uses full-match `1v1` behavior via `rlgym-tools` when available.

`rocket_league_bot_src/reporting.py`

- Generates the HTML graph report at `data/training_report.html`.

`rocket_league_bot_src/checkpoints.py`

- Shared checkpoint discovery and runtime metadata helpers.

`rocket_league_bot_src/export.py`

- Shared checkpoint-to-RLBot export logic.

`rocket_league_bot_src/league.py`

- Snapshot registry for future old-version self-play / league training.

`BotBoi_v1/src/bot.py`

- Standalone inference/runtime bot for RLBot.
- Reads `runtime_config.json` so training-side action repeat and network settings stay aligned with the packaged bot.

## Change Guidance

- For training behavior changes, start with `config.py`, `rewards.py`, `obs.py`, `mutators.py`, and `env.py`.
- For checkpoint resume/save behavior, inspect `train.py`.
- For inference/export compatibility issues, inspect `watch.py`, `rocket_league_bot_src/export.py`, and `BotBoi_v1/src/bot.py`.
- If you change observation features, action dimensions, or policy architecture, call that out explicitly because existing weights may become unusable.
- If you add scripts, keep them in `bin/` when they are operator-facing helpers.
- If you change logging or metrics, keep `bin/progress_dashboard`, `bin/manage_training status`, and `data/training_report.html` useful.
- If you change checkpoint save semantics, keep `--resume-latest` and RLBot auto-export working.

## Validation

There is no formal test suite in the repo at the moment. Use lightweight validation appropriate to the change:

- Syntax check targeted files with `python3 -m py_compile ...`.
- For training-path changes, run a short local smoke test rather than a long training job unless the user asks for more.
- For reporting-script changes, run the script directly against existing `data/` artifacts if available.
- For RLBot runtime changes, verify assumptions in `BotBoi_v1/src/bot.py` against the training-side observation/action code.
- For workflow changes, verify these still work:
  - `python3 bin/manage_training status`
  - `python3 bin/progress_dashboard`
  - `python3 bin/render_training_report`
  - `python3 bin/export_rlbot`
  - `python3 bin/validate_rlbot_package`

## Safety Notes

- `watch.py` auto-discovers the latest checkpoint. If it fails, inspect checkpoint discovery before hardcoding paths.
- `bin/stats` launches TensorBoard against `runs`. Confirm that path still matches actual logging output before changing monitoring workflows.
- `train.py` defaults to `--resume-latest`, so be careful not to accidentally continue from an incompatible checkpoint after architecture changes.
- `BotBoi_v1/src/runtime_config.json` is part of the training/runtime contract. If export metadata changes, update the RLBot runtime loader too.
- There are no other repo-local agent instruction files right now. If more are added later, keep them consistent with this file.
