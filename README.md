# Rocket League Bot

This repository trains and packages a Rocket League bot built with `rlgym`, `rlgym-ppo`, `rocketsim`, and `rlbot`.

## Repo Layout

- `train.py`: PPO training entry point.
- `watch.py`: loads the latest checkpoint and runs a local deterministic rollout.
- `rocket_league_bot_src/`: training environment, curriculum, observations, rewards, and reset scenarios.
- `BotBoi_v1/src/bot.py`: RLBot runtime bot.
- `bin/train`: convenience wrapper around `train.py`.
- `bin/train_tuned`: stronger multi-process training wrapper that resumes by default.
- `bin/train_tuned_fresh`: same tuned wrapper, but always starts fresh.
- `bin/progress_report`: summarizes checkpoint reward trends.
- `bin/metrics_report`: summarizes `data/training_metrics.csv`.
- `bin/evaluate_ladder`: evaluates the current checkpoint against a stable ladder of older checkpoints.

## Current Training Design

The training pipeline is intentionally staged:

1. `CONTACT`
   The bot learns to reach and touch the ball from controlled placements.
2. `DRIBBLE`
   The bot learns to keep pressure on the ball and move it through space with control.
3. `SHOOT`
   The bot learns to convert open-net and forward-ball scenarios into goals.
4. `SHOOT_CONTESTED`
   The bot learns to finish chances with a live defender between ball and goal.
5. `DEFEND`
   The bot learns first saves from dangerous goal-side starts.
6. `DEFEND_CLEAR`
   The bot learns to turn those saves into real clears and exits under pressure.
7. `DUEL`
   The bot learns short-form 1v1 conversions from replay-like attack and defense starts.
8. `SELF_PLAY`
   The bot trains in full-match 1v1 after the structured duel stage.

The current design lives primarily in:

- `rocket_league_bot_src/config.py`
- `rocket_league_bot_src/curriculum.py`
- `rocket_league_bot_src/mutators.py`
- `rocket_league_bot_src/rewards.py`
- `rocket_league_bot_src/env.py`

## Why The Training Was Reworked

The earlier version mixed many overlapping reward terms with a single generic reset pattern. That made it hard to tell what the agent was actually being trained to do, and it made curriculum behavior harder to inspect.

The current rewrite pushes the setup toward:

- fewer, clearer reward terms
- stage-specific reset scenarios
- explicit curriculum progression
- a progression that covers offense and defense before full self-play
- centralized training constants
- iteration metrics that show stage and difficulty
- snapshot registration for future league-style training against older checkpoints

## Setup

Typical local setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Commands

Start training:

```bash
bin/train
```

Start tuned training with better GPU/CPU utilization and auto-resume:

```bash
bin/train_tuned
```

Start tuned training without resuming from an older checkpoint:

```bash
bin/train_tuned_fresh
```

Start unattended background training:

```bash
bin/manage_training start
```

Check whether it is still running, what checkpoint it last saved, and the recent log tail:

```bash
bin/manage_training status
```

Follow the live training log:

```bash
bin/manage_training logs -f
```

Stop the background training process cleanly:

```bash
bin/manage_training stop
```

Train directly with custom flags:

```bash
python3 train.py --n-proc 8 --min-inference-size 8 --resume-latest
```

Resume with a frozen opponent checkpoint behind the current run:

```bash
python3 train.py --resume-latest --self-play-mode frozen --opponent-gap-ts 4000000
```

Resume with a fixed opponent checkpoint:

```bash
python3 train.py --resume-latest --self-play-mode frozen --opponent-checkpoint data/checkpoints/<run>/<ts>
```

The tuned wrappers default to:

- `n_proc=8`
- `min_inference_size=n_proc`
- `ts_per_iteration=100000`
- `ppo_batch_size=100000`
- `ppo_minibatch_size=20000`
- `exp_buffer_size=400000`

Override them per run with environment variables, for example:

```bash
N_PROC=10 PPO_MINIBATCH_SIZE=25000 bin/train_tuned
```

Watch the latest checkpoint:

```bash
./env/bin/python watch.py
```

Inspect saved checkpoints:

```bash
bin/progress_report data/checkpoints
```

Inspect live training metrics:

```bash
bin/metrics_report data/training_metrics.csv
```

Watch the full training/export dashboard live:

```bash
bin/progress_dashboard --watch 5
```

`bin/progress_dashboard` now auto-runs the evaluation ladder when the latest compatible checkpoint changes, so the dashboard keeps a checkpoint-vs-checkpoint progress signal without needing a separate eval command.

Run a checkpoint-vs-checkpoint evaluation ladder:

```bash
bin/evaluate_ladder
```

The evaluation ladder keeps the same anchor checkpoints for 10 million timesteps by default, then refreshes them forward as training advances.
That makes it easier to tell whether the current bot is actually improving instead of only tying itself in live self-play.
By default it evaluates at the current checkpoint's saved curriculum stage and difficulty.

Serve the auto-refreshing HTML training graphs locally:

```bash
bin/serve_training_report
```

Generate the HTML graph report manually:

```bash
bin/render_training_report
```

Export the latest checkpoint into the RLBot package:

```bash
bin/export_rlbot
```

Refresh the latest RLBot package and sync it into your RLBot botpack folder when one is detected:

```bash
bin/use_latest_rlbot
```

Validate the RLBot package before opening RLBot:

```bash
bin/validate_rlbot_package
```

The `bin/` entrypoints now prefer the repo-local `./env/bin/python` automatically and fall back to `python3` only if that env does not exist.

## Notes

- `watch.py` now discovers the latest checkpoint instead of using a hardcoded run path.
- Training uses `RepeatAction(LookupTableAction(), repeats=8)` to match common RLGym practice more closely than the old `repeats=2`.
- The observation contract now includes angular velocity and core car-state flags inspired by the standard RLGym observation builder. This changed `OBS_DIM`, so older checkpoints are intentionally incompatible with current training.
- Observation compatibility still matters. If you change `rocket_league_bot_src/obs.py`, review `BotBoi_v1/src/bot.py` as well.
- Later competitive stages now include a small dense attack-pressure shaping term so the bot gets credit for creating faster, more dangerous shots before sparse goal events arrive. Goals still dominate the reward mix.
- `DUEL` and `SELF_PLAY` now use competitive shaping, so non-goal reward is scored relative to the opponent team instead of being added symmetrically for both sides.
- Snapshot metadata for future old-version self-play is stored under `data/league/snapshots.json`.
- `bin/use_latest_rlbot` is the one-command way to refresh the package for in-game use. It exports the newest compatible checkpoint into `BotBoi_v1/src` and, when it finds an RLBot botpack directory, copies the `BotBoi_v1` package there too.
- If RLBot is installed in a non-standard location, set `RLBOT_BOTPACK_DIR` or pass `--botpack-dir` to `bin/use_latest_rlbot`.
- The RLBot package lives at `BotBoi_v1/src/bot.cfg`. If auto-install is skipped, load that bot config directly in RLBot GUI.
- `BotBoi_v1/src/runtime_config.json` is now the contract between training and the in-game bot package.
- During unattended training, PID 0 now auto-exports the newest checkpoint into the RLBot package when it detects a fresh save.
- Background run state is stored in `data/training_run.json` and logs go to `data/logs/train_latest.log`.
- The graph report is written to `data/training_report.html` whenever a new metrics row is logged.
- Default training uses current-policy vs current-policy self-play for throughput. Use `--self-play-mode frozen --opponent-gap-ts 4000000` when you want a slower but more stable old-checkpoint comparison target.

## Further Reading

Project-specific training notes are in [docs/training.md](/home/rafe/games/rocket-league-bot/docs/training.md).
