# Rocket League Bot

This repository trains and packages a Rocket League bot built with `rlgym`, `rlgym-ppo`, `rocketsim`, and `rlbot`.

## Repo Layout

- `train.py`: PPO training entry point.
- `watch.py`: loads the latest checkpoint and runs a local deterministic rollout.
- `rocket_league_bot_src/`: training environment, curriculum, observations, rewards, and reset scenarios.
- `BotBoi_v1/src/bot.py`: RLBot runtime bot.
- `bin/train`: convenience wrapper around `train.py`.
- `bin/progress_report`: summarizes checkpoint reward trends.
- `bin/metrics_report`: summarizes `data/training_metrics.csv`.

## Current Training Design

The training pipeline is intentionally staged:

1. `CONTACT`
   The bot learns to reach and touch the ball from controlled placements.
2. `DRIBBLE`
   The bot learns to keep pressure on the ball and move it through space with control.
3. `SHOOT`
   The bot learns to convert open-net and forward-ball scenarios into goals.
4. `DEFEND`
   The bot learns to clear dangerous balls and survive threat-heavy starts.
5. `SELF_PLAY`
   The bot trains in 1v1 with mixed resets and much sparser shaping.

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

Train directly with custom flags:

```bash
python3 train.py --n-proc 1 --resume-latest
```

Watch the latest checkpoint:

```bash
python3 watch.py
```

Inspect saved checkpoints:

```bash
bin/progress_report data/checkpoints
```

Inspect live training metrics:

```bash
bin/metrics_report data/training_metrics.csv
```

## Notes

- `watch.py` now discovers the latest checkpoint instead of using a hardcoded run path.
- Training uses `RepeatAction(LookupTableAction(), repeats=8)` to match common RLGym practice more closely than the old `repeats=2`.
- Observation compatibility still matters. If you change `rocket_league_bot_src/obs.py`, review `BotBoi_v1/src/bot.py` as well.
- Snapshot metadata for future old-version self-play is stored under `data/league/snapshots.json`.

## Further Reading

Project-specific training notes are in [docs/training.md](/Users/rafe/games/rocket-league-bot/docs/training.md).
