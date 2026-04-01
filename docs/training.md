# Training Notes

## Purpose

This document explains how this repository now structures Rocket League training, why that structure was chosen, and what the next research-backed improvements should be.

## External References

These local docs were informed by:

- RLGym Tools introduction: replay parsing for replay analysis and imitation learning, plus scoreboard-aware full-match training.
- Zealan's RLGym-PPO guide: rewards and learner settings should be chosen deliberately rather than copied from defaults.

Sources:

- https://rlgym.org/RLGym%20Tools/introduction/
- https://github.com/ZealanL/RLGym-PPO-Guide/blob/main/learner_settings.md
- https://github.com/ZealanL/RLGym-PPO-Guide/blob/main/rewards.md
- https://github.com/ZealanL/RLGym-PPO-Guide

## Design Principles

The current local training rewrite follows a few rules:

- Use fewer reward terms, and make each term legible.
- Use stage-specific scenarios instead of one reset pattern for everything.
- Make curriculum state explicit and inspectable.
- Teach offense and defense as separate subskills before relying on full self-play.
- Avoid hiding training assumptions across many files.
- Keep metrics tied to the curriculum so changes can be evaluated.

## What Changed Locally

The main training files are:

- `rocket_league_bot_src/config.py`
- `rocket_league_bot_src/curriculum.py`
- `rocket_league_bot_src/mutators.py`
- `rocket_league_bot_src/rewards.py`
- `rocket_league_bot_src/env.py`

### `config.py`

This is now the single place for:

- action repeat
- policy and critic layer sizes
- stage definitions
- reward weights
- reset scenario parameters

That means you can inspect stage behavior without reading the whole environment stack.

### `curriculum.py`

The curriculum is now intentionally simple:

1. `CONTACT`
2. `DRIBBLE`
3. `SHOOT`
4. `DEFEND`
5. `SELF_PLAY`

Each stage has its own difficulty progression. Difficulty affects reset geometry and ball speed rather than hiding complexity in many unrelated weights.

### `mutators.py`

The old reset logic centered on one easy pattern: place the ball near the car.

The new reset logic uses scenario families:

- `CONTACT`: controlled front-ball placements
- `DRIBBLE`: midfield carry and follow-up situations
- `SHOOT`: front-ball placements with a stronger bias toward useful scoring situations
- `DEFEND`: threat-heavy starts near goal that force touches, clears, and saves
- `SELF_PLAY`: mixed neutral and attacking resets

This matters because environment design is one of the main levers in Rocket League RL. If the bot almost never sees useful states, it cannot learn useful behaviors even with a good reward.

### `rewards.py`

The reward stack was reduced to a smaller set:

- signed goal reward
- touch reward
- speed toward ball
- face ball
- in-air reward
- touch-gated ball speed toward goal
- ball distance to goal delta
- small step penalty

This is still shaped, but it is more legible than the old system. The goal is not to make rewards perfectly minimal on day one. The goal is to keep them comprehensible enough that failures can be debugged.

### `env.py`

The environment builder now reads more directly:

- build stage-aware team sizing
- apply kickoff mutator
- apply stage-aware scenario reset mutator
- use the shared observation builder
- use the curriculum reward
- log iteration metrics with stage and difficulty

## Why This Should Train Better

This rewrite does not guarantee strong learning by itself, but it removes several likely blockers:

- `RepeatAction(..., repeats=2)` was unusually low for this kind of RLGym setup. The repo now uses `repeats=8`.
- The previous reward system had many overlapping terms, which made it hard to know which behaviors were actually being reinforced.
- The previous curriculum mixed stage logic and parameter mutation more tightly than necessary.
- The previous reset design did not clearly distinguish early contact learning from later shooting or self-play situations.

## What To Watch During Training

The most useful signals in `data/training_metrics.csv` are:

- `stage`
- `difficulty`
- `touch_rate`
- `goal_rate`
- `median_t_first`
- `median_t_goal`

Interpretation:

- In `CONTACT`, the first sign of life is rising `touch_rate` and falling `median_t_first`.
- In `DRIBBLE`, `touch_rate` should stay high while `goal_rate` begins to lift.
- In `SHOOT`, `goal_rate` should start moving before defense and self-play are introduced.
- In `DEFEND`, survival and clears matter more than pretty offense, so progress may look noisier.
- In `SELF_PLAY`, progress is slower and noisier, so stage-aware resets and evaluation become more important.

Practical monitoring commands:

```bash
python3 bin/progress_dashboard --watch 5
python3 bin/metrics_report data/training_metrics.csv
python3 bin/progress_report data/checkpoints
```

## Known Limits

This repo still does not have everything needed for a strong long-term training system.

Missing or incomplete areas:

- no replay-based imitation warm start
- no dedicated evaluation environment separate from training
- no scoreboard-aware full-match training
- no automated comparison between curriculum versions
- no documented hyperparameter sweep workflow
- no live multi-policy opponent injection yet

## Recommended Next Steps

### 1. Run short real experiments and tune from metrics

The new structure is easier to tune, but it still needs real runs. The first pass should focus on:

- stage transition thresholds
- reward weights in `SHOOT`
- reset probabilities in `SELF_PLAY`
- PPO settings such as `n_proc`, `ts_per_iteration`, and batch sizes

### 2. Add replay-based imitation learning

The RLGym Tools docs explicitly call out replay parsing for replay analysis and imitation learning. That makes replay-based warm starts a strong next step for this repo.

High-value uses:

- bootstrap contact and approach behavior
- seed realistic car-ball states
- compare learned policy behavior to replay-derived targets

### 3. Add scoreboard-aware full-match training or evaluation

RLGym Tools also documents `ScoreboardProvider` for standard Rocket League match rules. Even if training stays curriculum-driven, evaluation should eventually include scoreboard-aware match conditions.

### 4. Separate training and evaluation configs

Right now the repo is better structured, but training and evaluation are still operationally close. A future pass should add:

- a dedicated evaluation env builder
- fixed evaluation seeds or scenario mixes
- versioned curriculum configs

### 5. Finish the old-version self-play path

This repo now includes `rocket_league_bot_src/league.py`, which records promoted checkpoints into `data/league/snapshots.json`.

What it does today:

- registers milestone checkpoints as league snapshots
- creates a stable manifest for future opponent sampling

What it does not do yet:

- load different policies for blue and orange inside the current learner loop
- sample an old checkpoint as a live opponent during training

That limitation comes from the current training architecture using one shared policy across all agents. A future pass needs a multi-policy action path or an opponent-serving wrapper around inference.

## Exporting To RLBot

The in-game bot package is `BotBoi_v1/src/`.

After training:

```bash
python3 bin/export_rlbot
python3 bin/validate_rlbot_package
```

That export copies:

- latest `PPO_POLICY.pt`
- latest `BOOK_KEEPING_VARS.json`
- generated `runtime_config.json`

The RLBot runtime bot reads `runtime_config.json` to stay aligned with training-side values such as:

- observation size
- action repeat
- policy hidden sizes
- checkpoint provenance

Load `BotBoi_v1/src/bot.cfg` in RLBot GUI after export.

## Practical Guidance For Future Edits

- If you change stage behavior, start in `rocket_league_bot_src/config.py`.
- If you change progression logic, edit `rocket_league_bot_src/curriculum.py`.
- If you want new practice situations, add them in `rocket_league_bot_src/mutators.py`.
- If you want reward changes, keep them minimal and explicit in `rocket_league_bot_src/rewards.py`.
- If you change observation layout, also review `BotBoi_v1/src/bot.py`.

## Summary

The main goal of this rewrite was not to make the bot instantly good. It was to make the training logic understandable enough that improvement work can be deliberate.

That is the immediate value of the external sources here:

- RLGym Tools points toward replay- and match-aware infrastructure.
- The RLGym-PPO guide reinforces that environment and reward design matter more than copying stock examples.
- League-style improvement against older versions is the right long-term direction, but it needs explicit opponent-policy infrastructure, not just another reward tweak.
