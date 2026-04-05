# Training Notes

## Purpose

This document records the current training architecture, the main failures we have already seen, and the working assumptions behind the latest training rewrite.

It exists to preserve context across runs and across chats. If training looks strange again, start here before guessing.

## Current Bottom Line

The repo should not rely on raw full-match self-play as the main proof of learning.

What we observed locally:

- Current-vs-current self-play can drive `touch_rate` and `goal_rate` to `1.0` without producing clear improvement against older checkpoints.
- Full-match eval originally overcounted draws, so older reports may understate improvement.
- Throughput matters. Frozen-opponent self-play is slower, and auto-eval running next to training can make it much slower.
- Scenario quality matters. Resetting only the ball while leaving cars at kickoff was not close enough to realistic Rocket League play.
- Reward shaping matters. If rewards do not reinforce game-relevant mechanics, the policy can optimize easy metrics without learning stronger play.

The current strategy is:

1. Train subskills with scenario-based stages.
2. Use denser but legible game-relevant reward terms.
3. Add harder contested pre-duel stages before full-match `SELF_PLAY`.
4. Use eval against older checkpoints as the real signal of progress.
5. Keep `SELF_PLAY` sparse and zero-sum enough that both sides cannot farm the same shaping signal at once.

## Research Basis

These conclusions are aligned with common Rocket League bot training practice:

- RLGym encourages custom reward functions and custom state setters rather than relying on one generic environment reset.
- RLGym Tools supports replay parsing and scoreboard-aware match behavior, which points toward replay-like states and realistic evaluation.
- Community Rocket League bot training guidance consistently treats reward design and state distribution as first-class, not secondary, concerns.

References:

- https://rlgym.org/
- https://rlgym.org/RLGym%20Tools/introduction/
- https://github.com/ZealanL/RLGym-PPO-Guide
- https://github.com/Rolv-Arild/replay-pretraining

## Main Things We Learned

### 1. Training metrics can lie

`goal_rate=1.0` in training does not mean the policy is getting stronger at Rocket League.

Why:

- In self-play, both sides come from the same policy distribution.
- The agent can co-adapt to the current sparring partner.
- "A goal happened" is much weaker than "the current checkpoint beats older fixed checkpoints."

Working rule:

- Use training metrics to detect stage competence and stuckness.
- Use checkpoint-vs-checkpoint eval to judge actual improvement.

### 2. Evaluation had a real draw-counting bug

The original full-match eval logic treated many finished games as draws unless the terminal state itself had `goal_scored=True`.

That was wrong because full matches often end on timer or on the ball touching the ground after the last goal.

The fix in `rocket_league_bot_src/eval.py` now:

- reads final score from the shared scoreboard
- tracks wins separately from goal counts
- records `blue_win_rate` as wins per episode rather than goals per episode

Important caveat:

- old rows already written to `data/eval/results.csv` may still contain misleading draw-heavy history from before the fix

### 3. A reward bug was distorting self-play

Earlier shaping rewarded progress toward the wrong goal. That meant some ball-to-goal shaping was encouraging the agent to push the ball the wrong way.

That has already been corrected in `rocket_league_bot_src/rewards.py`.

### 4. Generic resets were too weak

Older stage resets mostly moved the ball but often left the cars in weak or unrealistic positions.

That makes it much harder to learn useful 1v1 behavior because the state distribution is poor.

The repo now uses more replay-like scenario resets that position:

- the ball
- the attacking car
- the defending car
- car yaw
- car velocity and boost where useful

### 5. Full self-play is not the right first teacher

Full-match `SELF_PLAY` is useful, but it is noisy and expensive.

It should come after the policy can already:

- reach the ball
- carry it
- convert chances
- clear threats
- survive short contested 1v1 situations

That is why the curriculum now inserts harder contested subskill stages before `DUEL` and `SELF_PLAY`.

### 6. Throughput tradeoffs are real

We observed a big speed drop when training and reporting/eval competed for resources.

Known causes:

- full-match stages are slower than short scenario stages
- frozen-opponent self-play runs a second policy and is slower than current-vs-current
- `bin/serve_training_report` and `bin/progress_dashboard` can auto-run eval unless disabled

Working rule:

- default training should optimize for throughput
- frozen-opponent training and ladder eval should be used intentionally, not constantly

### 7. Our observation set was too thin

Compared to the standard RLGym default observation builder, our older observation space was missing several features that are commonly used:

- car angular velocity
- ball angular velocity
- car state flags such as supersonic, jumped, double-jumped, and demoed

Those are now part of the local observation contract.

Working rule:

- prefer adding proven, high-signal state features used by established bot setups before increasing model size

## Current Curriculum

The current intended stage order is:

1. `CONTACT`
2. `DRIBBLE`
3. `SHOOT`
4. `SHOOT_CONTESTED`
5. `DEFEND`
6. `DEFEND_CLEAR`
7. `DUEL`
8. `SELF_PLAY`

Stage intent:

- `CONTACT`: learn to approach and touch quickly
- `DRIBBLE`: learn control and follow-up touches
- `SHOOT`: learn forward pressure and finishing in relatively open scenarios
- `SHOOT_CONTESTED`: learn to finish with a real defender present
- `DEFEND`: learn first-contact saves from dangerous starts
- `DEFEND_CLEAR`: learn to turn those saves into clear exits and counterpressure
- `DUEL`: learn short contested 1v1 attack/defense conversions from realistic setups
- `SELF_PLAY`: learn full-match adaptation only after the core mechanics already exist

Important implementation note:

- Stage transitions now reset stage EMAs instead of carrying them forward. This avoids later stages instantly promoting on inherited momentum from easier stages.

## Current Reward Philosophy

Rewards should be game-relevant, stage-aware, and readable enough to debug.

Current important reward types in `rocket_league_bot_src/rewards.py`:

- signed goal reward
- touch reward
- speed toward ball
- face ball
- in-air reward
- touch-gated ball speed toward opponent goal
- ball distance-to-goal delta
- hard-hit reward
- flip-touch reward
- save/clear reward
- boost gain reward
- boost-keep reward
- small step penalty

Why these exist:

- `hard_hit`: teaches that strong, useful touches matter
- `flip_touch`: reinforces dodge timing into the ball
- `save_clear`: reinforces turning dangerous defensive states into safer ones
- `boost_gain`: teaches boost collection as a useful objective
- `boost_keep`: discourages being empty all the time and rewards healthy resource state

Working rule:

- Do not add many new reward terms casually.
- If a reward is added, it should correspond to a real game behavior we want and should be inspectable in training outcomes.
- In `SELF_PLAY`, shaped rewards should be sparse and competitive. If both sides can farm them simultaneously, they are likely hurting match-strength learning.

## Current Observation Philosophy

Observations should include the core mechanical state needed to act well in Rocket League, not just coarse geometry.

The current observation set in `rocket_league_bot_src/obs.py` now includes:

- car orientation
- car linear velocity
- car angular velocity
- boost amount
- on-ground state
- supersonic state
- jumped state
- double-jumped state
- demoed state
- relative ball position
- relative ball linear velocity
- ball angular velocity
- goal-relative features
- nearest opponent relative features

This is still smaller than the full RLGym default observation because it does not yet include full boost-pad timer maps.

Working rule:

- add high-value state first
- avoid large observation explosions unless they are clearly justified
- treat observation-dimension changes as checkpoint compatibility breaks

## Current Reset Philosophy

State distribution is part of the training signal.

The scenario mutator in `rocket_league_bot_src/mutators.py` now tries to create meaningful Rocket League situations rather than generic ball placements.

Examples:

- attack starts with the attacker behind the ball and the defender between ball and goal
- defense starts with the defender near goal under immediate threat
- neutral starts place both cars around midfield in contestable positions

Working rule:

- if a stage underperforms, inspect the reset geometry before assuming PPO or reward weights are the main issue

For `DEFEND` specifically:

- the stage should be threat-heavy, not mostly neutral
- the defender should start between ball and goal often enough to practice saves and clears
- reward shaping should emphasize stopping danger more than immediate offensive conversion

## Self-Play Policy

Current default:

- train against the current policy for throughput

Optional mode:

- use `--self-play-mode frozen` with either `--opponent-checkpoint` or `--opponent-gap-ts`

Why the default is current-vs-current:

- much faster
- easier to keep the learner busy

Why frozen mode still matters:

- more stable opponent target
- better for deliberate comparison runs

Working rule:

- use current-vs-current for cheap training
- use frozen opponents and ladder eval to check whether that training transfers into actual strength

## Evaluation Policy

Evaluation is more important than raw training `goal_rate`.

What eval should answer:

- does the new checkpoint beat fixed older checkpoints more often over time
- where is improvement happening and where is it flat
- are we seeing actual wins or only draw-heavy games

Current reporting behavior:

- `rocket_league_bot_src/eval.py` maintains a ladder of older anchor checkpoints
- `rocket_league_bot_src/reporting.py` renders an Evaluation History section into `data/training_report.html`
- that history plots win rate over current checkpoint timesteps against older checkpoint anchors

Known limitations:

- the ladder does not evaluate against every historical checkpoint
- old CSV rows may still include pre-fix draw accounting
- report servers can consume resources if they auto-run eval too aggressively

## Operational Notes

### Monitoring

Useful commands:

```bash
bin/manage_training status
bin/manage_training logs -f
bin/metrics_report data/training_metrics.csv
bin/progress_dashboard --watch 5
bin/render_training_report --no-auto-eval
bin/evaluate_ladder
```

### Report server

`bin/serve_training_report` and `bin/server_training_report` can auto-refresh eval unless started with `--no-auto-eval`.

If training speed collapses unexpectedly, check whether report/eval processes are competing for resources before changing PPO settings.

### Background training

`train.py` now disables `KBHit` when stdin is not a TTY so unattended runs do not crash on terminal-control calls.

### Fresh runs

When reward semantics or stage design change materially, prefer a fresh run over trying to compare directly to old curves.

## What Probably Matters Next

If learning is still weak, the most likely next levers are:

1. Better scenario distributions, especially in `DUEL` and early `SELF_PLAY`
2. Tighter reward weights for game-relevant behaviors rather than more reward terms
3. Stronger fixed-opponent evaluation and milestone comparisons
4. Replay-derived starts or imitation warm-starts
5. Scoreboard-aware full-match evaluation and possibly training

## Things To Be Careful About

- Observation or action-layout changes can invalidate saved/exported policies.
- Eval history may mix old and new accounting unless old rows are regenerated.
- Training speed comparisons are meaningless if one run had report-server auto-eval and the other did not.
- A run that looks great in `CONTACT` or `DRIBBLE` has not proven match strength yet.
- A run that looks great in current-vs-current self-play may still fail against older fixed checkpoints.

## Source Files To Check First

- `rocket_league_bot_src/config.py`
- `rocket_league_bot_src/curriculum.py`
- `rocket_league_bot_src/mutators.py`
- `rocket_league_bot_src/rewards.py`
- `rocket_league_bot_src/eval.py`
- `rocket_league_bot_src/reporting.py`
- `train.py`
- `bin/manage_training`
