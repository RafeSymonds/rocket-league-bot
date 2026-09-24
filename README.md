# BotBoi

A Rocket League bot for 1v1 and 2v2, trained from scratch with self-play PPO in
RocketSim ([RLGym 2](https://rlgym.org) + [rlgym-learn](https://github.com/JPK314/rlgym-learn))
and played in game through [RLBot v5](https://rlbot.org).

| Path | What it is |
|---|---|
| `botboi/` | Python package: observations, actions, rewards, env, trainer, evaluation, export |
| `bot/` | RLBot v5 bot (`bot.toml`, `bot.py`, match configs, Windows setup script) |
| `bin/` | `setup`, `train`, `eval`, `export`, `test` |
| `tests/` | pytest suite, including sim-vs-RLBot observation parity |
| `runs/` | training output (gitignored): checkpoints, policy snapshots, `metrics.csv` |

## Setup (WSL2 + NVIDIA GPU)

1. In WSL, `nvidia-smi` must list the GPU. It comes from the Windows NVIDIA
   driver, so keep that current; do not install a Linux driver inside WSL.
2. Clone into the WSL filesystem (for example `~/rocket-league-bot`), not
   under `/mnt/c`, which is much slower.
3. `bin/setup` creates `./env` with Python 3.12 and CUDA PyTorch, then checks
   the GPU and the simulator. It installs `uv` first if it is missing.
4. `bin/test` runs the suite in about a minute. `bin/test -m "not slow"` skips
   the short training job.

Settings live in `botboi/config.py`. Run `bin/train --help` for command-line
overrides.

## Training

```bash
tmux new -s train     # training runs for hours; tmux keeps it alive
bin/train             # run "botboi", phase "early", resumes automatically
```

- Keys in the training terminal: `p` pause, `c` checkpoint now, `q` checkpoint
  and quit. Ctrl+C also saves a checkpoint.
- Running `bin/train` again resumes the run from its latest checkpoint.
  `bin/train --run <name>` starts or resumes a separately named run.
- It runs one env process per CPU core minus one (`--n-proc` to change).
  PPO updates run on the GPU. Actions during collection come from a CPU copy
  of the policy, which is faster than the GPU for these small batches.
- Output goes to `runs/botboi/`:
  - `metrics.csv` has one row per iteration.
  - `policies/<steps>.pt` is a policy snapshot every 50M steps, kept forever.
  - `checkpoints/` holds full checkpoints every 10M steps, with the last 5 kept.
  - `run.json` records each launch and its settings.
- `bin/train --wandb` also logs to Weights & Biases. Run `env/bin/wandb login`
  once first.

Each iteration prints two lines. These are real lines from a 10M-step 1v1 test
run on the MacBook (CPU, small network):

```
[9,950,798 steps] sps 26,577 (collect 46,959) | reward 0.1993 | entropy 3.531 | kl 0.00321 | clip 0.030 | critic loss 0.1089
    goals/min 0.71 | touches/min 17.1 | aerial touches/min 1.73 | ball speed 868 | in air 0.26 | boost 6 | 2v2 share 0.00
```

What healthy early training looks like:

- Entropy falls slowly from 4.50, the value for uniform random actions over 90.
- Car speed and touches per game minute rise within the first few million
  steps. In the test run, touches per minute went from 0.3 to 17 by 10M steps.
- KL stays around 0.001-0.02 and clip fraction stays below about 0.2.
- The bot first learns to hit the ball hard in any direction, then to aim.
  At 10M steps the test run still scored own goals about as often as real
  ones.

### Training phases

A phase bundles reward weights, discount factor, and 2v2 team spirit
(`PHASES` in `botboi/config.py`).

1. `early`: dense shaping teaches the bot to reach the ball, hit it hard, and
   score. Start here.
2. `main`: winning. Goals and ball-toward-goal velocity dominate, chasing
   rewards are off, gamma goes from 0.99 to 0.995, and 2v2 teammates share
   competitive credit.

Switch to `main` once touches per minute have plateaued and the bot scores in
open play. Expect that somewhere around 100-300M steps. Stop training, then
run `bin/train --phase main`. It resumes the same run, and later launches keep
the phase recorded in `run.json`.

### Speed

The desktop (i7-9700K, 7 env processes, RTX 2080 Super) runs at about 15k
steps/s overall, measured 2026-09-24. That is about 55M steps per hour, so
100M steps take about 2 hours and 1B steps about 19 hours. The limit is the
single learner process that picks every action, not the GPU.

## Evaluating progress

Training reward alone does not show improvement in self-play, so check
against older snapshots:

```bash
bin/eval botboi runs/botboi/policies/100000000.pt      # latest vs 100M
bin/eval runs/botboi/policies/300000000.pt runs/botboi/policies/200000000.pt --games 200
```

Each game starts from a kickoff and ends at the first goal, or is a draw after
30 s without a touch or 120 s total. The command prints A's score, counting a
draw as half a win, with a 95% confidence interval for 1v1 and 2v2. It also
prints the number of own goals. Once the bot aims, a new snapshot should
clearly beat older ones. Before that, while own goals are common, win rates
stay near 50% even as touches improve.

## Playing against it (Windows, RLBot v5)

RLBot starts Rocket League without Easy Anti-Cheat, so bot matches are
offline only.

1. From WSL, export a policy into a bot folder on the Windows drive:
   ```bash
   bin/export --dest /mnt/c/Users/<you>/Documents/BotBoi   # latest checkpoint
   bin/export runs/botboi/policies/500000000.pt --dest /mnt/c/Users/<you>/Documents/BotBoi
   ```
2. On Windows, install Python 3.12 and RLBot v5 from https://rlbot.org.
3. In that folder, run `powershell -ExecutionPolicy Bypass -File setup_windows.ps1`
   once. It creates `venv` with the bot's small runtime (CPU PyTorch).
4. Start a match with `venv\Scripts\python.exe run_match.py match_1v1.toml`
   (you vs BotBoi) or `match_2v2.toml` (you + BotBoi vs two BotBois). You can
   also add `bot.toml` in the RLBot v5 GUI. Set `launcher` in the match file to
   `Epic` if you play on Epic.

Re-exporting to the same folder keeps its `venv`. The bot refuses to load a
policy trained with a different observation layout.

## How it works

- **Self-play.** One policy drives every car. Episodes are 1v1 or 2v2 (50/50)
  and start from a kickoff or a random on-ground situation (50/50). An episode
  ends at a goal, after 30 s without a touch, or after 300 s.
- **Actions.** Each choice is one of 90 discrete controller combos (Necto's
  lookup table), held for 8 physics ticks, so the bot makes 15 decisions per
  second.
- **Observations** (`botboi/obs.py`, 163 floats):
  - Everything is seen from the car's team side, so orange sees a mirrored field.
  - Ball, boost pad timers, own car state, and jump/flip state.
  - Ball position and velocity in the car's own frame.
  - One teammate slot and two opponent slots, filled nearest first. Missing
    cars are zero slots with a presence flag.
- **Same code in game.** The RLBot bot runs the same observation code on a
  game state built by `rlgym-compat`. `tests/test_parity.py` replays simulator
  ticks as RLBot packets and checks the observations match.
- **Rewards** (`botboi/rewards.py`):
  - Goal: +1 for the scoring team, -1 for the conceding team.
  - Zero-sum terms, where one team's gain is the other's loss: ball velocity
    toward the opponent goal, hard touches, and demos.
  - Individual shaping: speed toward the ball, facing the ball, boost pickup,
    and boost kept.
  - Rewards are normalized by the running spread of returns.
- **Network.** A 512-512-256 policy and a 1024-1024-512-512 value network,
  both MLPs. The policy stays small because it runs on the CPU for every
  action during collection.

## Known gaps

- No training against older versions of the bot yet. Pure current-vs-current
  self-play can drift into habits that exploit only itself. `bin/eval`
  against old snapshots is how to catch that. rlgym-learn-algos'
  `MultiAgentController` is the route to add past-version opponents.
- rlgym-learn 2.0.0 was released in August 2026. Two of its rough edges are
  worked around here (see AGENTS.md).
