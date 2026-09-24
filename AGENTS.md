# AGENTS.md

## Purpose

Trains a Rocket League bot for 1v1 and 2v2 with self-play PPO in RocketSim
(RLGym 2, rlgym-learn) and runs it in game through RLBot v5. The rewrite of
2026-09-24 replaced the old rlgym-ppo / 12-stage-curriculum code. That code,
the vendored `necto/` and `reply-training/` trees, and the ballchasing replay
tools remain in git history. Commit `845afa3` is the last one that has them.

Training runs on the Windows desktop under WSL2 (`ssh desktop`, repo at
`~/games/botboi`). It has an i7-9700K (8 cores), an RTX 2080 Super, and a
40 GB WSL limit. `~/games/rocket-league-bot` on the desktop is the old
April-era checkout with 2.5 GB of replay data, so leave it alone. The Mac is
for development and short smoke runs only. `lobs` is an M4 Mac mini that runs
other jobs, so do not train there.

## Layout

- `botboi/obs.py`: `BotObs`, the 163-float observation. It is shared by
  training and the RLBot runtime. `OBS_VERSION` guards compatibility.
- `botboi/actions.py`: the 90-action lookup table, held for `TICK_SKIP = 8` ticks.
- `botboi/model.py`: `Policy`, the inference MLP (same parameter names as
  rlgym_learn_algos' `DiscreteFF`), plus `check_compatible`.
- `botboi/rlbot_obs.py`: `RLBotObsAdapter`, which turns RLBot packets into a
  game state (via rlgym_compat) and builds observations.
- `botboi/rewards.py`: reward terms and `build_reward`. Goal, zero-sum
  competitive terms, and individual shaping.
- `botboi/mutators.py`: `RandomGroundStateMutator` (episode starts besides kickoffs).
- `botboi/env.py`: `build_env`, plus `StatsProvider` for game stats.
- `botboi/config.py`: all settings. `PHASES` holds rewards, gamma, and team
  spirit. `TrainConfig`, `EnvConfig`, `SMOKE_OVERRIDES`.
- `botboi/train.py`: the rlgym-learn setup. `BotPPOController` handles
  snapshots, checkpoint metadata, and the step limit. `BotMetricsLogger`
  handles console output and `metrics.csv`.
- `botboi/checkpoints.py`: checkpoint discovery and `resolve_policy`.
- `botboi/evaluate.py`: head-to-head games between two policies.
- `botboi/export.py`: writes `bot/policy.pt` or builds a standalone bot folder.
- `bot/`: RLBot v5 package. `bot.py` imports `botboi` from its own folder
  (exported) or the repo root.
- `bin/`: `setup`, `train`, `eval`, `export`, `test`.

## Working rules

- Keep edits small and direct. No framework-style abstractions.
- Never modify or delete anything under `runs/` or a user's exported bot
  folder unless asked. `bot/policy.pt` is generated.
- Treat any change to `botboi/obs.py`, `botboi/actions.py`, or network input
  or output sizes as a fresh-training boundary. Bump `OBS_VERSION` for obs
  changes, and say so explicitly.
- Anything the RLBot runtime imports must stay out of the training stack: no
  rocketsim or rlgym_learn imports in `obs.py`, `actions.py`, `model.py`, or
  `rlbot_obs.py`. The exported bot copies exactly `export.RUNTIME_MODULES`.
- Keep resume working. `bin/train` must always pick up the latest checkpoint
  of the run.
- Update README.md and this file when the workflow or architecture changes.
- Avoid new dependencies. Pins live in `requirements.txt` (training),
  `bot/requirements.txt` (runtime), and `bin/setup` (torch).
- Never add `earl-pytorch`. Its wheel overwrites rlgym 2.x with stale copies.

## Validation

- `bin/test` runs the full suite in about a minute. `-m "not slow"` skips the
  training job.
- `tests/test_parity.py` must pass after any change to obs, `rlbot_obs.py`,
  or the bot. It replays simulator ticks as RLBot packets.
- `tests/test_bot.py` runs an exported bot folder in a clean interpreter.
- `tests/test_pipeline.py` covers train, resume with a phase change, and eval,
  using `--preset smoke`.
- For training changes, run `bin/train --run scratch --preset smoke` or a
  short custom run. Do not start long runs on the Mac.

## Gotchas (all verified 2026-09-24)

- **rlgym-learn 2.0.0 shared-info bug.** Setting `shared_info_serde_type`
  corrupts startup parsing: `collect_start_response_data` does not advance
  the offset past the shared info. Game stats therefore go through
  `runs/<run>/stats/<pid>.json` files (`StatsProvider` writes them,
  `BotMetricsLogger` reads them).
- **Allocation pool warning.** pyany-serde's numpy allocation pool warning
  (default 10k) runs a referrer scan on every allocation once an iteration
  passes 10k steps, which made collection about 15x slower. Obs and action
  serdes set `allocation_pool_warning_size=None`.
- **Step counting.** rlgym-learn's `timestep_limit` counts only the current
  launch, and differently from the controller. `BotPPOController` enforces
  the run total and stops by raising `StepLimitReached` (a
  `KeyboardInterrupt`, so rlgym-learn saves).
- **Shared-memory link files.** rlgym-learn writes them to `flinks_folder`
  (default `./shmem_flinks`) and deletes them on shutdown, which broke a
  second run started from the same directory. Each run uses
  `runs/<run>/shmem_flinks`.
- **Resume.** With `save_mid_iteration_data_in_checkpoint=False`, a resumed
  controller used to count phantom iteration steps. `_load_from_checkpoint`
  resets `iteration_timesteps`.
- **Boost pad timers.** RLBot reports seconds since pickup, while RocketSim
  and training use seconds until respawn. `RLBotObsAdapter.update` converts.
  Pad order: rlgym's `BOOST_LOCATIONS`, RocketSim's engine order, and
  rlgym_compat all agree in rlgym 2.0.1. Reversing the array mirrors the
  field.
- **Features rlgym_compat can only estimate.** `on_ground` is guessed during
  jump takeoff, and `is_boosting` lags one tick after a boost tap. The
  observation uses `is_boosting and boost > 0` (RLBot reports boosting on an
  empty tank) and a masked flip window instead of raw air time (RLBot stops
  updating it after a flip).
- **Action delay.** `RocketSimEngine(rlbot_delay=True)`, the default, applies
  actions one tick late, like RLBot. An RLBot packet's `last_input` equals
  the previous step's controls.
- **RLBot v5 package.** The pip package is `rlbot==2.0.0b55`, and a plain
  `rlbot` gets v4. `game_mode` must be `"Soccar"`. Importing `rlbot.config`
  before `rlbot.utils.logging` triggers a circular import.
- **Policy inference during collection.** On the WSL2 desktop, each GPU
  policy call for ~12 observations cost ~0.7 ms of launch and copy overhead
  whatever the network size, which capped collection at ~10k steps/s.
  `BotPPOController` therefore picks actions with a CPU copy of the actor,
  synced after every update. The 512-512-256 actor then takes ~0.3 ms per
  call, for ~15k steps/s overall. These did not help: 1024-wide actor on CPU
  (slower), two torch threads (no change), sampling in numpy (+5%),
  `min_frac_process_responses_per_collection=1.0` (5-50x slower), and more
  env processes than cores. The learner process is the bottleneck, so
  `n_proc` defaults to cores minus one.
- **WSL nvidia-smi.** It lives in `/usr/lib/wsl/lib`, which non-login
  shells (ssh commands) may lack on PATH. `bin/setup` checks there.
- **numpy.** rlgym 2.0.1 pins `numpy<2`, so rlviser-py (numpy>=2) cannot be
  installed alongside it.
- **Easy Anti-Cheat.** RLBot v5 launches Rocket League with EAC off, so bot
  matches are offline or LAN only.
