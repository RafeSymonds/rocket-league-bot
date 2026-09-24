"""Training settings. Everything tunable lives here.

A training "phase" bundles the reward weights with the settings that should
change together as the bot improves. Start a run with `early`, then resume
the same run with `--phase main` once the bot reliably hits the ball and
scores in open play (see README "Training phases").
"""

from __future__ import annotations

from dataclasses import dataclass, field

PHASES: dict[str, dict] = {
    # Learn to reach, hit, and score. Dense shaping dominates.
    "early": {
        "gamma": 0.99,
        "team_spirit": 0.0,
        "rewards": {
            "goal": 20.0,
            "touch": 0.5,
            "strong_touch": 8.0,
            "ball_toward_goal": 0.3,
            "speed_toward_ball": 0.5,
            "face_ball": 0.05,
            "in_air": 0.03,
            "boost_pickup": 1.5,
            "save_boost": 0.03,
            "demo": 8.0,
        },
    },
    # Play to win. Goals and ball-toward-goal dominate, chasing shaping is off,
    # and 2v2 teammates share competitive credit so both stop chasing.
    "main": {
        "gamma": 0.995,
        "team_spirit": 0.3,
        "rewards": {
            "goal": 20.0,
            "touch": 0.0,
            "strong_touch": 5.0,
            "ball_toward_goal": 0.6,
            "speed_toward_ball": 0.05,
            "face_ball": 0.0,
            "in_air": 0.0,
            "boost_pickup": 1.0,
            "save_boost": 0.03,
            "demo": 8.0,
        },
    },
}


@dataclass
class EnvConfig:
    phase: str = "early"
    # Episode mode weights: {team size: weight}. 1 = 1v1, 2 = 2v2.
    team_size_weights: dict[int, float] = field(default_factory=lambda: {1: 0.5, 2: 0.5})
    kickoff_prob: float = 0.5
    no_touch_timeout_seconds: float = 30.0
    episode_timeout_seconds: float = 300.0
    # Where env processes write game stats for the metrics logger (None = off).
    stats_dir: str | None = None


@dataclass
class TrainConfig:
    run_name: str = "botboi"
    runs_dir: str = "runs"
    phase: str = "early"
    n_proc: int = 0  # 0 = one env process per CPU core
    device: str = "auto"  # auto = cuda if available, else cpu
    timestep_limit: int = 10_000_000_000  # total steps for the run
    timesteps_per_iteration: int = 100_000
    batch_size: int = 100_000
    n_minibatches: int = 2
    n_epochs: int = 2
    experience_buffer_size: int = 200_000
    policy_lr: float = 2e-4
    critic_lr: float = 2e-4
    ent_coef: float = 0.01
    clip_range: float = 0.2
    gae_lambda: float = 0.95
    policy_layers: tuple[int, ...] = (1024, 1024, 512, 512)
    critic_layers: tuple[int, ...] = (1024, 1024, 512, 512)
    save_every_ts: int = 10_000_000
    checkpoints_to_keep: int = 5
    # Standalone policy snapshots (runs/<run>/policies/) for eval and export.
    # These are never pruned.
    snapshot_every_ts: int = 50_000_000
    seed: int = 123
    wandb: bool = False
    env: EnvConfig = field(default_factory=EnvConfig)


# Tiny settings that finish in about a minute on a laptop CPU. Used by
# tests and `bin/train --preset smoke` to prove the pipeline runs end to end.
SMOKE_OVERRIDES = {
    "n_proc": 2,
    "device": "cpu",
    "timestep_limit": 30_000,
    "timesteps_per_iteration": 5_000,
    "batch_size": 5_000,
    "n_minibatches": 1,
    "n_epochs": 1,
    "experience_buffer_size": 5_000,
    "policy_layers": (64, 64),
    "critic_layers": (64, 64),
    "save_every_ts": 10_000,
    "snapshot_every_ts": 10_000,
}
