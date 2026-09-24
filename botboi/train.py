"""PPO self-play training with rlgym-learn.

Every agent in every match (both teams, 1v1 and 2v2) is driven by the one
policy being trained. Run `python -m botboi.train --help` for options.
"""

from __future__ import annotations

import os

# Keep numpy single-threaded in env processes so they don't fight over cores.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import csv
import dataclasses
import functools
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rlgym_learn import (
    BaseConfigModel,
    LearningCoordinator,
    LearningCoordinatorConfigModel,
    ProcessConfigModel,
    SerdeTypesModel,
)
from rlgym_learn.pyany_serde import NumpySerdeConfig, PyAnySerdeType
from rlgym_learn_algos.ppo import (
    BasicCritic,
    DiscreteFF,
    ExperienceBufferConfigModel,
    GAETrajectoryProcessor,
    GAETrajectoryProcessorConfigModel,
    NumpyExperienceBuffer,
    PPOAgentController,
    PPOAgentControllerConfigModel,
    PPOLearnerConfigModel,
    PPOMetricsLogger,
    SeparateActorCritic,
)
from torch.optim import Adam

from .actions import TICK_SKIP
from .checkpoints import (
    checkpoint_timesteps,
    find_latest_checkpoint,
    policy_from_checkpoint,
    policy_meta,
)
from .config import PHASES, SMOKE_OVERRIDES, EnvConfig, TrainConfig
from .env import build_env
from .model import check_compatible
from .obs import OBS_SIZE

TICKS_PER_SECOND = 120
# Normalized rewards are clipped to this. High enough that goals are never cut.
REWARD_CLIP = 50.0


class StepLimitReached(KeyboardInterrupt):
    """Raised in the learning loop at the --timesteps limit. rlgym-learn treats
    KeyboardInterrupt as "save a checkpoint and shut down"."""


class BotPPOController(PPOAgentController):
    """PPOAgentController that writes botboi.json into every checkpoint, keeps
    never-pruned policy snapshots in runs/<run>/policies/, and stops the run
    at a total step count."""

    def __init__(self, *args, meta_dir: Path, snapshot_every_ts: int, stop_at_timesteps: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.policies_dir = meta_dir / "policies"
        self.snapshot_every_ts = snapshot_every_ts
        self.stop_at_timesteps = stop_at_timesteps
        existing = [int(p.stem) for p in self.policies_dir.glob("*.pt") if p.stem.isdigit()]
        self.last_snapshot_ts = max(existing, default=0)

    def _load_from_checkpoint(self):
        super()._load_from_checkpoint()
        # Mid-iteration trajectories are not saved (they are large), so start
        # the resumed iteration from zero instead of from the saved count.
        self.iteration_timesteps = 0

    def process_timestep_data(self, timestep_data):
        super().process_timestep_data(timestep_data)
        if self.cumulative_timesteps >= self.stop_at_timesteps:
            print(f"Reached {self.stop_at_timesteps:,} total steps.")
            raise StepLimitReached

    def save_checkpoint(self):
        super().save_checkpoint()
        # The parent names each checkpoint folder after time.time_ns().
        checkpoint = max(
            (p for p in Path(self.checkpoints_save_folder).iterdir() if p.name.isdigit()),
            key=lambda p: int(p.name),
        )
        meta = policy_meta(self.cumulative_timesteps, str(checkpoint))
        (checkpoint / "botboi.json").write_text(json.dumps(meta, indent=2))
        if self.cumulative_timesteps - self.last_snapshot_ts >= self.snapshot_every_ts:
            policy, _ = policy_from_checkpoint(checkpoint)
            self.policies_dir.mkdir(parents=True, exist_ok=True)
            policy.save(self.policies_dir / f"{self.cumulative_timesteps}.pt", meta)
            self.last_snapshot_ts = self.cumulative_timesteps
            print(f"Saved policy snapshot {self.cumulative_timesteps}.pt")


class BotMetricsLogger(PPOMetricsLogger):
    """Adds game stats to the PPO metrics, prints a compact summary each
    iteration, and appends it to runs/<run>/metrics.csv.

    Game stats come from the running totals each env process writes to
    stats_dir (see StatsProvider). Each iteration reports the change since
    the previous read."""

    def __init__(self, csv_path: Path, stats_dir: Path):
        super().__init__()
        self.csv_path = csv_path
        self.stats_dir = stats_dir
        self.prev_totals: dict[str, dict[str, float]] = {}

    def collect_env_metrics(self, data: list[dict[str, Any] | None]):
        totals: dict[str, float] = {}
        for path in self.stats_dir.glob("*.json"):
            try:
                current = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            previous = self.prev_totals.get(path.name, {})
            for key, value in current.items():
                totals[key] = totals.get(key, 0.0) + value - previous.get(key, 0.0)
            self.prev_totals[path.name] = current
        steps = totals.get("steps", 0.0)
        if steps == 0:
            self.state_metrics = {}
            return
        game_minutes = steps * TICK_SKIP / TICKS_PER_SECOND / 60
        self.state_metrics = {
            "Game": {
                "Goals per game minute": totals["goals"] / game_minutes,
                "Touches per game minute": totals["touches"] / game_minutes,
                "Aerial touches per game minute": totals["aerial_touches"] / game_minutes,
                "Demos per game minute": totals["demos"] / game_minutes,
                "Mean ball speed": totals["ball_speed"] / steps,
                "Mean car speed": totals["car_speed"] / steps,
                "Fraction of time in air": totals["in_air"] / steps,
                "Mean boost": totals["boost"] / steps,
                "Fraction of 2v2 steps": _fraction_2v2(totals["cars"], steps),
            }
        }

    def report_metrics(self):
        metrics = self.get_metrics()
        flat = {f"{group}/{key}": value for group, items in metrics.items() for key, value in items.items()}
        flat = {"unix_time": time.time(), **flat}
        self._append_csv(flat)
        game = metrics.get("Game", {})
        ppo = metrics["PPO Metrics"]
        timing = metrics["Timing"]
        steps = metrics["Timestep Collection"]["Cumulative Timesteps"]
        print(
            f"[{steps:,} steps] "
            f"sps {timing['Overall Steps per Second']:,.0f} "
            f"(collect {timing['Collected Steps per Second']:,.0f}) | "
            f"reward {ppo['Average Reward']:.4f} | "
            f"entropy {ppo['Actor Entropy']:.3f} | "
            f"kl {ppo['Mean KL Divergence']:.5f} | "
            f"clip {ppo['SB3 Clip Fraction']:.3f} | "
            f"critic loss {ppo['Critic Loss']:.4f}"
        )
        if game:
            print(
                f"    goals/min {game['Goals per game minute']:.2f} | "
                f"touches/min {game['Touches per game minute']:.1f} | "
                f"aerial touches/min {game['Aerial touches per game minute']:.2f} | "
                f"ball speed {game['Mean ball speed']:.0f} | "
                f"in air {game['Fraction of time in air']:.2f} | "
                f"boost {game['Mean boost']:.0f} | "
                f"2v2 share {game['Fraction of 2v2 steps']:.2f}"
            )

    def _append_csv(self, row: dict[str, float]):
        header: list[str] = []
        if self.csv_path.exists():
            with open(self.csv_path, newline="") as f:
                header = next(csv.reader(f), [])
        new_columns = [key for key in row if key not in header]
        if new_columns:
            # Game stats can be missing from the first rows. Add columns,
            # never drop them, and rewrite the file once with the wider header.
            old_rows = []
            if header:
                with open(self.csv_path, newline="") as f:
                    old_rows = list(csv.DictReader(f))
            header += new_columns
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                writer.writerows(old_rows)
        with open(self.csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=header).writerow(row)


def _fraction_2v2(total_cars: float, steps: float) -> float:
    # Each step has 2 cars (1v1) or 4 cars (2v2).
    return (total_cars / steps - 2) / 2


def parse_args() -> TrainConfig:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=defaults.run_name, help="run name, folder under --runs-dir")
    parser.add_argument("--runs-dir", default=defaults.runs_dir)
    parser.add_argument("--phase", choices=sorted(PHASES), default=None,
                        help="reward/gamma preset (default: early for a new run, else the run's last phase)")
    parser.add_argument("--n-proc", type=int, default=defaults.n_proc, help="env processes (0 = one per CPU core)")
    parser.add_argument("--device", default=defaults.device, help="auto, cuda, cuda:0, or cpu")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="stop when the run reaches this many total steps (or press q / Ctrl+C)")
    parser.add_argument("--ts-per-iteration", type=int, default=defaults.timesteps_per_iteration)
    parser.add_argument("--policy-lr", type=float, default=defaults.policy_lr)
    parser.add_argument("--critic-lr", type=float, default=defaults.critic_lr)
    parser.add_argument("--ent-coef", type=float, default=defaults.ent_coef)
    parser.add_argument("--team-size-weights", default="1:0.5,2:0.5",
                        help='episode mix, "size:weight,...". "1:1" = 1v1 only')
    parser.add_argument("--kickoff-prob", type=float, default=EnvConfig().kickoff_prob)
    parser.add_argument("--save-every-ts", type=int, default=defaults.save_every_ts)
    parser.add_argument("--wandb", action="store_true", help="log metrics to Weights & Biases")
    parser.add_argument("--preset", choices=["default", "smoke"], default="default",
                        help="smoke = tiny settings for a ~1 minute pipeline check")
    args = parser.parse_args()

    cfg = TrainConfig(
        run_name=args.run,
        runs_dir=args.runs_dir,
        n_proc=args.n_proc,
        device=args.device,
        timesteps_per_iteration=args.ts_per_iteration,
        policy_lr=args.policy_lr,
        critic_lr=args.critic_lr,
        ent_coef=args.ent_coef,
        save_every_ts=args.save_every_ts,
        wandb=args.wandb,
        env=EnvConfig(
            team_size_weights=_parse_team_sizes(args.team_size_weights),
            kickoff_prob=args.kickoff_prob,
        ),
    )
    if args.preset == "smoke":
        cfg = dataclasses.replace(cfg, **SMOKE_OVERRIDES)
    if args.timesteps is not None:
        cfg = dataclasses.replace(cfg, timestep_limit=args.timesteps)
    run_dir = Path(cfg.runs_dir) / cfg.run_name
    phase = args.phase or _last_phase(run_dir) or "early"
    return dataclasses.replace(cfg, phase=phase, env=dataclasses.replace(cfg.env, phase=phase))


def _parse_team_sizes(text: str) -> dict[int, float]:
    weights = {}
    for item in text.split(","):
        size, weight = item.split(":")
        weights[int(size)] = float(weight)
    if any(size not in (1, 2) for size in weights):
        raise SystemExit("team sizes must be 1 or 2 (the observation holds at most 2 per team)")
    return weights


def _last_phase(run_dir: Path) -> str | None:
    run_file = run_dir / "run.json"
    if not run_file.exists():
        return None
    history = json.loads(run_file.read_text()).get("history", [])
    return history[-1]["phase"] if history else None


def _record_run_start(run_dir: Path, cfg: TrainConfig, resumed_from: Path | None) -> None:
    """Append this launch to runs/<run>/run.json so phase changes are traceable."""
    run_file = run_dir / "run.json"
    data = json.loads(run_file.read_text()) if run_file.exists() else {"history": []}
    data["history"].append(
        {
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "phase": cfg.phase,
            "resumed_from": str(resumed_from) if resumed_from else None,
            "resumed_at_timesteps": checkpoint_timesteps(resumed_from) if resumed_from else 0,
            "config": dataclasses.asdict(cfg),
        }
    )
    run_file.write_text(json.dumps(data, indent=2, default=str))


def main() -> None:
    cfg = parse_args()
    run_dir = Path(cfg.runs_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    device = cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    n_proc = cfg.n_proc or os.cpu_count() or 1
    phase = PHASES[cfg.phase]
    resume_from = find_latest_checkpoint(run_dir)
    if resume_from:
        try:
            check_compatible(policy_from_checkpoint(resume_from)[1], str(resume_from))
        except ValueError as e:
            raise SystemExit(f"Cannot resume: {e}\nStart a new run with --run <new name>.")

    print(f"Run: {run_dir}  phase: {cfg.phase}  device: {device}  env processes: {n_proc}")
    print(f"Obs size: {OBS_SIZE}  policy: {cfg.policy_layers}  critic: {cfg.critic_layers}")
    if resume_from:
        print(f"Resuming from {resume_from} ({checkpoint_timesteps(resume_from):,} steps)")
    else:
        print("Starting a new run")
    _record_run_start(run_dir, cfg, resume_from)

    def actor_critic_factory(obs_space, action_space, dtype, torch_device, agent_controller_name):
        actor = DiscreteFF(obs_space[1], action_space[1], cfg.policy_layers, dtype, torch_device)
        critic = BasicCritic(obs_space[1], cfg.critic_layers, dtype, torch_device)
        return SeparateActorCritic(actor, critic)

    def optimizers_factory(actor_critic, param_group_kwargs, agent_controller_name):
        return [
            Adam(actor_critic.actor.parameters(), **param_group_kwargs["actor"]),
            Adam(actor_critic.critic.parameters(), **param_group_kwargs["critic"]),
        ]

    # Totals from a previous launch would be double counted.
    stats_dir = run_dir / "stats"
    shutil.rmtree(stats_dir, ignore_errors=True)
    stats_dir.mkdir()
    env_cfg = dataclasses.replace(cfg.env, stats_dir=str(stats_dir))

    metrics_logger = BotMetricsLogger(run_dir / "metrics.csv", stats_dir)
    metrics_logger_config = None
    if cfg.wandb:
        from rlgym_learn_algos.logging.wandb import (
            WandbMetricsLogger,
            WandbMetricsLoggerConfigModel,
        )

        metrics_logger = WandbMetricsLogger(metrics_logger)
        metrics_logger_config = WandbMetricsLoggerConfigModel(
            inner_metrics_logger_config=None,
            project="botboi",
            group=cfg.run_name,
            run=cfg.run_name,
        )

    controller_config = PPOAgentControllerConfigModel(
        timesteps_per_iteration=cfg.timesteps_per_iteration,
        save_every_ts=cfg.save_every_ts,
        run_name=cfg.run_name,
        checkpoint_load_folder=str(resume_from) if resume_from else None,
        n_checkpoints_to_keep=cfg.checkpoints_to_keep,
        random_seed=cfg.seed,
        # The buffer refills in a couple of iterations after a resume, and
        # skipping it keeps checkpoints small.
        save_mid_iteration_data_in_checkpoint=False,
        learner_config=PPOLearnerConfigModel(
            n_epochs=cfg.n_epochs,
            batch_size=cfg.batch_size,
            n_minibatches=cfg.n_minibatches,
            ent_coef=cfg.ent_coef,
            clip_range=cfg.clip_range,
            optimizer_named_parameter_group_kwargs={
                "actor": {"lr": cfg.policy_lr},
                "critic": {"lr": cfg.critic_lr},
            },
            device=device,
        ),
        experience_buffer_config=ExperienceBufferConfigModel(
            max_size=cfg.experience_buffer_size,
            device="cpu",
            save_experience_buffer_in_checkpoint=False,
            trajectory_processor_config=GAETrajectoryProcessorConfigModel(
                gamma=phase["gamma"],
                lmbda=cfg.gae_lambda,
                standardize_rewards=True,
                reward_clip=REWARD_CLIP,
            ),
        ),
        metrics_logger_config=metrics_logger_config,
    )

    config = LearningCoordinatorConfigModel(
        base_config=BaseConfigModel(
            serde_types=SerdeTypesModel(
                agent_id_serde_type=PyAnySerdeType.STRING(),
                # The pool legitimately holds one array per step of the
                # iteration. Its size warning (default 10k) scans object
                # referrers on every allocation and cuts collection speed ~15x.
                obs_serde_type=PyAnySerdeType.NUMPY(
                    np.float32,
                    config=NumpySerdeConfig.STATIC(shape=(OBS_SIZE,), allocation_pool_warning_size=None),
                ),
                action_serde_type=PyAnySerdeType.NUMPY(
                    np.int64,
                    config=NumpySerdeConfig.STATIC(shape=(1,), allocation_pool_warning_size=None),
                ),
                reward_serde_type=PyAnySerdeType.FLOAT(),
                obs_space_serde_type=PyAnySerdeType.TUPLE(
                    (PyAnySerdeType.STRING(), PyAnySerdeType.INT())
                ),
                action_space_serde_type=PyAnySerdeType.TUPLE(
                    (PyAnySerdeType.STRING(), PyAnySerdeType.INT())
                ),
            ),
            random_seed=cfg.seed,
            # Per-run folder: rlgym-learn deletes these link files on shutdown,
            # so a shared default folder breaks other runs started from the
            # same directory.
            flinks_folder=str(run_dir / "shmem_flinks"),
            # BotPPOController enforces cfg.timestep_limit on the run total.
            # rlgym-learn's own limit counts only this launch.
            timestep_limit=2**62,
        ),
        process_config=ProcessConfigModel(n_proc=n_proc),
        agent_controller_config=controller_config,
        agent_controller_save_folder=str(run_dir / "checkpoints"),
    )

    controller = BotPPOController(
        actor_critic_factory,
        optimizers_factory,
        NumpyExperienceBuffer(GAETrajectoryProcessor()),
        metrics_logger=metrics_logger,
        meta_dir=run_dir,
        snapshot_every_ts=cfg.snapshot_every_ts,
        stop_at_timesteps=cfg.timestep_limit,
    )
    coordinator = LearningCoordinator(
        env_create_function=functools.partial(build_env, env_cfg),
        agent_controller=controller,
        config=config,
    )
    coordinator.start()


if __name__ == "__main__":
    main()
