"""Finding checkpoints and turning them into standalone policies.

Layout written by training (rlgym-learn decides the middle two levels):
    runs/<run>/checkpoints/<run><suffix>/<time_ns>/ppo_learner/actor_critic.pt
    runs/<run>/checkpoints/<run><suffix>/<time_ns>/ppo_agent.json
    runs/<run>/policies/<timesteps>.pt      policy snapshots, never pruned
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .actions import N_ACTIONS, TICK_SKIP
from .model import Policy, check_compatible
from .obs import OBS_SIZE, OBS_VERSION


def find_checkpoints(run_dir: str | Path) -> list[Path]:
    """All full checkpoints of a run, oldest first."""
    found = [
        p.parents[1]
        for p in (Path(run_dir) / "checkpoints").rglob("ppo_learner/actor_critic.pt")
    ]
    # Checkpoint folders are named after time.time_ns(), so name order is age order.
    return sorted(found, key=lambda p: int(p.name))


def find_latest_checkpoint(run_dir: str | Path) -> Path | None:
    checkpoints = find_checkpoints(run_dir)
    return checkpoints[-1] if checkpoints else None


def checkpoint_timesteps(checkpoint_dir: str | Path) -> int:
    with open(Path(checkpoint_dir) / "ppo_agent.json") as f:
        return int(json.load(f)["cumulative_timesteps"])


def policy_meta(timesteps: int, source: str) -> dict:
    return {
        "obs_version": OBS_VERSION,
        "obs_size": OBS_SIZE,
        "n_actions": N_ACTIONS,
        "tick_skip": TICK_SKIP,
        "timesteps": int(timesteps),
        "source": source,
    }


def policy_from_checkpoint(checkpoint_dir: str | Path) -> tuple[Policy, dict]:
    checkpoint_dir = Path(checkpoint_dir)
    state = torch.load(
        checkpoint_dir / "ppo_learner" / "actor_critic.pt",
        map_location="cpu",
        weights_only=True,
    )
    actor = {k.removeprefix("actor."): v for k, v in state.items() if k.startswith("actor.")}
    meta_file = checkpoint_dir / "botboi.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text())
    else:
        meta = policy_meta(checkpoint_timesteps(checkpoint_dir), str(checkpoint_dir))
    return Policy.from_actor_state_dict(actor), meta


def resolve_policy(spec: str, runs_dir: str = "runs") -> tuple[Policy, dict]:
    """Load a policy from a .pt snapshot, a checkpoint folder, a run folder,
    or a run name under `runs_dir` (latest checkpoint of that run)."""
    path = Path(spec)
    if path.is_file():
        policy, meta = Policy.load(path)
    elif (path / "ppo_learner").is_dir():
        policy, meta = policy_from_checkpoint(path)
    else:
        run_dir = path if path.is_dir() else Path(runs_dir) / spec
        latest = find_latest_checkpoint(run_dir)
        if latest is None:
            raise FileNotFoundError(f"No policy, checkpoint, or run found for {spec!r}")
        policy, meta = policy_from_checkpoint(latest)
    check_compatible(meta, spec)
    return policy, meta
