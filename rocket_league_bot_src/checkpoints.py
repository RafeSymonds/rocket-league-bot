from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import ACTION_REPEAT, CRITIC_LAYER_SIZES, OBS_DIM, POLICY_LAYER_SIZES


def _list_checkpoint_candidates(checkpoint_root: str) -> list[tuple[int, float, str]]:
    root = Path(checkpoint_root)
    if not root.exists():
        return []

    candidates: list[tuple[int, float, str]] = []
    for book in root.rglob("BOOK_KEEPING_VARS.json"):
        try:
            data = json.loads(book.read_text())
            ts = int(data.get("cumulative_timesteps", 0))
        except Exception:
            ts = 0
        candidates.append((ts, book.stat().st_mtime, str(book.parent)))

    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates


def find_latest_checkpoint(checkpoint_root: str) -> str:
    candidates = _list_checkpoint_candidates(checkpoint_root)
    if not candidates:
        return ""
    return candidates[-1][2]


def load_checkpoint_book(checkpoint_dir: str) -> dict[str, Any]:
    book_path = Path(checkpoint_dir) / "BOOK_KEEPING_VARS.json"
    if not book_path.exists():
        return {}
    try:
        return json.loads(book_path.read_text())
    except Exception:
        return {}


def load_curriculum_state_from_checkpoint(checkpoint_dir: str) -> dict[str, Any]:
    book = load_checkpoint_book(checkpoint_dir)
    state = book.get("curriculum_state")
    return state if isinstance(state, dict) else {}


def _checkpoint_obs_dim(checkpoint_dir: str) -> int | None:
    book = load_checkpoint_book(checkpoint_dir)
    shape = book.get("obs_running_stats", {}).get("shape")
    if isinstance(shape, list) and len(shape) == 1:
        try:
            return int(shape[0])
        except Exception:
            return None
    return None


def find_latest_compatible_checkpoint(checkpoint_root: str, expected_obs_dim: int = OBS_DIM) -> str:
    candidates = _list_checkpoint_candidates(checkpoint_root)
    for _, _, checkpoint_dir in reversed(candidates):
        obs_dim = _checkpoint_obs_dim(checkpoint_dir)
        if obs_dim is None or obs_dim == int(expected_obs_dim):
            return checkpoint_dir
    return ""


def is_checkpoint_compatible(checkpoint_dir: str, expected_obs_dim: int = OBS_DIM) -> bool:
    obs_dim = _checkpoint_obs_dim(checkpoint_dir)
    return obs_dim is None or obs_dim == int(expected_obs_dim)


def build_runtime_config(checkpoint_dir: str) -> dict[str, Any]:
    book = load_checkpoint_book(checkpoint_dir)
    return {
        "checkpoint_dir": str(Path(checkpoint_dir)),
        "cumulative_timesteps": int(book.get("cumulative_timesteps", 0)),
        "policy_average_reward": float(book.get("policy_average_reward", 0.0)),
        "obs_dim": int(OBS_DIM),
        "action_repeat": int(ACTION_REPEAT),
        "policy_hidden_sizes": list(POLICY_LAYER_SIZES),
        "critic_hidden_sizes": list(CRITIC_LAYER_SIZES),
        "action_dim": None,
    }
