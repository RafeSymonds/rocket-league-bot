from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import ACTION_REPEAT, CRITIC_LAYER_SIZES, OBS_DIM, POLICY_LAYER_SIZES


def find_latest_checkpoint(checkpoint_root: str) -> str:
    root = Path(checkpoint_root)
    if not root.exists():
        return ""

    candidates: list[tuple[int, float, str]] = []
    for book in root.rglob("BOOK_KEEPING_VARS.json"):
        try:
            data = json.loads(book.read_text())
            ts = int(data.get("cumulative_timesteps", 0))
        except Exception:
            ts = 0
        candidates.append((ts, book.stat().st_mtime, str(book.parent)))

    if not candidates:
        return ""

    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[-1][2]


def load_checkpoint_book(checkpoint_dir: str) -> dict[str, Any]:
    book_path = Path(checkpoint_dir) / "BOOK_KEEPING_VARS.json"
    if not book_path.exists():
        return {}
    try:
        return json.loads(book_path.read_text())
    except Exception:
        return {}


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
