from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import (
    ACTION_REPEAT,
    CRITIC_LAYER_SIZES,
    OBS_DIM,
    POLICY_LAYER_SIZES,
    Stage,
)


_STAGE_ORDER = {
    Stage.CONTACT.value: 0,
    Stage.DRIBBLE.value: 1,
    Stage.SHOOT.value: 2,
    Stage.AERIAL_CONTACT.value: 3,
    Stage.AERIAL_SHOOT.value: 4,
    Stage.SHOOT_CONTESTED.value: 5,
    Stage.SHADOW_DEFEND.value: 6,
    Stage.DEFEND.value: 7,
    Stage.DEFEND_CLEAR.value: 8,
    Stage.POSITIONAL_DUEL.value: 9,
    Stage.DUEL.value: 10,
    Stage.SELF_PLAY.value: 11,
}


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


def list_compatible_checkpoints(
    checkpoint_root: str,
    expected_obs_dim: int = OBS_DIM,
) -> list[tuple[int, float, str]]:
    compatible: list[tuple[int, float, str]] = []
    for ts, mtime, checkpoint_dir in _list_checkpoint_candidates(checkpoint_root):
        obs_dim = _checkpoint_obs_dim(checkpoint_dir)
        if obs_dim is None or obs_dim == int(expected_obs_dim):
            compatible.append((ts, mtime, checkpoint_dir))
    return compatible


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


def _checkpoint_stage_rank(checkpoint_dir: str) -> int:
    book = load_checkpoint_book(checkpoint_dir)
    curriculum_state = book.get("curriculum_state", {})
    if isinstance(curriculum_state, dict):
        stage = curriculum_state.get("stage")
        if isinstance(stage, str):
            return _STAGE_ORDER.get(stage, -1)
    return -1


def find_latest_compatible_checkpoint(
    checkpoint_root: str, expected_obs_dim: int = OBS_DIM
) -> str:
    compatible: list[tuple[int, int, float, str]] = []
    for ts, mtime, checkpoint_dir in list_compatible_checkpoints(
        checkpoint_root, expected_obs_dim
    ):
        compatible.append(
            (_checkpoint_stage_rank(checkpoint_dir), ts, mtime, checkpoint_dir)
        )
    if not compatible:
        return ""
    compatible.sort(key=lambda item: (item[0], item[1], item[2]))
    return compatible[-1][3]


def is_checkpoint_compatible(
    checkpoint_dir: str, expected_obs_dim: int = OBS_DIM
) -> bool:
    obs_dim = _checkpoint_obs_dim(checkpoint_dir)
    return obs_dim is None or obs_dim == int(expected_obs_dim)


def find_opponent_checkpoint(
    checkpoint_root: str,
    current_ts: int,
    gap_ts: int,
    expected_obs_dim: int = OBS_DIM,
    exclude_checkpoint_dir: str = "",
) -> str:
    compatible = list_compatible_checkpoints(checkpoint_root, expected_obs_dim)
    if not compatible:
        return ""

    exclude_checkpoint_dir = (
        str(Path(exclude_checkpoint_dir)) if exclude_checkpoint_dir else ""
    )
    target_ts = int(current_ts) - int(gap_ts)

    eligible = [
        (ts, mtime, checkpoint_dir)
        for ts, mtime, checkpoint_dir in compatible
        if checkpoint_dir != exclude_checkpoint_dir and ts <= target_ts
    ]
    if eligible:
        eligible.sort(key=lambda item: (item[0], item[1]))
        return eligible[-1][2]

    fallback = [
        (ts, mtime, checkpoint_dir)
        for ts, mtime, checkpoint_dir in compatible
        if checkpoint_dir != exclude_checkpoint_dir and ts < int(current_ts)
    ]
    if fallback:
        fallback.sort(key=lambda item: (item[0], item[1]))
        return fallback[-1][2]
    return ""


def sample_opponent_checkpoint(
    checkpoint_root: str,
    current_ts: int,
    target_gap_ts: int,
    expected_obs_dim: int = OBS_DIM,
    exclude_checkpoint_dir: str = "",
    band_width_ts: int = 1_500_000,
    prefer_newest: bool = True,
) -> str:
    compatible = list_compatible_checkpoints(checkpoint_root, expected_obs_dim)
    exclude_checkpoint_dir = (
        str(Path(exclude_checkpoint_dir)) if exclude_checkpoint_dir else ""
    )
    compatible = [
        (ts, mtime, checkpoint_dir)
        for ts, mtime, checkpoint_dir in compatible
        if checkpoint_dir != exclude_checkpoint_dir and ts < int(current_ts)
    ]
    if not compatible:
        return ""

    target_ts = max(0, int(current_ts) - int(target_gap_ts))
    band = max(1, int(band_width_ts))
    in_band = [
        (ts, mtime, checkpoint_dir)
        for ts, mtime, checkpoint_dir in compatible
        if abs(ts - target_ts) <= band
    ]
    candidates = in_band if in_band else compatible
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[-1][2] if prefer_newest else candidates[0][2]


def select_eval_anchor_checkpoints(
    checkpoint_root: str,
    current_checkpoint_dir: str,
    count: int = 5,
    span_ts: int = 10_000_000,
    expected_obs_dim: int = OBS_DIM,
) -> list[dict[str, Any]]:
    current_book = load_checkpoint_book(current_checkpoint_dir)
    current_ts = int(current_book.get("cumulative_timesteps", 0))
    if current_ts <= 0 or count <= 0:
        return []

    compatible = list_compatible_checkpoints(checkpoint_root, expected_obs_dim)
    compatible = [
        (ts, mtime, checkpoint_dir)
        for ts, mtime, checkpoint_dir in compatible
        if checkpoint_dir != str(Path(current_checkpoint_dir)) and ts < current_ts
    ]
    if not compatible:
        return []

    compatible.sort(key=lambda item: (item[0], item[1]))
    step = max(1, int(span_ts) // int(count))
    anchors: list[dict[str, Any]] = []
    used_dirs: set[str] = set()

    for slot in range(1, int(count) + 1):
        target_ts = max(0, current_ts - (slot * step))
        chosen_idx = -1
        for idx in range(len(compatible) - 1, -1, -1):
            ts, _, checkpoint_dir = compatible[idx]
            if checkpoint_dir in used_dirs:
                continue
            if ts <= target_ts:
                chosen_idx = idx
                break
        if chosen_idx == -1:
            for idx in range(len(compatible) - 1, -1, -1):
                _, _, checkpoint_dir = compatible[idx]
                if checkpoint_dir not in used_dirs:
                    chosen_idx = idx
                    break
        if chosen_idx == -1:
            break

        ts, _, checkpoint_dir = compatible[chosen_idx]
        used_dirs.add(checkpoint_dir)
        anchors.append(
            {
                "slot": slot,
                "target_timesteps": int(target_ts),
                "checkpoint_dir": checkpoint_dir,
                "cumulative_timesteps": int(ts),
            }
        )

    anchors.sort(key=lambda item: item["slot"])
    return anchors


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
