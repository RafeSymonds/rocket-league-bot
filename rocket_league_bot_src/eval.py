from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from rlgym.api import RLGym
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.sim import RocketSimEngine

from .checkpoints import (
    find_latest_compatible_checkpoint,
    load_checkpoint_book,
    select_eval_anchor_checkpoints,
)
from .conditions import CurriculumDoneCondition, CurriculumTruncationCondition
from .config import ACTION_REPEAT, DEFAULT_CHECKPOINT_ROOT, Stage
from .curriculum import CurriculumManager
from .mutators import DynamicMatchMutator
from .obs import SharedObs
from .opponent import FrozenOpponentPolicy
from .rewards import CurriculumReward

try:
    from rlgym_tools.rocket_league.shared_info_providers.scoreboard_provider import (
        ScoreboardInfo,
        ScoreboardProvider,
    )
except Exception:  # pragma: no cover - optional dependency until installed
    ScoreboardInfo = None
    ScoreboardProvider = None


def _eval_dir(root: str = "data/eval") -> Path:
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temp_path.replace(path)


def resolve_eval_ladder(
    checkpoint_root: str = DEFAULT_CHECKPOINT_ROOT,
    current_checkpoint_dir: str = "",
    count: int = 5,
    span_ts: int = 10_000_000,
    state_path: str = "data/eval/ladder_state.json",
    force_refresh: bool = False,
) -> dict[str, Any]:
    current_checkpoint_dir = current_checkpoint_dir or find_latest_compatible_checkpoint(checkpoint_root)
    if not current_checkpoint_dir:
        raise RuntimeError("No compatible checkpoint found for evaluation")

    current_book = load_checkpoint_book(current_checkpoint_dir)
    current_ts = int(current_book.get("cumulative_timesteps", 0))
    state_file = Path(state_path)
    state = _load_json(state_file)

    anchors = state.get("anchors", [])
    refresh_after_ts = int(state.get("refresh_after_timesteps", 0))
    needs_refresh = force_refresh or not anchors
    if state.get("span_ts") != int(span_ts) or state.get("count") != int(count):
        needs_refresh = True
    if current_ts >= refresh_after_ts:
        needs_refresh = True
    if any(not Path(str(anchor.get("checkpoint_dir", ""))).exists() for anchor in anchors):
        needs_refresh = True

    if needs_refresh:
        anchors = select_eval_anchor_checkpoints(
            checkpoint_root=checkpoint_root,
            current_checkpoint_dir=current_checkpoint_dir,
            count=count,
            span_ts=span_ts,
        )
        state = {
            "reference_checkpoint_dir": str(current_checkpoint_dir),
            "reference_timesteps": int(current_ts),
            "refresh_after_timesteps": int(current_ts + int(span_ts)),
            "count": int(count),
            "span_ts": int(span_ts),
            "anchors": anchors,
            "updated_unix_time": time.time(),
        }
        _write_json(state_file, state)
    return state


def _identify_agents(env: RLGym, obs_dict: dict[Any, np.ndarray]) -> tuple[Any, Any]:
    blue_agent = None
    orange_agent = None
    for agent_id in obs_dict:
        car = env.state.cars[agent_id]
        if car.is_orange and orange_agent is None:
            orange_agent = agent_id
        elif not car.is_orange and blue_agent is None:
            blue_agent = agent_id
    if blue_agent is None or orange_agent is None:
        raise RuntimeError("Evaluation requires one blue and one orange agent")
    return blue_agent, orange_agent


def _make_eval_env(stage: str, difficulty: float) -> RLGym:
    curriculum_manager = CurriculumManager()
    curriculum_manager.load_dict(
        {
            "stage": str(stage).upper(),
            "difficulty": float(difficulty),
            "stage_iterations": 0,
            "ema_touch_rate": 0.0,
            "ema_goal_rate": 0.0,
        }
    )
    action_parser = RepeatAction(LookupTableAction(), repeats=ACTION_REPEAT)
    return RLGym(
        state_mutator=DynamicMatchMutator(curriculum_manager),
        obs_builder=SharedObs(),
        action_parser=action_parser,
        reward_fn=CurriculumReward(curriculum_manager),
        termination_cond=CurriculumDoneCondition(curriculum_manager),
        truncation_cond=CurriculumTruncationCondition(curriculum_manager),
        transition_engine=RocketSimEngine(),
        **(
            {"shared_info_provider": ScoreboardProvider()}
            if ScoreboardProvider is not None
            else {}
        ),
    )


def evaluate_checkpoint_matchup(
    current_checkpoint_dir: str,
    opponent_checkpoint_dir: str,
    episodes: int = 6,
    stage: str = Stage.SELF_PLAY.value,
    difficulty: float = 0.5,
    device: str = "gpu",
) -> dict[str, Any]:
    env = _make_eval_env(stage=stage, difficulty=float(difficulty))
    current_policy = FrozenOpponentPolicy(device=device, deterministic=True)
    opponent_policy = FrozenOpponentPolicy(device=device, deterministic=True)
    current_policy.load(current_checkpoint_dir)
    opponent_policy.load(opponent_checkpoint_dir)

    obs_dict = env.reset()
    blue_agent, orange_agent = _identify_agents(env, obs_dict)
    episode_count = 0
    blue_goals = 0
    orange_goals = 0
    blue_wins = 0
    orange_wins = 0
    draws = 0
    blue_returns: list[float] = []
    orange_returns: list[float] = []
    blue_episode_return = 0.0
    orange_episode_return = 0.0
    steps_total = 0
    episode_goal_steps: list[int] = []
    episode_step = 0

    while episode_count < int(episodes):
        prev_blue_agent = blue_agent
        prev_orange_agent = orange_agent
        blue_action = current_policy.act(np.asarray(obs_dict[blue_agent], dtype=np.float32))
        orange_action = opponent_policy.act(np.asarray(obs_dict[orange_agent], dtype=np.float32))
        obs_dict, reward_dict, terminated_dict, truncated_dict = env.step(
            {
                blue_agent: np.array([blue_action], dtype=np.int32),
                orange_agent: np.array([orange_action], dtype=np.int32),
            }
        )
        blue_episode_return += float(reward_dict.get(prev_blue_agent, 0.0))
        orange_episode_return += float(reward_dict.get(prev_orange_agent, 0.0))
        steps_total += 1
        episode_step += 1

        done = bool(
            terminated_dict.get(prev_blue_agent, False) or truncated_dict.get(prev_blue_agent, False)
        )
        if not done:
            blue_agent, orange_agent = _identify_agents(env, obs_dict)
            continue

        episode_count += 1
        blue_returns.append(blue_episode_return)
        orange_returns.append(orange_episode_return)
        scoreboard = env.shared_info.get("scoreboard")
        if isinstance(scoreboard, ScoreboardInfo) and str(stage).upper() == Stage.SELF_PLAY.value:
            blue_score = int(scoreboard.blue_score)
            orange_score = int(scoreboard.orange_score)
            blue_goals += blue_score
            orange_goals += orange_score
            if blue_score > orange_score:
                blue_wins += 1
            elif orange_score > blue_score:
                orange_wins += 1
            else:
                draws += 1
        elif env.state.goal_scored:
            episode_goal_steps.append(episode_step)
            if int(env.state.scoring_team) == 0:
                blue_goals += 1
                blue_wins += 1
            else:
                orange_goals += 1
                orange_wins += 1
        else:
            draws += 1

        blue_episode_return = 0.0
        orange_episode_return = 0.0
        episode_step = 0
        if episode_count < int(episodes):
            obs_dict = env.reset()
        blue_agent, orange_agent = _identify_agents(env, obs_dict)

    env.close()
    total_episodes = max(1, int(episode_count))
    return {
        "episodes": int(episode_count),
        "stage": str(stage).upper(),
        "difficulty": float(difficulty),
        "blue_goals": int(blue_goals),
        "orange_goals": int(orange_goals),
        "blue_wins": int(blue_wins),
        "orange_wins": int(orange_wins),
        "draws": int(draws),
        "blue_win_rate": float(blue_wins / total_episodes),
        "goal_diff_per_episode": float((blue_goals - orange_goals) / total_episodes),
        "avg_blue_return": float(np.mean(blue_returns) if blue_returns else 0.0),
        "avg_orange_return": float(np.mean(orange_returns) if orange_returns else 0.0),
        "avg_steps_per_episode": float(steps_total / total_episodes),
        "median_goal_steps": float(np.median(episode_goal_steps) if episode_goal_steps else -1.0),
    }


def append_eval_rows(
    rows: list[dict[str, Any]],
    csv_path: str = "data/eval/results.csv",
) -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "unix_time",
        "current_checkpoint_dir",
        "current_timesteps",
        "stage",
        "difficulty",
        "anchor_slot",
        "opponent_checkpoint_dir",
        "opponent_timesteps",
        "episodes",
        "blue_goals",
        "orange_goals",
        "blue_wins",
        "orange_wins",
        "draws",
        "blue_win_rate",
        "goal_diff_per_episode",
        "avg_blue_return",
        "avg_orange_return",
        "avg_steps_per_episode",
        "median_goal_steps",
    ]
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_eval_ladder(
    checkpoint_root: str = DEFAULT_CHECKPOINT_ROOT,
    current_checkpoint_dir: str = "",
    episodes: int = 6,
    stage: str = "auto",
    difficulty: float = -1.0,
    device: str = "gpu",
    anchor_count: int = 5,
    anchor_span_ts: int = 10_000_000,
    force_refresh: bool = False,
    state_path: str = "data/eval/ladder_state.json",
    csv_path: str = "data/eval/results.csv",
    latest_summary_path: str = "data/eval/latest_summary.json",
) -> dict[str, Any]:
    ladder = resolve_eval_ladder(
        checkpoint_root=checkpoint_root,
        current_checkpoint_dir=current_checkpoint_dir,
        count=anchor_count,
        span_ts=anchor_span_ts,
        state_path=state_path,
        force_refresh=force_refresh,
    )
    current_checkpoint_dir = str(ladder.get("reference_checkpoint_dir", current_checkpoint_dir))
    current_book = load_checkpoint_book(current_checkpoint_dir)
    current_ts = int(current_book.get("cumulative_timesteps", 0))
    current_curriculum = current_book.get("curriculum_state", {})
    if str(stage).lower() == "auto":
        saved_stage = current_curriculum.get("stage")
        stage = str(saved_stage or Stage.SELF_PLAY.value).upper()
    if float(difficulty) < 0.0:
        saved_difficulty = current_curriculum.get("difficulty")
        difficulty = float(saved_difficulty if saved_difficulty is not None else 0.5)
    now = time.time()

    results: list[dict[str, Any]] = []
    for anchor in ladder.get("anchors", []):
        opponent_checkpoint_dir = str(anchor.get("checkpoint_dir", ""))
        if not opponent_checkpoint_dir:
            continue
        matchup = evaluate_checkpoint_matchup(
            current_checkpoint_dir=current_checkpoint_dir,
            opponent_checkpoint_dir=opponent_checkpoint_dir,
            episodes=episodes,
            stage=stage,
            difficulty=float(difficulty),
            device=device,
        )
        row = {
            "unix_time": f"{now:.3f}",
            "current_checkpoint_dir": current_checkpoint_dir,
            "current_timesteps": int(current_ts),
            "stage": str(stage).upper(),
            "difficulty": float(difficulty),
            "anchor_slot": int(anchor.get("slot", 0)),
            "opponent_checkpoint_dir": opponent_checkpoint_dir,
            "opponent_timesteps": int(anchor.get("cumulative_timesteps", 0)),
            **matchup,
        }
        results.append(row)

    append_eval_rows(results, csv_path=csv_path)
    summary = {
        "unix_time": now,
        "current_checkpoint_dir": current_checkpoint_dir,
        "current_timesteps": int(current_ts),
        "stage": str(stage).upper(),
        "difficulty": float(difficulty),
        "anchor_count": int(len(results)),
        "anchor_span_ts": int(anchor_span_ts),
        "reference_timesteps": int(ladder.get("reference_timesteps", current_ts)),
        "refresh_after_timesteps": int(ladder.get("refresh_after_timesteps", current_ts + anchor_span_ts)),
        "avg_blue_win_rate": float(np.mean([row["blue_win_rate"] for row in results]) if results else 0.0),
        "avg_goal_diff_per_episode": float(
            np.mean([row["goal_diff_per_episode"] for row in results]) if results else 0.0
        ),
        "rows": results,
    }
    _write_json(Path(latest_summary_path), summary)
    return summary


def maybe_refresh_eval_summary(
    latest_checkpoint: str,
    episodes: int = 6,
    device: str = "gpu",
    disabled: bool = False,
    latest_summary_path: str = "data/eval/latest_summary.json",
) -> tuple[dict[str, Any] | None, str, bool]:
    if disabled or not latest_checkpoint:
        return None, "", False

    summary_path = Path(latest_summary_path)
    summary = _load_json(summary_path)
    latest_book = load_checkpoint_book(latest_checkpoint)
    latest_ts = int(latest_book.get("cumulative_timesteps", 0))
    refresh_after_ts = int(summary.get("refresh_after_timesteps", 0) or 0)
    summary_checkpoint = str(summary.get("current_checkpoint_dir", ""))

    needs_refresh = not summary or summary_checkpoint != str(latest_checkpoint)
    if latest_ts >= refresh_after_ts > 0:
        needs_refresh = True

    if not needs_refresh:
        return summary, "", False

    try:
        summary = run_eval_ladder(
            current_checkpoint_dir=latest_checkpoint,
            episodes=max(1, int(episodes)),
            device=str(device),
            latest_summary_path=latest_summary_path,
        )
        return summary, "", True
    except Exception as exc:
        return None, str(exc), False
