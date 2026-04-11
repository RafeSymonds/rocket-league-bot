from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

from rlgym.api import RLGym
from rlgym.rocket_league import common_values
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym_ppo.util.rlgym_v2_gym_wrapper import RLGymV2GymWrapper

from .action_parser import NectoAction
from .conditions import CurriculumDoneCondition, CurriculumTruncationCondition
from .config import ACTION_REPEAT, USE_DISCRETE_ACTIONS
from .curriculum import CurriculumManager
from .league import SnapshotLeague
from .obs import SharedObs
from .opponent import SelfPlayOpponentGymWrapper
from .reporting import write_training_report
from .rewards import CurriculumReward
from .utils import Stats
from .checkpoints import (
    find_opponent_checkpoint,
    load_checkpoint_book,
    sample_opponent_checkpoint,
)

try:
    from .mutators_with_replay import DynamicMatchMutatorWithReplay
except ImportError:
    DynamicMatchMutatorWithReplay = None

try:
    from .mutators import DynamicMatchMutator
except ImportError:
    DynamicMatchMutator = None

try:
    from rlgym_tools.rocket_league.shared_info_providers.scoreboard_provider import (
        ScoreboardProvider,
    )
except Exception:  # pragma: no cover - optional dependency until installed
    ScoreboardProvider = None


class ProcessIterationLogger:
    _METRICS_COLUMNS = [
        "unix_time",
        "stage",
        "difficulty",
        "sps",
        "episodes",
        "avg_return",
        "orange_avg_return",
        "total_avg_return",
        "touch_rate",
        "blue_touch_rate",
        "orange_touch_rate",
        "goal_rate",
        "blue_goal_rate",
        "orange_goal_rate",
        "median_t_first",
        "blue_median_t_first",
        "orange_median_t_first",
        "median_t_goal",
        "blue_median_t_goal",
        "orange_median_t_goal",
        "aerial_touch_rate",
        "goal_side_rate",
        "behind_ball_rate",
        "ema_touch",
        "ema_goal",
    ]

    def __init__(
        self,
        env,
        process_id: int,
        iteration_timesteps: int,
        curriculum_manager: CurriculumManager,
        checkpoint_root: str,
        curriculum_state_path: str,
        opponent_state_path: str,
        self_play_mode: str,
        opponent_gap_ts: int,
        current_checkpoint_dir: str,
        fixed_opponent_checkpoint: str,
    ):
        self.env = env
        self.pid = process_id
        self.iteration_ts = iteration_timesteps
        self.cm = curriculum_manager
        self.checkpoint_root = checkpoint_root
        self.curriculum_state_path = curriculum_state_path
        self.opponent_state_path = opponent_state_path
        self.self_play_mode = str(self_play_mode)
        self.opponent_gap_ts = int(opponent_gap_ts)
        self.current_checkpoint_dir = current_checkpoint_dir
        self.fixed_opponent_checkpoint = fixed_opponent_checkpoint
        self.league = SnapshotLeague()
        self._last_exported_checkpoint = ""
        self.log_counter = 0

        self._env_obs_space = env.observation_space
        self._env_act_space = env.action_space
        self._cached_action_space: Optional[object] = None
        self._cached_obs_space: Optional[object] = None
        self._spaces_resolved = False

        self._reset_iteration_stats()
        self._reset_episode_stats()
        self._init_metrics_file()
        self._sync_curriculum_state()
        self._write_opponent_state()

    def _init_metrics_file(self):
        self._metrics_path = None
        if self.pid != 0:
            return
        os.makedirs("data", exist_ok=True)
        self._metrics_path = os.path.join("data", "training_metrics.csv")
        if os.path.exists(self._metrics_path):
            self._migrate_metrics_file_if_needed()
            return
        with open(self._metrics_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(self._METRICS_COLUMNS)

    def _migrate_metrics_file_if_needed(self) -> None:
        if self._metrics_path is None:
            return
        try:
            with open(self._metrics_path, "r", newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
        except Exception:
            return
        if not rows:
            return

        header = rows[0]
        if header == self._METRICS_COLUMNS:
            return

        migrated_rows: list[list[str]] = [self._METRICS_COLUMNS]
        header_index = {name: idx for idx, name in enumerate(header)}
        for row in rows[1:]:
            if not row:
                continue
            migrated_rows.append(
                [
                    row[header_index[name]]
                    if name in header_index and header_index[name] < len(row)
                    else ""
                    for name in self._METRICS_COLUMNS
                ]
            )

        with open(self._metrics_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerows(migrated_rows)

    def _reset_iteration_stats(self):
        self.iteration_start_time = time.time()
        self.iteration_steps = 0
        self.iteration_episodes = 0
        self.iteration_return = 0.0
        self.iteration_orange_return = 0.0
        self.iteration_total_return = 0.0
        self.iteration_goals = 0
        self.iteration_blue_touch_eps = 0
        self.iteration_orange_touch_eps = 0
        self.iteration_blue_goals = 0
        self.iteration_orange_goals = 0
        self.iteration_success_eps = 0
        self.iteration_median_t_first = []
        self.iteration_blue_median_t_first = []
        self.iteration_orange_median_t_first = []
        self.iteration_median_t_goal = []
        self.iteration_blue_median_t_goal = []
        self.iteration_orange_median_t_goal = []
        self.iteration_blue_aerial_touch_eps = 0
        self.iteration_goal_side_rates = []
        self.iteration_behind_ball_rates = []

    def _reset_episode_stats(self):
        self.ep_return = 0.0
        self.ep_orange_return = 0.0
        self.ep_steps = 0
        self.ep_ball_touches = 0
        self.ep_blue_touches = 0
        self.ep_orange_touches = 0
        self.ep_first_touch_step = -1
        self.ep_blue_first_touch_step = -1
        self.ep_orange_first_touch_step = -1
        self.ep_goal_step = -1
        self.ep_blue_goal_step = -1
        self.ep_orange_goal_step = -1
        self.ep_blue_aerial_touch = False
        self.ep_blue_control_steps = 0
        self.ep_blue_goal_side_steps = 0
        self.ep_blue_behind_ball_steps = 0
        self._prev_touches = {}

    @staticmethod
    def _blue_car(state):
        for car in state.cars.values():
            if not car.is_orange:
                return car
        return None

    @staticmethod
    def _goal_side_value(car, ball_pos) -> float:
        own_goal = np.array([0.0, -common_values.BACK_NET_Y, 0.0], dtype=np.float32)
        lane = ball_pos - own_goal
        lane_norm_sq = float(np.dot(lane, lane))
        if lane_norm_sq < 1e-6:
            return 0.0
        goal_to_car = np.asarray(car.physics.position, dtype=np.float32) - own_goal
        proj = float(np.dot(goal_to_car, lane) / lane_norm_sq)
        if proj <= 0.02 or proj >= 1.05:
            return 0.0
        lateral = goal_to_car - proj * lane
        return float(np.clip(1.0 - (np.linalg.norm(lateral) / 2500.0), 0.0, 1.0))

    @staticmethod
    def _behind_ball_value(car, ball_pos) -> float:
        enemy_goal = np.array([0.0, common_values.BACK_NET_Y, 0.0], dtype=np.float32)
        ball_to_goal = enemy_goal - ball_pos
        norm = float(np.linalg.norm(ball_to_goal))
        if norm < 1e-6:
            return 0.0
        direction = ball_to_goal / norm
        ball_to_car = np.asarray(car.physics.position, dtype=np.float32) - ball_pos
        depth = float(np.dot(ball_to_car, direction))
        if depth >= -60.0:
            return 0.0
        lateral = ball_to_car - depth * direction
        depth_bonus = float(np.clip((-depth - 60.0) / 1200.0, 0.0, 1.0))
        return float(
            np.clip(depth_bonus * (1.0 - (np.linalg.norm(lateral) / 2600.0)), 0.0, 1.0)
        )

    def _append_metrics_row(
        self,
        stage: str,
        difficulty: float,
        sps: float,
        avg_return: float,
        orange_avg_return: float,
        total_avg_return: float,
        touch_rate: float,
        blue_touch_rate: float,
        orange_touch_rate: float,
        goal_rate: float,
        blue_goal_rate: float,
        orange_goal_rate: float,
        median_t_first: float,
        blue_median_t_first: float,
        orange_median_t_first: float,
        median_t_goal: float,
        blue_median_t_goal: float,
        orange_median_t_goal: float,
        aerial_touch_rate: float,
        goal_side_rate: float,
        behind_ball_rate: float,
    ) -> None:
        if self._metrics_path is None:
            return
        snap = self.cm.snapshot()
        with open(self._metrics_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    f"{time.time():.3f}",
                    stage,
                    f"{difficulty:.6f}",
                    f"{sps:.3f}",
                    int(self.iteration_episodes),
                    f"{avg_return:.6f}",
                    f"{orange_avg_return:.6f}",
                    f"{total_avg_return:.6f}",
                    f"{touch_rate:.6f}",
                    f"{blue_touch_rate:.6f}",
                    f"{orange_touch_rate:.6f}",
                    f"{goal_rate:.6f}",
                    f"{blue_goal_rate:.6f}",
                    f"{orange_goal_rate:.6f}",
                    f"{median_t_first:.3f}",
                    f"{blue_median_t_first:.3f}",
                    f"{orange_median_t_first:.3f}",
                    f"{median_t_goal:.3f}",
                    f"{blue_median_t_goal:.3f}",
                    f"{orange_median_t_goal:.3f}",
                    f"{aerial_touch_rate:.6f}",
                    f"{goal_side_rate:.6f}",
                    f"{behind_ball_rate:.6f}",
                    f"{snap.ema_touch:.6f}",
                    f"{snap.ema_goal:.6f}",
                ]
            )
        try:
            write_training_report(metrics_path=self._metrics_path)
        except Exception as exc:
            print(f"[report] failed to update training report: {exc}")

    def close(self, **kwargs):
        pass

    def reset(self, **kwargs):
        self._sync_curriculum_state()
        self._reset_episode_stats()
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}

        state = getattr(self.env, 'state', None)
        if state is not None:
            for agent, car in state.cars.items():
                self._prev_touches[agent] = int(car.ball_touches)
        self._resolve_spaces()
        return obs

    def _resolve_spaces(self):
        if self._spaces_resolved:
            return
        try:
            action_spaces = getattr(self.env, 'action_spaces', {}) or {}
            observation_spaces = getattr(self.env, 'observation_spaces', {}) or {}
            if action_spaces:
                first_act_info = list(action_spaces.values())[0]
                act_space = first_act_info[0] if isinstance(first_act_info, tuple) else first_act_info
                if hasattr(act_space, 'seed'):
                    self._cached_action_space = act_space
            if observation_spaces:
                first_obs_info = list(observation_spaces.values())[0]
                obs_space = first_obs_info[0] if isinstance(first_obs_info, tuple) else first_obs_info
                if hasattr(obs_space, 'shape'):
                    self._cached_obs_space = obs_space
            self._spaces_resolved = True
        except Exception:
            pass

    @property
    def action_space(self):
        if self._cached_action_space is not None:
            return self._cached_action_space
        return self._env_act_space

    @property
    def observation_space(self):
        if self._cached_obs_space is not None:
            return self._cached_obs_space
        return self._env_obs_space

    @property
    def action_spaces(self):
        return getattr(self.env, 'action_spaces', {})

    @property
    def observation_spaces(self):
        return getattr(self.env, 'observation_spaces', {})

    def step(self, action):
        if isinstance(action, np.ndarray):
            agents = getattr(self.env, 'agents', []) or []
            if agents and hasattr(self.env, 'action_space') and not hasattr(self.env.action_space, 'seed'):
                action_dict = {}
                for i, agent in enumerate(agents):
                    if i < action.shape[0] if len(action.shape) > 1 else 1:
                        action_dict[agent] = action[i] if len(action.shape) > 1 else action
                action = action_dict
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        else:
            obs, reward, done, info = result
            terminated = done
            truncated = False

        self.iteration_steps += 1
        self.ep_steps += 1
        state = getattr(self.env, 'state', None)
        blue_rewards: list[float] = []
        orange_rewards: list[float] = []
        if isinstance(reward, (list, tuple, np.ndarray)) and hasattr(
            self.env, "agent_map"
            ):
            for idx, rew in enumerate(reward):
                agent_id = self.env.agent_map.get(idx)
                if state is None:
                    continue
                car = state.cars.get(agent_id)
                if car is None:
                    continue
                if car.is_orange:
                    orange_rewards.append(float(rew))
                else:
                    blue_rewards.append(float(rew))
        elif isinstance(reward, dict):
            for agent_id, rew in reward.items():
                if state is None:
                    continue
                car = state.cars.get(agent_id)
                if car is None:
                    continue
                if car.is_orange:
                    orange_rewards.append(float(rew))
                else:
                    blue_rewards.append(float(rew))
        else:
            blue_rewards.append(float(np.mean(reward)))

        if blue_rewards:
            self.ep_return += float(np.mean(blue_rewards))
        if orange_rewards:
            self.ep_orange_return += float(np.mean(orange_rewards))

        if state is not None:
            blue_car = self._blue_car(state)
            if blue_car is not None:
                ball_pos = np.asarray(state.ball.position, dtype=np.float32)
                self.ep_blue_control_steps += 1
                self.ep_blue_goal_side_steps += int(
                    self._goal_side_value(blue_car, ball_pos) >= 0.55
                )
                self.ep_blue_behind_ball_steps += int(
                    self._behind_ball_value(blue_car, ball_pos) >= 0.35
                )
            for agent, car in state.cars.items():
                prev = self._prev_touches.get(agent, int(car.ball_touches))
                cur = int(car.ball_touches)
                self._prev_touches[agent] = cur
                if cur > prev:
                    self.ep_ball_touches += 1
                    if self.ep_first_touch_step == -1:
                        self.ep_first_touch_step = self.ep_steps
                    if car.is_orange:
                        self.ep_orange_touches += 1
                        if self.ep_orange_first_touch_step == -1:
                            self.ep_orange_first_touch_step = self.ep_steps
                    else:
                        self.ep_blue_touches += 1
                        if self.ep_blue_first_touch_step == -1:
                            self.ep_blue_first_touch_step = self.ep_steps
                        if (not car.on_ground) and float(
                            state.ball.position[2]
                        ) >= 150.0:
                            self.ep_blue_aerial_touch = True
            if state.goal_scored and self.ep_goal_step == -1:
                self.ep_goal_step = self.ep_steps
                if int(state.scoring_team) == 0:
                    self.ep_blue_goal_step = self.ep_steps
                elif int(state.scoring_team) == 1:
                    self.ep_orange_goal_step = self.ep_steps

        terminated_any = any(terminated.values()) if isinstance(terminated, dict) else bool(terminated)
        if terminated_any or truncated:
            self.iteration_episodes += 1
            self.iteration_return += self.ep_return
            self.iteration_orange_return += self.ep_orange_return
            self.iteration_total_return += self.ep_return + self.ep_orange_return

            if self.ep_ball_touches > 0:
                self.iteration_success_eps += 1
                self.iteration_median_t_first.append(self.ep_first_touch_step)
            if self.ep_blue_touches > 0:
                self.iteration_blue_touch_eps += 1
                self.iteration_blue_median_t_first.append(self.ep_blue_first_touch_step)
            if self.ep_blue_aerial_touch:
                self.iteration_blue_aerial_touch_eps += 1
            if self.ep_orange_touches > 0:
                self.iteration_orange_touch_eps += 1
                self.iteration_orange_median_t_first.append(
                    self.ep_orange_first_touch_step
                )
            if self.ep_blue_control_steps > 0:
                self.iteration_goal_side_rates.append(
                    self.ep_blue_goal_side_steps / self.ep_blue_control_steps
                )
                self.iteration_behind_ball_rates.append(
                    self.ep_blue_behind_ball_steps / self.ep_blue_control_steps
                )

            if self.ep_goal_step != -1:
                self.iteration_goals += 1
                self.iteration_median_t_goal.append(self.ep_goal_step)
                if state is not None and int(state.scoring_team) == 0:
                    self.iteration_blue_goals += 1
                    self.iteration_blue_median_t_goal.append(self.ep_blue_goal_step)
                elif state is not None and int(state.scoring_team) == 1:
                    self.iteration_orange_goals += 1
                    self.iteration_orange_median_t_goal.append(self.ep_orange_goal_step)

            self._reset_episode_stats()
            if state is not None:
                for agent, car in state.cars.items():
                    self._prev_touches[agent] = int(car.ball_touches)

        if self.iteration_steps >= self.iteration_ts:
            self._report_and_reset_iteration()

        return obs, reward, terminated, truncated, info

    def _report_and_reset_iteration(self):
        avg_return = (
            self.iteration_return / self.iteration_episodes
            if self.iteration_episodes > 0
            else 0.0
        )
        orange_avg_return = (
            self.iteration_orange_return / self.iteration_episodes
            if self.iteration_episodes > 0
            else 0.0
        )
        total_avg_return = (
            self.iteration_total_return / self.iteration_episodes
            if self.iteration_episodes > 0
            else 0.0
        )
        touch_rate = (
            self.iteration_success_eps / self.iteration_episodes
            if self.iteration_episodes > 0
            else 0.0
        )
        blue_touch_rate = (
            self.iteration_blue_touch_eps / self.iteration_episodes
            if self.iteration_episodes > 0
            else 0.0
        )
        orange_touch_rate = (
            self.iteration_orange_touch_eps / self.iteration_episodes
            if self.iteration_episodes > 0
            else 0.0
        )
        goal_rate = (
            self.iteration_goals / self.iteration_episodes
            if self.iteration_episodes > 0
            else 0.0
        )
        blue_goal_rate = (
            self.iteration_blue_goals / self.iteration_episodes
            if self.iteration_episodes > 0
            else 0.0
        )
        orange_goal_rate = (
            self.iteration_orange_goals / self.iteration_episodes
            if self.iteration_episodes > 0
            else 0.0
        )
        median_t_first = (
            np.median(self.iteration_median_t_first)
            if self.iteration_median_t_first
            else -1.0
        )
        blue_median_t_first = (
            np.median(self.iteration_blue_median_t_first)
            if self.iteration_blue_median_t_first
            else -1.0
        )
        orange_median_t_first = (
            np.median(self.iteration_orange_median_t_first)
            if self.iteration_orange_median_t_first
            else -1.0
        )
        median_t_goal = (
            np.median(self.iteration_median_t_goal)
            if self.iteration_median_t_goal
            else -1.0
        )
        blue_median_t_goal = (
            np.median(self.iteration_blue_median_t_goal)
            if self.iteration_blue_median_t_goal
            else -1.0
        )
        orange_median_t_goal = (
            np.median(self.iteration_orange_median_t_goal)
            if self.iteration_orange_median_t_goal
            else -1.0
        )
        aerial_touch_rate = (
            self.iteration_blue_aerial_touch_eps / self.iteration_episodes
            if self.iteration_episodes > 0
            else 0.0
        )
        goal_side_rate = (
            float(np.mean(self.iteration_goal_side_rates))
            if self.iteration_goal_side_rates
            else 0.0
        )
        behind_ball_rate = (
            float(np.mean(self.iteration_behind_ball_rates))
            if self.iteration_behind_ball_rates
            else 0.0
        )

        duration = max(time.time() - self.iteration_start_time, 1e-6)
        sps = self.iteration_steps / duration
        cfg = self.cm.current_config()
        snap = self.cm.snapshot()

        self._append_metrics_row(
            stage=cfg.stage.value,
            difficulty=snap.difficulty,
            sps=sps,
            avg_return=avg_return,
            orange_avg_return=orange_avg_return,
            total_avg_return=total_avg_return,
            touch_rate=touch_rate,
            blue_touch_rate=blue_touch_rate,
            orange_touch_rate=orange_touch_rate,
            goal_rate=goal_rate,
            blue_goal_rate=blue_goal_rate,
            orange_goal_rate=orange_goal_rate,
            median_t_first=float(median_t_first),
            blue_median_t_first=float(blue_median_t_first),
            orange_median_t_first=float(orange_median_t_first),
            median_t_goal=float(median_t_goal),
            blue_median_t_goal=float(blue_median_t_goal),
            orange_median_t_goal=float(orange_median_t_goal),
            aerial_touch_rate=aerial_touch_rate,
            goal_side_rate=goal_side_rate,
            behind_ball_rate=behind_ball_rate,
        )

        if self.pid == 0 and self.log_counter % 3 == 0:
            print(
                f"[P-{self.pid:02d} | {cfg.stage.value:<9}] "
                f"diff={snap.difficulty:0.2f} | "
                f"SPS={sps:7.1f} | "
                f"Eps={self.iteration_episodes:4d} | "
                f"AvgRet={avg_return:7.3f} | "
                f"OrangeRet={orange_avg_return:7.3f} | "
                f"Touch={touch_rate:0.2f} | "
                f"Goal={goal_rate:0.2f} | "
                f"Aerial={aerial_touch_rate:0.2f} | "
                f"GoalSide={goal_side_rate:0.2f} | "
                f"BlueGoal={blue_goal_rate:0.2f} | "
                f"OrangeGoal={orange_goal_rate:0.2f}"
            )
        self.log_counter += 1

        stats = Stats(
            touch_rate=touch_rate,
            goal_rate=goal_rate,
            blue_goal_rate=blue_goal_rate,
            orange_goal_rate=orange_goal_rate,
            median_t_first=float(median_t_first if median_t_first != -1 else 9999.0),
            median_t_goal=float(median_t_goal if median_t_goal != -1 else 9999.0),
            aerial_touch_rate=aerial_touch_rate,
            goal_side_rate=goal_side_rate,
            behind_ball_rate=behind_ball_rate,
        )
        if self.pid == 0:
            self.cm.maybe_advance(stats)
            self._write_curriculum_state()
            self._write_opponent_state(
                blue_goal_rate=blue_goal_rate,
                goal_rate=goal_rate,
            )
        else:
            self._sync_curriculum_state()
        self._maybe_register_league_snapshot()
        self._maybe_auto_export_latest_checkpoint()
        self._reset_iteration_stats()

    def _sync_curriculum_state(self) -> None:
        path = Path(self.curriculum_state_path)
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text())
        except Exception:
            return
        if isinstance(payload, dict):
            self.cm.load_dict(payload)

    def _write_curriculum_state(self) -> None:
        if self.pid != 0:
            return
        path = Path(self.curriculum_state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.cm.to_dict(), indent=2, sort_keys=True))

    def _persist_curriculum_state_to_checkpoint(self, checkpoint_dir: Path) -> None:
        book = checkpoint_dir / "BOOK_KEEPING_VARS.json"
        if not book.exists():
            return
        try:
            data = json.loads(book.read_text())
        except Exception:
            return
        data["curriculum_state"] = self.cm.to_dict()
        book.write_text(json.dumps(data, indent=4))

    def _maybe_register_league_snapshot(self) -> None:
        if self.pid != 0:
            return

        latest = self._find_latest_checkpoint()
        if latest is None:
            return

        book = latest / "BOOK_KEEPING_VARS.json"
        if not book.exists():
            return
        try:
            data = json.loads(book.read_text())
        except Exception:
            return

        ts = int(data.get("cumulative_timesteps", 0))
        if ts <= 0 or ts % 10_000_000 != 0:
            return

        cfg = self.cm.current_config()
        self.league.register_snapshot(
            checkpoint_dir=str(latest),
            cumulative_timesteps=ts,
            stage=cfg.stage.value,
            difficulty=self.cm.snapshot().difficulty,
        )

    def _maybe_auto_export_latest_checkpoint(self) -> None:
        if self.pid != 0:
            return
        latest = self._find_latest_checkpoint()
        if latest is None:
            return
        self.current_checkpoint_dir = str(latest)
        self._write_opponent_state()
        self._persist_curriculum_state_to_checkpoint(latest)
        latest_str = str(latest)
        if latest_str == self._last_exported_checkpoint:
            return
        self._last_exported_checkpoint = latest_str

        try:
            from .export import export_checkpoint_to_rlbot_package

            export_checkpoint_to_rlbot_package(latest_str)
        except Exception as exc:
            print(f"[export] failed to export latest checkpoint: {exc}")

    def _find_latest_checkpoint(self):
        root = Path(self.checkpoint_root)
        if not root.exists():
            return None

        candidates: list[tuple[int, float, Path]] = []
        for book in root.rglob("BOOK_KEEPING_VARS.json"):
            try:
                data = json.loads(book.read_text())
                ts = int(data.get("cumulative_timesteps", 0))
            except Exception:
                ts = 0
            candidates.append((ts, book.stat().st_mtime, book.parent))

        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates[-1][2]

    def _write_opponent_state(
        self,
        blue_goal_rate: float | None = None,
        goal_rate: float | None = None,
    ) -> None:
        if self.pid != 0:
            return
        path = Path(self.opponent_state_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.self_play_mode != "frozen":
            payload = {
                "enabled": False,
                "mode": self.self_play_mode,
                "checkpoint_dir": "",
                "gap_ts": 0,
                "base_gap_ts": int(self.opponent_gap_ts),
                "blue_goal_rate": None
                if blue_goal_rate is None
                else float(blue_goal_rate),
            }
            path.write_text(json.dumps(payload, indent=2, sort_keys=True))
            return

        checkpoint_dir = ""
        effective_gap_ts = int(self.opponent_gap_ts)
        if (
            blue_goal_rate is not None
            and goal_rate is not None
            and self.cm.current_config().full_match
        ):
            if blue_goal_rate >= 0.90 and goal_rate >= 0.90:
                effective_gap_ts = max(500_000, self.opponent_gap_ts // 4)
            elif blue_goal_rate >= 0.75 and goal_rate >= 0.80:
                effective_gap_ts = max(1_000_000, self.opponent_gap_ts // 2)
            elif blue_goal_rate <= 0.30 and goal_rate >= 0.70:
                effective_gap_ts = min(
                    self.opponent_gap_ts * 2, self.opponent_gap_ts + 4_000_000
                )
            elif blue_goal_rate <= 0.40:
                effective_gap_ts = min(self.opponent_gap_ts + 2_000_000, 8_000_000)

        if self.fixed_opponent_checkpoint:
            checkpoint_dir = self.fixed_opponent_checkpoint
        elif self.current_checkpoint_dir:
            book = load_checkpoint_book(self.current_checkpoint_dir)
            current_ts = int(book.get("cumulative_timesteps", 0))
            if blue_goal_rate is not None and goal_rate is not None:
                if blue_goal_rate >= 0.75 and goal_rate >= 0.80:
                    checkpoint_dir = sample_opponent_checkpoint(
                        self.checkpoint_root,
                        current_ts=current_ts,
                        target_gap_ts=effective_gap_ts,
                        exclude_checkpoint_dir=self.current_checkpoint_dir,
                        band_width_ts=1_000_000,
                        prefer_newest=True,
                    )
                elif blue_goal_rate <= 0.40:
                    checkpoint_dir = sample_opponent_checkpoint(
                        self.checkpoint_root,
                        current_ts=current_ts,
                        target_gap_ts=effective_gap_ts,
                        exclude_checkpoint_dir=self.current_checkpoint_dir,
                        band_width_ts=2_000_000,
                        prefer_newest=False,
                    )
                else:
                    checkpoint_dir = find_opponent_checkpoint(
                        self.checkpoint_root,
                        current_ts=current_ts,
                        gap_ts=effective_gap_ts,
                        exclude_checkpoint_dir=self.current_checkpoint_dir,
                    )
            else:
                checkpoint_dir = find_opponent_checkpoint(
                    self.checkpoint_root,
                    current_ts=current_ts,
                    gap_ts=effective_gap_ts,
                    exclude_checkpoint_dir=self.current_checkpoint_dir,
                )

        payload = {
            "enabled": bool(checkpoint_dir),
            "mode": self.self_play_mode,
            "checkpoint_dir": str(checkpoint_dir),
            "gap_ts": int(effective_gap_ts),
            "base_gap_ts": int(self.opponent_gap_ts),
            "blue_goal_rate": None if blue_goal_rate is None else float(blue_goal_rate),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))


class EnvBuilder:
    def __init__(
        self,
        iteration_timesteps: int,
        checkpoint_root: str = "data/checkpoints",
        n_proc: int = 1,
        initial_curriculum_state: dict[str, object] | None = None,
        current_checkpoint_dir: str = "",
        self_play_mode: str = "current",
        fixed_opponent_checkpoint: str = "",
        opponent_gap_ts: int = 4_000_000,
        opponent_device: str = "gpu",
        replay_folder: Optional[str] = None,
        use_discrete_actions: bool = USE_DISCRETE_ACTIONS,
    ):
        self.iteration_timesteps = iteration_timesteps
        self.checkpoint_root = checkpoint_root
        self.n_proc = max(1, int(n_proc))
        self.curriculum_manager = CurriculumManager()
        self.curriculum_manager.load_dict(initial_curriculum_state)
        self.curriculum_state_path = str(Path("data") / "curriculum_state.json")
        self.opponent_state_path = str(Path("data") / "opponent_state.json")
        self.current_checkpoint_dir = current_checkpoint_dir
        self.self_play_mode = str(self_play_mode)
        self.fixed_opponent_checkpoint = fixed_opponent_checkpoint
        self.opponent_gap_ts = int(opponent_gap_ts)
        self.opponent_device = opponent_device
        self.replay_folder = replay_folder
        self.use_discrete_actions = use_discrete_actions
        Path(self.curriculum_state_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.curriculum_state_path).write_text(
            json.dumps(self.curriculum_manager.to_dict(), indent=2, sort_keys=True)
        )

    def __call__(self, process_id: int | None = None):
        if process_id is None:
            process = mp.current_process()
            process_id = int(process._identity[0] - 1) if process._identity else 0

        curriculum_manager = self.curriculum_manager

        if self.use_discrete_actions:
            action_parser = NectoAction()
        else:
            action_parser = RepeatAction(LookupTableAction(), repeats=ACTION_REPEAT)

        if self.replay_folder and DynamicMatchMutatorWithReplay is not None:
            state_mutator = DynamicMatchMutatorWithReplay(
                curriculum_manager=curriculum_manager,
                replay_folder=self.replay_folder,
                replay_reset_probability=0.7,
                use_lazy_loading=True,
            )
        elif DynamicMatchMutator is not None:
            state_mutator = DynamicMatchMutator(curriculum_manager)
        else:
            raise RuntimeError("No state mutator available. Install rlgym-tools.")

        env = RLGym(
            state_mutator=state_mutator,
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

        if self.self_play_mode == "frozen":
            gym_env = SelfPlayOpponentGymWrapper(
                env,
                opponent_state_path=self.opponent_state_path,
                deterministic_opponent=False,
                device=self.opponent_device,
            )
        else:
            gym_env = env

        wrapped = ProcessIterationLogger(
            gym_env,
            process_id=process_id,
            iteration_timesteps=max(1, self.iteration_timesteps // self.n_proc),
            curriculum_manager=curriculum_manager,
            checkpoint_root=self.checkpoint_root,
            curriculum_state_path=self.curriculum_state_path,
            opponent_state_path=self.opponent_state_path,
            self_play_mode=self.self_play_mode,
            opponent_gap_ts=self.opponent_gap_ts,
            current_checkpoint_dir=self.current_checkpoint_dir,
            fixed_opponent_checkpoint=self.fixed_opponent_checkpoint,
        )
        return wrapped
