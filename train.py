from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from typing_extensions import override

import numpy as np
from gymnasium import spaces
from torch.utils.tensorboard.writer import SummaryWriter

from rlgym.api import (
    AgentID,
    DoneCondition,
    RLGym,
    RewardFunction,
    ObsBuilder,
    StateMutator,
)
from rlgym.api.typing import AgentID as AgentIDType
from rlgym.rocket_league import common_values
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.done_conditions import (
    AnyCondition,
    GoalCondition,
    NoTouchTimeoutCondition,
    TimeoutCondition,
)
from rlgym.rocket_league.reward_functions import TouchReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import (
    FixedTeamSizeMutator,
    KickoffMutator,
    MutatorSequence,
)
from rlgym_ppo import Learner
from rlgym_ppo.util import RLGymV2GymWrapper


# ==================================================
# Small utilities
# ==================================================


class TBLogger:
    def __init__(self, log_dir: str = "runs/rlgym") -> None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.w = SummaryWriter(log_dir=log_dir)

    def scalar(self, tag: str, value: float, step: int) -> None:
        self.w.add_scalar(tag, float(value), step)

    def text(self, tag: str, text: str, step: int) -> None:
        self.w.add_text(tag, text, step)

    def flush(self) -> None:
        self.w.flush()


class CurriculumValue:
    """A mutable scalar shared between curriculum + mutators (no restart needed)."""

    def __init__(self, value: float):
        self.value = float(value)

    def get(self) -> float:
        return float(self.value)

    def set(self, value: float) -> None:
        self.value = float(value)


class Stage(str, Enum):
    TOUCH = "touch"
    SCORE = "score"
    SELFPLAY = "selfplay"


@dataclass
class StageConfig:
    stage: Stage

    # teams
    blue_players: int
    orange_players: int

    # termination / truncation
    end_on_touch: bool
    end_on_goal: bool
    no_touch_timeout_s: int
    timeout_s: int

    # reward weights (single place to tune)
    # --- terminal-ish ---
    w_goal: float
    w_fast_goal: float

    # --- score-safe dense / touch-gated ---
    w_ball_vel_to_goal: float
    w_ball_dist_to_goal: float
    w_shot_commit: float
    w_align: float
    w_hard_hit: float

    # --- touch-stage shaping ---
    w_touch: float
    w_power: float
    w_approach: float

    # --- universal hygiene ---
    w_step_penalty: float
    w_notouch_pressure: float
    w_camp_penalty: float


# ==================================================
# Episode logging + curriculum handshake
# ==================================================


class EpisodeLogger:
    """
    Wraps the gym env and logs per-episode stats + pushes summary stats to curriculum.
    """

    def __init__(self, env, logger: TBLogger, print_every: int = 10):
        self.env = env
        self.logger = logger
        self.print_every = print_every

        self.global_ts = 0
        self.episode_idx = 0

        self.curriculum = None
        self.stage: Stage = Stage.TOUCH

        self.last_episodes: list[dict[str, Any]] = []
        self._reset_episode()

    def set_curriculum(self, curriculum) -> None:
        self.curriculum = curriculum

    def set_stage(self, stage: Stage) -> None:
        self.stage = stage

    def _reset_episode(self):
        self.ep_steps = 0
        self.ep_return = 0.0

        self.first_touch_step: Optional[int] = None
        self.ball_touches = 0
        self.shots = 0
        self.power_hits = 0

        self.goal_step: Optional[int] = None
        self.ended_by_goal = False
        self.ended_by_no_touch = False
        self.ended_by_timeout = False

        self._prev_ball_speed = 0.0
        self._prev_touches: Dict[AgentID, int] = {}
        self._prev_goal_scored = False

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)

        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}

        self._reset_episode()

        state = info.get("state")
        if state is not None:
            self._prev_ball_speed = float(np.linalg.norm(state.ball.linear_velocity))
            self._prev_goal_scored = bool(state.goal_scored)
            for a, car in state.cars.items():
                self._prev_touches[a] = int(car.ball_touches)

        return obs if not info else (obs, info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.global_ts += 1
        self.ep_steps += 1
        self.ep_return += float(np.sum(reward))

        state: GameState | None = info.get("state")

        # ----------------------------
        # Per-step skill signals
        # ----------------------------
        if state is not None:
            ball_speed = float(np.linalg.norm(state.ball.linear_velocity))

            for agent, car in state.cars.items():
                prev = self._prev_touches.get(agent, int(car.ball_touches))
                cur = int(car.ball_touches)
                self._prev_touches[agent] = cur
                if cur <= prev:
                    continue

                self.ball_touches += 1
                if self.first_touch_step is None:
                    self.first_touch_step = self.ep_steps

                if ball_speed - self._prev_ball_speed > 600.0:
                    self.power_hits += 1

                # shot proxy: ball velocity toward opponent goal after touch
                opp_goal_y = (
                    common_values.BACK_NET_Y
                    if car.is_orange
                    else -common_values.BACK_NET_Y
                )
                to_goal = np.array([0.0, opp_goal_y, 0.0]) - state.ball.position
                n = np.linalg.norm(to_goal)
                if n > 1e-6:
                    to_goal /= n
                    vel = np.dot(state.ball.linear_velocity, to_goal)
                    if vel > 500.0:
                        self.shots += 1

            self._prev_ball_speed = ball_speed

            goal_now = bool(state.goal_scored) and not bool(self._prev_goal_scored)
            self._prev_goal_scored = bool(state.goal_scored)
            if goal_now and self.goal_step is None:
                self.goal_step = self.ep_steps

        # ----------------------------
        # Episode termination
        # ----------------------------
        done = bool(np.any(terminated)) or bool(np.any(truncated))
        if done:
            self.episode_idx += 1

            if self.ball_touches == 0:
                self.ended_by_no_touch = True
            elif state is not None and state.goal_scored:
                self.ended_by_goal = True
            else:
                self.ended_by_timeout = True

            # TB logs
            self.logger.scalar("episode/return", self.ep_return, self.global_ts)
            self.logger.scalar("episode/length", self.ep_steps, self.global_ts)
            self.logger.scalar("episode/touches", self.ball_touches, self.global_ts)
            self.logger.scalar("episode/shots", self.shots, self.global_ts)
            self.logger.scalar("episode/power_hits", self.power_hits, self.global_ts)

            if self.first_touch_step is not None:
                self.logger.scalar(
                    "episode/time_to_first_touch", self.first_touch_step, self.global_ts
                )
            if self.goal_step is not None:
                self.logger.scalar(
                    "episode/time_to_goal", self.goal_step, self.global_ts
                )

            self.logger.scalar(
                "episode/ended_by_goal", int(self.ended_by_goal), self.global_ts
            )
            self.logger.scalar(
                "episode/ended_by_no_touch", int(self.ended_by_no_touch), self.global_ts
            )
            self.logger.scalar(
                "episode/ended_by_timeout", int(self.ended_by_timeout), self.global_ts
            )

            self.logger.text("curriculum/stage", str(self.stage.value), self.global_ts)
            self.logger.flush()

            if self.episode_idx % self.print_every == 0:
                t_first = (
                    self.first_touch_step if self.first_touch_step is not None else "-"
                )
                t_goal = self.goal_step if self.goal_step is not None else "-"
                print(
                    f"[ep {self.episode_idx:5d}] "
                    f"stage={self.stage.value:<8} "
                    f"ret={self.ep_return:8.1f} "
                    f"len={self.ep_steps:4d} "
                    f"touches={self.ball_touches:3d} "
                    f"t_first={t_first!s:>4} "
                    f"shots={self.shots:3d} "
                    f"t_goal={t_goal!s:>4} "
                    f"no_touch={int(self.ended_by_no_touch)}"
                )

            # rolling window for curriculum
            self.last_episodes.append(
                {
                    "touched": self.ball_touches > 0,
                    "t_first": self.first_touch_step
                    if self.first_touch_step is not None
                    else 9999,
                    "goal": bool(state.goal_scored) if state is not None else False,
                    "t_goal": self.goal_step if self.goal_step is not None else 9999,
                }
            )

            if len(self.last_episodes) >= 50 and self.curriculum is not None:
                recent = self.last_episodes[-50:]
                touch_rate = float(sum(e["touched"] for e in recent) / len(recent))
                goal_rate = float(sum(e["goal"] for e in recent) / len(recent))
                median_t_first = float(np.median([e["t_first"] for e in recent]))
                median_t_goal = float(np.median([e["t_goal"] for e in recent]))

                stats = type("Stats", (), {})()
                stats.touch_rate = touch_rate
                stats.goal_rate = goal_rate
                stats.median_t_first = median_t_first
                stats.median_t_goal = median_t_goal

                self.curriculum.maybe_advance(stats)

            self._reset_episode()

        return obs, reward, terminated, truncated, info

    def __getattr__(self, name):
        return getattr(self.env, name)


# ==================================================
# Observation (fixed-shape)
# ==================================================


class SharedObs(ObsBuilder):
    OBS_SIZE = 44

    def __init__(self):
        super().__init__()
        self.pos_coef = np.array(
            [
                1.0 / common_values.SIDE_WALL_X,
                1.0 / common_values.BACK_NET_Y,
                1.0 / common_values.CEILING_Z,
            ],
            dtype=np.float32,
        )
        self.car_vel_coef = 1.0 / common_values.CAR_MAX_SPEED
        self.ball_vel_coef = 1.0 / common_values.BALL_MAX_SPEED
        self.boost_coef = 1.0 / 100.0
        self.height_coef = 1.0 / common_values.CEILING_Z

        max_dist = float(
            np.linalg.norm(
                [
                    common_values.SIDE_WALL_X,
                    common_values.BACK_NET_Y,
                    common_values.CEILING_Z,
                ]
            )
        )
        self.dist_coef = 1.0 / max_dist

        self.blue_goal = np.array([0.0, -common_values.BACK_NET_Y, 0.0], np.float32)
        self.orange_goal = np.array([0.0, common_values.BACK_NET_Y, 0.0], np.float32)

    @staticmethod
    def _dir_dist(vec: np.ndarray):
        dist = float(np.linalg.norm(vec))
        if dist > 1e-6:
            return (vec / dist).astype(np.float32), dist
        return np.zeros(3, dtype=np.float32), 0.0

    def reset(self, agents, initial_state, shared_info):
        pass

    def get_obs_space(self, agent):
        return spaces.Box(
            low=-1.0, high=1.0, shape=(self.OBS_SIZE,), dtype=np.float32
        ), None

    def build_obs(self, agents, state: GameState, shared_info):
        obs: Dict[AgentID, np.ndarray] = {}

        for agent in agents:
            car = state.cars[agent]
            car_phys = car.inverted_physics if car.is_orange else car.physics
            ball_phys = state.inverted_ball if car.is_orange else state.ball

            my_goal = self.orange_goal if car.is_orange else self.blue_goal
            enemy_goal = self.blue_goal if car.is_orange else self.orange_goal

            forward = car_phys.forward.astype(np.float32)
            self_vel = car_phys.linear_velocity * self.car_vel_coef

            rel_ball_pos = ball_phys.position - car_phys.position
            rel_ball_vel = ball_phys.linear_velocity - car_phys.linear_velocity

            to_ball_dir, to_ball_dist = self._dir_dist(rel_ball_pos)

            ball_speed = np.linalg.norm(ball_phys.linear_velocity) * self.ball_vel_coef
            ball_height = ball_phys.position[2] * self.height_coef

            speed_toward_ball = (
                float(np.dot(car_phys.linear_velocity, to_ball_dir)) * self.car_vel_coef
            )
            cos_forward_to_ball = float(np.dot(forward, to_ball_dir))

            ball_to_goal = enemy_goal - ball_phys.position
            ball_to_goal_dir, _ = self._dir_dist(ball_to_goal)
            cos_ball_to_goal = float(np.dot(to_ball_dir, ball_to_goal_dir))

            to_my_goal_dir, to_my_goal_dist = self._dir_dist(
                my_goal - car_phys.position
            )
            to_enemy_goal_dir, to_enemy_goal_dist = self._dir_dist(
                enemy_goal - car_phys.position
            )

            closest_dist = float("inf")
            opp_rel_pos = np.zeros(3, np.float32)
            opp_rel_vel = np.zeros(3, np.float32)
            to_opp_dir = np.zeros(3, np.float32)
            to_opp_dist = 0.0

            for other in agents:
                if other == agent:
                    continue
                other_car = state.cars[other]
                if other_car.is_orange == car.is_orange:
                    continue

                other_phys = (
                    other_car.inverted_physics if car.is_orange else other_car.physics
                )
                rel_pos = other_phys.position - car_phys.position
                d = np.linalg.norm(rel_pos)

                if d < closest_dist:
                    closest_dist = d
                    opp_rel_pos = rel_pos
                    opp_rel_vel = other_phys.linear_velocity - car_phys.linear_velocity
                    to_opp_dir, to_opp_dist = self._dir_dist(rel_pos)

            vec = np.concatenate(
                [
                    forward,  # 3
                    car_phys.up.astype(np.float32),  # 3
                    self_vel,  # 3
                    np.array([car.boost_amount * self.boost_coef], np.float32),  # 1
                    np.array([float(car.on_ground)], np.float32),  # 1
                    rel_ball_pos * self.pos_coef,  # 3
                    rel_ball_vel * self.ball_vel_coef,  # 3
                    to_ball_dir,  # 3
                    np.array([to_ball_dist * self.dist_coef], np.float32),  # 1
                    np.array([ball_speed], np.float32),  # 1
                    np.array([ball_height], np.float32),  # 1
                    np.array([speed_toward_ball], np.float32),  # 1
                    np.array([cos_forward_to_ball], np.float32),  # 1
                    np.array([cos_ball_to_goal], np.float32),  # 1
                    to_my_goal_dir,  # 3
                    to_enemy_goal_dir,  # 3
                    np.array([to_my_goal_dist * self.dist_coef], np.float32),  # 1
                    np.array([to_enemy_goal_dist * self.dist_coef], np.float32),  # 1
                    opp_rel_pos * self.pos_coef,  # 3
                    opp_rel_vel * self.car_vel_coef,  # 3
                    to_opp_dir,  # 3
                    np.array([to_opp_dist * self.dist_coef], np.float32),  # 1
                ],
                dtype=np.float32,
            )

            if vec.shape != (self.OBS_SIZE,):
                raise ValueError(
                    f"Obs size mismatch: got {vec.shape}, expected {(self.OBS_SIZE,)}"
                )

            obs[agent] = vec

        return obs


# ==================================================
# Rewards
#   - TOUCH stage: allow simple shaping
#   - SCORE/SELFPLAY: replace exploitable shaping with touch-gated goal-directed rewards
# ==================================================


class GoalEventMixin:
    """Helper to make 'goal_scored' pay only once per goal event."""

    def reset(self, agents, initial_state, shared_info):
        self._prev_goal = False

    def goal_event(self, state: GameState) -> bool:
        now = bool(state.goal_scored)
        evt = now and not bool(self._prev_goal)
        self._prev_goal = now
        return evt


class SignedGoalReward(GoalEventMixin, RewardFunction):
    def reset(self, agents, initial_state, shared_info):
        GoalEventMixin.reset(self, agents, initial_state, shared_info)

    def get_rewards(
        self, agents, state: GameState, is_terminated, is_truncated, shared_info
    ):
        rewards = {a: 0.0 for a in agents}
        if not self.goal_event(state):
            return rewards

        scoring_team = state.scoring_team
        for agent in agents:
            car = state.cars[agent]
            agent_team = 1 if car.is_orange else 0
            rewards[agent] = 1.0 if agent_team == scoring_team else -1.0
        return rewards


class FastGoalBonus(GoalEventMixin, RewardFunction):
    def reset(self, agents, initial_state, shared_info):
        GoalEventMixin.reset(self, agents, initial_state, shared_info)
        self.steps = 0

    def get_rewards(
        self, agents, state: GameState, is_terminated, is_truncated, shared_info
    ):
        self.steps += 1
        rewards = {a: 0.0 for a in agents}
        if not self.goal_event(state):
            return rewards

        bonus = float(np.exp(-self.steps / 400.0))
        scoring_team = state.scoring_team
        for agent in agents:
            car = state.cars[agent]
            agent_team = 1 if car.is_orange else 0
            rewards[agent] = bonus if agent_team == scoring_team else -bonus
        return rewards


class TouchBasedRewardBase(RewardFunction):
    def reset(self, agents, initial_state, shared_info):
        self.prev_touches = (
            {a: int(initial_state.cars[a].ball_touches) for a in agents}
            if initial_state is not None
            else {a: 0 for a in agents}
        )

    def _new_touch(self, agent: AgentID, state: GameState) -> bool:
        car = state.cars[agent]
        cur = int(car.ball_touches)
        prev = int(self.prev_touches.get(agent, cur))
        self.prev_touches[agent] = cur
        return cur > prev


class ShotCommitReward(TouchBasedRewardBase):
    """
    Binary: after a touch, if ball velocity toward opponent goal is high enough, reward.
    This prevents sideways farming.
    """

    def __init__(self, threshold: float = 1400.0):
        super().__init__()
        self.threshold = float(threshold)

    def get_rewards(
        self,
        agents: List[AgentIDType],
        state: GameState,
        is_terminated,
        is_truncated,
        shared_info,
    ):
        rewards: Dict[AgentIDType, float] = {}
        ball = state.ball

        for agent in agents:
            if not self._new_touch(agent, state):
                rewards[agent] = 0.0
                continue

            car = state.cars[agent]
            opp_goal_y = (
                common_values.BACK_NET_Y if car.is_orange else -common_values.BACK_NET_Y
            )
            to_goal = np.array(
                [0.0, opp_goal_y, 0.0], dtype=np.float32
            ) - ball.position.astype(np.float32)
            n = float(np.linalg.norm(to_goal))
            if n < 1e-6:
                rewards[agent] = 0.0
                continue
            to_goal /= n

            vel_toward = float(np.dot(ball.linear_velocity.astype(np.float32), to_goal))
            rewards[agent] = 1.0 if vel_toward > self.threshold else 0.0

        return rewards


class BallVelocityTowardGoalReward(TouchBasedRewardBase):
    """
    Dense but touch-gated: after a touch, reward positive ball velocity component toward opponent net.
    """

    def __init__(self, scale: float = 3000.0, cap: float = 1.0):
        super().__init__()
        self.scale = float(scale)
        self.cap = float(cap)

    def get_rewards(
        self, agents, state: GameState, is_terminated, is_truncated, shared_info
    ):
        rewards: Dict[AgentID, float] = {}
        ball = state.ball

        for a in agents:
            if not self._new_touch(a, state):
                rewards[a] = 0.0
                continue

            car = state.cars[a]
            opp_goal_y = (
                common_values.BACK_NET_Y if car.is_orange else -common_values.BACK_NET_Y
            )
            goal = np.array([0.0, opp_goal_y, 0.0], dtype=np.float32)

            to_goal = goal - ball.position.astype(np.float32)
            n = float(np.linalg.norm(to_goal))
            if n < 1e-6:
                rewards[a] = 0.0
                continue
            to_goal /= n

            vel = float(np.dot(ball.linear_velocity.astype(np.float32), to_goal))
            rewards[a] = float(np.clip(max(vel, 0.0) / self.scale, 0.0, self.cap))

        return rewards


class TouchWindowBallDistanceToGoalDelta(RewardFunction):
    """
    After a touch, open a short window where we reward *reducing distance to goal*.
    This gives dense follow-through without allowing endless camping.
    """

    def __init__(self, window_steps: int = 15):
        super().__init__()
        self.window_steps = int(window_steps)

    def reset(self, agents, initial_state, shared_info):
        self.prev_touches = {a: int(initial_state.cars[a].ball_touches) for a in agents}
        self.window_left = {a: 0 for a in agents}
        self.prev_dist = {}

        if initial_state is not None:
            for a in agents:
                car = initial_state.cars[a]
                opp_goal_y = (
                    common_values.BACK_NET_Y
                    if car.is_orange
                    else -common_values.BACK_NET_Y
                )
                goal = np.array([0.0, opp_goal_y, 0.0], dtype=np.float32)
                self.prev_dist[a] = float(
                    np.linalg.norm(initial_state.ball.position - goal)
                )

    def get_rewards(
        self, agents, state: GameState, is_terminated, is_truncated, shared_info
    ):
        rewards: Dict[AgentID, float] = {}
        ball_pos = state.ball.position.astype(np.float32)

        for a in agents:
            car = state.cars[a]
            cur_t = int(car.ball_touches)
            if cur_t > self.prev_touches.get(a, cur_t):
                self.window_left[a] = self.window_steps
                opp_goal_y = (
                    common_values.BACK_NET_Y
                    if car.is_orange
                    else -common_values.BACK_NET_Y
                )
                goal = np.array([0.0, opp_goal_y, 0.0], dtype=np.float32)
                self.prev_dist[a] = float(np.linalg.norm(ball_pos - goal))

            self.prev_touches[a] = cur_t

            if self.window_left.get(a, 0) <= 0:
                rewards[a] = 0.0
                continue

            opp_goal_y = (
                common_values.BACK_NET_Y if car.is_orange else -common_values.BACK_NET_Y
            )
            goal = np.array([0.0, opp_goal_y, 0.0], dtype=np.float32)
            dist = float(np.linalg.norm(ball_pos - goal))

            prev = float(self.prev_dist.get(a, dist))
            delta = prev - dist  # positive is good (closer)
            self.prev_dist[a] = dist
            self.window_left[a] -= 1

            # small, capped, and ignores tiny jitter
            if abs(delta) < 10.0:
                delta = 0.0
            rewards[a] = float(np.clip(delta / 2000.0, -0.02, 0.05))

        return rewards


class GoalAlignmentReward(RewardFunction):
    """
    Dense, not touch-gated: rewards ball velocity direction aligning with the opponent goal.
    Kept low-weight. Discourages sideways dribbling.
    """

    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(
        self, agents, state: GameState, is_terminated, is_truncated, shared_info
    ):
        ball_v = state.ball.linear_velocity.astype(np.float32)
        speed = float(np.linalg.norm(ball_v))
        if speed < 300.0:
            return {a: 0.0 for a in agents}

        rewards: Dict[AgentID, float] = {}
        for a in agents:
            car = state.cars[a]
            opp_goal_y = (
                common_values.BACK_NET_Y if car.is_orange else -common_values.BACK_NET_Y
            )
            to_goal = np.array(
                [0.0, opp_goal_y, 0.0], dtype=np.float32
            ) - state.ball.position.astype(np.float32)
            n = float(np.linalg.norm(to_goal))
            if n < 1e-6:
                rewards[a] = 0.0
                continue
            to_goal /= n
            align = float(np.dot(ball_v / speed, to_goal))
            rewards[a] = float(np.clip(align, 0.0, 1.0))
        return rewards


class PowerHitReward(TouchBasedRewardBase):
    """
    Touch-gated: reward speed increase of the ball after the touch.
    Good for hard shots / clears, but capped.
    """

    def reset(self, agents, initial_state, shared_info):
        super().reset(agents, initial_state, shared_info)
        self.prev_ball_speed = (
            0.0
            if initial_state is None
            else float(np.linalg.norm(initial_state.ball.linear_velocity))
        )

    def get_rewards(
        self, agents, state: GameState, is_terminated, is_truncated, shared_info
    ):
        ball_speed = float(np.linalg.norm(state.ball.linear_velocity))
        delta_speed = ball_speed - float(self.prev_ball_speed)
        self.prev_ball_speed = ball_speed

        rewards: Dict[AgentID, float] = {}
        for a in agents:
            if not self._new_touch(a, state):
                rewards[a] = 0.0
                continue
            rewards[a] = float(np.clip(delta_speed / 3000.0, 0.0, 0.3))
        return rewards


class SpeedTowardBallReward(RewardFunction):
    """TOUCH-stage shaping: reward decreasing distance to ball (capped)."""

    def reset(self, agents, initial_state, shared_info):
        self.prev_dist: Dict[AgentID, float] = {}

    def get_rewards(
        self, agents, state: GameState, is_terminated, is_truncated, shared_info
    ):
        rewards: Dict[AgentID, float] = {}
        for a in agents:
            car = state.cars[a]
            car_phys = car.physics if not car.is_orange else car.inverted_physics
            ball_phys = state.ball if not car.is_orange else state.inverted_ball

            dist = float(np.linalg.norm(ball_phys.position - car_phys.position))
            prev = float(self.prev_dist.get(a, dist))
            delta = prev - dist

            fade = np.clip(dist / 1500.0, 0.0, 1.0)
            reward = (delta / 500.0) * fade
            rewards[a] = float(np.clip(reward, 0.0, 0.05))

            self.prev_dist[a] = dist
        return rewards


class NoTouchTimeoutPressure(RewardFunction):
    def reset(self, agents, initial_state, shared_info):
        self.steps_since_touch = {a: 0 for a in agents}
        self.prev_touches = {a: int(initial_state.cars[a].ball_touches) for a in agents}

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {a: 0.0 for a in agents}
        for a in agents:
            cur = int(state.cars[a].ball_touches)
            if cur > self.prev_touches[a]:
                self.steps_since_touch[a] = 0
            else:
                self.steps_since_touch[a] += 1
            self.prev_touches[a] = cur

            t = self.steps_since_touch[a]
            if t > 60:
                rewards[a] = -min(0.002 * (t - 60), 0.2)
        return rewards


class StepPenalty(RewardFunction):
    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(
        self, agents, state: GameState, is_terminated, is_truncated, shared_info
    ):
        return {agent: -0.001 for agent in agents}


class GoalMouthCampingPenalty(RewardFunction):
    """
    Penalize sitting deep in / behind the goal mouth.
    Helps prevent 'park in net' exploits in SCORE.
    """

    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(
        self, agents, state: GameState, is_terminated, is_truncated, shared_info
    ):
        rewards: Dict[AgentID, float] = {}
        for a in agents:
            car = state.cars[a]
            # absolute y, near back wall/net
            y = float(abs(car.physics.position[1]))
            # start penalizing in the last ~600uu of the field
            if y > common_values.BACK_NET_Y - 600:
                rewards[a] = -0.01
            else:
                rewards[a] = 0.0
        return rewards


# ==================================================
# Curriculum mutators
# ==================================================


class BallNearCarMutator(StateMutator):
    def __init__(
        self,
        min_dist: CurriculumValue,
        max_dist: CurriculumValue,
        max_angle_deg: CurriculumValue,
        ball_velocity: CurriculumValue,
    ):
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.max_angle = max_angle_deg
        self.ball_velocity = ball_velocity

    @override
    def apply(self, state: GameState, shared_info: dict[str, Any]) -> None:
        car = next(iter(state.cars.values()))
        phys = car.physics

        forward = phys.forward
        right = phys.right

        dist = np.random.uniform(self.min_dist.get(), self.max_dist.get())
        angle = np.deg2rad(
            np.random.uniform(-self.max_angle.get(), self.max_angle.get())
        )

        dir_vec = np.cos(angle) * forward + np.sin(angle) * right
        dir_vec /= np.linalg.norm(dir_vec)

        pos = phys.position + dir_vec * dist

        pos[0] = np.clip(
            pos[0], -common_values.SIDE_WALL_X + 200, common_values.SIDE_WALL_X - 200
        )
        pos[1] = np.clip(
            pos[1], -common_values.BACK_NET_Y + 200, common_values.BACK_NET_Y - 200
        )
        pos[2] = common_values.BALL_RADIUS

        state.ball.position[:] = pos
        state.ball.angular_velocity[:] = 0.0

        v = self.ball_velocity.get()
        if v > 0.0:
            vel_dir = np.array(
                [np.random.uniform(-1, 1), np.random.uniform(-1, 1), 0.0],
                dtype=np.float32,
            )
            vel_dir /= np.linalg.norm(vel_dir)
            state.ball.linear_velocity[:] = vel_dir * v
        else:
            state.ball.linear_velocity[:] = 0.0


class ProgressiveResetMutator(StateMutator):
    def __init__(self, easy_mutator: StateMutator, p_easy: CurriculumValue):
        self.easy_mutator = easy_mutator
        self.p_easy = p_easy

    @override
    def apply(self, state: GameState, shared_info: dict[str, Any]) -> None:
        if np.random.rand() < self.p_easy.get():
            self.easy_mutator.apply(state, shared_info)
        # else: leave kickoff as-is


class TouchDoneCondition(DoneCondition[AgentID, GameState]):
    def reset(
        self,
        agents: List[AgentID],
        initial_state: GameState,
        shared_info: Dict[str, Any],
    ) -> None:
        self.prev_touches = {a: int(initial_state.cars[a].ball_touches) for a in agents}

    def is_done(
        self, agents: List[AgentID], state: GameState, shared_info: Dict[str, Any]
    ) -> Dict[AgentID, bool]:
        touched = False
        for a in agents:
            cur = int(state.cars[a].ball_touches)
            if cur > self.prev_touches.get(a, cur):
                touched = True
            self.prev_touches[a] = cur
        return {a: touched for a in agents}


# ==================================================
# Curriculum manager: touch -> score -> selfplay
# ==================================================


class CurriculumManager:
    """
    Owns (a) stage transitions and (b) the knobs for the easy reset distribution.
    """

    def __init__(
        self,
        episode_logger: EpisodeLogger,
        min_dist: CurriculumValue,
        max_dist: CurriculumValue,
        max_angle: CurriculumValue,
        ball_velocity: CurriculumValue,
        p_easy_reset: CurriculumValue,
        stage_ref: "EnvBuilder",
    ):
        self.logger = episode_logger

        self.min_dist = min_dist
        self.max_dist = max_dist
        self.max_angle = max_angle
        self.ball_velocity = ball_velocity
        self.p_easy_reset = p_easy_reset

        self.stage_ref = stage_ref

    def _set_stage(self, stage: Stage) -> None:
        self.stage_ref.stage = stage
        self.logger.set_stage(stage)
        print(f"✅ Curriculum stage -> {stage.value}")

    def maybe_advance(self, stats) -> None:
        stage: Stage = self.stage_ref.stage

        # -------------------------
        # Stage 0: TOUCH curriculum
        # -------------------------
        if stage == Stage.TOUCH:
            # touch_rate in [0.3, 0.95] → difficulty in [0, 1]
            touch_factor = np.clip((stats.touch_rate - 0.3) / 0.65, 0.0, 1.0)

            # median_t_first in [150, 50] → speed factor in [0, 1]
            speed_factor = np.clip((150 - stats.median_t_first) / 100.0, 0.0, 1.0)

            # combine (touch reliability matters more than speed early)
            progress = 0.85 * touch_factor + 0.15 * speed_factor

            # targets as a function of progress
            target_min_dist = 300 + 600 * progress  # 300 → 900
            target_max_dist = 600 + 800 * progress  # 600 → 1400
            target_angle = 20 + 40 * progress  # 20° → 60°
            target_ball_vel = 0 + 600 * progress  # 0 → 600
            target_p_easy = 1.0 - progress  # 1.0 → 0.0

            # smoothing
            alpha = 0.15
            self.min_dist.set(
                (1 - alpha) * self.min_dist.get() + alpha * target_min_dist
            )
            self.max_dist.set(
                (1 - alpha) * self.max_dist.get() + alpha * target_max_dist
            )
            self.max_angle.set(
                (1 - alpha) * self.max_angle.get() + alpha * target_angle
            )
            self.ball_velocity.set(
                (1 - alpha) * self.ball_velocity.get() + alpha * target_ball_vel
            )
            self.p_easy_reset.set(
                (1 - alpha) * self.p_easy_reset.get() + alpha * target_p_easy
            )

            if np.random.rand() < 0.05:
                print(
                    f"📈 TOUCH progress={progress:.2f} "
                    f"touch={stats.touch_rate:.2f} "
                    f"t_first={stats.median_t_first:.0f} "
                    f"min={self.min_dist.get():.0f} "
                    f"max={self.max_dist.get():.0f} "
                    f"angle={self.max_angle.get():.0f} "
                    f"v={self.ball_velocity.get():.0f} "
                    f"p_easy={self.p_easy_reset.get():.2f}"
                )

            if progress > 0.75:
                self._set_stage(Stage.SCORE)

        # -------------------------
        # Stage 1: SCORE curriculum
        # -------------------------
        elif stage == Stage.SCORE:
            # graduate when scoring is somewhat consistent and not too slow
            if stats.goal_rate > 0.20 and stats.median_t_goal < 260:
                self._set_stage(Stage.SELFPLAY)

        # -------------------------
        # Stage 2: SELFPLAY
        # -------------------------
        elif stage == Stage.SELFPLAY:
            pass


# ==================================================
# Central env factory
# ==================================================


def make_stage_config(stage: Stage) -> StageConfig:
    """
    Single source of truth for how each phase behaves.
    """
    if stage == Stage.TOUCH:
        return StageConfig(
            stage=stage,
            blue_players=1,
            orange_players=0,
            end_on_touch=True,
            end_on_goal=False,
            no_touch_timeout_s=5,
            timeout_s=300,
            # rewards: learn contact + meaningful touch
            w_goal=0.0,
            w_fast_goal=0.0,
            w_ball_vel_to_goal=0.0,
            w_ball_dist_to_goal=0.0,
            w_shot_commit=0.0,
            w_align=0.0,
            w_hard_hit=0.0,
            w_touch=2.0,
            w_power=0.6,
            w_approach=0.25,
            w_step_penalty=1.0,
            w_notouch_pressure=0.10,
            w_camp_penalty=0.0,
        )

    if stage == Stage.SCORE:
        return StageConfig(
            stage=stage,
            blue_players=1,
            orange_players=0,
            end_on_touch=False,
            end_on_goal=True,
            no_touch_timeout_s=10,
            timeout_s=300,
            # rewards: force real shots / conversions (score-safe)
            w_goal=40.0,
            w_fast_goal=10.0,
            w_ball_vel_to_goal=8.0,
            w_ball_dist_to_goal=3.0,
            w_shot_commit=6.0,
            w_align=1.5,
            w_hard_hit=4.0,
            # turn off comfort shaping
            w_touch=0.0,
            w_power=0.0,
            w_approach=0.0,
            # punish stalling more than TOUCH
            w_step_penalty=1.5,
            w_notouch_pressure=0.25,
            w_camp_penalty=1.0,
        )

    # SELFPLAY
    return StageConfig(
        stage=stage,
        blue_players=1,
        orange_players=1,
        end_on_touch=False,
        end_on_goal=True,
        no_touch_timeout_s=10,
        timeout_s=300,
        # rewards: still goal-focused, slightly less dense to avoid farming
        w_goal=35.0,
        w_fast_goal=8.0,
        w_ball_vel_to_goal=6.0,
        w_ball_dist_to_goal=2.0,
        w_shot_commit=4.0,
        w_align=1.0,
        w_hard_hit=2.5,
        w_touch=0.0,
        w_power=0.0,
        w_approach=0.0,
        w_step_penalty=1.3,
        w_notouch_pressure=0.20,
        w_camp_penalty=1.0,
    )


class CurriculumReward(RewardFunction):
    def __init__(self, stage_ref: "EnvBuilder"):
        self.stage_ref = stage_ref

        # underlying rewards (created once)
        self.goal = SignedGoalReward()
        self.fast_goal = FastGoalBonus()

        # score-safe, goal-directed rewards
        self.ball_vel_to_goal = BallVelocityTowardGoalReward()
        self.ball_dist_to_goal = TouchWindowBallDistanceToGoalDelta(window_steps=15)
        self.shot_commit = ShotCommitReward(threshold=1400.0)
        self.align = GoalAlignmentReward()
        self.hard_hit = PowerHitReward()  # already touch-gated and based on delta speed

        # touch-stage shaping
        self.touch = TouchReward()
        self.approach = SpeedTowardBallReward()
        self.power = PowerHitReward()

        # hygiene
        self.step = StepPenalty()
        self.notouch = NoTouchTimeoutPressure()
        self.camp = GoalMouthCampingPenalty()

    def reset(self, agents, initial_state, shared_info):
        for r in (
            self.goal,
            self.fast_goal,
            self.ball_vel_to_goal,
            self.ball_dist_to_goal,
            self.shot_commit,
            self.align,
            self.hard_hit,
            self.touch,
            self.approach,
            self.power,
            self.step,
            self.notouch,
            self.camp,
        ):
            r.reset(agents, initial_state, shared_info)

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        cfg = make_stage_config(self.stage_ref.stage)
        rewards = {a: 0.0 for a in agents}

        def add(rwd, weight: float) -> None:
            if weight == 0.0:
                return
            vals = rwd.get_rewards(
                agents, state, is_terminated, is_truncated, shared_info
            )
            for a in agents:
                rewards[a] += weight * float(vals[a])

        # terminal-ish
        add(self.goal, cfg.w_goal)
        add(self.fast_goal, cfg.w_fast_goal)

        # score-safe, goal-directed
        add(self.ball_vel_to_goal, cfg.w_ball_vel_to_goal)
        add(self.ball_dist_to_goal, cfg.w_ball_dist_to_goal)
        add(self.shot_commit, cfg.w_shot_commit)
        add(self.align, cfg.w_align)
        add(self.hard_hit, cfg.w_hard_hit)

        # touch-stage shaping only
        add(self.touch, cfg.w_touch)
        add(self.power, cfg.w_power)
        add(self.approach, cfg.w_approach)

        # hygiene
        add(self.step, cfg.w_step_penalty)
        add(self.notouch, cfg.w_notouch_pressure)
        add(self.camp, cfg.w_camp_penalty)

        return rewards


class CurriculumDoneCondition(DoneCondition[AgentID, GameState]):
    def __init__(self, stage_ref: "EnvBuilder"):
        self.stage_ref = stage_ref
        self.touch_done = TouchDoneCondition()
        self.goal_done = GoalCondition()

    def reset(self, agents, initial_state, shared_info):
        self.touch_done.reset(agents, initial_state, shared_info)
        self.goal_done.reset(agents, initial_state, shared_info)

    def is_done(self, agents, state, shared_info):
        stage = self.stage_ref.stage
        if stage == Stage.TOUCH:
            return self.touch_done.is_done(agents, state, shared_info)
        return self.goal_done.is_done(agents, state, shared_info)


class EnvBuilder:
    def __init__(self):
        # curriculum knobs
        self.min_dist = CurriculumValue(300)
        self.max_dist = CurriculumValue(600)
        self.max_angle = CurriculumValue(20)
        self.ball_velocity = CurriculumValue(0.0)
        self.p_easy_reset = CurriculumValue(1.0)

        # stage is mutable and shared across resets
        self.stage: Stage = Stage.TOUCH

    def __call__(self):
        cfg = make_stage_config(self.stage)

        action_parser = RepeatAction(LookupTableAction(), repeats=2)
        reward_fn = CurriculumReward(stage_ref=self)

        termination_cond = CurriculumDoneCondition(stage_ref=self)
        truncation_cond = AnyCondition(
            NoTouchTimeoutCondition(cfg.no_touch_timeout_s),
            TimeoutCondition(cfg.timeout_s),
        )

        reset_mutator = ProgressiveResetMutator(
            easy_mutator=BallNearCarMutator(
                self.min_dist,
                self.max_dist,
                self.max_angle,
                self.ball_velocity,
            ),
            p_easy=self.p_easy_reset,
        )

        env = RLGym(
            state_mutator=MutatorSequence(
                FixedTeamSizeMutator(cfg.blue_players, cfg.orange_players),
                KickoffMutator(),
                reset_mutator,
            ),
            obs_builder=SharedObs(),
            action_parser=action_parser,
            reward_fn=reward_fn,
            termination_cond=termination_cond,
            truncation_cond=truncation_cond,
            transition_engine=RocketSimEngine(),
        )

        base = RLGymV2GymWrapper(env)

        logger = TBLogger("runs/shot_bot_v3_progressive_fixed")
        wrapped = EpisodeLogger(base, logger, print_every=10)
        wrapped.set_stage(cfg.stage)

        curriculum = CurriculumManager(
            episode_logger=wrapped,
            min_dist=self.min_dist,
            max_dist=self.max_dist,
            max_angle=self.max_angle,
            ball_velocity=self.ball_velocity,
            p_easy_reset=self.p_easy_reset,
            stage_ref=self,
        )
        wrapped.set_curriculum(curriculum)

        return wrapped


# ==================================================
# Training
# ==================================================


def main():
    env_builder = EnvBuilder()

    learner = Learner(
        env_builder,
        n_proc=16,
        min_inference_size=12,
        policy_layer_sizes=(512, 512, 256),
        critic_layer_sizes=(512, 512, 256),
        ppo_batch_size=100_000,
        ppo_minibatch_size=25_000,
        ppo_epochs=2,
        ppo_ent_coef=0.01,
        policy_lr=3e-4,
        critic_lr=3e-4,
        ts_per_iteration=100_000,
        exp_buffer_size=400_000,
        timestep_limit=1_000_000_000,
        log_to_wandb=False,
        save_every_ts=5_000_000,
    )

    learner.learn()


if __name__ == "__main__":
    main()
