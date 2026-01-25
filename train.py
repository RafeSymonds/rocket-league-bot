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
from rlgym.rocket_league.reward_functions import CombinedReward, TouchReward
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
        # tensorboard supports add_text; SummaryWriter has add_text
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
    w_goal: float
    ball_to_net: float
    w_fast_goal: float
    w_shot: float
    w_touch: float
    w_power: float
    w_progress: float
    w_approach: float
    w_dist: float
    w_step_penalty: float
    w_notouch_pressure: float


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
                self.goal_step = self.ep_steps
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
# Observation (unchanged, fixed-shape)
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
# Rewards (reuse your existing ones)
# ==================================================


class SignedGoalReward(RewardFunction):
    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(
        self, agents, state: GameState, is_terminated, is_truncated, shared_info
    ):
        rewards = {a: 0.0 for a in agents}
        if not state.goal_scored:
            return rewards

        scoring_team = state.scoring_team
        for agent in agents:
            car = state.cars[agent]
            agent_team = 1 if car.is_orange else 0
            rewards[agent] = 1.0 if agent_team == scoring_team else -1.0
        return rewards


class FastGoalBonus(RewardFunction):
    def reset(self, agents, initial_state, shared_info):
        self.steps = 0

    def get_rewards(
        self, agents, state: GameState, is_terminated, is_truncated, shared_info
    ):
        self.steps += 1
        rewards = {a: 0.0 for a in agents}
        if not state.goal_scored:
            return rewards

        bonus = float(np.exp(-self.steps / 400.0))
        scoring_team = state.scoring_team
        for agent in agents:
            car = state.cars[agent]
            agent_team = 1 if car.is_orange else 0
            rewards[agent] = bonus if agent_team == scoring_team else -bonus
        return rewards


class BallNetProgressReward(RewardFunction):
    def reset(self, agents, initial_state, shared_info):
        self.prev_ball_dist: Dict[AgentID, float] = {}

    def get_rewards(
        self, agents, state: GameState, is_terminated, is_truncated, shared_info
    ):
        rewards: Dict[AgentID, float] = {}
        ball = state.ball

        for agent in agents:
            car = state.cars[agent]
            goal_y = (
                common_values.BACK_NET_Y if car.is_orange else -common_values.BACK_NET_Y
            )
            goal_pos = np.array([0.0, goal_y, 0.0], dtype=np.float32)

            dist = float(np.linalg.norm(ball.position.astype(np.float32) - goal_pos))
            prev = self.prev_ball_dist.get(agent, dist)
            delta = prev - dist
            if abs(delta) < 5.0:
                delta = 0.0

            rewards[agent] = float(np.clip(delta / 1000.0, -0.05, 0.05))
            self.prev_ball_dist[agent] = dist

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


class ShotReward(TouchBasedRewardBase):
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

            vel = float(np.dot(ball.linear_velocity.astype(np.float32), to_goal))
            rewards[agent] = max(vel / common_values.BALL_MAX_SPEED, 0.0)

        return rewards


class PowerHitReward(TouchBasedRewardBase):
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
            rewards[a] = float(np.clip(delta_speed / 3000.0, 0.0, 0.2))
        return rewards


class SpeedTowardBallReward(RewardFunction):
    def reset(self, agents, initial_state, shared_info):
        self.prev_dist: Dict[AgentID, float] = {}

    def get_rewards(
        self, agents, state: GameState, is_terminated, is_truncated, shared_info
    ):
        rewards: Dict[AgentID, float] = {}

        for agent in agents:
            car = state.cars[agent]
            car_phys = car.physics if not car.is_orange else car.inverted_physics
            ball_phys = state.ball if not car.is_orange else state.inverted_ball

            dist = float(np.linalg.norm(ball_phys.position - car_phys.position))
            prev = float(self.prev_dist.get(agent, dist))
            delta = prev - dist

            fade = np.clip(dist / 1500.0, 0.0, 1.0)
            reward = (delta / 500.0) * fade
            rewards[agent] = float(np.clip(reward, 0.0, 0.05))

            self.prev_dist[agent] = dist

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


class BallTowardNetReward(RewardFunction):
    """
    Reward change in ball distance toward opponent net.
    Dense, directional, and non-exploitable.
    """

    def reset(self, agents, initial_state, shared_info):
        self.prev_dist = {}
        if initial_state is None:
            return

        for a in agents:
            car = initial_state.cars[a]
            goal_y = (
                common_values.BACK_NET_Y if car.is_orange else -common_values.BACK_NET_Y
            )
            goal = np.array([0.0, goal_y, 0.0], dtype=np.float32)
            self.prev_dist[a] = float(
                np.linalg.norm(initial_state.ball.position - goal)
            )

    def get_rewards(self, agents, state, *_):
        rewards = {}

        for a in agents:
            car = state.cars[a]
            goal_y = (
                common_values.BACK_NET_Y if car.is_orange else -common_values.BACK_NET_Y
            )
            goal = np.array([0.0, goal_y, 0.0], dtype=np.float32)

            dist = float(np.linalg.norm(state.ball.position - goal))
            prev = self.prev_dist.get(a, dist)

            delta = prev - dist  # positive = closer to net

            # deadzone to avoid jitter farming
            if abs(delta) < 10.0:
                delta = 0.0

            # scale + clamp
            rewards[a] = float(np.clip(delta / 2000.0, -0.05, 0.05))
            self.prev_dist[a] = dist

        return rewards


class StepPenalty(RewardFunction):
    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(
        self, agents, state: GameState, is_terminated, is_truncated, shared_info
    ):
        return {agent: -0.001 for agent in agents}


class DistanceReductionReward(RewardFunction):
    def reset(self, agents, initial_state, shared_info):
        self.prev_dist = {}

    def get_rewards(
        self, agents, state: GameState, is_terminated, is_truncated, shared_info
    ):
        rewards = {}
        for a in agents:
            car = state.cars[a]
            ball = state.ball
            d = np.linalg.norm(car.physics.position - ball.position)
            prev = self.prev_dist.get(a, d)
            rewards[a] = np.clip((prev - d) / 1000.0, 0.0, 0.05)
            self.prev_dist[a] = d
        return rewards


# ==================================================
# Curriculum mutators (same spirit, just cleaner wiring)
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
        # pick "a" car to place ball near; in selfplay you could do per-team logic,
        # but keeping it simple: first car in dict.
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
        stage_ref: EnvBuilder,
    ):
        self.logger = episode_logger

        self.min_dist = min_dist
        self.max_dist = max_dist
        self.max_angle = max_angle
        self.ball_velocity = ball_velocity
        self.p_easy_reset = p_easy_reset

        # stage_ref is a mutable dict so env factory can read the current stage
        self.stage_ref = stage_ref

    def _set_stage(self, stage: Stage) -> None:
        self.stage_ref.stage = stage
        self.logger.set_stage(stage)
        print(f"✅ Curriculum stage -> {stage.value}")

    def maybe_advance(self, stats) -> None:
        stage: Stage = self.stage_ref.stage

        # Stage 0: learn to touch quickly & reliably
        if stage == Stage.TOUCH:
            # tighten: once touch_rate is high, make resets harder until p_easy=0
            if stats.touch_rate > 0.85 and self.p_easy_reset.get() > 0.0:
                self.min_dist.set(min(self.min_dist.get() + 100, 900))
                self.max_dist.set(min(self.max_dist.get() + 150, 1400))
                self.max_angle.set(min(self.max_angle.get() + 10, 60))
                self.ball_velocity.set(min(self.ball_velocity.get() + 50, 600))
                self.p_easy_reset.set(max(self.p_easy_reset.get() - 0.2, 0.0))
                print(
                    f"➡️ touch curriculum harder: "
                    f"min={self.min_dist.get():.0f} max={self.max_dist.get():.0f} "
                    f"angle={self.max_angle.get():.0f} v={self.ball_velocity.get():.0f} p_easy={self.p_easy_reset.get():.2f}"
                )

            # switch to scoring curriculum when touch is consistent AND first touch is fast
            if (
                stats.touch_rate > 0.90
                and stats.median_t_first < 200
                and self.p_easy_reset.get() <= 0.2
            ):
                self._set_stage(Stage.SCORE)

        # Stage 1: learn to score (goal condition + goal-centric reward)
        elif stage == Stage.SCORE:
            # once it can score with some frequency and not take forever, go to selfplay
            if stats.goal_rate > 0.25 and stats.median_t_goal < 250:
                self._set_stage(Stage.SELFPLAY)

        # Stage 2: selfplay is open-ended; you can add snapshot gating here later
        elif stage == Stage.SELFPLAY:
            pass


# ==================================================
# Optional hook for "vs old versions" (stub)
# ==================================================


class OpponentSnapshotManager:
    """
    Placeholder. Use this once you implement a mixed-policy action pipeline.

    Strategy you likely want:
      - periodically select a checkpoint from ./checkpoints/
      - load it into a frozen opponent policy
      - during rollouts, use current policy for blue and frozen for orange
    """

    def __init__(self, checkpoints_dir: str = "checkpoints", prob_old: float = 0.5):
        self.dir = Path(checkpoints_dir)
        self.prob_old = prob_old

    def choose_opponent_checkpoint(self) -> Optional[Path]:
        if np.random.rand() > self.prob_old:
            return None
        if not self.dir.exists():
            return None
        ckpts = sorted(self.dir.glob("**/*"), key=lambda p: p.stat().st_mtime)
        # pick an older one (not the newest)
        if len(ckpts) < 2:
            return None
        idx = np.random.randint(0, max(1, len(ckpts) - 1))
        return ckpts[idx]


# ==================================================
# Central env factory (the big win)
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
            no_touch_timeout_s=10,
            timeout_s=300,
            # reward weights: touch/shot dominate, goal is irrelevant
            w_goal=0.0,
            ball_to_net=0.0,
            w_fast_goal=0.0,
            w_shot=6.0,
            w_touch=2.0,
            w_power=0.5,
            w_progress=0.2,
            w_approach=0.2,
            w_dist=0.1,
            w_step_penalty=1.0,
            w_notouch_pressure=0.1,
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
            # reward weights: goal dominates now
            w_goal=40.0,
            ball_to_net=1,
            w_fast_goal=10.0,
            w_shot=10.0,
            w_touch=2.0,
            w_power=0.5,
            w_progress=0.2,
            w_approach=0.2,
            w_dist=0.1,
            w_step_penalty=1.0,
            w_notouch_pressure=0.1,
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
        # reward weights: goal still king, but shaping lower to avoid farming
        w_goal=35.0,
        ball_to_net=0.4,
        w_fast_goal=8.0,
        w_shot=6.0,
        w_touch=1.0,
        w_power=0.3,
        w_progress=0.15,
        w_approach=0.10,
        w_dist=0.05,
        w_step_penalty=1.0,
        w_notouch_pressure=0.1,
    )


class CurriculumReward(RewardFunction):
    def __init__(self, stage_ref):
        self.stage_ref = stage_ref

        # underlying rewards (created once)
        self.goal = SignedGoalReward()
        self.fast_goal = FastGoalBonus()
        self.shot = ShotReward()
        self.touch = TouchReward()
        self.power = PowerHitReward()
        self.progress = BallNetProgressReward()
        self.approach = SpeedTowardBallReward()
        self.dist = DistanceReductionReward()
        self.step = StepPenalty()
        self.notouch = NoTouchTimeoutPressure()
        self.ball_to_net = BallTowardNetReward()

    def reset(self, agents, initial_state, shared_info):
        for r in (
            self.goal,
            self.fast_goal,
            self.shot,
            self.touch,
            self.power,
            self.progress,
            self.approach,
            self.dist,
            self.step,
            self.notouch,
        ):
            r.reset(agents, initial_state, shared_info)

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        cfg = make_stage_config(self.stage_ref.stage)

        rewards = {a: 0.0 for a in agents}

        def add(rwd, weight):
            if weight == 0.0:
                return
            vals = rwd.get_rewards(
                agents, state, is_terminated, is_truncated, shared_info
            )
            for a in agents:
                rewards[a] += weight * float(vals[a])

        add(self.goal, cfg.w_goal)
        add(self.ball_to_net, cfg.ball_to_net)
        add(self.fast_goal, cfg.w_fast_goal)
        add(self.shot, cfg.w_shot)
        add(self.touch, cfg.w_touch)
        add(self.power, cfg.w_power)
        add(self.progress, cfg.w_progress)
        add(self.approach, cfg.w_approach)
        add(self.dist, cfg.w_dist)
        add(self.step, cfg.w_step_penalty)
        add(self.notouch, cfg.w_notouch_pressure)

        return rewards


class CurriculumDoneCondition(DoneCondition[AgentID, GameState]):
    def __init__(self, stage_ref):
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

        # SCORE and SELFPLAY
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

        logger = TBLogger("runs/shot_bot_v2_curriculum")
        wrapped = EpisodeLogger(base, logger, print_every=10)
        wrapped.set_stage(cfg.stage)

        curriculum = CurriculumManager(
            episode_logger=wrapped,
            min_dist=self.min_dist,
            max_dist=self.max_dist,
            max_angle=self.max_angle,
            ball_velocity=self.ball_velocity,
            p_easy_reset=self.p_easy_reset,
            stage_ref=self,  # pass *this* object
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
