from __future__ import annotations
from typing import Any, Dict, List

import numpy as np

from rlgym_ppo import Learner
from rlgym.api import AgentID, RLGym, RewardFunction, ObsBuilder
from rlgym.api.typing import AgentID, RewardType, StateType
from rlgym.rocket_league import common_values
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.done_conditions import (
    AnyCondition,
    GoalCondition,
    NoTouchTimeoutCondition,
    TimeoutCondition,
)
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import (
    FixedTeamSizeMutator,
    KickoffMutator,
    MutatorSequence,
)
from rlgym_ppo.util import RLGymV2GymWrapper


from gymnasium import spaces


from dataclasses import dataclass, field
from pathlib import Path

from torch.utils.tensorboard.writer import SummaryWriter


# --------------------------------------------------
# METRICS
# --------------------------------------------------


@dataclass
class EpisodeStats:
    ep: int = 0
    steps: int = 0
    return_sum: float = 0.0

    goals: int = 0
    ball_touches: int = 0
    shots: int = 0  # touches that push ball toward opponent goal

    def reset(self) -> None:
        self.steps = 0
        self.return_sum = 0.0
        self.goals = 0
        self.ball_touches = 0
        self.shots = 0


class TBLogger:
    """Independent of rlgym_ppo internals."""

    def __init__(self, log_dir: str = "runs/rlgym") -> None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.w = SummaryWriter(log_dir=log_dir)

    def scalar(self, tag: str, value: float, step: int) -> None:
        self.w.add_scalar(tag, float(value), step)

    def flush(self) -> None:
        self.w.flush()


class GymEpisodeLoggingWrapper:
    """
    Logs *actual Rocket League skill signals* per episode:
      - goals
      - ball touches
      - shots (touches toward opponent goal)
      - episodic return
      - episode length
    """

    def __init__(self, env, logger: TBLogger, print_every_episodes: int = 10):
        self.env = env
        self.logger = logger
        self.print_every = print_every_episodes

        self.global_ts = 0
        self.stats = EpisodeStats()

        # Track last-touch state so we only count new touches
        self._last_ball_touch = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_ball_touch = False
        self.stats = EpisodeStats()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        r = float(np.sum(reward))
        self.stats.steps += 1
        self.stats.return_sum += r
        self.global_ts += 1

        # -----------------------------
        # Pull GameState from info
        # -----------------------------
        state: GameState | None = info.get("state")
        if state is not None:
            for car in state.cars.values():
                ball = state.ball

                # -----------------------------
                # Ball touches
                # -----------------------------
                touched = car.ball_touches > 0
                if touched and not self._last_ball_touch:
                    self.stats.ball_touches += 1

                    # -----------------------------
                    # Shot detection
                    # -----------------------------
                    goal_y = (
                        -common_values.BACK_NET_Y
                        if car.is_orange
                        else common_values.BACK_NET_Y
                    )

                    to_goal = np.array([0.0, goal_y, 0.0]) - ball.position
                    to_goal /= np.linalg.norm(to_goal) + 1e-6

                    vel_toward_goal = np.dot(ball.linear_velocity, to_goal)
                    if vel_toward_goal > 500:  # conservative threshold
                        self.stats.shots += 1

                    self._last_ball_touch = touched

            if state.goal_scored:
                self.stats.goals += 1

            # -----------------------------
            # Episode end
            # -----------------------------
            done = bool(np.any(terminated)) or bool(np.any(truncated))
            if done:
                self.stats.ep += 1

                # -----------------------------
                # TensorBoard logging
                # -----------------------------
                self.logger.scalar(
                    "episode/return", self.stats.return_sum, self.global_ts
                )
                self.logger.scalar("episode/length", self.stats.steps, self.global_ts)
                self.logger.scalar("episode/goals", self.stats.goals, self.global_ts)
                self.logger.scalar(
                    "episode/ball_touches", self.stats.ball_touches, self.global_ts
                )
                self.logger.scalar("episode/shots", self.stats.shots, self.global_ts)

                # Derived metric (most important)
                if self.stats.ball_touches > 0:
                    self.logger.scalar(
                        "episode/shot_rate",
                        self.stats.shots / self.stats.ball_touches,
                        self.global_ts,
                    )

                self.logger.flush()

                # -----------------------------
                # stdout
                # -----------------------------
                if self.stats.ep % self.print_every == 0:
                    print(
                        f"[ep {self.stats.ep:6d}] "
                        f"ret={self.stats.return_sum:7.1f} "
                        f"len={self.stats.steps:4d} "
                        f"goals={self.stats.goals} "
                        f"touches={self.stats.ball_touches} "
                        f"shots={self.stats.shots} "
                        f"ts={self.global_ts}"
                    )

                self.stats.reset()
                self._last_ball_touch = False

        return obs, reward, terminated, truncated, info

    def __getattr__(self, name):
        return getattr(self.env, name)


# --------------------------------------------------
# REWARDS
# --------------------------------------------------


class ShotReward(RewardFunction):
    """Reward ball velocity toward opponent goal AFTER a touch."""

    def reset(
        self,
        agents: list,
        initial_state: Any,
        shared_info: dict[str, Any],
    ) -> None:
        pass

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: dict[AgentID, bool],
        is_truncated: dict[AgentID, bool],
        shared_info: dict[str, Any],
    ) -> dict[AgentID, RewardType]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            if not car.ball_touches > 0:
                rewards[agent] = 0.0
                continue

            ball = state.ball
            goal_y = (
                -common_values.BACK_NET_Y if car.is_orange else common_values.BACK_NET_Y
            )

            to_goal = np.array([0.0, goal_y, 0.0]) - ball.position
            to_goal /= np.linalg.norm(to_goal) + 1e-6

            vel = np.dot(ball.linear_velocity, to_goal)
            rewards[agent] = max(vel / common_values.BALL_MAX_SPEED, 0.0)

        return rewards


class SpeedTowardBallReward(RewardFunction):
    """Encourage accelerating toward the ball, scaled by distance."""

    def reset(
        self,
        agents: list,
        initial_state: Any,
        shared_info: dict[str, Any],
    ) -> None:
        pass

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: dict[AgentID, bool],
        is_truncated: dict[AgentID, bool],
        shared_info: dict[str, Any],
    ) -> dict[AgentID, RewardType]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            car_phys = car.physics if not car.is_orange else car.inverted_physics
            ball_phys = state.ball if not car.is_orange else state.inverted_ball

            delta = ball_phys.position - car_phys.position
            dist = np.linalg.norm(delta)
            if dist < 1e-6:
                rewards[agent] = 0.0
                continue

            dir_to_ball = delta / dist
            speed = max(np.dot(car_phys.linear_velocity, dir_to_ball), 0.0)
            dist_factor = np.exp(-dist / 3000)

            rewards[agent] = (speed / common_values.CAR_MAX_SPEED) * dist_factor

        return rewards


class BallControlReward(RewardFunction):
    """Encourage staying close to the ball."""

    def reset(
        self,
        agents: list,
        initial_state: Any,
        shared_info: dict[str, Any],
    ) -> None:
        pass

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: dict[AgentID, bool],
        is_truncated: dict[AgentID, bool],
        shared_info: dict[str, Any],
    ) -> dict[AgentID, RewardType]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            dist = np.linalg.norm(car.physics.position - state.ball.position)
            rewards[agent] = np.exp(-dist / 1200)
        return rewards


class GoalSideReward(RewardFunction):
    """Reward staying between the ball and your own goal."""

    def reset(
        self,
        agents: list,
        initial_state: Any,
        shared_info: dict[str, Any],
    ) -> None:
        pass

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: dict[AgentID, bool],
        is_truncated: dict[AgentID, bool],
        shared_info: dict[str, Any],
    ) -> dict[AgentID, RewardType]:
        rewards = {}

        for agent in agents:
            car = state.cars[agent]
            ball = state.ball

            goal_y = (
                -common_values.BACK_NET_Y if car.is_orange else common_values.BACK_NET_Y
            )

            car_dist = abs(car.physics.position[1] - goal_y)
            ball_dist = abs(ball.position[1] - goal_y)

            rewards[agent] = float(car_dist < ball_dist)

        return rewards


class BoostEfficiencyReward(RewardFunction):
    """Encourage useful boost usage."""

    def reset(
        self,
        agents: list,
        initial_state: Any,
        shared_info: dict[str, Any],
    ) -> None:
        pass

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: dict[AgentID, bool],
        is_truncated: dict[AgentID, bool],
        shared_info: dict[str, Any],
    ) -> dict[AgentID, RewardType]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            speed = np.linalg.norm(car.physics.linear_velocity)
            rewards[agent] = (speed / common_values.CAR_MAX_SPEED) * (
                car.boost_amount / 100.0
            )
        return rewards


# --------------------------------------------------
# OBSERVATION (DEPLOYABLE IN RLBOT)
# --------------------------------------------------


class SharedObs(ObsBuilder):
    """
    Explicit, symmetric observation vector.
    Easy to reimplement in RLBot.
    """

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
        self.ang_vel_coef = 1.0 / common_values.CAR_MAX_ANG_VEL
        self.boost_coef = 1.0 / 100.0

        max_dist = np.linalg.norm(
            [
                common_values.SIDE_WALL_X,
                common_values.BACK_NET_Y,
                common_values.CEILING_Z,
            ]
        )
        self.dist_coef = 1.0 / max_dist

    def _dir_dist(self, a: np.ndarray, b: np.ndarray):
        d = b - a
        dist = np.linalg.norm(d)
        if dist > 1e-6:
            return d / dist, dist
        return np.zeros(3, dtype=np.float32), 0.0

    def reset(self, agents, initial_state, shared_info):
        pass

    def get_obs_space(self, agent):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(48,),
            dtype=np.float32,
        ), None

    def build_obs(self, agents, state: GameState, shared_info):
        obs = {}

        for agent in agents:
            car = state.cars[agent]

            car_phys = car.physics if not car.is_orange else car.inverted_physics
            ball_phys = state.ball if not car.is_orange else state.inverted_ball

            opp = next(a for a in agents if a != agent)
            opp_car = state.cars[opp]
            opp_phys = (
                opp_car.physics if not car.is_orange else opp_car.inverted_physics
            )

            to_ball_dir, to_ball_dist = self._dir_dist(
                car_phys.position, ball_phys.position
            )
            to_opp_dir, to_opp_dist = self._dir_dist(
                car_phys.position, opp_phys.position
            )

            goal_y = (
                -common_values.BACK_NET_Y if car.is_orange else common_values.BACK_NET_Y
            )
            goal_pos = np.array([0.0, goal_y, 0.0], dtype=np.float32)
            ball_to_goal_dir, ball_to_goal_dist = self._dir_dist(
                ball_phys.position, goal_pos
            )

            rel_ball_vel = (
                ball_phys.linear_velocity - car_phys.linear_velocity
            ) * self.ball_vel_coef

            approach_speed = (
                np.dot(car_phys.linear_velocity, to_ball_dir)
                / common_values.CAR_MAX_SPEED
            )

            obs[agent] = np.concatenate(
                [
                    # self
                    car_phys.position * self.pos_coef,
                    car_phys.linear_velocity * self.car_vel_coef,
                    car_phys.forward,
                    car_phys.up,
                    car_phys.angular_velocity * self.ang_vel_coef,
                    [car.boost_amount * self.boost_coef],
                    [float(car.on_ground)],
                    # ball
                    ball_phys.position * self.pos_coef,
                    ball_phys.linear_velocity * self.ball_vel_coef,
                    rel_ball_vel,
                    # opponent (relative)
                    (opp_phys.position - car_phys.position) * self.pos_coef,
                    (opp_phys.linear_velocity - car_phys.linear_velocity)
                    * self.car_vel_coef,
                    [opp_car.boost_amount * self.boost_coef],
                    [float(opp_car.on_ground)],
                    # relations
                    to_ball_dir,
                    [to_ball_dist * self.dist_coef],
                    to_opp_dir,
                    [to_opp_dist * self.dist_coef],
                    ball_to_goal_dir,
                    [ball_to_goal_dist * self.dist_coef],
                    [approach_speed],
                ],
                dtype=np.float32,
            )

        return obs


# --------------------------------------------------
# ENV BUILDER
# --------------------------------------------------


def build_rlgym_v2_env():
    action_parser = RepeatAction(LookupTableAction(), repeats=8)

    reward_fn = CombinedReward(
        (GoalReward(), 10.0),
        (ShotReward(), 1.0),
        (BallControlReward(), 0.3),
        (GoalSideReward(), 0.2),
        (SpeedTowardBallReward(), 0.05),
        (BoostEfficiencyReward(), 0.02),
    )

    env = RLGym(
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(1, 1),
            KickoffMutator(),
        ),
        obs_builder=SharedObs(),
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=GoalCondition(),
        truncation_cond=AnyCondition(
            NoTouchTimeoutCondition(30),
            TimeoutCondition(300),
        ),
        transition_engine=RocketSimEngine(),
    )

    base = RLGymV2GymWrapper(env)

    logger = TBLogger("runs/shot_bot_v1")
    wrapped = GymEpisodeLoggingWrapper(base, logger, print_every_episodes=10)

    return wrapped


# --------------------------------------------------
# TRAINING
# --------------------------------------------------


def main():
    learner = Learner(
        build_rlgym_v2_env,
        n_proc=48,
        min_inference_size=12,
        policy_layer_sizes=(512, 512, 256),
        critic_layer_sizes=(512, 512, 256),
        ppo_batch_size=100_000,
        ppo_minibatch_size=25_000,
        ppo_epochs=2,
        ppo_ent_coef=0.01,
        policy_lr=1e-4,
        critic_lr=1e-4,
        ts_per_iteration=100_000,
        exp_buffer_size=400_000,
        timestep_limit=1_000_000_000,
        log_to_wandb=False,
        save_every_ts=5_000_000,
    )

    learner.learn()


if __name__ == "__main__":
    main()
