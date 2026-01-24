from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from gymnasium import spaces
from torch.utils.tensorboard.writer import SummaryWriter

from rlgym.api import AgentID, RLGym, RewardFunction, ObsBuilder
from rlgym.api.typing import AgentID as AgentIDType
from rlgym.api.typing import RewardType
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
# TensorBoard logging (independent of rlgym_ppo)
# ==================================================


class TBLogger:
    def __init__(self, log_dir: str = "runs/rlgym") -> None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.w = SummaryWriter(log_dir=log_dir)

    def scalar(self, tag: str, value: float, step: int) -> None:
        self.w.add_scalar(tag, float(value), step)

    def flush(self) -> None:
        self.w.flush()


# ==================================================
# Episode metrics + logging wrapper
# ==================================================


@dataclass
class EpisodeStats:
    ep: int = 0
    steps: int = 0
    return_sum: float = 0.0

    goals: int = 0
    ball_touches: int = 0

    shots: int = 0  # touches that create a shot toward opponent goal
    power_hits: int = 0  # touches that significantly increase ball speed
    defenses: int = 0  # touches that avert a near-term threat to own goal

    power_sum: float = 0.0  # sum of "power" values for analysis (avg power)

    def reset_episode_counters(self) -> None:
        self.steps = 0
        self.return_sum = 0.0
        self.goals = 0
        self.ball_touches = 0
        self.shots = 0
        self.power_hits = 0
        self.defenses = 0
        self.power_sum = 0.0


class GymEpisodeLoggingWrapper:
    """
    Logs per-episode skill signals:
      - goals (team goals during the episode)
      - touches (new touches only)
      - shots (new touch + ball velocity toward opponent goal)
      - power hits (new touch + ball speed jump)
      - successful defenses (new touch that stops a dangerous ball toward own net)
      - episodic return + length
    """

    def __init__(self, env, logger: TBLogger, print_every_episodes: int = 10):
        self.env = env
        self.logger = logger
        self.print_every = print_every_episodes

        self.global_ts = 0
        self.stats = EpisodeStats()

        # per-agent touch tracking so "touch" means "new touch this step"
        self._prev_touches: Dict[AgentID, int] = {}

        # ball speed tracking for "power hit"
        self._prev_ball_speed: float = 0.0

        # threat/defense tracking (near own goal + moving toward it)
        self._prev_ball_dist_to_blue: float = 0.0
        self._prev_ball_dist_to_orange: float = 0.0
        self._prev_ball_vel_y: float = 0.0

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)

        # rlgym_ppo may return obs OR (obs, info)
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, None

        self.stats = EpisodeStats()
        self._prev_touches = {}
        self._prev_ball_speed = 0.0
        self._prev_ball_vel_y = 0.0

        if isinstance(info, dict):
            state = info.get("state")
        else:
            state = None

        if state is not None:
            self._prev_ball_speed = float(np.linalg.norm(state.ball.linear_velocity))
            self._prev_ball_vel_y = float(state.ball.linear_velocity[1])

            self._prev_ball_dist_to_blue = float(
                np.linalg.norm(
                    state.ball.position - np.array([0.0, common_values.BACK_NET_Y, 0.0])
                )
            )
            self._prev_ball_dist_to_orange = float(
                np.linalg.norm(
                    state.ball.position
                    - np.array([0.0, -common_values.BACK_NET_Y, 0.0])
                )
            )

            for a, car in state.cars.items():
                self._prev_touches[a] = int(car.ball_touches)

        return obs if info is None else (obs, info)

    @staticmethod
    def _goal_y_for_team(team_is_orange: bool) -> float:
        # blue scores at +Y, orange scores at -Y
        return -common_values.BACK_NET_Y if team_is_orange else common_values.BACK_NET_Y

    @staticmethod
    def _dist_to_goal(ball_pos: np.ndarray, goal_y: float) -> float:
        return float(np.linalg.norm(ball_pos - np.array([0.0, goal_y, 0.0])))

    @staticmethod
    def _vel_toward_goal(
        ball_vel: np.ndarray, ball_pos: np.ndarray, goal_y: float
    ) -> float:
        to_goal = np.array([0.0, goal_y, 0.0], dtype=np.float32) - ball_pos.astype(
            np.float32
        )
        n = float(np.linalg.norm(to_goal))
        if n < 1e-6:
            return 0.0
        to_goal /= n
        return float(np.dot(ball_vel, to_goal))

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        r = float(np.sum(reward))
        self.stats.steps += 1
        self.stats.return_sum += r
        self.global_ts += 1

        state: GameState | None = info.get("state")

        # If wrapper isn't passing state, we can still log return/len, but not skill metrics.
        if state is not None:
            ball = state.ball
            ball_speed = float(np.linalg.norm(ball.linear_velocity))

            # Precompute ball distances to each net (world frame)
            dist_to_blue = self._dist_to_goal(ball.position, common_values.BACK_NET_Y)
            dist_to_orange = self._dist_to_goal(
                ball.position, -common_values.BACK_NET_Y
            )
            vel_y = float(ball.linear_velocity[1])

            # Detect goal scored (team event)
            if state.goal_scored:
                self.stats.goals += 1

            # For each car, detect new touches and derived events
            for agent, car in state.cars.items():
                prev_t = self._prev_touches.get(agent, int(car.ball_touches))
                cur_t = int(car.ball_touches)
                new_touch = cur_t > prev_t
                self._prev_touches[agent] = cur_t

                if not new_touch:
                    continue

                # Count touch
                self.stats.ball_touches += 1

                # ---- Power hit detection (speed jump after touch)
                # We use last-step ball speed as baseline; after touch, speed should spike.
                speed_delta = ball_speed - self._prev_ball_speed
                # conservative threshold; tune if needed
                if speed_delta > 600.0:
                    self.stats.power_hits += 1
                    self.stats.power_sum += speed_delta

                # ---- Shot detection (touch + ball velocity toward opponent goal)
                # If the toucher is orange, opponent goal is +Y (blue net). If blue, opponent is -Y.
                opp_goal_y = (
                    common_values.BACK_NET_Y
                    if car.is_orange
                    else -common_values.BACK_NET_Y
                )
                vel_toward_opp = self._vel_toward_goal(
                    ball.linear_velocity, ball.position, opp_goal_y
                )
                if vel_toward_opp > 500.0:
                    self.stats.shots += 1

                # ---- Successful defense detection
                # A "danger" is: ball near your own net AND moving toward it (roughly via Y direction),
                # and a touch occurs that reduces that danger immediately.
                #
                # For blue net at +Y: ball moving toward it => vel_y > 0
                # For orange net at -Y: ball moving toward it => vel_y < 0
                own_goal_y = (
                    -common_values.BACK_NET_Y
                    if car.is_orange
                    else common_values.BACK_NET_Y
                )
                own_dist_prev = (
                    self._prev_ball_dist_to_orange
                    if car.is_orange
                    else self._prev_ball_dist_to_blue
                )
                own_dist_now = dist_to_orange if car.is_orange else dist_to_blue

                moving_toward_own = (
                    (self._prev_ball_vel_y < -300.0)
                    if car.is_orange
                    else (self._prev_ball_vel_y > 300.0)
                )
                dangerous_close = own_dist_prev < 2500.0

                # A defense is: was dangerous + moving toward own net, and after touch either:
                #  - distance to own net increases meaningfully, OR
                #  - ball's Y-velocity flips away from your goal (or at least reduces strongly)
                dist_increase = (own_dist_now - own_dist_prev) > 250.0
                vel_flip_away = (vel_y > 200.0) if car.is_orange else (vel_y < -200.0)
                vel_reduced = abs(vel_y) < abs(self._prev_ball_vel_y) - 300.0

                if (
                    dangerous_close
                    and moving_toward_own
                    and (dist_increase or vel_flip_away or vel_reduced)
                ):
                    self.stats.defenses += 1

            # Update ball baselines for next step
            self._prev_ball_speed = ball_speed
            self._prev_ball_vel_y = vel_y
            self._prev_ball_dist_to_blue = dist_to_blue
            self._prev_ball_dist_to_orange = dist_to_orange

            done = bool(np.any(terminated)) or bool(np.any(truncated))
            if done:
                self.stats.ep += 1

                self.logger.scalar(
                    "episode/return", self.stats.return_sum, self.global_ts
                )
                self.logger.scalar("episode/length", self.stats.steps, self.global_ts)
                self.logger.scalar("episode/goals", self.stats.goals, self.global_ts)
                self.logger.scalar(
                    "episode/ball_touches", self.stats.ball_touches, self.global_ts
                )
                self.logger.scalar("episode/shots", self.stats.shots, self.global_ts)
                self.logger.scalar(
                    "episode/power_hits", self.stats.power_hits, self.global_ts
                )
                self.logger.scalar(
                    "episode/defenses", self.stats.defenses, self.global_ts
                )

                if self.stats.ball_touches > 0:
                    self.logger.scalar(
                        "episode/shot_rate",
                        self.stats.shots / self.stats.ball_touches,
                        self.global_ts,
                    )
                    self.logger.scalar(
                        "episode/power_hit_rate",
                        self.stats.power_hits / self.stats.ball_touches,
                        self.global_ts,
                    )
                    self.logger.scalar(
                        "episode/defense_rate",
                        self.stats.defenses / self.stats.ball_touches,
                        self.global_ts,
                    )

                if self.stats.power_hits > 0:
                    self.logger.scalar(
                        "episode/avg_power_delta",
                        self.stats.power_sum / self.stats.power_hits,
                        self.global_ts,
                    )

                self.logger.flush()

                if self.stats.ep % self.print_every == 0:
                    avg_pow = (
                        (self.stats.power_sum / self.stats.power_hits)
                        if self.stats.power_hits
                        else 0.0
                    )
                    print(
                        f"[ep {self.stats.ep:6d}] "
                        f"ret={self.stats.return_sum:8.1f} "
                        f"len={self.stats.steps:4d} "
                        f"goals={self.stats.goals} "
                        f"touches={self.stats.ball_touches} "
                        f"shots={self.stats.shots} "
                        f"power_hits={self.stats.power_hits} "
                        f"avg_powΔ={avg_pow:6.1f} "
                        f"defenses={self.stats.defenses} "
                        f"ts={self.global_ts}"
                    )

                self.stats.reset_episode_counters()

        return obs, reward, terminated, truncated, info

    def __getattr__(self, name):
        return getattr(self.env, name)


# ==================================================
# Rewards
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

        scoring_team = state.scoring_team  # 0 blue, 1 orange
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
    """
    Reward ball progress toward opponent goal, per-agent (fixed).
    """

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
                -common_values.BACK_NET_Y if car.is_orange else common_values.BACK_NET_Y
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
    """
    Utility base to provide 'new touch this step' detection per agent.
    """

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
    """
    Reward ball velocity toward opponent goal AFTER a *new* touch.
    """

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
    """
    Reward 'power' on a *new* touch: increase in ball speed since last step.
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
            # only reward meaningful positive speed deltas
            rewards[a] = float(np.clip(delta_speed / 3000.0, 0.0, 0.2))
        return rewards


class SuccessfulDefenseReward(TouchBasedRewardBase):
    """
    Reward a *new* touch that reduces immediate danger to own net.
    (Same core heuristic as the logger, but continuous and small.)
    """

    def reset(self, agents, initial_state, shared_info):
        super().reset(agents, initial_state, shared_info)
        self.prev_ball_vel_y = (
            0.0
            if initial_state is None
            else float(initial_state.ball.linear_velocity[1])
        )
        self.prev_dist_to_blue = (
            0.0
            if initial_state is None
            else float(
                np.linalg.norm(
                    initial_state.ball.position
                    - np.array([0.0, common_values.BACK_NET_Y, 0.0])
                )
            )
        )
        self.prev_dist_to_orange = (
            0.0
            if initial_state is None
            else float(
                np.linalg.norm(
                    initial_state.ball.position
                    - np.array([0.0, -common_values.BACK_NET_Y, 0.0])
                )
            )
        )

    def get_rewards(
        self, agents, state: GameState, is_terminated, is_truncated, shared_info
    ):
        ball = state.ball
        vel_y = float(ball.linear_velocity[1])
        dist_to_blue = float(
            np.linalg.norm(
                ball.position - np.array([0.0, common_values.BACK_NET_Y, 0.0])
            )
        )
        dist_to_orange = float(
            np.linalg.norm(
                ball.position - np.array([0.0, -common_values.BACK_NET_Y, 0.0])
            )
        )

        rewards: Dict[AgentID, float] = {}

        for a in agents:
            if not self._new_touch(a, state):
                rewards[a] = 0.0
                continue

            car = state.cars[a]
            own_dist_prev = (
                self.prev_dist_to_orange if car.is_orange else self.prev_dist_to_blue
            )
            own_dist_now = dist_to_orange if car.is_orange else dist_to_blue

            moving_toward_own = (
                (self.prev_ball_vel_y < -300.0)
                if car.is_orange
                else (self.prev_ball_vel_y > 300.0)
            )
            dangerous_close = own_dist_prev < 2500.0

            dist_increase = (own_dist_now - own_dist_prev) > 250.0
            vel_flip_away = (vel_y > 200.0) if car.is_orange else (vel_y < -200.0)
            vel_reduced = abs(vel_y) < abs(self.prev_ball_vel_y) - 300.0

            if (
                dangerous_close
                and moving_toward_own
                and (dist_increase or vel_flip_away or vel_reduced)
            ):
                rewards[a] = 0.1
            else:
                rewards[a] = 0.0

        # update baselines once per step
        self.prev_ball_vel_y = vel_y
        self.prev_dist_to_blue = dist_to_blue
        self.prev_dist_to_orange = dist_to_orange

        return rewards


class SpeedTowardBallReward(RewardFunction):
    """
    Reward reducing distance to the ball, with deadzone.
    """

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

            if abs(delta) < 5.0:
                delta = 0.0

            rewards[agent] = float(np.clip(delta / 500.0, 0.0, 0.05))
            self.prev_dist[agent] = dist

        return rewards


class NoTouchProximityPenalty(RewardFunction):
    """
    Penalize stalling near the ball WITHOUT touching it, gated by low speed.
    (This is intentionally mild so it doesn't teach "avoid the ball".)
    """

    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(
        self, agents, state: GameState, is_terminated, is_truncated, shared_info
    ):
        rewards: Dict[AgentID, float] = {}

        for agent in agents:
            car = state.cars[agent]
            dist = float(np.linalg.norm(car.physics.position - state.ball.position))
            speed = float(np.linalg.norm(car.physics.linear_velocity))

            # Only penalize if close, no touches yet, AND basically stalling.
            if dist < 800.0 and int(car.ball_touches) == 0 and speed < 300.0:
                rewards[agent] = -0.002
            else:
                rewards[agent] = 0.0

        return rewards


class StepPenalty(RewardFunction):
    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(
        self, agents, state: GameState, is_terminated, is_truncated, shared_info
    ):
        return {agent: -0.001 for agent in agents}


# ==================================================
# Observation (FIXED SHAPE)
# ==================================================


class SharedObs(ObsBuilder):
    """
    Explicit, symmetric observation vector.
    """

    OBS_SIZE = 47

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

    @staticmethod
    def _dir_dist(a: np.ndarray, b: np.ndarray):
        d = b - a
        dist = float(np.linalg.norm(d))
        if dist > 1e-6:
            return (d / dist).astype(np.float32), float(dist)
        return np.zeros(3, dtype=np.float32), 0.0

    def reset(self, agents, initial_state, shared_info):
        pass

    def get_obs_space(self, agent):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.OBS_SIZE,),
            dtype=np.float32,
        ), None

    def build_obs(self, agents, state: GameState, shared_info):
        obs: Dict[AgentID, np.ndarray] = {}

        for agent in agents:
            car = state.cars[agent]

            car_phys = car.physics if not car.is_orange else car.inverted_physics
            ball_phys = state.ball if not car.is_orange else state.inverted_ball

            # --------------------------------------------------
            # Find closest opponent (if any)
            # --------------------------------------------------
            closest_opp = None
            closest_opp_dist = float("inf")

            for other in agents:
                if other == agent:
                    continue

                other_car = state.cars[other]
                if other_car.is_orange == car.is_orange:
                    continue  # same team → ignore

                other_phys = (
                    other_car.physics
                    if not car.is_orange
                    else other_car.inverted_physics
                )

                d = np.linalg.norm(other_phys.position - car_phys.position)
                if d < closest_opp_dist:
                    closest_opp_dist = d
                    closest_opp = other

            # --------------------------------------------------
            # Opponent features (zero-filled if none)
            # --------------------------------------------------
            if closest_opp is not None:
                opp_car = state.cars[closest_opp]
                opp_phys = (
                    opp_car.physics if not car.is_orange else opp_car.inverted_physics
                )

                opp_pos = (opp_phys.position - car_phys.position) * self.pos_coef
                opp_vel = (
                    opp_phys.linear_velocity - car_phys.linear_velocity
                ) * self.car_vel_coef
                opp_boost = np.array(
                    [opp_car.boost_amount * self.boost_coef], dtype=np.float32
                )
                opp_ground = np.array([float(opp_car.on_ground)], dtype=np.float32)

                to_opp_dir, to_opp_dist = self._dir_dist(
                    car_phys.position, opp_phys.position
                )
                to_opp_dist = np.array([to_opp_dist * self.dist_coef], dtype=np.float32)
            else:
                # single-agent or no opponents
                opp_pos = np.zeros(3, dtype=np.float32)
                opp_vel = np.zeros(3, dtype=np.float32)
                opp_boost = np.zeros(1, dtype=np.float32)
                opp_ground = np.zeros(1, dtype=np.float32)
                to_opp_dir = np.zeros(3, dtype=np.float32)
                to_opp_dist = np.zeros(1, dtype=np.float32)

            # --------------------------------------------------
            # Ball relations
            # --------------------------------------------------
            to_ball_dir, to_ball_dist = self._dir_dist(
                car_phys.position, ball_phys.position
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

            approach_speed = np.array(
                [
                    np.dot(car_phys.linear_velocity, to_ball_dir)
                    / common_values.CAR_MAX_SPEED
                ],
                dtype=np.float32,
            )

            # --------------------------------------------------
            # Final observation vector (FIXED SIZE)
            # --------------------------------------------------
            vec = np.concatenate(
                [
                    # self (17)
                    car_phys.position * self.pos_coef,  # 3
                    car_phys.linear_velocity * self.car_vel_coef,  # 3
                    car_phys.forward.astype(np.float32),  # 3
                    car_phys.up.astype(np.float32),  # 3
                    car_phys.angular_velocity * self.ang_vel_coef,  # 3
                    np.array([car.boost_amount * self.boost_coef], np.float32),  # 1
                    np.array([float(car.on_ground)], np.float32),  # 1
                    # ball (9)
                    ball_phys.position * self.pos_coef,  # 3
                    ball_phys.linear_velocity * self.ball_vel_coef,  # 3
                    rel_ball_vel.astype(np.float32),  # 3
                    # closest opponent (8)
                    opp_pos,  # 3
                    opp_vel,  # 3
                    opp_boost,  # 1
                    opp_ground,  # 1
                    # relations (13)
                    to_ball_dir.astype(np.float32),  # 3
                    np.array([to_ball_dist * self.dist_coef], np.float32),  # 1
                    to_opp_dir.astype(np.float32),  # 3
                    to_opp_dist,  # 1
                    ball_to_goal_dir.astype(np.float32),  # 3
                    np.array([ball_to_goal_dist * self.dist_coef], np.float32),  # 1
                    approach_speed,  # 1
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
# Env builder
# ==================================================


def build_rlgym_v2_env():
    # Lower repeats helps a *lot* with early learning/control
    action_parser = RepeatAction(LookupTableAction(), repeats=4)

    reward_fn = CombinedReward(
        # win condition
        (SignedGoalReward(), 10.0),
        (FastGoalBonus(), 5.0),
        # shaping toward scoring
        (BallNetProgressReward(), 1.0),
        # commitment signals (new-touch based)
        (ShotReward(), 0.5),
        (PowerHitReward(), 0.25),
        (TouchReward(), 0.5),
        # defense shaping (small)
        (SuccessfulDefenseReward(), 0.25),
        # approach shaping
        (SpeedTowardBallReward(), 0.1),
        # anti-stall (mild + gated)
        (NoTouchProximityPenalty(), 0.2),
        (StepPenalty(), 1.0),
    )

    env = RLGym(
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(1, 0),  # how many players
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

    logger = TBLogger("runs/shot_bot_v2")
    wrapped = GymEpisodeLoggingWrapper(base, logger, print_every_episodes=10)
    return wrapped


# ==================================================
# Training
# ==================================================


def main():
    learner = Learner(
        build_rlgym_v2_env,
        n_proc=100,
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
