"""Reward functions and the weighted combination used for training.

Weights live in botboi/config.py (REWARD_PHASES). The combination has three
groups:

- goal: +1 for the scoring team, -1 for the conceding team.
- competitive terms are zero-sum between teams (DistributeRewardsWrapper), so
  pushing the ball toward the opponent's goal is exactly the opponent's loss.
  `team_spirit` shares these terms between teammates in 2v2.
- individual terms are plain shaping that each car earns for itself.

Every per-step term below is scaled to roughly [-1, 1] per decision step, so
the weights are directly comparable.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from rlgym.api import RewardFunction
from rlgym.rocket_league.common_values import (
    BACK_NET_Y,
    BALL_MAX_SPEED,
    CAR_MAX_SPEED,
    ORANGE_TEAM,
)
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
from rlgym_tools.rocket_league.reward_functions.advanced_touch_reward import (
    AdvancedTouchReward,
)
from rlgym_tools.rocket_league.reward_functions.boost_change_reward import (
    BoostChangeReward,
)
from rlgym_tools.rocket_league.reward_functions.demo_reward import DemoReward
from rlgym_tools.rocket_league.reward_functions.distribute_rewards_wrapper import (
    DistributeRewardsWrapper,
)

ORANGE_GOAL = np.array([0, BACK_NET_Y, 0], dtype=np.float32)
BLUE_GOAL = -ORANGE_GOAL


class _PerAgentReward(RewardFunction):
    """Base for stateless rewards computed independently per agent."""

    def reset(self, agents, initial_state, shared_info: dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        return {agent: self.reward(state.cars[agent], state) for agent in agents}

    def reward(self, car, state) -> float:
        raise NotImplementedError


class SpeedTowardBallReward(_PerAgentReward):
    """Car velocity toward the ball / max car speed, floored at 0."""

    def reward(self, car, state) -> float:
        to_ball = state.ball.position - car.physics.position
        dist = np.linalg.norm(to_ball)
        if dist < 1e-6:
            return 0.0
        speed = float(car.physics.linear_velocity @ to_ball) / dist
        return max(0.0, speed / CAR_MAX_SPEED)


class FaceBallReward(_PerAgentReward):
    """Cosine between the car's nose and the direction to the ball."""

    def reward(self, car, state) -> float:
        to_ball = state.ball.position - car.physics.position
        dist = np.linalg.norm(to_ball)
        if dist < 1e-6:
            return 0.0
        return float(car.physics.forward @ to_ball) / dist


class InAirReward(_PerAgentReward):
    """1 while the car is off the ground. Tiny weight: it only breaks the early
    habit of never jumping."""

    def reward(self, car, state) -> float:
        return 0.0 if car.on_ground else 1.0


class BallTowardGoalReward(_PerAgentReward):
    """Ball velocity toward the opponent's goal / max ball speed."""

    def reward(self, car, state) -> float:
        target = BLUE_GOAL if car.team_num == ORANGE_TEAM else ORANGE_GOAL
        to_goal = target - state.ball.position
        dist = np.linalg.norm(to_goal)
        if dist < 1e-6:
            return 0.0
        return float(state.ball.linear_velocity @ to_goal) / dist / BALL_MAX_SPEED


class SaveBoostReward(_PerAgentReward):
    """sqrt(boost / 100): values the first boost in the tank the most."""

    def reward(self, car, state) -> float:
        return math.sqrt(car.boost_amount / 100)


def _weighted(terms: dict[str, tuple[RewardFunction, float]]) -> CombinedReward | None:
    active = [(fn, weight) for fn, weight in terms.values() if weight != 0]
    return CombinedReward(*active) if active else None


def build_reward(weights: dict[str, float], team_spirit: float) -> RewardFunction:
    """Combine the reward terms using the weights of one REWARD_PHASES entry."""
    w = weights
    competitive = _weighted(
        {
            "ball_toward_goal": (BallTowardGoalReward(), w["ball_toward_goal"]),
            # Touch acceleration: |delta ball velocity| / max ball speed on touch.
            "strong_touch": (
                AdvancedTouchReward(touch_reward=0.0, acceleration_reward=1.0),
                w["strong_touch"],
            ),
            "demo": (DemoReward(attacker_reward=1.0, victim_punishment=0.0), w["demo"]),
        }
    )
    individual = _weighted(
        {
            "touch": (TouchReward(), w["touch"]),
            "speed_toward_ball": (SpeedTowardBallReward(), w["speed_toward_ball"]),
            "face_ball": (FaceBallReward(), w["face_ball"]),
            "in_air": (InAirReward(), w["in_air"]),
            # sqrt-scaled boost gained this step, no penalty for spending it.
            "boost_pickup": (
                BoostChangeReward(gain_weight=1.0, lose_weight=0.0),
                w["boost_pickup"],
            ),
            "save_boost": (SaveBoostReward(), w["save_boost"]),
        }
    )

    parts: list[tuple[RewardFunction, float]] = [(GoalReward(), w["goal"])]
    if competitive is not None:
        # With the default team_coef = opp_coef = 0.5 each car gets
        # 0.5 * (own-team term - opponent-team term), which sums to zero.
        parts.append(
            (DistributeRewardsWrapper(competitive, selflessness=team_spirit), 1.0)
        )
    if individual is not None:
        parts.append((individual, 1.0))
    return CombinedReward(*parts)
