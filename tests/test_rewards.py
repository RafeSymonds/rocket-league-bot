from __future__ import annotations

import copy

import numpy as np
import pytest
from rlgym.rocket_league.common_values import BACK_NET_Y, BALL_RADIUS

from botboi.config import PHASES, EnvConfig
from botboi.env import build_env
from botboi.rewards import build_reward

ZERO_WEIGHTS = {name: 0.0 for name in PHASES["early"]["rewards"]}


def env_with_reward(team_size: int, weights: dict, team_spirit: float = 0.0):
    env = build_env(EnvConfig(team_size_weights={team_size: 1.0}, kickoff_prob=0.0))
    env.reward_fn = build_reward(weights, team_spirit)
    return env


@pytest.mark.parametrize("team_size", [1, 2])
def test_goal_rewards_scorer_and_punishes_conceder(team_size):
    env = env_with_reward(team_size, {**ZERO_WEIGHTS, "goal": 20.0})
    env.reset()
    state = copy.deepcopy(env.state)
    # Ball just short of the orange goal line, moving in: blue scores next step.
    state.ball.position = np.array([0.0, BACK_NET_Y - 900.0 + BALL_RADIUS, BALL_RADIUS + 50], dtype=np.float32)
    state.ball.linear_velocity = np.array([0.0, 4000.0, 0.0], dtype=np.float32)
    env.transition_engine.set_state(state, {})
    env.reward_fn.reset(env.agents, env.state, env.shared_info)
    for _ in range(30):
        _, rewards, terminated, _ = env.step({a: np.array([0]) for a in env.agents})
        if any(terminated.values()):
            break
    assert any(terminated.values()), "ball never reached the goal"
    for agent, reward in rewards.items():
        expected = 20.0 if env.state.cars[agent].is_blue else -20.0
        assert reward == pytest.approx(expected)


@pytest.mark.parametrize("team_size", [1, 2])
@pytest.mark.parametrize("team_spirit", [0.0, 0.3])
def test_competitive_terms_are_zero_sum(team_size, team_spirit):
    weights = {**ZERO_WEIGHTS, "ball_toward_goal": 1.0, "strong_touch": 5.0, "demo": 8.0}
    env = env_with_reward(team_size, weights, team_spirit)
    rng = np.random.default_rng(0)
    for _ in range(3):
        env.reset()
        for _ in range(150):
            _, rewards, terminated, truncated = env.step(
                {a: np.array([rng.integers(90)]) for a in env.agents}
            )
            assert sum(rewards.values()) == pytest.approx(0.0, abs=1e-6)
            if any(terminated.values()) or any(truncated.values()):
                break


@pytest.mark.parametrize("phase", sorted(PHASES))
def test_phase_rewards_are_finite_and_bounded(phase):
    env = build_env(EnvConfig(phase=phase))
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(4):
        env.reset()
        for _ in range(200):
            _, rewards, terminated, truncated = env.step(
                {a: np.array([rng.integers(90)]) for a in env.agents}
            )
            values = np.array(list(rewards.values()))
            assert np.all(np.isfinite(values))
            worst = max(worst, float(np.abs(values).max()))
            if any(terminated.values()) or any(truncated.values()):
                break
    # Nothing except a goal should come close to the goal reward.
    assert worst <= PHASES[phase]["rewards"]["goal"] * 1.5
