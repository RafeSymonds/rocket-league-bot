from __future__ import annotations

import copy
import random

import numpy as np
import pytest
from rlgym.rocket_league.common_values import BLUE_TEAM, ORANGE_TEAM

from botboi.config import EnvConfig
from botboi.env import build_env
from botboi.obs import (
    CAR_BLOCK_SIZE,
    NUM_BOOST_PADS,
    OBS_SIZE,
    SELF_EXTRAS_SIZE,
    SLOT_SIZE,
    BotObs,
)

FIRST_SLOT = 9 + NUM_BOOST_PADS + CAR_BLOCK_SIZE + SELF_EXTRAS_SIZE + 6
BALL_LOCAL = FIRST_SLOT - 6


def make_env(team_size: int, kickoff_prob: float = 0.0):
    return build_env(EnvConfig(team_size_weights={team_size: 1.0}, kickoff_prob=kickoff_prob))


def slot(obs: np.ndarray, k: int) -> np.ndarray:
    return obs[FIRST_SLOT + k * SLOT_SIZE : FIRST_SLOT + (k + 1) * SLOT_SIZE]


@pytest.mark.parametrize("team_size", [1, 2])
def test_size_and_slots(team_size):
    env = make_env(team_size)
    obs = env.reset()
    assert len(obs) == 2 * team_size
    for vector in obs.values():
        assert vector.shape == (OBS_SIZE,) and vector.dtype == np.float32
        assert np.all(np.isfinite(vector))
        # Slot 0 is the teammate, slots 1-2 are opponents; presence flag first.
        presence = [slot(vector, k)[0] for k in range(3)]
        if team_size == 1:
            assert presence == [0.0, 1.0, 0.0]
            assert not slot(vector, 0).any() and not slot(vector, 2).any()
        else:
            assert presence == [1.0, 1.0, 1.0]


def test_extra_cars_are_dropped():
    # A 3v3 match (never used in training) still yields a valid obs.
    env = make_env(1)
    env.reset()
    state = copy.deepcopy(env.state)
    for name in ("blue-1", "blue-2", "orange-1", "orange-2"):
        state.cars[name] = copy.deepcopy(state.cars["blue-0" if name.startswith("blue") else "orange-0"])
        state.cars[name].physics.position = state.cars[name].physics.position + [300.0 * len(state.cars), 0, 0]
    obs = BotObs().build_obs(list(state.cars), state, {})
    assert all(v.shape == (OBS_SIZE,) for v in obs.values())


def test_ball_in_front_is_positive_local_x():
    env = make_env(1)
    env.reset()
    state = env.state
    car = state.cars["blue-0"]
    car.physics.position = np.array([0.0, 0.0, 17.0], dtype=np.float32)
    car.physics.euler_angles = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # facing +x
    state.ball.position = np.array([1000.0, 0.0, 93.0], dtype=np.float32)
    state = env.transition_engine.set_state(state, {})
    vector = BotObs().build_obs(["blue-0"], state, {})["blue-0"]
    local = vector[BALL_LOCAL : BALL_LOCAL + 3]
    assert local[0] > 0.2 and abs(local[1]) < 1e-3


def mirrored(state):
    """Same situation with the teams swapped and the field rotated 180 degrees."""
    mirror = copy.deepcopy(state)
    for car in mirror.cars.values():
        car.physics = car.physics.inverted()
        car.team_num = ORANGE_TEAM if car.team_num == BLUE_TEAM else BLUE_TEAM
    mirror.ball = mirror.ball.inverted()
    mirror.boost_pad_timers = state.boost_pad_timers[::-1].copy()
    return mirror


@pytest.mark.parametrize("team_size", [1, 2])
def test_orange_sees_the_mirror_of_blue(team_size):
    random.seed(3)
    np.random.seed(3)
    env = make_env(team_size)
    env.reset()
    for _ in range(40):
        env.step({a: np.array([np.random.randint(90)]) for a in env.agents})
    state = copy.deepcopy(env.state)
    state.boost_pad_timers = np.random.uniform(0, 10, NUM_BOOST_PADS).astype(np.float32)
    # Round-trip through the engine so cached inverted values are rebuilt.
    state = env.transition_engine.set_state(state, {})
    mirror = env.transition_engine.set_state(mirrored(state), {})
    obs = BotObs().build_obs(list(state.cars), state, {})
    mirror_obs = BotObs().build_obs(list(mirror.cars), mirror, {})
    for agent in state.cars:
        np.testing.assert_allclose(obs[agent], mirror_obs[agent], atol=2e-3)
