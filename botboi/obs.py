"""Observation builder shared by training and the RLBot runtime.

Training passes an rlgym ``GameState`` from RocketSim. The RLBot bot passes an
``rlgym_compat.GameState`` built from game packets. Both expose the same
attribute names, so this exact code runs in both places, and the parity tests
in ``tests/test_parity.py`` check that they agree.

Everything is team-relative: orange agents see the field rotated 180 degrees
about the z axis, so every agent attacks toward +y.

Layout (OBS_SIZE floats):
    ball              9   position, linear velocity, angular velocity
    boost pads       34   seconds until the pad respawns / 10 (0 = available)
    self car         21   car block, see _car_block
    self extras       9   jump and flip state (see _self_extras)
    ball vs self      6   ball position and velocity in the car's local frame
    teammate slots   28 each (MAX_TEAM_SIZE - 1 slots)
    opponent slots   28 each (MAX_TEAM_SIZE slots)

Changing the layout invalidates every trained checkpoint. Bump OBS_VERSION
when you do, so export and the bot refuse to mix versions.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from rlgym.api import AgentID, ObsBuilder
from rlgym.rocket_league.common_values import ORANGE_TEAM

OBS_VERSION = 1
MAX_TEAM_SIZE = 2
NUM_BOOST_PADS = 34

POS_COEF = np.array([1 / 4096, 1 / 5120, 1 / 2044], dtype=np.float32)
VEL_COEF = 1 / 2300
ANG_VEL_COEF = 1 / math.pi
LOCAL_POS_COEF = 1 / 4096
PAD_TIMER_COEF = 1 / 10
DEMO_TIMER_COEF = 1 / 3
FLIP_WINDOW_SECONDS = 1.25

CAR_BLOCK_SIZE = 21
SELF_EXTRAS_SIZE = 9
SLOT_SIZE = 1 + CAR_BLOCK_SIZE + 6
OBS_SIZE = (
    9
    + NUM_BOOST_PADS
    + CAR_BLOCK_SIZE
    + SELF_EXTRAS_SIZE
    + 6
    + (2 * MAX_TEAM_SIZE - 1) * SLOT_SIZE
)

_EMPTY_SLOT = np.zeros(SLOT_SIZE, dtype=np.float32)


def _car_block(car, physics) -> np.ndarray:
    return np.concatenate(
        [
            physics.position * POS_COEF,
            physics.forward,
            physics.up,
            physics.linear_velocity * VEL_COEF,
            physics.angular_velocity * ANG_VEL_COEF,
            [
                car.boost_amount / 100,
                car.on_ground,
                car.has_flip,
                car.demo_respawn_timer * DEMO_TIMER_COEF,
                # RLBot reports boosting whenever boost is held, even on an
                # empty tank; the simulator does not. This form agrees in both.
                car.is_boosting and car.boost_amount > 0,
                car.is_supersonic,
            ],
        ],
        dtype=np.float32,
    )


def _flip_window_left(car) -> float:
    """Fraction of the post-jump flip window remaining, 0 once no flip is
    available. (Raw air time is not used: RLBot stops updating it after a
    flip, so it would differ from training.)"""
    if not (car.has_jumped and car.has_flip):
        return 0.0
    return 1.0 - min(car.air_time_since_jump, FLIP_WINDOW_SECONDS) / FLIP_WINDOW_SECONDS


def _self_extras(car) -> np.ndarray:
    return np.array(
        [
            car.is_holding_jump,
            car.handbrake,
            car.has_jumped,
            car.is_jumping,
            car.has_flipped,
            car.is_flipping,
            car.has_double_jumped,
            car.can_flip,
            _flip_window_left(car),
        ],
        dtype=np.float32,
    )


class BotObs(ObsBuilder[AgentID, np.ndarray, Any, tuple[str, int]]):
    def get_obs_space(self, agent: AgentID) -> tuple[str, int]:
        return "real", OBS_SIZE

    def reset(self, agents, initial_state, shared_info: dict[str, Any]) -> None:
        pass

    def build_obs(self, agents, state, shared_info: dict[str, Any]) -> dict:
        # Car blocks depend only on (car, frame), so share them across agents.
        block_cache: dict[tuple[Any, bool], np.ndarray] = {}
        return {agent: self._build(agent, state, block_cache) for agent in agents}

    def _build(self, agent, state, block_cache) -> np.ndarray:
        car = state.cars[agent]
        inverted = car.team_num == ORANGE_TEAM
        ball = state.inverted_ball if inverted else state.ball
        pads = state.inverted_boost_pad_timers if inverted else state.boost_pad_timers
        phys = car.inverted_physics if inverted else car.physics
        # Columns are forward, right, up, so `v @ rot` expresses v in the car's
        # local frame. Local-frame values do not change under the field flip.
        rot = phys.rotation_mtx

        def block(car_id, other_car, other_phys):
            key = (car_id, inverted)
            if key not in block_cache:
                block_cache[key] = _car_block(other_car, other_phys)
            return block_cache[key]

        allies = []
        enemies = []
        for other_id, other in state.cars.items():
            if other_id == agent:
                continue
            other_phys = other.inverted_physics if inverted else other.physics
            rel_pos = other_phys.position - phys.position
            slot = np.concatenate(
                [
                    [1.0],
                    block(other_id, other, other_phys),
                    (rel_pos @ rot) * LOCAL_POS_COEF,
                    ((other_phys.linear_velocity - phys.linear_velocity) @ rot) * VEL_COEF,
                ],
                dtype=np.float32,
            )
            group = allies if other.team_num == car.team_num else enemies
            group.append((float(rel_pos @ rel_pos), slot))

        parts = [
            ball.position * POS_COEF,
            ball.linear_velocity * VEL_COEF,
            ball.angular_velocity * ANG_VEL_COEF,
            np.asarray(pads, dtype=np.float32) * PAD_TIMER_COEF,
            block(agent, car, phys),
            _self_extras(car),
            ((ball.position - phys.position) @ rot) * LOCAL_POS_COEF,
            ((ball.linear_velocity - phys.linear_velocity) @ rot) * VEL_COEF,
        ]
        # Nearest cars fill the slots first. Extra cars (e.g. a 3v3 match) are
        # dropped, and missing cars are zero slots with presence 0.
        for group, n_slots in ((allies, MAX_TEAM_SIZE - 1), (enemies, MAX_TEAM_SIZE)):
            group.sort(key=lambda item: item[0])
            parts.extend(slot for _, slot in group[:n_slots])
            parts.extend(_EMPTY_SLOT for _ in range(n_slots - min(len(group), n_slots)))

        return np.concatenate(parts, dtype=np.float32)
