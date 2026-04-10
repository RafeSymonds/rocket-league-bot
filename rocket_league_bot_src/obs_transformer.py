from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from gymnasium import spaces
from rlgym.api import AgentID, ObsBuilder
from rlgym.rocket_league import common_values
from rlgym.rocket_league.api import GameState

from .config import (
    EARL_EMBED_DIM,
    EARL_ENTITY_COUNT,
    EARL_KV_FEATURES,
    EARL_NUM_HEADS,
    EARL_NUM_LAYERS,
    EARL_QUERY_FEATURES,
    NUM_BOOSTS,
    MAX_PLAYERS,
)


IS_SELF, IS_MATE, IS_OPP, IS_BALL, IS_BOOST = range(5)
POS = slice(5, 8)
LIN_VEL = slice(8, 11)
FW = slice(11, 14)
UP = slice(14, 17)
ANG_VEL = slice(17, 20)
BOOST, DEMO, ON_GROUND, HAS_FLIP, HAS_JUMP = range(20, 25)
ACTIONS = slice(25, 33)
GOAL_DIFF, TIME_LEFT, IS_OVERTIME = range(33, 36)


BOOST_LOCATIONS = np.array(
    [
        [0, -4096, 0],
        [0, 4096, 0],
        [-1024, -2560, 0],
        [1024, -2560, 0],
        [-1024, 2560, 0],
        [1024, 2560, 0],
        [-2048, 0, 0],
        [2048, 0, 0],
        [-3072, -1638, 0],
        [3072, -1638, 0],
        [-3072, 1638, 0],
        [3072, 1638, 0],
        [-4096, -2560, 0],
        [0, -2560, 0],
        [4096, -2560, 0],
        [-4096, 2560, 0],
        [0, 2560, 0],
        [4096, 2560, 0],
        [-1872, -3706, 0],
        [1872, -3706, 0],
        [-1872, 3706, 0],
        [1872, 3706, 0],
        [-3584, -496, 0],
        [3584, -496, 0],
        [-3584, 496, 0],
        [3584, 496, 0],
        [-496, -4688, 0],
        [496, -4688, 0],
        [-496, 4688, 0],
        [496, 4688, 0],
        [-2648, -1176, 0],
        [2648, -1176, 0],
        [-2648, 1176, 0],
        [2648, 1176, 0],
    ],
    dtype=np.float32,
)


NORM = np.array(
    [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        2300.0,
        2300.0,
        2300.0,
        2300.0,
        2300.0,
        2300.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        5.5,
        5.5,
        5.5,
        1.0,
        10.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ],
    dtype=np.float32,
)


INVERT = np.array(
    [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ],
    dtype=np.float32,
)


class TransformerObs(ObsBuilder):
    def __init__(self):
        super().__init__()
        self._boost_locations = BOOST_LOCATIONS
        self.demo_timers = None
        self.boost_timers = None
        self.blue_score = 0
        self.orange_score = 0
        self._prev_goal_scored = False

    def _reset(self, initial_state: GameState):
        n_players = len(initial_state.cars)
        self.demo_timers = np.zeros(n_players, dtype=np.float32)
        self.boost_timers = np.zeros(len(self._boost_locations), dtype=np.float32)
        self.blue_score = 0
        self.orange_score = 0
        self._prev_goal_scored = False

    def reset(self, agents, initial_state: GameState, shared_info):
        self._reset(initial_state)

    def get_obs_space(self, agent) -> Tuple[spaces.Box, None]:
        return spaces.Tuple(
            (
                spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(1, 1, EARL_QUERY_FEATURES),
                    dtype=np.float32,
                ),
                spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(EARL_ENTITY_COUNT, EARL_KV_FEATURES),
                    dtype=np.float32,
                ),
                spaces.Box(
                    low=0.0, high=1.0, shape=(EARL_ENTITY_COUNT,), dtype=np.float32
                ),
            )
        ), None

    def build_obs(
        self, agents, state: GameState, shared_info
    ) -> Dict[AgentID, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if state.goal_scored and not self._prev_goal_scored:
            if state.scoring_team == 0:  # BLUE
                self.blue_score += 1
            else:  # ORANGE
                self.orange_score += 1
        self._prev_goal_scored = bool(state.goal_scored)

        obs = {}
        n_agents = len(agents)
        n_entities = n_agents + 1 + len(self._boost_locations)

        ball_phys = state.ball
        inverted_ball = state.inverted_ball

        blue_score = float(self.blue_score)
        orange_score = float(self.orange_score)
        # RLGym v2 GameState doesn't have ticks_left easily accessible
        ticks_left = float("inf") 
        is_overtime = False
        goal_diff = np.clip(blue_score - orange_score, -5, 5) / 5.0
        time_left = 0.0 # Default since we don't know match duration easily

        q = np.zeros((n_agents, 1, 1, EARL_QUERY_FEATURES), dtype=np.float32)
        kv = np.zeros((n_agents, n_entities, EARL_KV_FEATURES), dtype=np.float32)
        m = np.zeros((n_agents, n_entities), dtype=np.float32)

        ball_index = n_agents
        boost_start = ball_index + 1

        agent_to_idx = {agent: idx for idx, agent in enumerate(agents)}
        teams = np.array([int(state.cars[a].is_orange) for a in agents], dtype=np.int32)

        for agent_idx, agent in enumerate(agents):
            car = state.cars[agent]
            car_phys = car.inverted_physics if car.is_orange else car.physics
            is_orange = int(car.is_orange)

            forward = car_phys.forward.astype(np.float32)
            up = car_phys.up.astype(np.float32)
            pos = np.array(car_phys.position, dtype=np.float32)
            lin_vel = np.array(car_phys.linear_velocity, dtype=np.float32)
            ang_vel = np.array(car_phys.angular_velocity, dtype=np.float32)

            if is_orange:
                pos[..., :2] *= -1
                lin_vel[..., :2] *= -1
                ang_vel[..., :2] *= -1
                forward[..., :2] *= -1
                up[..., :2] *= -1

            kv[agent_idx, agent_idx, IS_SELF] = 1
            kv[agent_idx, agent_idx, POS] = pos / 2300.0
            kv[agent_idx, agent_idx, LIN_VEL] = lin_vel / 2300.0
            kv[agent_idx, agent_idx, FW] = forward
            kv[agent_idx, agent_idx, UP] = up
            kv[agent_idx, agent_idx, ANG_VEL] = ang_vel / 5.5
            kv[agent_idx, agent_idx, BOOST] = np.clip(car.boost_amount, 0, 100) / 100.0
            kv[agent_idx, agent_idx, DEMO] = float(car.is_demoed)
            kv[agent_idx, agent_idx, ON_GROUND] = float(car.on_ground)
            kv[agent_idx, agent_idx, HAS_FLIP] = float(getattr(car, "has_flip", True))
            kv[agent_idx, agent_idx, HAS_JUMP] = float(
                car.has_jumped or car.has_double_jumped
            )
            m[agent_idx, agent_idx] = 1.0

            q[agent_idx, 0, 0, IS_SELF:IS_MATE] = 1
            q[agent_idx, 0, 0, POS] = pos / 2300.0
            q[agent_idx, 0, 0, LIN_VEL] = lin_vel / 2300.0
            q[agent_idx, 0, 0, FW] = forward
            q[agent_idx, 0, 0, UP] = up
            q[agent_idx, 0, 0, ANG_VEL] = ang_vel / 5.5
            q[agent_idx, 0, 0, BOOST] = np.clip(car.boost_amount, 0, 100) / 100.0
            q[agent_idx, 0, 0, DEMO] = float(car.is_demoed)
            q[agent_idx, 0, 0, ON_GROUND] = float(car.on_ground)
            q[agent_idx, 0, 0, HAS_FLIP] = float(getattr(car, "has_flip", True))
            q[agent_idx, 0, 0, HAS_JUMP] = float(
                car.has_jumped or car.has_double_jumped
            )

            q[agent_idx, 0, 0, GOAL_DIFF] = goal_diff if not is_orange else -goal_diff
            q[agent_idx, 0, 0, TIME_LEFT] = time_left
            q[agent_idx, 0, 0, IS_OVERTIME] = float(is_overtime)

        for agent_idx, agent in enumerate(agents):
            car = state.cars[agent]
            is_orange = int(car.is_orange)

            cur_ball_phys = inverted_ball if is_orange else ball_phys

            ball_pos = np.array(cur_ball_phys.position, dtype=np.float32)
            ball_vel = np.array(cur_ball_phys.linear_velocity, dtype=np.float32)
            ball_ang_vel = np.array(cur_ball_phys.angular_velocity, dtype=np.float32)

            if is_orange:
                ball_pos[..., :2] *= -1
                ball_vel[..., :2] *= -1
                ball_ang_vel[..., :2] *= -1

            kv[agent_idx, ball_index, IS_BALL] = 1
            kv[agent_idx, ball_index, POS] = ball_pos / 2300.0
            kv[agent_idx, ball_index, LIN_VEL] = ball_vel / 2300.0
            kv[agent_idx, ball_index, FW] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            kv[agent_idx, ball_index, UP] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            kv[agent_idx, ball_index, ANG_VEL] = ball_ang_vel / 5.5
            m[agent_idx, ball_index] = 1.0

            for boost_idx, boost_loc in enumerate(self._boost_locations):
                boost_pos = boost_loc.copy()
                if is_orange:
                    boost_pos[..., :2] *= -1
                boost_available = 1.0 if self.boost_timers[boost_idx] <= 0 else 0.0
                kv[agent_idx, boost_start + boost_idx, IS_BOOST] = 1
                kv[agent_idx, boost_start + boost_idx, POS] = boost_pos / 2300.0
                kv[agent_idx, boost_start + boost_idx, DEMO] = boost_available
                m[agent_idx, boost_start + boost_idx] = 1.0

        for agent_idx, agent in enumerate(agents):
            car = state.cars[agent]
            is_orange = int(car.is_orange)

            for other_idx, other in enumerate(agents):
                if other == agent:
                    continue
                other_car = state.cars[other]
                other_is_orange = int(other_car.is_orange)

                is_teammate = is_orange == other_is_orange

                other_phys = (
                    other_car.inverted_physics if is_orange else other_car.physics
                )
                other_pos = np.array(other_phys.position, dtype=np.float32)
                other_vel = np.array(other_phys.linear_velocity, dtype=np.float32)
                other_ang_vel = np.array(other_phys.angular_velocity, dtype=np.float32)
                other_fwd = other_phys.forward.astype(np.float32)
                other_up = other_phys.up.astype(np.float32)

                if is_orange:
                    other_pos[..., :2] *= -1
                    other_vel[..., :2] *= -1
                    other_ang_vel[..., :2] *= -1
                    other_fwd[..., :2] *= -1
                    other_up[..., :2] *= -1

                kv[agent_idx, other_idx, IS_MATE if is_teammate else IS_OPP] = 1
                kv[agent_idx, other_idx, POS] = other_pos / 2300.0
                kv[agent_idx, other_idx, LIN_VEL] = other_vel / 2300.0
                kv[agent_idx, other_idx, FW] = other_fwd
                kv[agent_idx, other_idx, UP] = other_up
                kv[agent_idx, other_idx, ANG_VEL] = other_ang_vel / 5.5
                kv[agent_idx, other_idx, BOOST] = (
                    np.clip(other_car.boost_amount, 0, 100) / 100.0
                )
                kv[agent_idx, other_idx, DEMO] = float(other_car.is_demoed)
                kv[agent_idx, other_idx, ON_GROUND] = float(other_car.on_ground)
                kv[agent_idx, other_idx, HAS_FLIP] = float(
                    getattr(other_car, "has_flip", True)
                )
                kv[agent_idx, other_idx, HAS_JUMP] = float(
                    other_car.has_jumped or other_car.has_double_jumped
                )
                m[agent_idx, other_idx] = 1.0

        for agent_idx in range(n_agents):
            kv[agent_idx] *= INVERT
            kv[agent_idx] /= NORM

        for agent_idx, agent in enumerate(agents):
            obs[agent] = (q[agent_idx], kv[agent_idx], m[agent_idx])

        return obs
