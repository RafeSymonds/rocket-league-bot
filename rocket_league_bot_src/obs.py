from __future__ import annotations

from typing import Dict

import numpy as np
from gymnasium import spaces

from rlgym.api import AgentID, ObsBuilder
from rlgym.rocket_league import common_values
from rlgym.rocket_league.api import GameState

from .config import OBS_DIM


class SharedObs(ObsBuilder):
    OBS_SIZE = OBS_DIM

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
