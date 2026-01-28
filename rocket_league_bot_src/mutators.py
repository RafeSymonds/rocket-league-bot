from __future__ import annotations

from typing import Any

import numpy as np
from rlgym.api import StateMutator
from rlgym.rocket_league import common_values
from rlgym.rocket_league.api import GameState
from typing_extensions import override

from .utils import CurriculumValue


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
