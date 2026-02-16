from __future__ import annotations

from typing import Any

import numpy as np
from rlgym.api import StateMutator
from rlgym.rocket_league import common_values
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator
from typing_extensions import override

from .config import make_stage_config
from .utils import CurriculumValue


class DynamicTeamSizeMutator(StateMutator):
    """Apply stage-dependent team sizes at reset time."""

    def __init__(self, stage_ref: "EnvBuilder"):
        self.stage_ref = stage_ref
        self._cache: dict[tuple[int, int], FixedTeamSizeMutator] = {}

    @override
    def apply(self, state: GameState, shared_info: dict[str, Any]) -> None:
        cfg = make_stage_config(self.stage_ref.stage)
        key = (cfg.blue_players, cfg.orange_players)
        if key not in self._cache:
            self._cache[key] = FixedTeamSizeMutator(*key)
        self._cache[key].apply(state, shared_info)


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

        min_d = float(self.min_dist.get())
        max_d = float(self.max_dist.get())
        if max_d < min_d:
            max_d = min_d

        dist = np.random.uniform(min_d, max_d)
        max_angle = max(0.0, float(self.max_angle.get()))
        angle = np.deg2rad(np.random.uniform(-max_angle, max_angle))

        dir_vec = np.cos(angle) * forward + np.sin(angle) * right
        n = float(np.linalg.norm(dir_vec))
        if n < 1e-6:
            dir_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            dir_vec = dir_vec / n

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

        v = max(0.0, float(self.ball_velocity.get()))
        if v > 0.0:
            vel_dir = np.array(
                [np.random.uniform(-1, 1), np.random.uniform(-1, 1), 0.0],
                dtype=np.float32,
            )
            vel_norm = float(np.linalg.norm(vel_dir))
            if vel_norm < 1e-6:
                vel_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            else:
                vel_dir /= vel_norm
            state.ball.linear_velocity[:] = vel_dir * v
        else:
            state.ball.linear_velocity[:] = 0.0


class ProgressiveResetMutator(StateMutator):
    def __init__(self, easy_mutator: StateMutator, p_easy: CurriculumValue):
        self.easy_mutator = easy_mutator
        self.p_easy = p_easy

    @override
    def apply(self, state: GameState, shared_info: dict[str, Any]) -> None:
        if np.random.rand() < float(self.p_easy.get()):
            self.easy_mutator.apply(state, shared_info)
        # else: keep kickoff/default reset
