"""Episode start states."""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
from rlgym.api import StateMutator
from rlgym.rocket_league.common_values import BALL_RADIUS

CAR_REST_Z = 17.0
FIELD_X = 3500.0  # inside the side walls with margin
FIELD_Y = 4500.0  # inside the back walls with margin
MIN_CAR_BALL_GAP = 400.0
MIN_CAR_CAR_GAP = 300.0


class RandomGroundStateMutator(StateMutator):
    """Cars upright on the ground at random spots, headings, speeds, and boost.
    The ball rests or rolls on the ground most of the time and is sometimes
    thrown into the air, so the policy sees a spread of real game situations
    beyond kickoffs.

    Apply after a team-size mutator (it only moves cars that already exist).
    """

    def __init__(self, air_ball_prob: float = 0.3):
        self.air_ball_prob = air_ball_prob

    def apply(self, state, shared_info: dict[str, Any]) -> None:
        ball = state.ball
        ball_xy = np.array(
            [random.uniform(-FIELD_X, FIELD_X), random.uniform(-FIELD_Y, FIELD_Y)]
        )
        if random.random() < self.air_ball_prob:
            ball_z = random.uniform(300.0, 1400.0)
            ball_vel = [random.uniform(-1000, 1000), random.uniform(-1000, 1000), random.uniform(-300, 600)]
        else:
            ball_z = BALL_RADIUS
            ball_vel = [random.uniform(-1200, 1200), random.uniform(-1200, 1200), 0.0]
        ball.position = np.array([*ball_xy, ball_z], dtype=np.float32)
        ball.linear_velocity = np.array(ball_vel, dtype=np.float32)
        ball.angular_velocity = np.zeros(3, dtype=np.float32)

        placed: list[np.ndarray] = []
        for car in state.cars.values():
            while True:
                xy = np.array(
                    [random.uniform(-FIELD_X, FIELD_X), random.uniform(-FIELD_Y, FIELD_Y)]
                )
                if np.linalg.norm(xy - ball_xy) < MIN_CAR_BALL_GAP:
                    continue
                if any(np.linalg.norm(xy - other) < MIN_CAR_CAR_GAP for other in placed):
                    continue
                break
            placed.append(xy)
            yaw = random.uniform(-math.pi, math.pi)
            speed = random.uniform(0.0, 1800.0) if random.random() < 0.8 else 0.0
            car.physics.position = np.array([*xy, CAR_REST_Z], dtype=np.float32)
            car.physics.euler_angles = np.array([0.0, yaw, 0.0], dtype=np.float32)
            car.physics.linear_velocity = np.array(
                [math.cos(yaw) * speed, math.sin(yaw) * speed, 0.0], dtype=np.float32
            )
            car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
            car.boost_amount = random.uniform(0.0, 100.0)
