from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Stats:
    touch_rate: float = 0.0
    goal_rate: float = 0.0
    blue_goal_rate: float = 0.0
    orange_goal_rate: float = 0.0
    median_t_first: float = 0.0
    median_t_goal: float = 0.0
    aerial_touch_rate: float = 0.0
    goal_side_rate: float = 0.0
    behind_ball_rate: float = 0.0
