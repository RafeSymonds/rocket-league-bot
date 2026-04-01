from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Stats:
    touch_rate: float = 0.0
    goal_rate: float = 0.0
    median_t_first: float = 0.0
    median_t_goal: float = 0.0
