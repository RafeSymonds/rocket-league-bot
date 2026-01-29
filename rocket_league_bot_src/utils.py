from __future__ import annotations
from dataclasses import dataclass


class CurriculumValue:
    def __init__(self, initial_value):
        self._value = initial_value

    def get(self):
        return self._value

    def set(self, new_value):
        self._value = new_value


@dataclass
class Stats:
    touch_rate: float = 0
    goal_rate: float = 0
    median_t_first: float = 0
    median_t_goal: float = 0
