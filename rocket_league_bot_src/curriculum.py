from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Stage, StageConfig, build_stage_config


@dataclass
class CurriculumSnapshot:
    stage_iters: float
    ema_touch: float
    ema_goal: float
    difficulty: float


class CurriculumManager:
    """
    Explicit stage controller:

    CONTACT -> learn to reach and touch the ball from varied placements.
    SHOOT -> learn to convert open-net scenarios.
    SELF_PLAY -> continue from mixed resets in 1v1.
    """

    def __init__(self):
        self.stage = Stage.CONTACT
        self.difficulty = 0.0
        self.stage_iterations = 0
        self.ema_touch_rate = 0.0
        self.ema_goal_rate = 0.0

    def current_config(self) -> StageConfig:
        return build_stage_config(self.stage, self.difficulty)

    @staticmethod
    def _smooth(current: float, target: float, alpha: float = 0.18) -> float:
        return (1.0 - alpha) * current + alpha * target

    def _set_stage(self, stage: Stage) -> None:
        if self.stage == stage:
            return
        self.stage = stage
        self.difficulty = 0.0
        self.stage_iterations = 0
        print(f"Curriculum stage -> {stage.value}")

    def _update_contact(self, stats) -> None:
        touch_skill = np.clip((stats.touch_rate - 0.30) / 0.60, 0.0, 1.0)
        speed_skill = np.clip((180.0 - stats.median_t_first) / 120.0, 0.0, 1.0)
        target_difficulty = 0.75 * touch_skill + 0.25 * speed_skill
        self.difficulty = self._smooth(self.difficulty, float(target_difficulty))

        ready = self.ema_touch_rate >= 0.82 and stats.median_t_first <= 110.0
        rescue = self.stage_iterations >= 22 and self.ema_touch_rate >= 0.70
        if ready or rescue:
            self._set_stage(Stage.SHOOT)

    def _update_shoot(self, stats) -> None:
        goal_skill = np.clip((stats.goal_rate - 0.05) / 0.30, 0.0, 1.0)
        touch_skill = np.clip((stats.touch_rate - 0.55) / 0.35, 0.0, 1.0)
        speed_skill = np.clip((280.0 - stats.median_t_goal) / 160.0, 0.0, 1.0)
        target_difficulty = 0.6 * goal_skill + 0.25 * touch_skill + 0.15 * speed_skill
        self.difficulty = self._smooth(self.difficulty, float(target_difficulty))

        ready = self.ema_goal_rate >= 0.18 and stats.median_t_goal <= 220.0
        rescue = self.stage_iterations >= 30 and self.ema_goal_rate >= 0.10
        if ready or rescue:
            self._set_stage(Stage.SELF_PLAY)

    def _update_self_play(self, stats) -> None:
        goal_skill = np.clip((stats.goal_rate - 0.02) / 0.16, 0.0, 1.0)
        touch_skill = np.clip((stats.touch_rate - 0.45) / 0.40, 0.0, 1.0)
        target_difficulty = 0.7 * goal_skill + 0.3 * touch_skill
        self.difficulty = self._smooth(self.difficulty, float(target_difficulty), alpha=0.12)

    def maybe_advance(self, stats) -> None:
        self.stage_iterations += 1
        self.ema_touch_rate = self._smooth(self.ema_touch_rate, float(stats.touch_rate), alpha=0.20)
        self.ema_goal_rate = self._smooth(self.ema_goal_rate, float(stats.goal_rate), alpha=0.20)

        if self.stage == Stage.CONTACT:
            self._update_contact(stats)
        elif self.stage == Stage.SHOOT:
            self._update_shoot(stats)
        else:
            self._update_self_play(stats)

    def snapshot(self) -> CurriculumSnapshot:
        return CurriculumSnapshot(
            stage_iters=float(self.stage_iterations),
            ema_touch=float(self.ema_touch_rate),
            ema_goal=float(self.ema_goal_rate),
            difficulty=float(self.difficulty),
        )
