from __future__ import annotations

import numpy as np

from .config import Stage
from .utils import CurriculumValue


class CurriculumManager:
    """
    Simple, explicit curriculum controller.

    TOUCH:
      - Progressively increases reset difficulty from easy -> hard.
      - Graduates to SCORE after sustained touch reliability.

    SCORE:
      - Graduates to SELFPLAY after sustained scoring reliability.
    """

    def __init__(
        self,
        min_dist: CurriculumValue,
        max_dist: CurriculumValue,
        max_angle: CurriculumValue,
        ball_velocity: CurriculumValue,
        p_easy_reset: CurriculumValue,
        stage_ref: "EnvBuilder",
    ):
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.max_angle = max_angle
        self.ball_velocity = ball_velocity
        self.p_easy_reset = p_easy_reset
        self.stage_ref = stage_ref

        self._touch_pass_streak = 0
        self._score_pass_streak = 0
        self._stage_iters = 0
        self._ema_touch_rate = 0.0
        self._ema_goal_rate = 0.0

    def _set_stage(self, stage: Stage) -> None:
        if self.stage_ref.stage != stage:
            self.stage_ref.stage = stage
            self._stage_iters = 0
            print(f"✅ Curriculum stage -> {stage.value}")

    @staticmethod
    def _smooth(current: float, target: float, alpha: float = 0.12) -> float:
        return (1.0 - alpha) * current + alpha * target

    def _update_touch_difficulty(self, stats) -> None:
        # 0 -> 1 skill estimate for touch stage.
        touch_factor = np.clip((stats.touch_rate - 0.35) / 0.55, 0.0, 1.0)
        speed_factor = np.clip((150.0 - stats.median_t_first) / 110.0, 0.0, 1.0)
        skill = 0.75 * touch_factor + 0.25 * speed_factor

        # Curriculum knobs (easy -> hard).
        target_min_dist = 250.0 + 850.0 * skill
        target_max_dist = 550.0 + 1050.0 * skill
        target_angle = 15.0 + 45.0 * skill
        target_ball_vel = 0.0 + 800.0 * skill
        target_p_easy = 1.0 - 0.95 * skill

        self.min_dist.set(self._smooth(self.min_dist.get(), target_min_dist))
        self.max_dist.set(self._smooth(self.max_dist.get(), target_max_dist))
        self.max_angle.set(self._smooth(self.max_angle.get(), target_angle))
        self.ball_velocity.set(self._smooth(self.ball_velocity.get(), target_ball_vel))
        self.p_easy_reset.set(self._smooth(self.p_easy_reset.get(), target_p_easy))

    def maybe_advance(self, stats) -> None:
        stage: Stage = self.stage_ref.stage
        self._stage_iters += 1
        self._ema_touch_rate = self._smooth(self._ema_touch_rate, float(stats.touch_rate), alpha=0.2)
        self._ema_goal_rate = self._smooth(self._ema_goal_rate, float(stats.goal_rate), alpha=0.2)

        if stage == Stage.TOUCH:
            self._update_touch_difficulty(stats)

            touch_pass = self._ema_touch_rate >= 0.74 and stats.median_t_first <= 130.0
            self._touch_pass_streak = self._touch_pass_streak + 1 if touch_pass else 0

            rescue_pass = self._stage_iters >= 25 and self._ema_touch_rate >= 0.62

            if self._touch_pass_streak >= 4 or rescue_pass:
                self._set_stage(Stage.SCORE)
                self._touch_pass_streak = 0
                self._score_pass_streak = 0

        elif stage == Stage.SCORE:
            score_pass = self._ema_goal_rate >= 0.14 and stats.median_t_goal <= 250.0
            self._score_pass_streak = self._score_pass_streak + 1 if score_pass else 0

            rescue_pass = self._stage_iters >= 35 and self._ema_goal_rate >= 0.10

            if self._score_pass_streak >= 6 or rescue_pass:
                self._set_stage(Stage.SELFPLAY)
                self._score_pass_streak = 0

        elif stage == Stage.SELFPLAY:
            pass

    def snapshot(self) -> dict[str, float]:
        return {
            "stage_iters": float(self._stage_iters),
            "ema_touch": float(self._ema_touch_rate),
            "ema_goal": float(self._ema_goal_rate),
            "min_dist": float(self.min_dist.get()),
            "max_dist": float(self.max_dist.get()),
            "max_angle": float(self.max_angle.get()),
            "ball_velocity": float(self.ball_velocity.get()),
            "p_easy": float(self.p_easy_reset.get()),
        }
