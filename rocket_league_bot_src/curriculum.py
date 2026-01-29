from __future__ import annotations

import numpy as np

from .config import Stage
from .utils import CurriculumValue


class CurriculumManager:
    """
    Owns (a) stage transitions and (b) the knobs for the easy reset distribution.
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

    def _set_stage(self, stage: Stage) -> None:
        self.stage_ref.stage = stage
        print(f"✅ Curriculum stage -> {stage.value}")

    def maybe_advance(self, stats) -> None:
        stage: Stage = self.stage_ref.stage

        # -------------------------
        # Stage 0: TOUCH curriculum
        # -------------------------
        if stage == Stage.TOUCH:
            # touch_rate in [0.3, 0.95] → difficulty in [0, 1]
            touch_factor = np.clip((stats.touch_rate - 0.3) / 0.65, 0.0, 1.0)

            # median_t_first in [150, 50] → speed factor in [0, 1]
            speed_factor = np.clip((150 - stats.median_t_first) / 100.0, 0.0, 1.0)

            # combine (touch reliability matters more than speed early)
            progress = 0.85 * touch_factor + 0.15 * speed_factor

            # targets as a function of progress
            target_min_dist = 300 + 600 * progress  # 300 → 900
            target_max_dist = 600 + 800 * progress  # 600 → 1400
            target_angle = 20 + 40 * progress  # 20° → 60°
            target_ball_vel = 0 + 600 * progress  # 0 → 600
            target_p_easy = 1.0 - progress  # 1.0 → 0.0

            # smoothing
            alpha = 0.15
            self.min_dist.set(
                (1 - alpha) * self.min_dist.get() + alpha * target_min_dist
            )
            self.max_dist.set(
                (1 - alpha) * self.max_dist.get() + alpha * target_max_dist
            )
            self.max_angle.set(
                (1 - alpha) * self.max_angle.get() + alpha * target_angle
            )
            self.ball_velocity.set(
                (1 - alpha) * self.ball_velocity.get() + alpha * target_ball_vel
            )
            self.p_easy_reset.set(
                (1 - alpha) * self.p_easy_reset.get() + alpha * target_p_easy
            )

            if np.random.rand() < 0.00001:
                print(
                    f"📈 TOUCH progress={progress:.2f} "
                    f"touch={stats.touch_rate:.2f} "
                    f"t_first={stats.median_t_first:.0f} "
                    f"min={self.min_dist.get():.0f} "
                    f"max={self.max_dist.get():.0f} "
                    f"angle={self.max_angle.get():.0f} "
                    f"v={self.ball_velocity.get():.0f} "
                    f"p_easy={self.p_easy_reset.get():.2f}"
                )

            # if stats.touch_rate > 0.85:
            #     self._set_stage(Stage.SCORE)

        # -------------------------
        # Stage 1: SCORE curriculum
        # -------------------------
        elif stage == Stage.SCORE:
            # graduate when scoring is somewhat consistent and not too slow
            if stats.goal_rate > 0.20 and stats.median_t_goal < 260:
                self._set_stage(Stage.SELFPLAY)

        # -------------------------
        # Stage 2: SELFPLAY
        # -------------------------
        elif stage == Stage.SELFPLAY:
            pass
