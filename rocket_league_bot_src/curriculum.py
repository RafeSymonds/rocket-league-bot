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
    DRIBBLE -> learn to keep pressure and advance the ball.
    SHOOT -> learn to convert open attacking scenarios.
    SHOOT_CONTESTED -> learn to finish with a live defender present.
    DEFEND -> learn first saves from dangerous positions.
    DEFEND_CLEAR -> learn to turn saves into real clears and exits.
    DUEL -> learn short 1v1 conversions from replay-like attack/defense starts.
    SELF_PLAY -> continue from full-match 1v1.
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
        self.ema_touch_rate = 0.0
        self.ema_goal_rate = 0.0
        print(f"Curriculum stage -> {stage.value}")

    def _update_contact(self, stats) -> None:
        ema_gate = np.clip((self.ema_touch_rate - 0.10) / 0.30, 0.0, 1.0)
        touch_skill = np.clip((stats.touch_rate - 0.08) / 0.42, 0.0, 1.0)
        speed_skill = np.clip((140.0 - stats.median_t_first) / 90.0, 0.0, 1.0)
        target_difficulty = ema_gate * (0.75 * touch_skill + 0.25 * speed_skill)
        self.difficulty = self._smooth(self.difficulty, float(target_difficulty))

        min_iters = self.stage_iterations >= 6
        ready = (
            min_iters
            and self.difficulty >= 0.12
            and self.ema_touch_rate >= 0.38
            and stats.touch_rate >= 0.55
            and stats.median_t_first <= 70.0
        )
        rescue = self.stage_iterations >= 14 and self.ema_touch_rate >= 0.16
        stalled = self.stage_iterations >= 30 and self.ema_touch_rate < 0.10
        if ready or rescue:
            self._set_stage(Stage.DRIBBLE)
        elif stalled:
            self._set_stage(Stage.DRIBBLE)

    def _update_dribble(self, stats) -> None:
        touch_skill = np.clip((stats.touch_rate - 0.55) / 0.30, 0.0, 1.0)
        goal_skill = np.clip((stats.goal_rate - 0.03) / 0.18, 0.0, 1.0)
        goal_time_skill = np.clip((260.0 - stats.median_t_goal) / 180.0, 0.0, 1.0)
        target_difficulty = 0.5 * touch_skill + 0.3 * goal_skill + 0.2 * goal_time_skill
        self.difficulty = self._smooth(self.difficulty, float(target_difficulty))

        ready = self.ema_touch_rate >= 0.78 and self.ema_goal_rate >= 0.10
        rescue = self.stage_iterations >= 26 and self.ema_goal_rate >= 0.06
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
            self._set_stage(Stage.SHOOT_CONTESTED)

    def _update_shoot_contested(self, stats) -> None:
        touch_skill = np.clip((stats.touch_rate - 0.62) / 0.28, 0.0, 1.0)
        goal_skill = np.clip((stats.blue_goal_rate - 0.10) / 0.24, 0.0, 1.0)
        speed_skill = np.clip((240.0 - stats.median_t_goal) / 150.0, 0.0, 1.0)
        target_difficulty = 0.45 * touch_skill + 0.40 * goal_skill + 0.15 * speed_skill
        self.difficulty = self._smooth(self.difficulty, float(target_difficulty), alpha=0.16)

        min_iters = self.stage_iterations >= 6
        ready = (
            min_iters
            and self.difficulty >= 0.14
            and self.ema_touch_rate >= 0.70
            and stats.blue_goal_rate >= 0.16
            and stats.orange_goal_rate <= 0.22
        )
        rescue = self.stage_iterations >= 24 and stats.blue_goal_rate >= 0.10
        if ready or rescue:
            self._set_stage(Stage.DEFEND)

    def _update_defend(self, stats) -> None:
        save_skill = np.clip((stats.touch_rate - 0.58) / 0.24, 0.0, 1.0)
        concede_skill = np.clip((0.18 - stats.orange_goal_rate) / 0.18, 0.0, 1.0)
        clear_bonus = np.clip(stats.blue_goal_rate / 0.08, 0.0, 1.0)
        target_difficulty = 0.60 * save_skill + 0.32 * concede_skill + 0.08 * clear_bonus
        self.difficulty = self._smooth(self.difficulty, float(target_difficulty), alpha=0.15)

        min_iters = self.stage_iterations >= 8
        ready = (
            min_iters
            and self.difficulty >= 0.22
            and self.ema_touch_rate >= 0.64
            and stats.touch_rate >= 0.78
            and stats.orange_goal_rate <= 0.12
            and stats.median_t_first <= 30.0
        )
        rescue = (
            self.stage_iterations >= 28
            and self.difficulty >= 0.14
            and self.ema_touch_rate >= 0.62
            and stats.orange_goal_rate <= 0.18
        )
        if ready or rescue:
            self._set_stage(Stage.DEFEND_CLEAR)

    def _update_defend_clear(self, stats) -> None:
        touch_skill = np.clip((stats.touch_rate - 0.62) / 0.24, 0.0, 1.0)
        clear_skill = np.clip((stats.blue_goal_rate - 0.08) / 0.16, 0.0, 1.0)
        concede_skill = np.clip((0.18 - stats.orange_goal_rate) / 0.18, 0.0, 1.0)
        speed_skill = np.clip((160.0 - stats.median_t_first) / 90.0, 0.0, 1.0)
        target_difficulty = 0.40 * touch_skill + 0.28 * clear_skill + 0.24 * concede_skill + 0.08 * speed_skill
        self.difficulty = self._smooth(self.difficulty, float(target_difficulty), alpha=0.15)

        min_iters = self.stage_iterations >= 8
        ready = (
            min_iters
            and self.difficulty >= 0.24
            and self.ema_touch_rate >= 0.68
            and stats.touch_rate >= 0.80
            and stats.blue_goal_rate >= 0.12
            and stats.orange_goal_rate <= 0.12
        )
        rescue = (
            self.stage_iterations >= 28
            and self.ema_touch_rate >= 0.64
            and stats.orange_goal_rate <= 0.16
        )
        if ready or rescue:
            self._set_stage(Stage.DUEL)

    def _update_duel(self, stats) -> None:
        touch_skill = np.clip((stats.touch_rate - 0.65) / 0.25, 0.0, 1.0)
        score_skill = np.clip((stats.blue_goal_rate - 0.16) / 0.22, 0.0, 1.0)
        concede_skill = np.clip((0.22 - stats.orange_goal_rate) / 0.22, 0.0, 1.0)
        speed_skill = np.clip((240.0 - stats.median_t_goal) / 160.0, 0.0, 1.0)
        target_difficulty = 0.34 * touch_skill + 0.30 * score_skill + 0.22 * concede_skill + 0.14 * speed_skill
        self.difficulty = self._smooth(self.difficulty, float(target_difficulty), alpha=0.14)

        min_iters = self.stage_iterations >= 8
        ready = (
            min_iters
            and self.difficulty >= 0.22
            and self.ema_touch_rate >= 0.76
            and stats.touch_rate >= 0.82
            and stats.blue_goal_rate >= 0.22
            and stats.orange_goal_rate <= 0.18
        )
        rescue = (
            self.stage_iterations >= 28
            and self.difficulty >= 0.16
            and stats.blue_goal_rate >= 0.16
            and stats.orange_goal_rate <= 0.20
        )
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
        elif self.stage == Stage.DRIBBLE:
            self._update_dribble(stats)
        elif self.stage == Stage.SHOOT:
            self._update_shoot(stats)
        elif self.stage == Stage.SHOOT_CONTESTED:
            self._update_shoot_contested(stats)
        elif self.stage == Stage.DEFEND:
            self._update_defend(stats)
        elif self.stage == Stage.DEFEND_CLEAR:
            self._update_defend_clear(stats)
        elif self.stage == Stage.DUEL:
            self._update_duel(stats)
        else:
            self._update_self_play(stats)

    def snapshot(self) -> CurriculumSnapshot:
        return CurriculumSnapshot(
            stage_iters=float(self.stage_iterations),
            ema_touch=float(self.ema_touch_rate),
            ema_goal=float(self.ema_goal_rate),
            difficulty=float(self.difficulty),
        )

    def to_dict(self) -> dict[str, float | str]:
        return {
            "stage": self.stage.value,
            "difficulty": float(self.difficulty),
            "stage_iterations": int(self.stage_iterations),
            "ema_touch_rate": float(self.ema_touch_rate),
            "ema_goal_rate": float(self.ema_goal_rate),
        }

    def load_dict(self, payload: dict[str, float | str] | None) -> None:
        if not payload:
            return
        stage_name = str(payload.get("stage", self.stage.value))
        try:
            self.stage = Stage(stage_name)
        except Exception:
            self.stage = Stage.CONTACT
        self.difficulty = float(payload.get("difficulty", 0.0))
        self.stage_iterations = int(payload.get("stage_iterations", 0))
        self.ema_touch_rate = float(payload.get("ema_touch_rate", 0.0))
        self.ema_goal_rate = float(payload.get("ema_goal_rate", 0.0))
