from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class Stage(Enum):
    TOUCH = "TOUCH"
    SCORE = "SCORE"
    SELFPLAY = "SELFPLAY"


@dataclass
class StageConfig:
    stage: Stage
    blue_players: int
    orange_players: int
    end_on_touch: bool
    end_on_goal: bool
    no_touch_timeout_s: int
    timeout_s: int
    w_goal: float
    w_fast_goal: float
    w_ball_vel_to_goal: float
    w_ball_dist_to_goal: float
    w_shot_commit: float
    w_align: float
    w_hard_hit: float
    w_touch: float
    w_power: float
    w_approach: float
    w_face_ball: float
    w_ball_dist: float
    w_step_penalty: float
    w_notouch_pressure: float
    w_camp_penalty: float


def make_stage_config(stage: Stage) -> StageConfig:
    """
    Single source of truth for how each phase behaves.
    """
    if stage == Stage.TOUCH:
        return StageConfig(
            stage=stage,
            blue_players=1,
            orange_players=0,
            end_on_touch=True,
            end_on_goal=False,
            no_touch_timeout_s=8,
            timeout_s=300,
            # rewards: learn contact + meaningful touch
            w_goal=0.0,
            w_fast_goal=0.0,
            w_ball_vel_to_goal=0.0,
            w_ball_dist_to_goal=0.0,
            w_shot_commit=0.0,
            w_align=0.0,
            w_hard_hit=0.0,
            w_touch=5.0,
            w_power=0.0,
            w_approach=0.2,
            w_face_ball=0.1,
            w_ball_dist=0.0,
            w_step_penalty=15.0,
            w_notouch_pressure=0.0,
            w_camp_penalty=0.0,
        )

    if stage == Stage.SCORE:
        return StageConfig(
            stage=stage,
            blue_players=1,
            orange_players=0,
            end_on_touch=False,
            end_on_goal=True,
            no_touch_timeout_s=10,
            timeout_s=300,
            # rewards: force real shots / conversions (score-safe)
            w_goal=40.0,
            w_fast_goal=10.0,
            w_ball_vel_to_goal=8.0,
            w_ball_dist_to_goal=3.0,
            w_shot_commit=6.0,
            w_align=1.5,
            w_hard_hit=4.0,
            # turn off comfort shaping
            w_touch=0.0,
            w_power=0.0,
            w_approach=0.0,
            w_face_ball=0.0,
            w_ball_dist=0.0,
            # punish stalling more than TOUCH
            w_step_penalty=1.5,
            w_notouch_pressure=0.25,
            w_camp_penalty=1.0,
        )

    # SELFPLAY
    return StageConfig(
        stage=stage,
        blue_players=1,
        orange_players=1,
        end_on_touch=False,
        end_on_goal=True,
        no_touch_timeout_s=10,
        timeout_s=300,
        # rewards: still goal-focused, slightly less dense to avoid farming
        w_goal=35.0,
        w_fast_goal=8.0,
        w_ball_vel_to_goal=6.0,
        w_ball_dist_to_goal=2.0,
        w_shot_commit=4.0,
        w_align=1.0,
        w_hard_hit=2.5,
        w_touch=0.0,
        w_power=0.0,
        w_approach=0.0,
        w_face_ball=0.0,
        w_ball_dist=0.0,
        w_step_penalty=1.3,
        w_notouch_pressure=0.20,
        w_camp_penalty=1.0,
    )