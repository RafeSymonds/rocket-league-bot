from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


ACTION_REPEAT = 8
OBS_DIM = 44
POLICY_LAYER_SIZES = (512, 512, 256)
CRITIC_LAYER_SIZES = (512, 512, 256)
DEFAULT_CHECKPOINT_ROOT = "data/checkpoints"


class Stage(Enum):
    CONTACT = "CONTACT"
    DRIBBLE = "DRIBBLE"
    SHOOT = "SHOOT"
    DEFEND = "DEFEND"
    SELF_PLAY = "SELF_PLAY"


@dataclass(frozen=True)
class RewardWeights:
    goal: float = 0.0
    touch: float = 0.0
    speed_to_ball: float = 0.0
    face_ball: float = 0.0
    in_air: float = 0.0
    ball_speed_to_goal: float = 0.0
    ball_distance_to_goal: float = 0.0
    step_penalty: float = 0.0


@dataclass(frozen=True)
class StageConfig:
    stage: Stage
    blue_players: int
    orange_players: int
    full_match: bool
    end_on_touch: bool
    end_on_goal: bool
    no_touch_timeout_s: int
    timeout_s: int
    reward_weights: RewardWeights
    touch_min_dist: float
    touch_max_dist: float
    touch_max_angle_deg: float
    ball_speed_max: float
    kickoff_reset_prob: float
    neutral_reset_prob: float
    attack_reset_prob: float


def _lerp(lo: float, hi: float, alpha: float) -> float:
    alpha = max(0.0, min(1.0, float(alpha)))
    return lo + (hi - lo) * alpha


def build_stage_config(stage: Stage, difficulty: float) -> StageConfig:
    d = max(0.0, min(1.0, float(difficulty)))

    if stage == Stage.CONTACT:
        return StageConfig(
            stage=stage,
            blue_players=1,
            orange_players=0,
            full_match=False,
            end_on_touch=True,
            end_on_goal=False,
            no_touch_timeout_s=6,
            timeout_s=12,
            reward_weights=RewardWeights(
                touch=4.5,
                speed_to_ball=0.55,
                face_ball=0.06,
                in_air=0.01,
                step_penalty=0.75,
            ),
            touch_min_dist=_lerp(120.0, 500.0, d),
            touch_max_dist=_lerp(320.0, 1200.0, d),
            touch_max_angle_deg=_lerp(5.0, 35.0, d),
            ball_speed_max=_lerp(0.0, 300.0, d),
            kickoff_reset_prob=_lerp(0.02, 0.0, d),
            neutral_reset_prob=1.0,
            attack_reset_prob=0.0,
        )

    if stage == Stage.DRIBBLE:
        return StageConfig(
            stage=stage,
            blue_players=1,
            orange_players=0,
            full_match=False,
            end_on_touch=False,
            end_on_goal=True,
            no_touch_timeout_s=10,
            timeout_s=24,
            reward_weights=RewardWeights(
                goal=6.0,
                touch=0.8,
                speed_to_ball=0.10,
                face_ball=0.04,
                in_air=0.015,
                ball_speed_to_goal=0.8,
                ball_distance_to_goal=0.45,
                step_penalty=1.0,
            ),
            touch_min_dist=_lerp(250.0, 900.0, d),
            touch_max_dist=_lerp(800.0, 2200.0, d),
            touch_max_angle_deg=_lerp(10.0, 55.0, d),
            ball_speed_max=_lerp(100.0, 1100.0, d),
            kickoff_reset_prob=0.02,
            neutral_reset_prob=_lerp(0.90, 0.55, d),
            attack_reset_prob=_lerp(0.08, 0.43, d),
        )

    if stage == Stage.SHOOT:
        return StageConfig(
            stage=stage,
            blue_players=1,
            orange_players=0,
            full_match=False,
            end_on_touch=False,
            end_on_goal=True,
            no_touch_timeout_s=10,
            timeout_s=30,
            reward_weights=RewardWeights(
                goal=10.0,
                touch=0.5,
                speed_to_ball=0.06,
                face_ball=0.03,
                in_air=0.01,
                ball_speed_to_goal=1.4,
                ball_distance_to_goal=0.3,
                step_penalty=1.0,
            ),
            touch_min_dist=_lerp(450.0, 1000.0, d),
            touch_max_dist=_lerp(1600.0, 3200.0, d),
            touch_max_angle_deg=_lerp(15.0, 75.0, d),
            ball_speed_max=_lerp(250.0, 1600.0, d),
            kickoff_reset_prob=0.05,
            neutral_reset_prob=_lerp(0.75, 0.45, d),
            attack_reset_prob=_lerp(0.20, 0.50, d),
        )

    if stage == Stage.DEFEND:
        return StageConfig(
            stage=stage,
            blue_players=1,
            orange_players=1,
            full_match=False,
            end_on_touch=False,
            end_on_goal=True,
            no_touch_timeout_s=12,
            timeout_s=30,
            reward_weights=RewardWeights(
                goal=10.0,
                touch=0.35,
                speed_to_ball=0.08,
                face_ball=0.02,
                in_air=0.01,
                ball_speed_to_goal=0.9,
                ball_distance_to_goal=0.35,
                step_penalty=1.0,
            ),
            touch_min_dist=_lerp(500.0, 1200.0, d),
            touch_max_dist=_lerp(1500.0, 3000.0, d),
            touch_max_angle_deg=_lerp(20.0, 75.0, d),
            ball_speed_max=_lerp(600.0, 1800.0, d),
            kickoff_reset_prob=_lerp(0.15, 0.05, d),
            neutral_reset_prob=_lerp(0.20, 0.15, d),
            attack_reset_prob=_lerp(0.65, 0.80, d),
        )

    return StageConfig(
        stage=stage,
        blue_players=1,
        orange_players=1,
        full_match=True,
        end_on_touch=False,
        end_on_goal=True,
        no_touch_timeout_s=12,
        timeout_s=35,
        reward_weights=RewardWeights(
            goal=14.0,
            touch=0.08,
            speed_to_ball=0.015,
            face_ball=0.005,
            in_air=0.002,
            ball_speed_to_goal=0.45,
            ball_distance_to_goal=0.08,
            step_penalty=1.15,
        ),
        touch_min_dist=_lerp(700.0, 1400.0, d),
        touch_max_dist=_lerp(2000.0, 3800.0, d),
        touch_max_angle_deg=_lerp(20.0, 80.0, d),
        ball_speed_max=_lerp(400.0, 1800.0, d),
        kickoff_reset_prob=_lerp(0.35, 0.20, d),
        neutral_reset_prob=_lerp(0.35, 0.25, d),
        attack_reset_prob=_lerp(0.30, 0.55, d),
    )
