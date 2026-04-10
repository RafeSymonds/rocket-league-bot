from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


ACTION_REPEAT = 8
OBS_DIM = 54
NUM_DISCRETE_ACTIONS = 124
POLICY_LAYER_SIZES = (512, 512, 256)
CRITIC_LAYER_SIZES = (512, 512, 256)
DEFAULT_CHECKPOINT_ROOT = "data/checkpoints"

USE_DISCRETE_ACTIONS = True


class Stage(Enum):
    CONTACT = "CONTACT"
    DRIBBLE = "DRIBBLE"
    SHOOT = "SHOOT"
    AERIAL_CONTACT = "AERIAL_CONTACT"
    AERIAL_SHOOT = "AERIAL_SHOOT"
    SHOOT_CONTESTED = "SHOOT_CONTESTED"
    SHADOW_DEFEND = "SHADOW_DEFEND"
    DEFEND = "DEFEND"
    DEFEND_CLEAR = "DEFEND_CLEAR"
    POSITIONAL_DUEL = "POSITIONAL_DUEL"
    DUEL = "DUEL"
    SELF_PLAY = "SELF_PLAY"


@dataclass(frozen=True)
class RewardWeights:
    goal: float = 0.0
    touch: float = 0.0
    speed_to_ball: float = 0.0
    face_ball: float = 0.0
    forward_drive: float = 0.0
    in_air: float = 0.0
    aerial_control: float = 0.0
    aerial_touch: float = 0.0
    ball_speed_to_goal: float = 0.0
    ball_distance_to_goal: float = 0.0
    hard_hit: float = 0.0
    flip_touch: float = 0.0
    save_clear: float = 0.0
    attack_pressure: float = 0.0
    goal_side: float = 0.0
    behind_ball: float = 0.0
    boost_gain: float = 0.0
    boost_keep: float = 0.0
    step_penalty: float = 0.0
    win_prob: float = 0.0


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
                forward_drive=0.10,
                hard_hit=0.08,
                in_air=0.01,
                step_penalty=0.75,
            ),
            touch_min_dist=_lerp(120.0, 500.0, d),
            touch_max_dist=_lerp(320.0, 1200.0, d),
            touch_max_angle_deg=_lerp(5.0, 35.0, d),
            ball_speed_max=_lerp(0.0, 300.0, d),
            kickoff_reset_prob=0.0,
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
                forward_drive=0.10,
                in_air=0.015,
                aerial_control=0.02,
                ball_speed_to_goal=0.8,
                ball_distance_to_goal=0.45,
                hard_hit=0.18,
                flip_touch=0.06,
                boost_gain=0.03,
                boost_keep=0.003,
                step_penalty=1.0,
            ),
            touch_min_dist=_lerp(250.0, 900.0, d),
            touch_max_dist=_lerp(800.0, 2200.0, d),
            touch_max_angle_deg=_lerp(10.0, 55.0, d),
            ball_speed_max=_lerp(100.0, 1100.0, d),
            kickoff_reset_prob=0.0,
            neutral_reset_prob=1.0,
            attack_reset_prob=0.0,
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
                forward_drive=0.08,
                in_air=0.01,
                aerial_control=0.05,
                ball_speed_to_goal=1.4,
                ball_distance_to_goal=0.3,
                hard_hit=0.30,
                flip_touch=0.14,
                boost_gain=0.04,
                boost_keep=0.004,
                step_penalty=1.0,
            ),
            touch_min_dist=_lerp(450.0, 1000.0, d),
            touch_max_dist=_lerp(1600.0, 3200.0, d),
            touch_max_angle_deg=_lerp(15.0, 75.0, d),
            ball_speed_max=_lerp(250.0, 1600.0, d),
            kickoff_reset_prob=0.0,
            neutral_reset_prob=0.0,
            attack_reset_prob=1.0,
        )

    if stage == Stage.AERIAL_CONTACT:
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
                goal=8.0,
                touch=0.28,
                speed_to_ball=0.04,
                face_ball=0.02,
                in_air=0.02,
                aerial_touch=1.7,
                ball_speed_to_goal=0.85,
                ball_distance_to_goal=0.14,
                hard_hit=0.22,
                flip_touch=0.14,
                behind_ball=0.06,
                boost_gain=0.02,
                boost_keep=0.003,
                step_penalty=1.0,
            ),
            touch_min_dist=_lerp(500.0, 1100.0, d),
            touch_max_dist=_lerp(1400.0, 2600.0, d),
            touch_max_angle_deg=_lerp(12.0, 48.0, d),
            ball_speed_max=_lerp(250.0, 1250.0, d),
            kickoff_reset_prob=0.0,
            neutral_reset_prob=0.0,
            attack_reset_prob=1.0,
        )

    if stage == Stage.AERIAL_SHOOT:
        return StageConfig(
            stage=stage,
            blue_players=1,
            orange_players=0,
            full_match=False,
            end_on_touch=False,
            end_on_goal=True,
            no_touch_timeout_s=10,
            timeout_s=28,
            reward_weights=RewardWeights(
                goal=12.0,
                touch=0.18,
                speed_to_ball=0.03,
                face_ball=0.02,
                in_air=0.02,
                aerial_touch=1.9,
                ball_speed_to_goal=1.25,
                ball_distance_to_goal=0.22,
                hard_hit=0.30,
                flip_touch=0.20,
                behind_ball=0.08,
                boost_gain=0.02,
                boost_keep=0.004,
                step_penalty=1.02,
            ),
            touch_min_dist=_lerp(700.0, 1400.0, d),
            touch_max_dist=_lerp(1600.0, 3000.0, d),
            touch_max_angle_deg=_lerp(14.0, 55.0, d),
            ball_speed_max=_lerp(350.0, 1500.0, d),
            kickoff_reset_prob=0.0,
            neutral_reset_prob=0.0,
            attack_reset_prob=1.0,
        )

    if stage == Stage.SHOOT_CONTESTED:
        return StageConfig(
            stage=stage,
            blue_players=1,
            orange_players=1,
            full_match=False,
            end_on_touch=False,
            end_on_goal=True,
            no_touch_timeout_s=10,
            timeout_s=26,
            reward_weights=RewardWeights(
                goal=12.0,
                touch=0.30,
                speed_to_ball=0.08,
                face_ball=0.02,
                forward_drive=0.06,
                in_air=0.008,
                aerial_control=0.06,
                ball_speed_to_goal=0.90,
                ball_distance_to_goal=0.20,
                hard_hit=0.32,
                flip_touch=0.18,
                save_clear=0.15,
                boost_gain=0.04,
                boost_keep=0.004,
                step_penalty=1.02,
            ),
            touch_min_dist=_lerp(500.0, 1200.0, d),
            touch_max_dist=_lerp(1500.0, 3200.0, d),
            touch_max_angle_deg=_lerp(18.0, 70.0, d),
            ball_speed_max=_lerp(400.0, 1700.0, d),
            kickoff_reset_prob=0.0,
            neutral_reset_prob=0.0,
            attack_reset_prob=1.0,
        )

    if stage == Stage.SHADOW_DEFEND:
        return StageConfig(
            stage=stage,
            blue_players=1,
            orange_players=1,
            full_match=False,
            end_on_touch=False,
            end_on_goal=True,
            no_touch_timeout_s=12,
            timeout_s=22,
            reward_weights=RewardWeights(
                goal=7.0,
                touch=0.22,
                speed_to_ball=0.06,
                face_ball=0.02,
                in_air=0.008,
                ball_speed_to_goal=0.18,
                ball_distance_to_goal=0.04,
                hard_hit=0.22,
                flip_touch=0.10,
                save_clear=1.10,
                goal_side=0.42,
                behind_ball=0.04,
                boost_gain=0.02,
                boost_keep=0.003,
                step_penalty=1.06,
            ),
            touch_min_dist=_lerp(500.0, 1200.0, d),
            touch_max_dist=_lerp(1400.0, 3000.0, d),
            touch_max_angle_deg=_lerp(15.0, 55.0, d),
            ball_speed_max=_lerp(700.0, 1800.0, d),
            kickoff_reset_prob=0.0,
            neutral_reset_prob=0.15,
            attack_reset_prob=0.85,
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
            timeout_s=20,
            reward_weights=RewardWeights(
                goal=6.0,
                touch=0.52,
                speed_to_ball=0.16,
                face_ball=0.04,
                forward_drive=0.04,
                in_air=0.01,
                aerial_control=0.03,
                ball_speed_to_goal=0.18,
                ball_distance_to_goal=0.03,
                hard_hit=0.28,
                flip_touch=0.12,
                save_clear=1.55,
                goal_side=0.24,
                boost_gain=0.02,
                boost_keep=0.002,
                step_penalty=1.08,
            ),
            touch_min_dist=_lerp(400.0, 900.0, d),
            touch_max_dist=_lerp(1200.0, 2600.0, d),
            touch_max_angle_deg=_lerp(18.0, 60.0, d),
            ball_speed_max=_lerp(900.0, 2100.0, d),
            kickoff_reset_prob=0.0,
            neutral_reset_prob=0.0,
            attack_reset_prob=1.0,
        )

    if stage == Stage.DEFEND_CLEAR:
        return StageConfig(
            stage=stage,
            blue_players=1,
            orange_players=1,
            full_match=False,
            end_on_touch=False,
            end_on_goal=True,
            no_touch_timeout_s=12,
            timeout_s=22,
            reward_weights=RewardWeights(
                goal=8.0,
                touch=0.34,
                speed_to_ball=0.12,
                face_ball=0.025,
                forward_drive=0.05,
                in_air=0.01,
                aerial_control=0.05,
                ball_speed_to_goal=0.40,
                ball_distance_to_goal=0.08,
                hard_hit=0.30,
                flip_touch=0.12,
                save_clear=1.30,
                attack_pressure=0.08,
                goal_side=0.16,
                behind_ball=0.10,
                boost_gain=0.03,
                boost_keep=0.004,
                step_penalty=1.08,
            ),
            touch_min_dist=_lerp(500.0, 1100.0, d),
            touch_max_dist=_lerp(1400.0, 2800.0, d),
            touch_max_angle_deg=_lerp(18.0, 60.0, d),
            ball_speed_max=_lerp(900.0, 2200.0, d),
            kickoff_reset_prob=0.0,
            neutral_reset_prob=0.0,
            attack_reset_prob=1.0,
        )

    if stage == Stage.POSITIONAL_DUEL:
        return StageConfig(
            stage=stage,
            blue_players=1,
            orange_players=1,
            full_match=False,
            end_on_touch=False,
            end_on_goal=True,
            no_touch_timeout_s=12,
            timeout_s=22,
            reward_weights=RewardWeights(
                goal=14.0,
                touch=0.10,
                speed_to_ball=0.03,
                face_ball=0.01,
                in_air=0.004,
                ball_speed_to_goal=0.24,
                ball_distance_to_goal=0.05,
                hard_hit=0.18,
                flip_touch=0.08,
                save_clear=0.45,
                goal_side=0.14,
                behind_ball=0.20,
                boost_gain=0.02,
                boost_keep=0.002,
                step_penalty=1.10,
            ),
            touch_min_dist=_lerp(600.0, 1300.0, d),
            touch_max_dist=_lerp(1700.0, 3200.0, d),
            touch_max_angle_deg=_lerp(18.0, 70.0, d),
            ball_speed_max=_lerp(500.0, 1800.0, d),
            kickoff_reset_prob=0.0,
            neutral_reset_prob=0.25,
            attack_reset_prob=0.75,
        )

    if stage == Stage.DUEL:
        return StageConfig(
            stage=stage,
            blue_players=1,
            orange_players=1,
            full_match=False,
            end_on_touch=False,
            end_on_goal=True,
            no_touch_timeout_s=12,
            timeout_s=20,
            reward_weights=RewardWeights(
                goal=22.0,
                touch=0.0,
                speed_to_ball=0.01,
                face_ball=0.0,
                forward_drive=0.04,
                in_air=0.0,
                aerial_control=0.03,
                ball_speed_to_goal=0.32,
                ball_distance_to_goal=0.04,
                hard_hit=0.10,
                flip_touch=0.08,
                save_clear=0.28,
                attack_pressure=0.22,
                goal_side=0.08,
                behind_ball=0.12,
                boost_gain=0.02,
                boost_keep=0.002,
                step_penalty=1.10,
            ),
            touch_min_dist=_lerp(600.0, 1400.0, d),
            touch_max_dist=_lerp(1800.0, 3400.0, d),
            touch_max_angle_deg=_lerp(18.0, 70.0, d),
            ball_speed_max=_lerp(500.0, 1900.0, d),
            kickoff_reset_prob=0.0,
            neutral_reset_prob=0.10,
            attack_reset_prob=0.90,
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
            goal=26.0,
            touch=0.0,
            speed_to_ball=0.0,
            face_ball=0.0,
            forward_drive=0.03,
            in_air=0.0,
            aerial_control=0.02,
            ball_speed_to_goal=0.14,
            ball_distance_to_goal=0.03,
            hard_hit=0.05,
            flip_touch=0.02,
            save_clear=0.12,
            attack_pressure=0.08,
            goal_side=0.04,
            behind_ball=0.04,
            boost_gain=0.0,
            boost_keep=0.0,
            step_penalty=1.20,
        ),
        touch_min_dist=_lerp(700.0, 1400.0, d),
        touch_max_dist=_lerp(2000.0, 3800.0, d),
        touch_max_angle_deg=_lerp(20.0, 80.0, d),
        ball_speed_max=_lerp(400.0, 1800.0, d),
        kickoff_reset_prob=_lerp(0.35, 0.20, d),
        neutral_reset_prob=_lerp(0.35, 0.25, d),
        attack_reset_prob=_lerp(0.30, 0.55, d),
    )
