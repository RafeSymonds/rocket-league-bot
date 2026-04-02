from __future__ import annotations

from typing import Any

import numpy as np
from rlgym.api import StateMutator
from rlgym.rocket_league import common_values
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, KickoffMutator
from typing_extensions import override

from .config import Stage

try:
    from rlgym_tools.rocket_league.state_mutators.game_mutator import GameMutator
    from rlgym_tools.rocket_league.shared_info_providers.scoreboard_provider import (
        ScoreboardInfo,
    )
except Exception:  # pragma: no cover - optional dependency until installed
    GameMutator = None
    ScoreboardInfo = None


class DynamicTeamSizeMutator(StateMutator):
    def __init__(self, curriculum_manager):
        self.curriculum_manager = curriculum_manager
        self._cache: dict[tuple[int, int], FixedTeamSizeMutator] = {}

    @override
    def apply(self, state: GameState, shared_info: dict[str, Any]) -> None:
        cfg = self.curriculum_manager.current_config()
        key = (cfg.blue_players, cfg.orange_players)
        if key not in self._cache:
            self._cache[key] = FixedTeamSizeMutator(*key)
        self._cache[key].apply(state, shared_info)


class ScenarioResetMutator(StateMutator):
    """
    Replaces the old single "ball near car" reset with stage-aware scenario families.
    Each family widens as curriculum difficulty rises.
    """

    def __init__(self, curriculum_manager):
        self.curriculum_manager = curriculum_manager

    @staticmethod
    def _blue_car(state: GameState):
        for car in state.cars.values():
            if not car.is_orange:
                return car
        return next(iter(state.cars.values()))

    @staticmethod
    def _clip_ball(pos: np.ndarray) -> np.ndarray:
        pos[0] = np.clip(
            pos[0], -common_values.SIDE_WALL_X + 250.0, common_values.SIDE_WALL_X - 250.0
        )
        pos[1] = np.clip(
            pos[1], -common_values.BACK_NET_Y + 250.0, common_values.BACK_NET_Y - 250.0
        )
        pos[2] = max(common_values.BALL_RADIUS, min(common_values.CEILING_Z - 50.0, pos[2]))
        return pos

    @staticmethod
    def _set_ball(state: GameState, position: np.ndarray, velocity: np.ndarray) -> None:
        state.ball.position[:] = position
        state.ball.linear_velocity[:] = velocity
        state.ball.angular_velocity[:] = 0.0

    def _front_ball_reset(self, state: GameState, cfg, toward_goal_bias: float) -> None:
        car = self._blue_car(state)
        phys = car.physics

        min_dist = float(cfg.touch_min_dist)
        max_dist = max(min_dist, float(cfg.touch_max_dist))
        dist = np.random.uniform(min_dist, max_dist)
        angle = np.deg2rad(np.random.uniform(-cfg.touch_max_angle_deg, cfg.touch_max_angle_deg))

        forward = phys.forward.astype(np.float32)
        right = phys.right.astype(np.float32)
        direction = np.cos(angle) * forward + np.sin(angle) * right
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm < 1e-6:
            direction = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            direction = direction / direction_norm

        ball_pos = phys.position + direction * dist
        ball_pos[2] = common_values.BALL_RADIUS

        enemy_goal = np.array([0.0, common_values.BACK_NET_Y, 0.0], dtype=np.float32)
        to_goal = enemy_goal - ball_pos
        to_goal_norm = float(np.linalg.norm(to_goal))
        if to_goal_norm > 1e-6:
            to_goal = to_goal / to_goal_norm

        random_dir = np.array(
            [np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), 0.0],
            dtype=np.float32,
        )
        random_norm = float(np.linalg.norm(random_dir))
        if random_norm < 1e-6:
            random_dir = direction
        else:
            random_dir /= random_norm

        speed = np.random.uniform(0.0, cfg.ball_speed_max)
        vel_dir = toward_goal_bias * to_goal + (1.0 - toward_goal_bias) * random_dir
        vel_norm = float(np.linalg.norm(vel_dir))
        if vel_norm > 1e-6:
            vel_dir /= vel_norm

        self._set_ball(state, self._clip_ball(ball_pos), vel_dir * speed)

    def _neutral_self_play_reset(self, state: GameState, cfg) -> None:
        pos = np.array(
            [
                np.random.uniform(-1400.0, 1400.0),
                np.random.uniform(-1200.0, 1200.0),
                common_values.BALL_RADIUS,
            ],
            dtype=np.float32,
        )
        vel = np.array(
            [
                np.random.uniform(-cfg.ball_speed_max, cfg.ball_speed_max),
                np.random.uniform(-cfg.ball_speed_max, cfg.ball_speed_max),
                0.0,
            ],
            dtype=np.float32,
        )
        self._set_ball(state, self._clip_ball(pos), vel)

    def _midfield_dribble_reset(self, state: GameState, cfg) -> None:
        car = self._blue_car(state)
        phys = car.physics
        forward = phys.forward.astype(np.float32)

        offset = np.array(
            [
                np.random.uniform(-400.0, 400.0),
                np.random.uniform(450.0, 1200.0),
                0.0,
            ],
            dtype=np.float32,
        )
        pos = phys.position + offset
        pos[2] = common_values.BALL_RADIUS

        base_speed = np.random.uniform(100.0, cfg.ball_speed_max)
        vel_dir = 0.7 * forward + 0.3 * np.array(
            [np.random.uniform(-1.0, 1.0), np.random.uniform(-0.2, 1.0), 0.0],
            dtype=np.float32,
        )
        norm = float(np.linalg.norm(vel_dir))
        if norm > 1e-6:
            vel_dir /= norm
        self._set_ball(state, self._clip_ball(pos), vel_dir * base_speed)

    def _defense_reset(self, state: GameState, cfg) -> None:
        defend_blue_side = np.random.rand() < 0.5
        sign = -1.0 if defend_blue_side else 1.0

        pos = np.array(
            [
                np.random.uniform(-1400.0, 1400.0),
                sign * np.random.uniform(2400.0, 4200.0),
                common_values.BALL_RADIUS,
            ],
            dtype=np.float32,
        )
        threatened_goal = np.array([0.0, sign * common_values.BACK_NET_Y, 0.0], dtype=np.float32)
        to_goal = threatened_goal - pos
        norm = float(np.linalg.norm(to_goal))
        if norm > 1e-6:
            to_goal /= norm
        speed = np.random.uniform(0.35 * cfg.ball_speed_max, cfg.ball_speed_max)
        self._set_ball(state, self._clip_ball(pos), to_goal * speed)

    def _attack_self_play_reset(self, state: GameState, cfg) -> None:
        sign = -1.0 if np.random.rand() < 0.5 else 1.0
        pos = np.array(
            [
                np.random.uniform(-1800.0, 1800.0),
                sign * np.random.uniform(500.0, 2600.0),
                common_values.BALL_RADIUS,
            ],
            dtype=np.float32,
        )
        target_y = -common_values.BACK_NET_Y if sign < 0.0 else common_values.BACK_NET_Y
        to_goal = np.array([0.0, target_y, 0.0], dtype=np.float32) - pos
        norm = float(np.linalg.norm(to_goal))
        if norm > 1e-6:
            to_goal /= norm
        speed = np.random.uniform(0.25 * cfg.ball_speed_max, cfg.ball_speed_max)
        self._set_ball(state, self._clip_ball(pos), to_goal * speed)

    @override
    def apply(self, state: GameState, shared_info: dict[str, Any]) -> None:
        cfg = self.curriculum_manager.current_config()
        roll = np.random.rand()

        if roll < cfg.kickoff_reset_prob:
            return

        if cfg.stage == Stage.CONTACT:
            self._front_ball_reset(state, cfg, toward_goal_bias=0.1)
            return

        if cfg.stage == Stage.DRIBBLE:
            if roll < cfg.kickoff_reset_prob + cfg.neutral_reset_prob:
                self._midfield_dribble_reset(state, cfg)
            else:
                self._front_ball_reset(state, cfg, toward_goal_bias=0.55)
            return

        if cfg.stage == Stage.SHOOT:
            if roll < cfg.kickoff_reset_prob + cfg.neutral_reset_prob:
                self._front_ball_reset(state, cfg, toward_goal_bias=0.35)
            else:
                self._front_ball_reset(state, cfg, toward_goal_bias=0.85)
            return

        if cfg.stage == Stage.DEFEND:
            if roll < cfg.kickoff_reset_prob + cfg.neutral_reset_prob:
                self._neutral_self_play_reset(state, cfg)
            else:
                self._defense_reset(state, cfg)
            return

        if roll < cfg.kickoff_reset_prob + cfg.neutral_reset_prob:
            self._neutral_self_play_reset(state, cfg)
        else:
            self._attack_self_play_reset(state, cfg)


class DynamicMatchMutator(StateMutator):
    def __init__(self, curriculum_manager):
        self.curriculum_manager = curriculum_manager
        self.team_size_mutator = DynamicTeamSizeMutator(curriculum_manager)
        self.kickoff_mutator = KickoffMutator()
        self.scenario_mutator = ScenarioResetMutator(curriculum_manager)
        self.game_mutator = GameMutator() if GameMutator is not None else None

    def _ensure_scoreboard(self, shared_info: dict[str, Any], *, full_match: bool) -> None:
        if ScoreboardInfo is None:
            return

        scoreboard = shared_info.get("scoreboard")
        if not isinstance(scoreboard, ScoreboardInfo):
            scoreboard = ScoreboardInfo(
                game_timer_seconds=float("inf"),
                kickoff_timer_seconds=0.0,
                blue_score=0,
                orange_score=0,
                go_to_kickoff=False,
                is_over=False,
            )
            shared_info["scoreboard"] = scoreboard

        if full_match:
            scoreboard.game_timer_seconds = 300.0
            scoreboard.kickoff_timer_seconds = 5.0
            scoreboard.blue_score = 0
            scoreboard.orange_score = 0
            scoreboard.go_to_kickoff = True
            scoreboard.is_over = False
            return

        scoreboard.game_timer_seconds = float("inf")
        scoreboard.kickoff_timer_seconds = 0.0
        scoreboard.blue_score = 0
        scoreboard.orange_score = 0
        scoreboard.go_to_kickoff = False
        scoreboard.is_over = False

    @override
    def apply(self, state: GameState, shared_info: dict[str, Any]) -> None:
        cfg = self.curriculum_manager.current_config()
        self.team_size_mutator.apply(state, shared_info)
        self._ensure_scoreboard(shared_info, full_match=cfg.full_match)
        if cfg.full_match:
            if self.game_mutator is None:
                raise RuntimeError(
                    "Full-match training requires rlgym-tools. Install it with: pip install -U rlgym-tools"
                )
            self.game_mutator.apply(state, shared_info)
            return

        self.kickoff_mutator.apply(state, shared_info)
        self.scenario_mutator.apply(state, shared_info)
