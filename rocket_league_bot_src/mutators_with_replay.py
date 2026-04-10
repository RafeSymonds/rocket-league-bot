"""
Integrated curriculum mutator that combines procedural scenarios with replay-based resets.

This integrates ReplayStateSetter into the curriculum system so that replay resets
happen with configurable probability in later training stages (matching Necto's 70%).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from rlgym.api import StateMutator
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, KickoffMutator
from typing_extensions import override

from .config import Stage
from .curriculum import CurriculumManager
from .replay_setter import ReplayStateSetter


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


class DynamicMatchMutatorWithReplay(StateMutator):
    """
    Extended DynamicMatchMutator that supports replay-based resets.

    This adds replay resets on top of the existing procedural scenario resets,
    weighted by replay_reset_probability (default 0.7 = 70% like Necto).

    Usage:
        mutator = DynamicMatchMutatorWithReplay(
            curriculum_manager=curriculum_manager,
            replay_folder="data/replays/ranked-duels",
            replay_reset_probability=0.7
        )
    """

    def __init__(
        self,
        curriculum_manager: CurriculumManager,
        replay_folder: Optional[str] = None,
        replay_reset_probability: float = 0.7,
        use_lazy_loading: bool = True,
    ):
        self.curriculum_manager = curriculum_manager
        self.replay_reset_probability = replay_reset_probability

        self.team_size_mutator = DynamicTeamSizeMutator(curriculum_manager)
        self.kickoff_mutator = KickoffMutator()
        self.scenario_mutator = ScenarioResetMutator(curriculum_manager)

        self.replay_setter: Optional[ReplayStateSetter] = None
        if replay_folder:
            if use_lazy_loading:
                from .replay_setter import ReplayStateSetterV2 as ReplayStateSetterClass
            else:
                from .replay_setter import ReplayStateSetter as ReplayStateSetterClass

            self.replay_setter = ReplayStateSetterClass(
                replay_folder=replay_folder,
                probability=replay_reset_probability,
            )

    @override
    def apply(self, state: GameState, shared_info: dict[str, Any]) -> None:
        cfg = self.curriculum_manager.current_config()

        self.team_size_mutator.apply(state, shared_info)

        if cfg.full_match:
            self._apply_full_match(state, shared_info, cfg)
            return

        self.kickoff_mutator.apply(state, shared_info)

        roll = np.random.rand()

        if roll < cfg.kickoff_reset_prob:
            return

        if (
            self.replay_setter
            and roll < cfg.kickoff_reset_prob + self.replay_reset_probability
        ):
            self.replay_setter.apply(state, shared_info)
            return

        self.scenario_mutator.apply(state, shared_info)

    def _apply_full_match(
        self, state: GameState, shared_info: dict[str, Any], cfg
    ) -> None:
        try:
            from rlgym_tools.rocket_league.state_mutators.game_mutator import (
                GameMutator,
            )
            from rlgym_tools.rocket_league.shared_info_providers.scoreboard_provider import (
                ScoreboardInfo,
            )
        except ImportError:
            raise RuntimeError(
                "Full-match training requires rlgym-tools. Install with: pip install -U rlgym-tools"
            )

        game_mutator = GameMutator()

        scoreboard = shared_info.get("scoreboard")
        if not isinstance(scoreboard, ScoreboardInfo):
            scoreboard = ScoreboardInfo(
                game_timer_seconds=300.0,
                kickoff_timer_seconds=5.0,
                blue_score=0,
                orange_score=0,
                go_to_kickoff=True,
                is_over=False,
            )
            shared_info["scoreboard"] = scoreboard

        game_mutator.apply(state, shared_info)


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
    def _cars(state: GameState) -> tuple[list[Any], list[Any]]:
        blue = [car for car in state.cars.values() if not car.is_orange]
        orange = [car for car in state.cars.values() if car.is_orange]
        return blue, orange

    @staticmethod
    def _clip_ball(pos: np.ndarray) -> np.ndarray:
        pos[0] = np.clip(
            pos[0],
            -common_values.SIDE_WALL_X + 250.0,
            common_values.SIDE_WALL_X - 250.0,
        )
        pos[1] = np.clip(
            pos[1], -common_values.BACK_NET_Y + 250.0, common_values.BACK_NET_Y - 250.0
        )
        pos[2] = max(
            common_values.BALL_RADIUS, min(common_values.CEILING_Z - 50.0, pos[2])
        )
        return pos

    @staticmethod
    def _set_ball(state: GameState, position: np.ndarray, velocity: np.ndarray) -> None:
        state.ball.position[:] = position
        state.ball.linear_velocity[:] = velocity
        state.ball.angular_velocity[:] = 0.0

    @staticmethod
    def _maybe_loft_ball(
        position: np.ndarray,
        velocity: np.ndarray,
        *,
        chance: float,
        min_height: float,
        max_height: float,
        min_up_speed: float,
        max_up_speed: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        pos = np.asarray(position, dtype=np.float32).copy()
        vel = np.asarray(velocity, dtype=np.float32).copy()
        if np.random.rand() >= chance:
            return pos, vel
        pos[2] = np.random.uniform(min_height, max_height)
        vel[2] = np.random.uniform(min_up_speed, max_up_speed)
        return pos, vel

    @staticmethod
    def _yaw_toward(src: np.ndarray, dst: np.ndarray) -> float:
        delta = np.asarray(dst, dtype=np.float32) - np.asarray(src, dtype=np.float32)
        return float(np.arctan2(delta[1], delta[0]))

    @staticmethod
    def _set_car(
        car,
        position: np.ndarray,
        yaw: float,
        *,
        velocity: np.ndarray | None = None,
        boost: float = 33.0,
    ) -> None:
        car.physics.position[:] = np.asarray(position, dtype=np.float32)
        car.physics.linear_velocity[:] = (
            np.asarray(velocity, dtype=np.float32) if velocity is not None else 0.0
        )
        car.physics.angular_velocity[:] = 0.0
        car.physics.euler_angles = np.array([0.0, yaw, 0.0], dtype=np.float32)
        car.on_ground = True
        car.boost_amount = float(np.clip(boost, 0.0, 100.0))
        car.supersonic_time = 0.0
        car.boost_active_time = 0.0
        car.handbrake = 0.0
        car.has_jumped = False
        car.is_holding_jump = False
        car.is_jumping = False
        car.jump_time = 0.0
        car.has_flipped = False
        car.has_double_jumped = False
        car.air_time_since_jump = 0.0
        car.flip_time = 0.0
        car.flip_torque[:] = 0.0
        car.is_autoflipping = False
        car.autoflip_timer = 0.0
        car.autoflip_direction = 0.0
        car.ball_touches = 0

    def _front_ball_reset(self, state: GameState, cfg, toward_goal_bias: float) -> None:
        car = self._blue_car(state)
        phys = car.physics

        min_dist = float(cfg.touch_min_dist)
        max_dist = max(min_dist, float(cfg.touch_max_dist))
        dist = np.random.uniform(min_dist, max_dist)
        angle = np.deg2rad(
            np.random.uniform(-cfg.touch_max_angle_deg, cfg.touch_max_angle_deg)
        )

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

    def _contact_reset(self, state: GameState, cfg) -> None:
        car = self._blue_car(state)
        car_pos = np.array(
            [
                np.random.uniform(-600.0, 600.0),
                np.random.uniform(-2200.0, -1200.0),
                17.0,
            ],
            dtype=np.float32,
        )
        ball_pos = np.array(
            [
                car_pos[0] + np.random.uniform(-120.0, 120.0),
                car_pos[1] + np.random.uniform(cfg.touch_min_dist, cfg.touch_max_dist),
                common_values.BALL_RADIUS,
            ],
            dtype=np.float32,
        )
        self._set_car(
            car,
            car_pos,
            self._yaw_toward(car_pos, ball_pos),
            boost=np.random.uniform(20.0, 45.0),
        )
        self._set_ball(
            state,
            self._clip_ball(ball_pos),
            np.array(
                [
                    np.random.uniform(-40.0, 40.0),
                    np.random.uniform(-40.0, 120.0),
                    0.0,
                ],
                dtype=np.float32,
            ),
        )

    def _dribble_reset(self, state: GameState, cfg) -> None:
        car = self._blue_car(state)
        car_pos = np.array(
            [
                np.random.uniform(-900.0, 900.0),
                np.random.uniform(-1800.0, -600.0),
                17.0,
            ],
            dtype=np.float32,
        )
        ball_pos = np.array(
            [
                car_pos[0] + np.random.uniform(-140.0, 140.0),
                car_pos[1] + np.random.uniform(700.0, 1200.0),
                common_values.BALL_RADIUS,
            ],
            dtype=np.float32,
        )
        self._set_car(
            car,
            car_pos,
            self._yaw_toward(car_pos, ball_pos),
            velocity=np.array(
                [0.0, np.random.uniform(150.0, 450.0), 0.0], dtype=np.float32
            ),
            boost=np.random.uniform(25.0, 55.0),
        )
        self._set_ball(
            state,
            self._clip_ball(ball_pos),
            np.array(
                [
                    np.random.uniform(-80.0, 80.0),
                    np.random.uniform(150.0, cfg.ball_speed_max),
                    0.0,
                ],
                dtype=np.float32,
            ),
        )

    def _shoot_open_reset(self, state: GameState, cfg) -> None:
        car = self._blue_car(state)
        ball_pos = np.array(
            [
                np.random.uniform(-900.0, 900.0),
                np.random.uniform(1200.0, 2600.0),
                common_values.BALL_RADIUS,
            ],
            dtype=np.float32,
        )
        car_pos = np.array(
            [
                ball_pos[0] + np.random.uniform(-300.0, 300.0),
                ball_pos[1] - np.random.uniform(900.0, 1500.0),
                17.0,
            ],
            dtype=np.float32,
        )
        goal = np.array([0.0, common_values.BACK_NET_Y, 0.0], dtype=np.float32)
        to_goal = goal - ball_pos
        norm = float(np.linalg.norm(to_goal))
        if norm > 1e-6:
            to_goal /= norm
        self._set_car(
            car,
            car_pos,
            self._yaw_toward(car_pos, ball_pos),
            velocity=to_goal * np.random.uniform(250.0, 550.0),
            boost=np.random.uniform(25.0, 65.0),
        )
        ball_vel = to_goal * np.random.uniform(
            0.20 * cfg.ball_speed_max, 0.55 * cfg.ball_speed_max
        )
        ball_pos, ball_vel = self._maybe_loft_ball(
            ball_pos,
            ball_vel,
            chance=0.25,
            min_height=220.0,
            max_height=560.0,
            min_up_speed=120.0,
            max_up_speed=520.0,
        )
        self._set_ball(state, self._clip_ball(ball_pos), ball_vel)

    def _aerial_contact_reset(self, state: GameState, cfg) -> None:
        car = self._blue_car(state)
        ball_pos = np.array(
            [
                np.random.uniform(-1100.0, 1100.0),
                np.random.uniform(900.0, 2300.0),
                np.random.uniform(
                    180.0, 340.0 + 150.0 * self.curriculum_manager.difficulty
                ),
            ],
            dtype=np.float32,
        )
        car_pos = np.array(
            [
                ball_pos[0] + np.random.uniform(-220.0, 220.0),
                ball_pos[1] - np.random.uniform(700.0, 1250.0),
                17.0,
            ],
            dtype=np.float32,
        )
        goal = np.array([0.0, common_values.BACK_NET_Y, 0.0], dtype=np.float32)
        to_goal = goal - ball_pos
        norm = float(np.linalg.norm(to_goal))
        if norm > 1e-6:
            to_goal /= norm
        self._set_car(
            car,
            car_pos,
            self._yaw_toward(car_pos, ball_pos),
            velocity=np.array(
                [
                    np.random.uniform(-120.0, 120.0),
                    np.random.uniform(400.0, 850.0),
                    0.0,
                ],
                dtype=np.float32,
            ),
            boost=np.random.uniform(28.0, 65.0),
        )
        self._set_ball(
            state,
            self._clip_ball(ball_pos),
            np.array(
                [
                    np.random.uniform(-160.0, 160.0),
                    to_goal[1] * np.random.uniform(180.0, 520.0),
                    np.random.uniform(-240.0, -40.0),
                ],
                dtype=np.float32,
            ),
        )

    def _aerial_shoot_reset(self, state: GameState, cfg) -> None:
        car = self._blue_car(state)
        ball_pos = np.array(
            [
                np.random.uniform(-950.0, 950.0),
                np.random.uniform(1500.0, 3000.0),
                np.random.uniform(
                    220.0, 420.0 + 140.0 * self.curriculum_manager.difficulty
                ),
            ],
            dtype=np.float32,
        )
        car_pos = np.array(
            [
                ball_pos[0] + np.random.uniform(-260.0, 260.0),
                ball_pos[1] - np.random.uniform(850.0, 1500.0),
                17.0,
            ],
            dtype=np.float32,
        )
        goal = np.array([0.0, common_values.BACK_NET_Y, 0.0], dtype=np.float32)
        to_goal = goal - ball_pos
        norm = float(np.linalg.norm(to_goal))
        if norm > 1e-6:
            to_goal /= norm
        self._set_car(
            car,
            car_pos,
            self._yaw_toward(car_pos, ball_pos),
            velocity=to_goal * np.random.uniform(500.0, 900.0),
            boost=np.random.uniform(35.0, 75.0),
        )
        self._set_ball(
            state,
            self._clip_ball(ball_pos),
            np.array(
                [
                    np.random.uniform(-140.0, 140.0),
                    to_goal[1] * np.random.uniform(260.0, 700.0),
                    np.random.uniform(-180.0, 40.0),
                ],
                dtype=np.float32,
            ),
        )

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
        blue, orange = self._cars(state)
        if blue:
            blue_pos = np.array(
                [
                    pos[0] + np.random.uniform(-350.0, 350.0),
                    pos[1] - np.random.uniform(900.0, 1500.0),
                    17.0,
                ],
                dtype=np.float32,
            )
            self._set_car(
                blue[0],
                blue_pos,
                self._yaw_toward(blue_pos, pos),
                boost=np.random.uniform(25.0, 60.0),
            )
        if orange:
            orange_pos = np.array(
                [
                    pos[0] + np.random.uniform(-350.0, 350.0),
                    pos[1] + np.random.uniform(900.0, 1500.0),
                    17.0,
                ],
                dtype=np.float32,
            )
            self._set_car(
                orange[0],
                orange_pos,
                self._yaw_toward(orange_pos, pos),
                boost=np.random.uniform(25.0, 60.0),
            )

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

        goal_y = sign * common_values.BACK_NET_Y
        pos = np.array(
            [
                np.random.uniform(-700.0, 700.0),
                sign * np.random.uniform(3500.0, 4450.0),
                common_values.BALL_RADIUS,
            ],
            dtype=np.float32,
        )
        threatened_goal = np.array([0.0, goal_y, 0.0], dtype=np.float32)
        to_goal = threatened_goal - pos
        norm = float(np.linalg.norm(to_goal))
        if norm > 1e-6:
            to_goal /= norm
        speed = np.random.uniform(0.80 * cfg.ball_speed_max, cfg.ball_speed_max)
        ball_vel = to_goal * speed
        pos, ball_vel = self._maybe_loft_ball(
            pos,
            ball_vel,
            chance=0.18,
            min_height=180.0,
            max_height=420.0,
            min_up_speed=80.0,
            max_up_speed=360.0,
        )
        self._set_ball(state, self._clip_ball(pos), ball_vel)
        blue, orange = self._cars(state)
        defenders = blue if defend_blue_side else orange
        attackers = orange if defend_blue_side else blue
        if defenders:
            defender_goal = np.array([0.0, goal_y, 0.0], dtype=np.float32)
            defender_pos = np.array(
                [
                    np.clip(pos[0] + np.random.uniform(-200.0, 200.0), -1800.0, 1800.0),
                    pos[1] - sign * np.random.uniform(320.0, 620.0),
                    17.0,
                ],
                dtype=np.float32,
            )
            self._set_car(
                defenders[0],
                defender_pos,
                self._yaw_toward(defender_pos, pos),
                velocity=to_goal * np.random.uniform(260.0, 600.0),
                boost=np.random.uniform(12.0, 35.0),
            )
        if attackers:
            attacker_pos = np.array(
                [
                    np.clip(pos[0] + np.random.uniform(-220.0, 220.0), -2000.0, 2000.0),
                    pos[1] + sign * np.random.uniform(420.0, 860.0),
                    17.0,
                ],
                dtype=np.float32,
            )
            self._set_car(
                attackers[0],
                attacker_pos,
                self._yaw_toward(attacker_pos, defender_goal),
                velocity=to_goal * np.random.uniform(420.0, 980.0),
                boost=np.random.uniform(35.0, 80.0),
            )

    def _shadow_defend_reset(self, state: GameState, cfg) -> None:
        attack_blue = np.random.rand() < 0.5
        sign = 1.0 if attack_blue else -1.0
        target_goal_y = (
            common_values.BACK_NET_Y if attack_blue else -common_values.BACK_NET_Y
        )
        defend_goal_y = -target_goal_y
        ball_pos = np.array(
            [
                np.random.uniform(-1400.0, 1400.0),
                sign * np.random.uniform(600.0, 2100.0),
                common_values.BALL_RADIUS,
            ],
            dtype=np.float32,
        )
        to_goal = np.array([0.0, target_goal_y, 0.0], dtype=np.float32) - ball_pos
        norm = float(np.linalg.norm(to_goal))
        if norm > 1e-6:
            to_goal /= norm
        self._set_ball(
            state,
            self._clip_ball(ball_pos),
            to_goal
            * np.random.uniform(0.20 * cfg.ball_speed_max, 0.55 * cfg.ball_speed_max),
        )
        blue, orange = self._cars(state)
        attacker = blue[0] if attack_blue and blue else orange[0] if orange else None
        defender = orange[0] if attack_blue and orange else blue[0] if blue else None
        if attacker is not None:
            attacker_pos = np.array(
                [
                    np.clip(
                        ball_pos[0] + np.random.uniform(-180.0, 180.0), -2400.0, 2400.0
                    ),
                    ball_pos[1] - sign * np.random.uniform(260.0, 620.0),
                    17.0,
                ],
                dtype=np.float32,
            )
            self._set_car(
                attacker,
                attacker_pos,
                self._yaw_toward(
                    attacker_pos, np.array([0.0, target_goal_y, 0.0], dtype=np.float32)
                ),
                velocity=to_goal * np.random.uniform(320.0, 760.0),
                boost=np.random.uniform(28.0, 72.0),
            )
        if defender is not None:
            defender_pos = np.array(
                [
                    np.clip(
                        ball_pos[0] + np.random.uniform(-280.0, 280.0), -2200.0, 2200.0
                    ),
                    ball_pos[1] - sign * np.random.uniform(900.0, 1500.0),
                    17.0,
                ],
                dtype=np.float32,
            )
            defend_goal = np.array([0.0, defend_goal_y, 0.0], dtype=np.float32)
            self._set_car(
                defender,
                defender_pos,
                self._yaw_toward(defender_pos, ball_pos),
                velocity=(ball_pos - defender_pos) * np.random.uniform(0.15, 0.30),
                boost=np.random.uniform(18.0, 55.0),
            )
            defender.physics.position[1] = np.clip(
                defender.physics.position[1],
                min(ball_pos[1] - sign * 220.0, defend_goal[1] + sign * 200.0),
                max(ball_pos[1] - sign * 220.0, defend_goal[1] + sign * 200.0),
            )

    def _attack_self_play_reset(self, state: GameState, cfg) -> None:
        attack_blue = np.random.rand() < 0.5
        sign = 1.0 if attack_blue else -1.0
        pos = np.array(
            [
                np.random.uniform(-1800.0, 1800.0),
                sign * np.random.uniform(900.0, 2600.0),
                common_values.BALL_RADIUS,
            ],
            dtype=np.float32,
        )
        target_y = (
            common_values.BACK_NET_Y if attack_blue else -common_values.BACK_NET_Y
        )
        to_goal = np.array([0.0, target_y, 0.0], dtype=np.float32) - pos
        norm = float(np.linalg.norm(to_goal))
        if norm > 1e-6:
            to_goal /= norm
        speed = np.random.uniform(0.25 * cfg.ball_speed_max, cfg.ball_speed_max)
        ball_vel = to_goal * speed
        pos, ball_vel = self._maybe_loft_ball(
            pos,
            ball_vel,
            chance=0.28,
            min_height=220.0,
            max_height=620.0,
            min_up_speed=120.0,
            max_up_speed=520.0,
        )
        self._set_ball(state, self._clip_ball(pos), ball_vel)
        blue, orange = self._cars(state)
        attacker_back = -1.0 if attack_blue else 1.0
        defender_forward = 1.0 if attack_blue else -1.0
        if blue:
            blue_pos = np.array(
                [
                    np.clip(pos[0] + np.random.uniform(-500.0, 500.0), -3000.0, 3000.0),
                    pos[1]
                    + (attacker_back if attack_blue else defender_forward)
                    * np.random.uniform(700.0, 1500.0),
                    17.0,
                ],
                dtype=np.float32,
            )
            self._set_car(
                blue[0],
                blue_pos,
                self._yaw_toward(
                    blue_pos,
                    pos
                    if attack_blue
                    else np.array(
                        [0.0, -common_values.BACK_NET_Y, 0.0], dtype=np.float32
                    ),
                ),
                boost=np.random.uniform(20.0, 75.0),
            )
        if orange:
            orange_pos = np.array(
                [
                    np.clip(pos[0] + np.random.uniform(-500.0, 500.0), -3000.0, 3000.0),
                    pos[1]
                    + (defender_forward if attack_blue else attacker_back)
                    * np.random.uniform(700.0, 1500.0),
                    17.0,
                ],
                dtype=np.float32,
            )
            self._set_car(
                orange[0],
                orange_pos,
                self._yaw_toward(
                    orange_pos,
                    np.array([0.0, common_values.BACK_NET_Y, 0.0], dtype=np.float32)
                    if attack_blue
                    else pos,
                ),
                boost=np.random.uniform(20.0, 75.0),
            )

    def _contested_shoot_reset(self, state: GameState, cfg) -> None:
        self._attack_self_play_reset(state, cfg)
        blue, orange = self._cars(state)
        if not blue or not orange:
            return
        ball_pos = np.asarray(state.ball.position, dtype=np.float32)
        attack_blue = ball_pos[1] >= 0.0
        attacker = blue[0] if attack_blue else orange[0]
        defender = orange[0] if attack_blue else blue[0]
        goal_y = common_values.BACK_NET_Y if attack_blue else -common_values.BACK_NET_Y
        defender_pos = np.array(
            [
                np.clip(
                    ball_pos[0] + np.random.uniform(-250.0, 250.0), -2200.0, 2200.0
                ),
                ball_pos[1]
                + (-1.0 if attack_blue else 1.0) * np.random.uniform(450.0, 900.0),
                17.0,
            ],
            dtype=np.float32,
        )
        self._set_car(
            defender,
            defender_pos,
            self._yaw_toward(defender_pos, ball_pos),
            boost=np.random.uniform(20.0, 55.0),
        )
        attacker_pos = np.array(
            [
                np.clip(
                    ball_pos[0] + np.random.uniform(-350.0, 350.0), -2600.0, 2600.0
                ),
                ball_pos[1]
                + (1.0 if attack_blue else -1.0) * np.random.uniform(700.0, 1300.0),
                17.0,
            ],
            dtype=np.float32,
        )
        self._set_car(
            attacker,
            attacker_pos,
            self._yaw_toward(
                attacker_pos, np.array([0.0, goal_y, 0.0], dtype=np.float32)
            ),
            boost=np.random.uniform(25.0, 70.0),
        )

    def _defend_clear_reset(self, state: GameState, cfg) -> None:
        self._defense_reset(state, cfg)
        blue, orange = self._cars(state)
        ball_pos = np.asarray(state.ball.position, dtype=np.float32)
        defend_blue_side = ball_pos[1] < 0.0
        defenders = blue if defend_blue_side else orange
        attackers = orange if defend_blue_side else blue
        sign = -1.0 if defend_blue_side else 1.0
        if defenders:
            defender = defenders[0]
            defender_pos = np.array(
                [
                    np.clip(
                        ball_pos[0] + np.random.uniform(-120.0, 120.0), -1600.0, 1600.0
                    ),
                    ball_pos[1] - sign * np.random.uniform(220.0, 420.0),
                    17.0,
                ],
                dtype=np.float32,
            )
            self._set_car(
                defender,
                defender_pos,
                self._yaw_toward(defender_pos, ball_pos),
                boost=np.random.uniform(10.0, 30.0),
            )
        if attackers:
            attacker = attackers[0]
            attacker_pos = np.array(
                [
                    np.clip(
                        ball_pos[0] + np.random.uniform(-180.0, 180.0), -1800.0, 1800.0
                    ),
                    ball_pos[1] + sign * np.random.uniform(380.0, 760.0),
                    17.0,
                ],
                dtype=np.float32,
            )
            self._set_car(
                attacker,
                attacker_pos,
                self._yaw_toward(attacker_pos, ball_pos),
                boost=np.random.uniform(35.0, 78.0),
            )

    def _duel_reset(self, state: GameState, cfg) -> None:
        attack_blue = np.random.rand() < 0.5
        sign = 1.0 if attack_blue else -1.0
        ball_pos = np.array(
            [
                np.random.uniform(-1200.0, 1200.0),
                sign * np.random.uniform(1200.0, 2400.0),
                common_values.BALL_RADIUS,
            ],
            dtype=np.float32,
        )
        target_goal_y = (
            common_values.BACK_NET_Y if attack_blue else -common_values.BACK_NET_Y
        )
        to_goal = np.array([0.0, target_goal_y, 0.0], dtype=np.float32) - ball_pos
        norm = float(np.linalg.norm(to_goal))
        if norm > 1e-6:
            to_goal /= norm
        ball_vel = to_goal * np.random.uniform(
            0.18 * cfg.ball_speed_max, 0.45 * cfg.ball_speed_max
        )
        ball_pos, ball_vel = self._maybe_loft_ball(
            ball_pos,
            ball_vel,
            chance=0.30,
            min_height=260.0,
            max_height=700.0,
            min_up_speed=150.0,
            max_up_speed=600.0,
        )
        self._set_ball(state, self._clip_ball(ball_pos), ball_vel)
        blue, orange = self._cars(state)
        attacker = blue[0] if attack_blue and blue else orange[0] if orange else None
        defender = orange[0] if attack_blue and orange else blue[0] if blue else None
        if attacker is not None:
            attacker_pos = np.array(
                [
                    np.clip(
                        ball_pos[0] + np.random.uniform(-220.0, 220.0), -2200.0, 2200.0
                    ),
                    ball_pos[1] - sign * np.random.uniform(520.0, 980.0),
                    17.0,
                ],
                dtype=np.float32,
            )
            self._set_car(
                attacker,
                attacker_pos,
                self._yaw_toward(attacker_pos, ball_pos),
                boost=np.random.uniform(18.0, 50.0),
            )
        if defender is not None:
            defender_pos = np.array(
                [
                    np.clip(
                        ball_pos[0] + np.random.uniform(-180.0, 180.0), -1800.0, 1800.0
                    ),
                    ball_pos[1] + sign * np.random.uniform(360.0, 760.0),
                    17.0,
                ],
                dtype=np.float32,
            )
            self._set_car(
                defender,
                defender_pos,
                self._yaw_toward(defender_pos, ball_pos),
                boost=np.random.uniform(15.0, 40.0),
            )

    def _positional_duel_reset(self, state: GameState, cfg) -> None:
        attack_blue = np.random.rand() < 0.5
        sign = 1.0 if attack_blue else -1.0
        ball_pos = np.array(
            [
                np.random.uniform(-1700.0, 1700.0),
                sign * np.random.uniform(300.0, 1800.0),
                common_values.BALL_RADIUS,
            ],
            dtype=np.float32,
        )
        target_goal_y = (
            common_values.BACK_NET_Y if attack_blue else -common_values.BACK_NET_Y
        )
        to_goal = np.array([0.0, target_goal_y, 0.0], dtype=np.float32) - ball_pos
        norm = float(np.linalg.norm(to_goal))
        if norm > 1e-6:
            to_goal /= norm
        self._set_ball(
            state,
            self._clip_ball(ball_pos),
            to_goal
            * np.random.uniform(0.12 * cfg.ball_speed_max, 0.35 * cfg.ball_speed_max),
        )
        blue, orange = self._cars(state)
        attacker = blue[0] if attack_blue and blue else orange[0] if orange else None
        defender = orange[0] if attack_blue and orange else blue[0] if blue else None
        if attacker is not None:
            attacker_pos = np.array(
                [
                    np.clip(
                        ball_pos[0] + np.random.uniform(-320.0, 320.0), -2500.0, 2500.0
                    ),
                    ball_pos[1] - sign * np.random.uniform(700.0, 1300.0),
                    17.0,
                ],
                dtype=np.float32,
            )
            self._set_car(
                attacker,
                attacker_pos,
                self._yaw_toward(attacker_pos, ball_pos),
                boost=np.random.uniform(20.0, 55.0),
            )
        if defender is not None:
            defender_pos = np.array(
                [
                    np.clip(
                        ball_pos[0] + np.random.uniform(-260.0, 260.0), -2300.0, 2300.0
                    ),
                    ball_pos[1] + sign * np.random.uniform(550.0, 1050.0),
                    17.0,
                ],
                dtype=np.float32,
            )
            self._set_car(
                defender,
                defender_pos,
                self._yaw_toward(defender_pos, ball_pos),
                boost=np.random.uniform(18.0, 45.0),
            )

    @override
    def apply(self, state: GameState, shared_info: dict[str, Any]) -> None:
        cfg = self.curriculum_manager.current_config()
        roll = np.random.rand()

        if roll < cfg.kickoff_reset_prob:
            return

        if cfg.stage == Stage.CONTACT:
            self._contact_reset(state, cfg)
            return

        if cfg.stage == Stage.DRIBBLE:
            self._dribble_reset(state, cfg)
            return

        if cfg.stage == Stage.SHOOT:
            self._shoot_open_reset(state, cfg)
            return

        if cfg.stage == Stage.AERIAL_CONTACT:
            self._aerial_contact_reset(state, cfg)
            return

        if cfg.stage == Stage.AERIAL_SHOOT:
            self._aerial_shoot_reset(state, cfg)
            return

        if cfg.stage == Stage.SHOOT_CONTESTED:
            self._contested_shoot_reset(state, cfg)
            return

        if cfg.stage == Stage.SHADOW_DEFEND:
            if roll < cfg.kickoff_reset_prob + cfg.neutral_reset_prob:
                self._neutral_self_play_reset(state, cfg)
            else:
                self._shadow_defend_reset(state, cfg)
            return

        if cfg.stage == Stage.DEFEND:
            if roll < cfg.kickoff_reset_prob + cfg.neutral_reset_prob:
                self._neutral_self_play_reset(state, cfg)
            else:
                self._defense_reset(state, cfg)
            return

        if cfg.stage == Stage.DEFEND_CLEAR:
            self._defend_clear_reset(state, cfg)
            return

        if cfg.stage == Stage.DUEL:
            self._duel_reset(state, cfg)
            return

        if cfg.stage == Stage.POSITIONAL_DUEL:
            if roll < cfg.kickoff_reset_prob + cfg.neutral_reset_prob:
                self._neutral_self_play_reset(state, cfg)
            else:
                self._positional_duel_reset(state, cfg)
            return

        if roll < cfg.kickoff_reset_prob + cfg.neutral_reset_prob:
            self._neutral_self_play_reset(state, cfg)
        else:
            self._attack_self_play_reset(state, cfg)


from rlgym.rocket_league import common_values
