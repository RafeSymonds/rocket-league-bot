from __future__ import annotations

from typing import Dict

import numpy as np

from rlgym.api import AgentID, RewardFunction
from rlgym.rocket_league import common_values
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.reward_functions import TouchReward

from .config import Stage


def _opponent_goal_y(is_orange: bool) -> float:
    return -common_values.BACK_NET_Y if is_orange else common_values.BACK_NET_Y


class GoalEventMixin:
    def reset(self, agents, initial_state, shared_info):
        self._prev_goal = False

    def goal_event(self, state: GameState) -> bool:
        now = bool(state.goal_scored)
        event = now and not bool(self._prev_goal)
        self._prev_goal = now
        return event


class SignedGoalReward(GoalEventMixin, RewardFunction):
    def reset(self, agents, initial_state, shared_info):
        GoalEventMixin.reset(self, agents, initial_state, shared_info)

    def get_rewards(self, agents, state: GameState, is_terminated, is_truncated, shared_info):
        rewards = {a: 0.0 for a in agents}
        if not self.goal_event(state):
            return rewards

        scoring_team = state.scoring_team
        for agent in agents:
            team = 1 if state.cars[agent].is_orange else 0
            rewards[agent] = 1.0 if team == scoring_team else -1.0
        return rewards


class TouchStatefulReward(RewardFunction):
    def reset(self, agents, initial_state, shared_info):
        if initial_state is None:
            self.prev_touches = {a: 0 for a in agents}
            return
        self.prev_touches = {a: int(initial_state.cars[a].ball_touches) for a in agents}

    def _is_new_touch(self, agent: AgentID, state: GameState) -> bool:
        cur = int(state.cars[agent].ball_touches)
        prev = int(self.prev_touches.get(agent, cur))
        self.prev_touches[agent] = cur
        return cur > prev


class BallSpeedTowardGoalReward(TouchStatefulReward):
    def __init__(self, scale: float = 3000.0):
        super().__init__()
        self.scale = float(scale)

    def get_rewards(self, agents, state: GameState, is_terminated, is_truncated, shared_info):
        rewards: Dict[AgentID, float] = {}
        ball = state.ball
        for agent in agents:
            if not self._is_new_touch(agent, state):
                rewards[agent] = 0.0
                continue

            car = state.cars[agent]
            target_y = _opponent_goal_y(car.is_orange)
            to_goal = np.array([0.0, target_y, 0.0], dtype=np.float32) - ball.position
            norm = float(np.linalg.norm(to_goal))
            if norm < 1e-6:
                rewards[agent] = 0.0
                continue

            speed_toward_goal = float(np.dot(ball.linear_velocity, to_goal / norm))
            rewards[agent] = float(np.clip(speed_toward_goal / self.scale, -0.2, 1.0))
        return rewards


class BallDistanceToGoalDeltaReward(RewardFunction):
    def reset(self, agents, initial_state, shared_info):
        self.prev_dist = {}
        if initial_state is None:
            return
        for agent in agents:
            car = initial_state.cars[agent]
            target_y = _opponent_goal_y(car.is_orange)
            goal = np.array([0.0, target_y, 0.0], dtype=np.float32)
            self.prev_dist[agent] = float(np.linalg.norm(initial_state.ball.position - goal))

    def get_rewards(self, agents, state: GameState, is_terminated, is_truncated, shared_info):
        rewards: Dict[AgentID, float] = {}
        for agent in agents:
            car = state.cars[agent]
            target_y = _opponent_goal_y(car.is_orange)
            goal = np.array([0.0, target_y, 0.0], dtype=np.float32)
            dist = float(np.linalg.norm(state.ball.position - goal))
            prev = float(self.prev_dist.get(agent, dist))
            self.prev_dist[agent] = dist
            rewards[agent] = float(np.clip((prev - dist) / 3000.0, -0.03, 0.03))
        return rewards


class SpeedTowardBallReward(RewardFunction):
    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(self, agents, state: GameState, is_terminated, is_truncated, shared_info):
        rewards: Dict[AgentID, float] = {}
        for agent in agents:
            car = state.cars[agent]
            car_phys = car.inverted_physics if car.is_orange else car.physics
            ball_phys = state.inverted_ball if car.is_orange else state.ball
            to_ball = ball_phys.position - car_phys.position
            norm = float(np.linalg.norm(to_ball))
            if norm < 1e-6:
                rewards[agent] = 0.0
                continue
            speed_toward_ball = float(np.dot(car_phys.linear_velocity, to_ball / norm))
            rewards[agent] = float(np.clip(speed_toward_ball / common_values.CAR_MAX_SPEED, -0.2, 1.0))
        return rewards


class FaceBallReward(RewardFunction):
    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(self, agents, state: GameState, is_terminated, is_truncated, shared_info):
        rewards: Dict[AgentID, float] = {}
        for agent in agents:
            car = state.cars[agent]
            car_phys = car.inverted_physics if car.is_orange else car.physics
            ball_phys = state.inverted_ball if car.is_orange else state.ball
            to_ball = ball_phys.position - car_phys.position
            norm = float(np.linalg.norm(to_ball))
            if norm < 1e-6:
                rewards[agent] = 0.0
                continue
            rewards[agent] = float(np.clip(np.dot(car_phys.forward, to_ball / norm), -1.0, 1.0))
        return rewards


class ForwardDriveReward(RewardFunction):
    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(self, agents, state: GameState, is_terminated, is_truncated, shared_info):
        rewards: Dict[AgentID, float] = {}
        for agent in agents:
            car = state.cars[agent]
            car_phys = car.inverted_physics if car.is_orange else car.physics
            ball_phys = state.inverted_ball if car.is_orange else state.ball
            to_ball = ball_phys.position - car_phys.position
            norm = float(np.linalg.norm(to_ball))
            if norm < 1e-6:
                rewards[agent] = 0.0
                continue
            facing_ball = max(0.0, float(np.dot(car_phys.forward, to_ball / norm)))
            forward_speed = float(np.dot(car_phys.linear_velocity, car_phys.forward))
            rewards[agent] = float(
                np.clip((forward_speed / common_values.CAR_MAX_SPEED) * facing_ball, -1.0, 1.0)
            )
        return rewards


class InAirReward(RewardFunction):
    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(self, agents, state: GameState, is_terminated, is_truncated, shared_info):
        return {agent: 0.0 if state.cars[agent].on_ground else 1.0 for agent in agents}


class AerialControlReward(RewardFunction):
    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(self, agents, state: GameState, is_terminated, is_truncated, shared_info):
        rewards: Dict[AgentID, float] = {}
        for agent in agents:
            car = state.cars[agent]
            if car.on_ground:
                rewards[agent] = 0.0
                continue

            car_phys = car.inverted_physics if car.is_orange else car.physics
            ball_phys = state.inverted_ball if car.is_orange else state.ball
            to_ball = ball_phys.position - car_phys.position
            norm = float(np.linalg.norm(to_ball))
            if norm < 1e-6:
                rewards[agent] = 0.0
                continue

            to_ball_dir = to_ball / norm
            forward_align = max(0.0, float(np.dot(car_phys.forward, to_ball_dir)))
            up_align = max(0.0, float(np.dot(car_phys.up, to_ball_dir)))
            closing_speed = max(
                0.0,
                float(np.dot(car_phys.linear_velocity, to_ball_dir)) / common_values.CAR_MAX_SPEED,
            )
            ball_height = float(
                np.clip(
                    (ball_phys.position[2] - 180.0) / (common_values.CEILING_Z - 180.0),
                    0.0,
                    1.0,
                )
            )
            car_height = float(
                np.clip(
                    (car_phys.position[2] - 80.0) / (common_values.CEILING_Z - 80.0),
                    0.0,
                    1.0,
                )
            )
            rewards[agent] = float(
                np.clip(ball_height * (0.55 * forward_align + 0.25 * up_align + 0.20 * closing_speed), 0.0, 1.0)
                * max(0.35, car_height)
            )
        return rewards


class StepPenalty(RewardFunction):
    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(self, agents, state: GameState, is_terminated, is_truncated, shared_info):
        return {agent: -0.001 for agent in agents}


class HardHitReward(TouchStatefulReward):
    def reset(self, agents, initial_state, shared_info):
        super().reset(agents, initial_state, shared_info)
        if initial_state is None:
            self.prev_ball_speed = 0.0
            return
        self.prev_ball_speed = float(np.linalg.norm(initial_state.ball.linear_velocity))

    def get_rewards(self, agents, state: GameState, is_terminated, is_truncated, shared_info):
        rewards: Dict[AgentID, float] = {}
        cur_ball_speed = float(np.linalg.norm(state.ball.linear_velocity))
        speed_gain = max(0.0, cur_ball_speed - float(self.prev_ball_speed))
        for agent in agents:
            if not self._is_new_touch(agent, state):
                rewards[agent] = 0.0
                continue
            rewards[agent] = float(
                np.clip((speed_gain / 1400.0) + (cur_ball_speed / 5000.0), 0.0, 1.0)
            )
        self.prev_ball_speed = cur_ball_speed
        return rewards


class FlipTouchReward(TouchStatefulReward):
    def get_rewards(self, agents, state: GameState, is_terminated, is_truncated, shared_info):
        rewards: Dict[AgentID, float] = {}
        for agent in agents:
            if not self._is_new_touch(agent, state):
                rewards[agent] = 0.0
                continue
            car = state.cars[agent]
            rewards[agent] = 1.0 if car.is_flipping or (not car.on_ground and car.has_flipped) else 0.0
        return rewards


class SaveClearReward(TouchStatefulReward):
    def reset(self, agents, initial_state, shared_info):
        super().reset(agents, initial_state, shared_info)
        self.prev_ball_pos = (
            np.asarray(initial_state.ball.position, dtype=np.float32).copy()
            if initial_state is not None
            else np.zeros(3, dtype=np.float32)
        )
        self.prev_ball_vel = (
            np.asarray(initial_state.ball.linear_velocity, dtype=np.float32).copy()
            if initial_state is not None
            else np.zeros(3, dtype=np.float32)
        )

    def get_rewards(self, agents, state: GameState, is_terminated, is_truncated, shared_info):
        rewards: Dict[AgentID, float] = {}
        cur_ball_pos = np.asarray(state.ball.position, dtype=np.float32)
        cur_ball_vel = np.asarray(state.ball.linear_velocity, dtype=np.float32)

        for agent in agents:
            if not self._is_new_touch(agent, state):
                rewards[agent] = 0.0
                continue

            car = state.cars[agent]
            own_goal_y = common_values.BACK_NET_Y if car.is_orange else -common_values.BACK_NET_Y
            own_goal = np.array([0.0, own_goal_y, 0.0], dtype=np.float32)

            prev_to_goal = own_goal - self.prev_ball_pos
            prev_dist = float(np.linalg.norm(prev_to_goal))
            if prev_dist < 1e-6:
                rewards[agent] = 0.0
                continue

            prev_toward_own_goal = float(np.dot(self.prev_ball_vel, prev_to_goal / prev_dist))
            own_goal_to_ball = cur_ball_pos - own_goal
            away_norm = float(np.linalg.norm(own_goal_to_ball))
            if away_norm < 1e-6:
                rewards[agent] = 0.0
                continue

            cur_away_speed = float(np.dot(cur_ball_vel, own_goal_to_ball / away_norm))
            defensive_touch = prev_dist <= 3200.0 and prev_toward_own_goal >= 250.0
            if not defensive_touch:
                rewards[agent] = 0.0
                continue

            rewards[agent] = float(np.clip(cur_away_speed / 2500.0, 0.0, 1.0))

        self.prev_ball_pos = cur_ball_pos.copy()
        self.prev_ball_vel = cur_ball_vel.copy()
        return rewards


class AttackPressureReward(RewardFunction):
    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(self, agents, state: GameState, is_terminated, is_truncated, shared_info):
        rewards: Dict[AgentID, float] = {}
        for agent in agents:
            car = state.cars[agent]
            ball = state.inverted_ball if car.is_orange else state.ball

            enemy_goal = np.array([0.0, common_values.BACK_NET_Y, 0.0], dtype=np.float32)
            own_goal = np.array([0.0, -common_values.BACK_NET_Y, 0.0], dtype=np.float32)

            to_enemy_goal = enemy_goal - ball.position
            enemy_dist = float(np.linalg.norm(to_enemy_goal))
            if enemy_dist > 1e-6:
                toward_enemy_goal = float(np.dot(ball.linear_velocity, to_enemy_goal / enemy_dist))
            else:
                toward_enemy_goal = 0.0

            to_own_goal = own_goal - ball.position
            own_dist = float(np.linalg.norm(to_own_goal))
            if own_dist > 1e-6:
                toward_own_goal = float(np.dot(ball.linear_velocity, to_own_goal / own_dist))
            else:
                toward_own_goal = 0.0

            attack_half = float(np.clip((ball.position[1] - 200.0) / 3200.0, 0.0, 1.0))
            own_half = float(np.clip((-ball.position[1] - 200.0) / 3200.0, 0.0, 1.0))
            goal_proximity = float(
                np.clip((common_values.BACK_NET_Y - enemy_dist) / common_values.BACK_NET_Y, 0.0, 1.0)
            )
            shot_speed = float(np.clip(toward_enemy_goal / 2500.0, 0.0, 1.0))
            own_goal_danger = own_half * float(np.clip(toward_own_goal / 2500.0, 0.0, 1.0))

            pressure = attack_half * (0.55 + 0.45 * goal_proximity) * shot_speed
            rewards[agent] = float(np.clip(pressure - (0.65 * own_goal_danger), -1.0, 1.0))
        return rewards


class BoostGainReward(RewardFunction):
    def reset(self, agents, initial_state, shared_info):
        self.prev_boost = {
            agent: float(initial_state.cars[agent].boost_amount) if initial_state is not None else 0.0
            for agent in agents
        }

    def get_rewards(self, agents, state: GameState, is_terminated, is_truncated, shared_info):
        rewards: Dict[AgentID, float] = {}
        for agent in agents:
            current = float(state.cars[agent].boost_amount)
            prev = float(self.prev_boost.get(agent, current))
            self.prev_boost[agent] = current
            rewards[agent] = float(np.clip((current - prev) / 45.0, 0.0, 1.0))
        return rewards


class BoostKeepReward(RewardFunction):
    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(self, agents, state: GameState, is_terminated, is_truncated, shared_info):
        rewards: Dict[AgentID, float] = {}
        for agent in agents:
            boost_frac = float(np.clip(state.cars[agent].boost_amount / 100.0, 0.0, 1.0))
            rewards[agent] = float(np.sqrt(boost_frac))
        return rewards


class CurriculumReward(RewardFunction):
    def __init__(self, curriculum_manager):
        self.curriculum_manager = curriculum_manager
        self.goal = SignedGoalReward()
        self.touch = TouchReward()
        self.speed_to_ball = SpeedTowardBallReward()
        self.face_ball = FaceBallReward()
        self.forward_drive = ForwardDriveReward()
        self.in_air = InAirReward()
        self.aerial_control = AerialControlReward()
        self.ball_speed_to_goal = BallSpeedTowardGoalReward()
        self.ball_distance_to_goal = BallDistanceToGoalDeltaReward()
        self.hard_hit = HardHitReward()
        self.flip_touch = FlipTouchReward()
        self.save_clear = SaveClearReward()
        self.attack_pressure = AttackPressureReward()
        self.boost_gain = BoostGainReward()
        self.boost_keep = BoostKeepReward()
        self.step_penalty = StepPenalty()
        self._sources = (
            self.goal,
            self.touch,
            self.speed_to_ball,
            self.face_ball,
            self.forward_drive,
            self.in_air,
            self.aerial_control,
            self.ball_speed_to_goal,
            self.ball_distance_to_goal,
            self.hard_hit,
            self.flip_touch,
            self.save_clear,
            self.attack_pressure,
            self.boost_gain,
            self.boost_keep,
            self.step_penalty,
        )

    def reset(self, agents, initial_state, shared_info):
        for source in self._sources:
            source.reset(agents, initial_state, shared_info)

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        cfg = self.curriculum_manager.current_config()
        weights = cfg.reward_weights
        rewards = {agent: 0.0 for agent in agents}
        shaping_rewards = {agent: 0.0 for agent in agents}

        def add(source: RewardFunction, weight: float, *, is_goal: bool = False) -> None:
            if weight == 0.0:
                return
            values = source.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
            for agent in agents:
                weighted = float(weight) * float(values[agent])
                if is_goal:
                    rewards[agent] += weighted
                else:
                    shaping_rewards[agent] += weighted

        add(self.goal, weights.goal, is_goal=True)
        add(self.touch, weights.touch)
        add(self.speed_to_ball, weights.speed_to_ball)
        add(self.face_ball, weights.face_ball)
        add(self.forward_drive, weights.forward_drive)
        add(self.in_air, weights.in_air)
        add(self.aerial_control, weights.aerial_control)
        add(self.ball_speed_to_goal, weights.ball_speed_to_goal)
        add(self.ball_distance_to_goal, weights.ball_distance_to_goal)
        add(self.hard_hit, weights.hard_hit)
        add(self.flip_touch, weights.flip_touch)
        add(self.save_clear, weights.save_clear)
        add(self.attack_pressure, weights.attack_pressure)
        add(self.boost_gain, weights.boost_gain)
        add(self.boost_keep, weights.boost_keep)
        add(self.step_penalty, weights.step_penalty)

        if cfg.stage in (Stage.DUEL, Stage.SELF_PLAY):
            team_agents = {
                False: [agent for agent in agents if not state.cars[agent].is_orange],
                True: [agent for agent in agents if state.cars[agent].is_orange],
            }
            team_means = {}
            for is_orange, members in team_agents.items():
                if members:
                    team_means[is_orange] = float(
                        np.mean([shaping_rewards[agent] for agent in members])
                    )
                else:
                    team_means[is_orange] = 0.0

            for agent in agents:
                is_orange = bool(state.cars[agent].is_orange)
                own = shaping_rewards[agent]
                opp = team_means[not is_orange]
                rewards[agent] += own - opp
        else:
            for agent in agents:
                rewards[agent] += shaping_rewards[agent]

        for agent in agents:
            rewards[agent] = float(np.clip(rewards[agent], -5.0, 5.0))
        return rewards
