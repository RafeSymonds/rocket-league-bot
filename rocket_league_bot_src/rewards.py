from __future__ import annotations

from typing import Dict

import numpy as np

from rlgym.api import AgentID, RewardFunction
from rlgym.rocket_league import common_values
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.reward_functions import TouchReward


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


class InAirReward(RewardFunction):
    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(self, agents, state: GameState, is_terminated, is_truncated, shared_info):
        return {agent: 0.0 if state.cars[agent].on_ground else 1.0 for agent in agents}


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
        self.in_air = InAirReward()
        self.ball_speed_to_goal = BallSpeedTowardGoalReward()
        self.ball_distance_to_goal = BallDistanceToGoalDeltaReward()
        self.hard_hit = HardHitReward()
        self.flip_touch = FlipTouchReward()
        self.save_clear = SaveClearReward()
        self.boost_gain = BoostGainReward()
        self.boost_keep = BoostKeepReward()
        self.step_penalty = StepPenalty()
        self._sources = (
            self.goal,
            self.touch,
            self.speed_to_ball,
            self.face_ball,
            self.in_air,
            self.ball_speed_to_goal,
            self.ball_distance_to_goal,
            self.hard_hit,
            self.flip_touch,
            self.save_clear,
            self.boost_gain,
            self.boost_keep,
            self.step_penalty,
        )

    def reset(self, agents, initial_state, shared_info):
        for source in self._sources:
            source.reset(agents, initial_state, shared_info)

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        weights = self.curriculum_manager.current_config().reward_weights
        rewards = {agent: 0.0 for agent in agents}

        def add(source: RewardFunction, weight: float) -> None:
            if weight == 0.0:
                return
            values = source.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
            for agent in agents:
                rewards[agent] += float(weight) * float(values[agent])

        add(self.goal, weights.goal)
        add(self.touch, weights.touch)
        add(self.speed_to_ball, weights.speed_to_ball)
        add(self.face_ball, weights.face_ball)
        add(self.in_air, weights.in_air)
        add(self.ball_speed_to_goal, weights.ball_speed_to_goal)
        add(self.ball_distance_to_goal, weights.ball_distance_to_goal)
        add(self.hard_hit, weights.hard_hit)
        add(self.flip_touch, weights.flip_touch)
        add(self.save_clear, weights.save_clear)
        add(self.boost_gain, weights.boost_gain)
        add(self.boost_keep, weights.boost_keep)
        add(self.step_penalty, weights.step_penalty)

        for agent in agents:
            rewards[agent] = float(np.clip(rewards[agent], -5.0, 5.0))
        return rewards
