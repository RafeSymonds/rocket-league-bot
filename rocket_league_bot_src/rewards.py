from __future__ import annotations

from typing import Dict

import numpy as np

from rlgym.api import AgentID, RewardFunction
from rlgym.rocket_league import common_values
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.reward_functions import TouchReward


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
            target_y = common_values.BACK_NET_Y if car.is_orange else -common_values.BACK_NET_Y
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
            target_y = common_values.BACK_NET_Y if car.is_orange else -common_values.BACK_NET_Y
            goal = np.array([0.0, target_y, 0.0], dtype=np.float32)
            self.prev_dist[agent] = float(np.linalg.norm(initial_state.ball.position - goal))

    def get_rewards(self, agents, state: GameState, is_terminated, is_truncated, shared_info):
        rewards: Dict[AgentID, float] = {}
        for agent in agents:
            car = state.cars[agent]
            target_y = common_values.BACK_NET_Y if car.is_orange else -common_values.BACK_NET_Y
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
        self.step_penalty = StepPenalty()
        self._sources = (
            self.goal,
            self.touch,
            self.speed_to_ball,
            self.face_ball,
            self.in_air,
            self.ball_speed_to_goal,
            self.ball_distance_to_goal,
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
        add(self.step_penalty, weights.step_penalty)

        for agent in agents:
            rewards[agent] = float(np.clip(rewards[agent], -5.0, 5.0))
        return rewards
