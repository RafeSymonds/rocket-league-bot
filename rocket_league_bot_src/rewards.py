from __future__ import annotations

from typing import Dict, List

import numpy as np

from rlgym.api import AgentID, RewardFunction
from rlgym.api.typing import AgentID as AgentIDType
from rlgym.rocket_league import common_values
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.reward_functions import TouchReward

from .config import make_stage_config


class GoalEventMixin:
    """Make `goal_scored` pay exactly once per event."""

    def reset(self, agents, initial_state, shared_info):
        self._prev_goal = False

    def goal_event(self, state: GameState) -> bool:
        now = bool(state.goal_scored)
        evt = now and not bool(self._prev_goal)
        self._prev_goal = now
        return evt


class SignedGoalReward(GoalEventMixin, RewardFunction):
    def reset(self, agents, initial_state, shared_info):
        GoalEventMixin.reset(self, agents, initial_state, shared_info)

    def get_rewards(
        self, agents, state: GameState, is_terminated, is_truncated, shared_info
    ):
        rewards = {a: 0.0 for a in agents}
        if not self.goal_event(state):
            return rewards

        scoring_team = state.scoring_team
        for agent in agents:
            agent_team = 1 if state.cars[agent].is_orange else 0
            rewards[agent] = 1.0 if agent_team == scoring_team else -1.0
        return rewards


class FastGoalBonus(GoalEventMixin, RewardFunction):
    def reset(self, agents, initial_state, shared_info):
        GoalEventMixin.reset(self, agents, initial_state, shared_info)
        self.steps = 0

    def get_rewards(
        self, agents, state: GameState, is_terminated, is_truncated, shared_info
    ):
        self.steps += 1
        rewards = {a: 0.0 for a in agents}
        if not self.goal_event(state):
            return rewards

        bonus = float(np.exp(-self.steps / 350.0))
        scoring_team = state.scoring_team
        for agent in agents:
            agent_team = 1 if state.cars[agent].is_orange else 0
            rewards[agent] = bonus if agent_team == scoring_team else -bonus
        return rewards


class TouchBasedRewardBase(RewardFunction):
    def reset(self, agents, initial_state, shared_info):
        if initial_state is None:
            self.prev_touches = {a: 0 for a in agents}
            return
        self.prev_touches = {a: int(initial_state.cars[a].ball_touches) for a in agents}

    def _new_touch(self, agent: AgentID, state: GameState) -> bool:
        cur = int(state.cars[agent].ball_touches)
        prev = int(self.prev_touches.get(agent, cur))
        self.prev_touches[agent] = cur
        return cur > prev


class ShotCommitReward(TouchBasedRewardBase):
    """Binary reward when a touch sends the ball strongly toward opponent goal."""

    def __init__(self, threshold: float = 1400.0):
        super().__init__()
        self.threshold = float(threshold)

    def get_rewards(
        self,
        agents: List[AgentIDType],
        state: GameState,
        is_terminated,
        is_truncated,
        shared_info,
    ):
        rewards: Dict[AgentIDType, float] = {}
        ball = state.ball

        for agent in agents:
            if not self._new_touch(agent, state):
                rewards[agent] = 0.0
                continue

            car = state.cars[agent]
            opp_goal_y = (
                common_values.BACK_NET_Y if car.is_orange else -common_values.BACK_NET_Y
            )
            to_goal = np.array([0.0, opp_goal_y, 0.0], dtype=np.float32) - ball.position
            n = float(np.linalg.norm(to_goal))
            if n < 1e-6:
                rewards[agent] = 0.0
                continue

            vel_toward = float(np.dot(ball.linear_velocity, to_goal / n))
            rewards[agent] = 1.0 if vel_toward > self.threshold else 0.0

        return rewards


class BallVelocityTowardGoalReward(TouchBasedRewardBase):
    """Touch-gated dense reward for positive ball speed toward opponent goal."""

    def __init__(self, scale: float = 3000.0, cap: float = 1.0):
        super().__init__()
        self.scale = float(scale)
        self.cap = float(cap)

    def get_rewards(
        self, agents, state: GameState, is_terminated, is_truncated, shared_info
    ):
        rewards: Dict[AgentID, float] = {}
        ball = state.ball

        for a in agents:
            if not self._new_touch(a, state):
                rewards[a] = 0.0
                continue

            car = state.cars[a]
            opp_goal_y = (
                common_values.BACK_NET_Y if car.is_orange else -common_values.BACK_NET_Y
            )
            to_goal = np.array([0.0, opp_goal_y, 0.0], dtype=np.float32) - ball.position
            n = float(np.linalg.norm(to_goal))
            if n < 1e-6:
                rewards[a] = 0.0
                continue

            vel = float(np.dot(ball.linear_velocity, to_goal / n))
            rewards[a] = float(np.clip(max(vel, 0.0) / self.scale, 0.0, self.cap))

        return rewards


class TouchWindowBallDistanceToGoalDelta(RewardFunction):
    """After a touch, reward moving ball closer to opponent goal for a short window."""

    def __init__(self, window_steps: int = 15):
        super().__init__()
        self.window_steps = int(window_steps)

    def reset(self, agents, initial_state, shared_info):
        self.prev_touches = (
            {a: int(initial_state.cars[a].ball_touches) for a in agents}
            if initial_state is not None
            else {a: 0 for a in agents}
        )
        self.window_left = {a: 0 for a in agents}
        self.prev_dist = {a: 0.0 for a in agents}

    def get_rewards(
        self, agents, state: GameState, is_terminated, is_truncated, shared_info
    ):
        rewards: Dict[AgentID, float] = {}
        ball_pos = state.ball.position

        for a in agents:
            car = state.cars[a]
            cur_touches = int(car.ball_touches)
            prev_touches = self.prev_touches.get(a, cur_touches)

            opp_goal_y = (
                common_values.BACK_NET_Y if car.is_orange else -common_values.BACK_NET_Y
            )
            goal = np.array([0.0, opp_goal_y, 0.0], dtype=np.float32)
            dist = float(np.linalg.norm(ball_pos - goal))

            if cur_touches > prev_touches:
                self.window_left[a] = self.window_steps
                self.prev_dist[a] = dist

            self.prev_touches[a] = cur_touches

            if self.window_left.get(a, 0) <= 0:
                rewards[a] = 0.0
                continue

            delta = float(self.prev_dist.get(a, dist) - dist)
            self.prev_dist[a] = dist
            self.window_left[a] -= 1

            rewards[a] = float(np.clip(delta / 1800.0, -0.02, 0.06))

        return rewards


class BallDistanceToGoalDeltaReward(RewardFunction):
    """Dense reward for reducing ball distance to opponent goal each step."""

    def reset(self, agents, initial_state, shared_info):
        self.prev_dist = {}
        if initial_state is None:
            return
        for a in agents:
            car = initial_state.cars[a]
            opp_goal_y = (
                common_values.BACK_NET_Y if car.is_orange else -common_values.BACK_NET_Y
            )
            goal = np.array([0.0, opp_goal_y, 0.0], dtype=np.float32)
            self.prev_dist[a] = float(np.linalg.norm(initial_state.ball.position - goal))

    def get_rewards(self, agents, state: GameState, is_terminated, is_truncated, shared_info):
        rewards: Dict[AgentID, float] = {}
        ball_pos = state.ball.position
        for a in agents:
            car = state.cars[a]
            opp_goal_y = (
                common_values.BACK_NET_Y if car.is_orange else -common_values.BACK_NET_Y
            )
            goal = np.array([0.0, opp_goal_y, 0.0], dtype=np.float32)
            dist = float(np.linalg.norm(ball_pos - goal))
            prev = float(self.prev_dist.get(a, dist))
            self.prev_dist[a] = dist
            rewards[a] = float(np.clip((prev - dist) / 2500.0, -0.03, 0.03))
        return rewards


class GoalAlignmentReward(RewardFunction):
    """Small dense reward for ball velocity alignment toward opponent goal."""

    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(
        self, agents, state: GameState, is_terminated, is_truncated, shared_info
    ):
        ball_v = state.ball.linear_velocity.astype(np.float32)
        speed = float(np.linalg.norm(ball_v))
        if speed < 250.0:
            return {a: 0.0 for a in agents}

        rewards: Dict[AgentID, float] = {}
        for a in agents:
            car = state.cars[a]
            opp_goal_y = (
                common_values.BACK_NET_Y if car.is_orange else -common_values.BACK_NET_Y
            )
            to_goal = np.array([0.0, opp_goal_y, 0.0], dtype=np.float32) - state.ball.position
            n = float(np.linalg.norm(to_goal))
            if n < 1e-6:
                rewards[a] = 0.0
                continue
            align = float(np.dot(ball_v / speed, to_goal / n))
            rewards[a] = float(np.clip(align, 0.0, 1.0))
        return rewards


class PowerHitReward(TouchBasedRewardBase):
    """Touch-gated reward for increasing ball speed."""

    def reset(self, agents, initial_state, shared_info):
        super().reset(agents, initial_state, shared_info)
        self.prev_ball_speed = (
            float(np.linalg.norm(initial_state.ball.linear_velocity))
            if initial_state is not None
            else 0.0
        )

    def get_rewards(
        self, agents, state: GameState, is_terminated, is_truncated, shared_info
    ):
        ball_speed = float(np.linalg.norm(state.ball.linear_velocity))
        delta_speed = max(0.0, ball_speed - float(self.prev_ball_speed))
        self.prev_ball_speed = ball_speed

        rewards: Dict[AgentID, float] = {}
        for a in agents:
            rewards[a] = (
                float(np.clip(delta_speed / 3200.0, 0.0, 0.25))
                if self._new_touch(a, state)
                else 0.0
            )
        return rewards


class SpeedTowardBallReward(RewardFunction):
    """Dense shaping: reward moving faster toward the ball."""

    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(
        self, agents, state: GameState, is_terminated, is_truncated, shared_info
    ):
        rewards: Dict[AgentID, float] = {}
        for a in agents:
            car = state.cars[a]
            car_phys = car.inverted_physics if car.is_orange else car.physics
            ball_phys = state.inverted_ball if car.is_orange else state.ball

            to_ball = ball_phys.position - car_phys.position
            n = float(np.linalg.norm(to_ball))
            if n < 1e-6:
                rewards[a] = 0.0
                continue

            vel_toward = float(np.dot(car_phys.linear_velocity, to_ball / n))
            rewards[a] = float(np.clip(vel_toward / 2300.0, -0.2, 1.0))
        return rewards


class FaceBallReward(RewardFunction):
    """Small dense shaping for facing the ball. Stable and bounded."""

    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(
        self, agents, state: GameState, is_terminated, is_truncated, shared_info
    ):
        rewards: Dict[AgentID, float] = {}
        for a in agents:
            car = state.cars[a]
            car_phys = car.inverted_physics if car.is_orange else car.physics
            ball_phys = state.inverted_ball if car.is_orange else state.ball

            to_ball = ball_phys.position - car_phys.position
            n = float(np.linalg.norm(to_ball))
            if n < 1e-6:
                rewards[a] = 0.0
                continue

            cos_sim = float(np.dot(car_phys.forward, to_ball / n))
            rewards[a] = float(np.clip(cos_sim, -1.0, 1.0))

        return rewards


class NoTouchTimeoutPressure(RewardFunction):
    def reset(self, agents, initial_state, shared_info):
        self.steps_since_touch = {a: 0 for a in agents}
        self.prev_touches = (
            {a: int(initial_state.cars[a].ball_touches) for a in agents}
            if initial_state is not None
            else {a: 0 for a in agents}
        )

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {a: 0.0 for a in agents}
        for a in agents:
            cur = int(state.cars[a].ball_touches)
            if cur > self.prev_touches[a]:
                self.steps_since_touch[a] = 0
            else:
                self.steps_since_touch[a] += 1
            self.prev_touches[a] = cur

            t = self.steps_since_touch[a]
            if t > 45:
                rewards[a] = -min(0.002 * (t - 45), 0.2)
        return rewards


class StepPenalty(RewardFunction):
    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(
        self, agents, state: GameState, is_terminated, is_truncated, shared_info
    ):
        return {agent: -0.001 for agent in agents}


class GoalMouthCampingPenalty(RewardFunction):
    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(
        self, agents, state: GameState, is_terminated, is_truncated, shared_info
    ):
        rewards: Dict[AgentID, float] = {}
        for a in agents:
            y = float(abs(state.cars[a].physics.position[1]))
            rewards[a] = -0.01 if y > common_values.BACK_NET_Y - 700 else 0.0
        return rewards


class DistanceToBallReward(RewardFunction):
    def __init__(self) -> None:
        super().__init__()
        self.prev_dist = {}

    def reset(self, agents, initial_state, shared_info):
        self.prev_dist = {}

    def get_rewards(self, agents, state, *_):
        rewards = {}
        for a in agents:
            d = float(np.linalg.norm(state.cars[a].physics.position - state.ball.position))
            prev = float(self.prev_dist.get(a, d))
            rewards[a] = float(np.clip((prev - d) / 500.0, 0.0, 0.05))
            self.prev_dist[a] = d
        return rewards


class CurriculumReward(RewardFunction):
    def __init__(self, stage_ref: "EnvBuilder"):
        self.stage_ref = stage_ref

        self.goal = SignedGoalReward()
        self.fast_goal = FastGoalBonus()
        self.ball_vel_to_goal = BallVelocityTowardGoalReward()
        self.ball_dist_to_goal = TouchWindowBallDistanceToGoalDelta(window_steps=15)
        self.ball_progress = BallDistanceToGoalDeltaReward()
        self.shot_commit = ShotCommitReward(threshold=1300.0)
        self.align = GoalAlignmentReward()
        self.hard_hit = PowerHitReward()

        self.touch = TouchReward()
        self.approach = SpeedTowardBallReward()
        self.power = PowerHitReward()
        self.face_ball = FaceBallReward()

        self.step = StepPenalty()
        self.notouch = NoTouchTimeoutPressure()
        self.camp = GoalMouthCampingPenalty()

        self.ball_distance = DistanceToBallReward()

        self._reward_sources = (
            self.goal,
            self.fast_goal,
            self.ball_vel_to_goal,
            self.ball_dist_to_goal,
            self.ball_progress,
            self.shot_commit,
            self.align,
            self.hard_hit,
            self.touch,
            self.approach,
            self.power,
            self.face_ball,
            self.step,
            self.notouch,
            self.camp,
            self.ball_distance,
        )

    def reset(self, agents, initial_state, shared_info):
        for r in self._reward_sources:
            r.reset(agents, initial_state, shared_info)

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        cfg = make_stage_config(self.stage_ref.stage)
        rewards = {a: 0.0 for a in agents}

        def add(rwd, weight: float) -> None:
            if weight == 0.0:
                return
            vals = rwd.get_rewards(
                agents, state, is_terminated, is_truncated, shared_info
            )
            for a in agents:
                rewards[a] += weight * float(vals[a])

        add(self.goal, cfg.w_goal)
        add(self.fast_goal, cfg.w_fast_goal)

        add(self.ball_vel_to_goal, cfg.w_ball_vel_to_goal)
        add(self.ball_dist_to_goal, cfg.w_ball_dist_to_goal)
        add(self.ball_progress, cfg.w_ball_progress)
        add(self.shot_commit, cfg.w_shot_commit)
        add(self.align, cfg.w_align)
        add(self.hard_hit, cfg.w_hard_hit)

        add(self.touch, cfg.w_touch)
        add(self.power, cfg.w_power)
        add(self.approach, cfg.w_approach)
        add(self.face_ball, cfg.w_face_ball)

        add(self.step, cfg.w_step_penalty)
        add(self.notouch, cfg.w_notouch_pressure)
        add(self.camp, cfg.w_camp_penalty)

        add(self.ball_distance, cfg.w_ball_dist)

        # Keep per-step rewards bounded to reduce unstable PPO updates.
        for a in agents:
            rewards[a] = float(np.clip(rewards[a], -10.0, 10.0))

        return rewards
