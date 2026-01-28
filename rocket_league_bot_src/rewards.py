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
    """Helper to make 'goal_scored' pay only once per goal event."""

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
            car = state.cars[agent]
            agent_team = 1 if car.is_orange else 0
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

        bonus = float(np.exp(-self.steps / 400.0))
        scoring_team = state.scoring_team
        for agent in agents:
            car = state.cars[agent]
            agent_team = 1 if car.is_orange else 0
            rewards[agent] = bonus if agent_team == scoring_team else -bonus
        return rewards


class TouchBasedRewardBase(RewardFunction):
    def reset(self, agents, initial_state, shared_info):
        self.prev_touches = (
            {a: int(initial_state.cars[a].ball_touches) for a in agents}
            if initial_state is not None
            else {a: 0 for a in agents}
        )

    def _new_touch(self, agent: AgentID, state: GameState) -> bool:
        car = state.cars[agent]
        cur = int(car.ball_touches)
        prev = int(self.prev_touches.get(agent, cur))
        self.prev_touches[agent] = cur
        return cur > prev


class ShotCommitReward(TouchBasedRewardBase):
    """
    Binary: after a touch, if ball velocity toward opponent goal is high enough, reward.
    This prevents sideways farming.
    """

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
            to_goal = np.array(
                [0.0, opp_goal_y, 0.0], dtype=np.float32
            ) - ball.position.astype(np.float32)
            n = float(np.linalg.norm(to_goal))
            if n < 1e-6:
                rewards[agent] = 0.0
                continue
            to_goal /= n

            vel_toward = float(np.dot(ball.linear_velocity.astype(np.float32), to_goal))
            rewards[agent] = 1.0 if vel_toward > self.threshold else 0.0

        return rewards


class BallVelocityTowardGoalReward(TouchBasedRewardBase):
    """
    Dense but touch-gated: after a touch, reward positive ball velocity component toward opponent net.
    """

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
            goal = np.array([0.0, opp_goal_y, 0.0], dtype=np.float32)

            to_goal = goal - ball.position.astype(np.float32)
            n = float(np.linalg.norm(to_goal))
            if n < 1e-6:
                rewards[a] = 0.0
                continue
            to_goal /= n

            vel = float(np.dot(ball.linear_velocity.astype(np.float32), to_goal))
            rewards[a] = float(np.clip(max(vel, 0.0) / self.scale, 0.0, self.cap))

        return rewards


class TouchWindowBallDistanceToGoalDelta(RewardFunction):
    """
    After a touch, open a short window where we reward *reducing distance to goal*.
    This gives dense follow-through without allowing endless camping.
    """

    def __init__(self, window_steps: int = 15):
        super().__init__()
        self.window_steps = int(window_steps)

    def reset(self, agents, initial_state, shared_info):
        self.prev_touches = {a: int(initial_state.cars[a].ball_touches) for a in agents}
        self.window_left = {a: 0 for a in agents}
        self.prev_dist = {}

        if initial_state is not None:
            for a in agents:
                car = initial_state.cars[a]
                opp_goal_y = (
                    common_values.BACK_NET_Y
                    if car.is_orange
                    else -common_values.BACK_NET_Y
                )
                goal = np.array([0.0, opp_goal_y, 0.0], dtype=np.float32)
                self.prev_dist[a] = float(
                    np.linalg.norm(initial_state.ball.position - goal)
                )

    def get_rewards(
        self, agents, state: GameState, is_terminated, is_truncated, shared_info
    ):
        rewards: Dict[AgentID, float] = {}
        ball_pos = state.ball.position.astype(np.float32)

        for a in agents:
            car = state.cars[a]
            cur_t = int(car.ball_touches)
            if cur_t > self.prev_touches.get(a, cur_t):
                self.window_left[a] = self.window_steps
                opp_goal_y = (
                    common_values.BACK_NET_Y
                    if car.is_orange
                    else -common_values.BACK_NET_Y
                )
                goal = np.array([0.0, opp_goal_y, 0.0], dtype=np.float32)
                self.prev_dist[a] = float(np.linalg.norm(ball_pos - goal))

            self.prev_touches[a] = cur_t

            if self.window_left.get(a, 0) <= 0:
                rewards[a] = 0.0
                continue

            opp_goal_y = (
                common_values.BACK_NET_Y if car.is_orange else -common_values.BACK_NET_Y
            )
            goal = np.array([0.0, opp_goal_y, 0.0], dtype=np.float32)
            dist = float(np.linalg.norm(ball_pos - goal))

            prev = float(self.prev_dist.get(a, dist))
            delta = prev - dist  # positive is good (closer)
            self.prev_dist[a] = dist
            self.window_left[a] -= 1

            # small, capped, and ignores tiny jitter
            if abs(delta) < 10.0:
                delta = 0.0
            rewards[a] = float(np.clip(delta / 2000.0, -0.02, 0.05))

        return rewards


class GoalAlignmentReward(RewardFunction):
    """
    Dense, not touch-gated: rewards ball velocity direction aligning with the opponent goal.
    Kept low-weight. Discourages sideways dribbling.
    """

    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(
        self, agents, state: GameState, is_terminated, is_truncated, shared_info
    ):
        ball_v = state.ball.linear_velocity.astype(np.float32)
        speed = float(np.linalg.norm(ball_v))
        if speed < 300.0:
            return {a: 0.0 for a in agents}

        rewards: Dict[AgentID, float] = {}
        for a in agents:
            car = state.cars[a]
            opp_goal_y = (
                common_values.BACK_NET_Y if car.is_orange else -common_values.BACK_NET_Y
            )
            to_goal = np.array(
                [0.0, opp_goal_y, 0.0], dtype=np.float32
            ) - state.ball.position.astype(np.float32)
            n = float(np.linalg.norm(to_goal))
            if n < 1e-6:
                rewards[a] = 0.0
                continue
            to_goal /= n
            align = float(np.dot(ball_v / speed, to_goal))
            rewards[a] = float(np.clip(align, 0.0, 1.0))
        return rewards


class PowerHitReward(TouchBasedRewardBase):
    """
    Touch-gated: reward speed increase of the ball after the touch.
    Good for hard shots / clears, but capped.
    """

    def reset(self, agents, initial_state, shared_info):
        super().reset(agents, initial_state, shared_info)
        self.prev_ball_speed = (
            0.0
            if initial_state is None
            else float(np.linalg.norm(initial_state.ball.linear_velocity))
        )

    def get_rewards(
        self, agents, state: GameState, is_terminated, is_truncated, shared_info
    ):
        ball_speed = float(np.linalg.norm(state.ball.linear_velocity))
        delta_speed = ball_speed - float(self.prev_ball_speed)
        self.prev_ball_speed = ball_speed

        rewards: Dict[AgentID, float] = {}
        for a in agents:
            if not self._new_touch(a, state):
                rewards[a] = 0.0
                continue
            rewards[a] = float(np.clip(delta_speed / 3000.0, 0.0, 0.3))
        return rewards


class SpeedTowardBallReward(RewardFunction):
    """TOUCH-stage shaping: reward decreasing distance to ball (capped)."""

    def reset(self, agents, initial_state, shared_info):
        self.prev_dist: Dict[AgentID, float] = {}

    def get_rewards(
        self, agents, state: GameState, is_terminated, is_truncated, shared_info
    ):
        rewards: Dict[AgentID, float] = {}
        for a in agents:
            car = state.cars[a]
            car_phys = car.physics if not car.is_orange else car.inverted_physics
            ball_phys = state.ball if not car.is_orange else state.inverted_ball

            dist = float(np.linalg.norm(ball_phys.position - car_phys.position))
            prev = float(self.prev_dist.get(a, dist))
            delta = prev - dist

            reward = delta / 500.0
            rewards[a] = float(np.clip(reward, 0.0, 0.1))

            self.prev_dist[a] = dist
        return rewards


class FaceBallReward(RewardFunction):
    """TOUCH-stage shaping: reward facing the ball."""

    def reset(self, agents, initial_state, shared_info):
        self.prev_cos_sim = {a: 0.0 for a in agents}

    def get_rewards(
        self, agents, state: GameState, is_terminated, is_truncated, shared_info
    ):
        rewards: Dict[AgentID, float] = {}
        for a in agents:
            car = state.cars[a]
            car_phys = car.physics if not car.is_orange else car.inverted_physics
            ball_phys = state.ball if not car.is_orange else state.inverted_ball

            to_ball = ball_phys.position - car_phys.position
            to_ball_dir = to_ball / np.linalg.norm(to_ball)

            cos_sim = np.dot(car_phys.forward, to_ball_dir)
            
            delta = cos_sim - self.prev_cos_sim.get(a, cos_sim)
            rewards[a] = delta
            self.prev_cos_sim[a] = cos_sim
            
        return rewards



class NoTouchTimeoutPressure(RewardFunction):
    def reset(self, agents, initial_state, shared_info):
        self.steps_since_touch = {a: 0 for a in agents}
        self.prev_touches = {a: int(initial_state.cars[a].ball_touches) for a in agents}

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
            if t > 60:
                rewards[a] = -min(0.002 * (t - 60), 0.2)
        return rewards


class StepPenalty(RewardFunction):
    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(
        self, agents, state: GameState, is_terminated, is_truncated, shared_info
    ):
        return {agent: -0.001 for agent in agents}


class GoalMouthCampingPenalty(RewardFunction):
    """
    Penalize sitting deep in / behind the goal mouth.
    Helps prevent 'park in net' exploits in SCORE.
    """

    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(
        self, agents, state: GameState, is_terminated, is_truncated, shared_info
    ):
        rewards: Dict[AgentID, float] = {}
        for a in agents:
            car = state.cars[a]
            # absolute y, near back wall/net
            y = float(abs(car.physics.position[1]))
            # start penalizing in the last ~600uu of the field
            if y > common_values.BACK_NET_Y - 600:
                rewards[a] = -0.01
            else:
                rewards[a] = 0.0
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
            car = state.cars[a]
            ball = state.ball
            d = np.linalg.norm(car.physics.position - ball.position)
            prev = self.prev_dist.get(a, d)
            rewards[a] = np.clip((prev - d) / 500.0, 0.0, 0.05)
            self.prev_dist[a] = d
        return rewards


class CurriculumReward(RewardFunction):
    def __init__(self, stage_ref: "EnvBuilder"):
        self.stage_ref = stage_ref

        # underlying rewards (created once)
        self.goal = SignedGoalReward()
        self.fast_goal = FastGoalBonus()

        # score-safe, goal-directed rewards
        self.ball_vel_to_goal = BallVelocityTowardGoalReward()
        self.ball_dist_to_goal = TouchWindowBallDistanceToGoalDelta(window_steps=15)
        self.shot_commit = ShotCommitReward(threshold=1400.0)
        self.align = GoalAlignmentReward()
        self.hard_hit = PowerHitReward()  # already touch-gated and based on delta speed

        # touch-stage shaping
        self.touch = TouchReward()
        self.approach = SpeedTowardBallReward()
        self.power = PowerHitReward()
        self.face_ball = FaceBallReward()

        # hygiene
        self.step = StepPenalty()
        self.notouch = NoTouchTimeoutPressure()
        self.camp = GoalMouthCampingPenalty()

        self.ball_distance = DistanceToBallReward()

    def reset(self, agents, initial_state, shared_info):
        for r in (
            self.goal,
            self.fast_goal,
            self.ball_vel_to_goal,
            self.ball_dist_to_goal,
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
        ):
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

        # terminal-ish
        add(self.goal, cfg.w_goal)
        add(self.fast_goal, cfg.w_fast_goal)

        # score-safe, goal-directed
        add(self.ball_vel_to_goal, cfg.w_ball_vel_to_goal)
        add(self.ball_dist_to_goal, cfg.w_ball_dist_to_goal)
        add(self.shot_commit, cfg.w_shot_commit)
        add(self.align, cfg.w_align)
        add(self.hard_hit, cfg.w_hard_hit)

        # touch-stage shaping only
        add(self.touch, cfg.w_touch)
        add(self.power, cfg.w_power)
        add(self.approach, cfg.w_approach)
        add(self.face_ball, cfg.w_face_ball)

        # hygiene
        add(self.step, cfg.w_step_penalty)
        add(self.notouch, cfg.w_notouch_pressure)
        add(self.camp, cfg.w_camp_penalty)

        add(self.ball_distance, cfg.w_ball_dist)

        return rewards
