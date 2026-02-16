from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np

from rlgym.api import RLGym
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import (
    KickoffMutator,
    MutatorSequence,
)
from rlgym_ppo.util import RLGymV2GymWrapper

from .config import Stage
from .conditions import CurriculumDoneCondition, CurriculumTruncationCondition
from .curriculum import CurriculumManager
from .mutators import BallNearCarMutator, DynamicTeamSizeMutator, ProgressiveResetMutator
from .obs import SharedObs
from .rewards import CurriculumReward
from .utils import CurriculumValue, Stats


class ProcessIterationLogger:  # No longer inherits from gym.Wrapper
    def __init__(
        self,
        env,
        process_id: int,
        iteration_timesteps: int,
        curriculum_manager: "CurriculumManager",
    ):
        self.env = env
        self.pid = process_id
        self.iteration_ts = iteration_timesteps
        self.cm = curriculum_manager
        self.log_counter = 0

        # Manually expose observation_space and action_space
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self._reset_iteration_stats()
        self._reset_episode_stats()

    def _reset_iteration_stats(self):
        self.iteration_start_time = time.time()
        self.iteration_steps = 0
        self.iteration_episodes = 0
        self.iteration_return = 0.0

        self.iteration_goals = 0
        self.iteration_touches = 0
        self.iteration_success_eps = 0  # Touched or scored
        self.iteration_median_t_first = []
        self.iteration_median_t_goal = []

    def _reset_episode_stats(self):
        self.ep_return = 0.0
        self.ep_steps = 0
        self.ep_ball_touches = 0
        self.ep_first_touch_step = -1
        self.ep_goal_step = -1
        self._prev_touches = {}

    def close(self, **kwargs):
        pass

    def reset(self, **kwargs):
        # This is called at the start of a new episode
        self._reset_episode_stats()

        res = self.env.reset(**kwargs)
        if len(res) == 2:  # gymnasium API
            obs, info = res
        else:  # old gym API
            obs = res
            info = {}

        # rlgym-ppo expects a list of observations, not a single concatenated array.
        # It also expects env.reset() to return just the observation, not a tuple of (obs, info).
        # By returning just `obs` here, we comply with what rlgym-ppo's batched agent expects.

        state = info.get("state")
        if state is not None:
            for a, car in state.cars.items():
                self._prev_touches[a] = int(car.ball_touches)

        return obs

    def step(self, action):
        res = self.env.step(action)
        if len(res) == 5:  # gymnasium API
            obs, reward, terminated, truncated, info = res
        else:  # old gym API
            obs, reward, done, info = res
            terminated = done
            truncated = (
                False  # Assume no truncation for now, or infer from info if available
            )

        self.iteration_steps += 1
        self.ep_steps += 1
        self.ep_return += float(np.sum(reward))

        state = info.get("state")
        if state:
            for agent, car in state.cars.items():
                prev = self._prev_touches.get(agent, int(car.ball_touches))
                cur = int(car.ball_touches)
                self._prev_touches[agent] = cur
                if cur > prev:
                    self.ep_ball_touches += 1
                    if self.ep_first_touch_step == -1:
                        self.ep_first_touch_step = self.ep_steps

            if state.goal_scored and self.ep_goal_step == -1:
                self.ep_goal_step = self.ep_steps

        done = terminated or truncated
        if done:
            self.iteration_episodes += 1
            self.iteration_return += self.ep_return

            if self.ep_ball_touches > 0:
                self.iteration_success_eps += 1
                self.iteration_median_t_first.append(self.ep_first_touch_step)

            if self.ep_goal_step != -1:
                self.iteration_goals += 1
                self.iteration_median_t_goal.append(self.ep_goal_step)

            self._reset_episode_stats()

        if self.iteration_steps >= self.iteration_ts:
            self._report_and_reset_iteration()

        return obs, reward, terminated, truncated, info

    def _report_and_reset_iteration(self):
        # Calculate stats
        avg_return = (
            self.iteration_return / self.iteration_episodes
            if self.iteration_episodes > 0
            else 0
        )
        touch_rate = (
            self.iteration_success_eps / self.iteration_episodes
            if self.iteration_episodes > 0
            else 0
        )
        goal_rate = (
            self.iteration_goals / self.iteration_episodes
            if self.iteration_episodes > 0
            else 0
        )

        median_t_first = (
            np.median(self.iteration_median_t_first)
            if self.iteration_median_t_first
            else -1
        )
        median_t_goal = (
            np.median(self.iteration_median_t_goal)
            if self.iteration_median_t_goal
            else -1
        )

        # Report
        duration = time.time() - self.iteration_start_time
        sps = self.iteration_steps / duration

        stage = self.cm.stage_ref.stage.value

        self.log_counter += 1
        if self.pid == 0 and self.log_counter % 5 == 1:
            print(
                f"[P-{self.pid:02d} | {stage:<8}] "
                f"SPS: {sps:7.1f} | "
                f"Eps: {self.iteration_episodes:4d} | "
                f"Avg Ret: {avg_return:8.2f} | "
                f"Touch Rate: {touch_rate:5.2f} | "
                f"Goal Rate: {goal_rate:5.2f} | "
                f"Med T_Touch: {median_t_first:5.0f} | "
                f"Med T_Goal: {median_t_goal:5.0f}"
            )

        # Advance curriculum
        if self.cm:
            stats = Stats()
            stats.touch_rate = touch_rate
            stats.goal_rate = goal_rate
            stats.median_t_first = float(
                median_t_first if median_t_first != -1 else 9999
            )
            stats.median_t_goal = float(median_t_goal if median_t_goal != -1 else 9999)
            self.cm.maybe_advance(stats)

        # Reset
        self._reset_iteration_stats()


class EnvBuilder:
    def __init__(self, iteration_timesteps: int):
        # curriculum knobs
        self.min_dist = CurriculumValue(300)
        self.max_dist = CurriculumValue(400)
        self.max_angle = CurriculumValue(10)
        self.ball_velocity = CurriculumValue(0.0)
        self.p_easy_reset = CurriculumValue(1.0)

        # stage is mutable and shared across resets
        self.stage: Stage = Stage.TOUCH
        self.iteration_timesteps = iteration_timesteps

    def __call__(self, process_id: int):
        action_parser = RepeatAction(LookupTableAction(), repeats=2)
        reward_fn = CurriculumReward(stage_ref=self)

        termination_cond = CurriculumDoneCondition(stage_ref=self)
        truncation_cond = CurriculumTruncationCondition(stage_ref=self)

        reset_mutator = ProgressiveResetMutator(
            easy_mutator=BallNearCarMutator(
                self.min_dist,
                self.max_dist,
                self.max_angle,
                self.ball_velocity,
            ),
            p_easy=self.p_easy_reset,
        )

        env = RLGym(
            state_mutator=MutatorSequence(
                DynamicTeamSizeMutator(self),
                KickoffMutator(),
                reset_mutator,
            ),
            obs_builder=SharedObs(),
            action_parser=action_parser,
            reward_fn=reward_fn,
            termination_cond=termination_cond,
            truncation_cond=truncation_cond,
            transition_engine=RocketSimEngine(),
        )

        base = RLGymV2GymWrapper(env)

        curriculum_manager = CurriculumManager(
            min_dist=self.min_dist,
            max_dist=self.max_dist,
            max_angle=self.max_angle,
            ball_velocity=self.ball_velocity,
            p_easy_reset=self.p_easy_reset,
            stage_ref=self,
        )

        wrapped = ProcessIterationLogger(
            base,
            process_id=process_id,
            iteration_timesteps=self.iteration_timesteps,
            curriculum_manager=curriculum_manager,
        )

        return wrapped
