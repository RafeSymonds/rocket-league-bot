"""
Train a PPO policy in RLGym with a CUSTOM observation that is easy to reproduce in RLBot.

Key changes vs your original:
- Replace DefaultObs(...) with SharedObs(...) which produces an explicit, stable vector.
- Keep LookupTableAction (+ RepeatAction) the same.
- Keep rewards / mutators / termination the same.

To deploy in RLBot:
- Recompute the SAME observation vector from GameTickPacket using the same normalization.
"""

from __future__ import annotations

import numpy as np

from rlgym_ppo import Learner
from rlgym.api import AgentID, RLGym, RewardFunction
from rlgym.rocket_league import common_values
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.done_conditions import (
    AnyCondition,
    GoalCondition,
    NoTouchTimeoutCondition,
    TimeoutCondition,
)
from rlgym.api import ObsBuilder
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import (
    FixedTeamSizeMutator,
    KickoffMutator,
    MutatorSequence,
)
from rlgym_ppo.util import RLGymV2GymWrapper

from gymnasium import spaces

# -------------------- REWARDS --------------------


class SpeedTowardBallReward(RewardFunction[AgentID, GameState, float]):
    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(self, agents, state, *_):
        rewards: dict[AgentID, float] = {}
        for agent in agents:
            car = state.cars[agent]
            car_physics = car.physics if not car.is_orange else car.inverted_physics
            ball_physics = state.ball if not car.is_orange else state.inverted_ball

            pos_diff = ball_physics.position - car_physics.position
            dist = np.linalg.norm(pos_diff)
            if dist == 0:
                rewards[agent] = 0.0
                continue

            dir_to_ball = pos_diff / dist
            speed_toward_ball = float(np.dot(car_physics.linear_velocity, dir_to_ball))
            rewards[agent] = max(speed_toward_ball / common_values.CAR_MAX_SPEED, 0.0)

        return rewards


class InAirReward(RewardFunction[AgentID, GameState, float]):
    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(self, agents, state, *_):
        return {agent: float(not state.cars[agent].on_ground) for agent in agents}


class VelocityBallToGoalReward(RewardFunction[AgentID, GameState, float]):
    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(self, agents, state, *_):
        rewards: dict[AgentID, float] = {}
        for agent in agents:
            car = state.cars[agent]
            ball = state.ball

            goal_y = (
                -common_values.BACK_NET_Y if car.is_orange else common_values.BACK_NET_Y
            )

            dir_to_goal = (
                np.array([0.0, float(goal_y), 0.0], dtype=np.float32) - ball.position
            )
            dist = np.linalg.norm(dir_to_goal)
            if dist == 0:
                rewards[agent] = 0.0
                continue

            dir_to_goal /= dist
            vel_toward_goal = float(np.dot(ball.linear_velocity, dir_to_goal))
            rewards[agent] = max(vel_toward_goal / common_values.BALL_MAX_SPEED, 0.0)

        return rewards


# -------------------- CUSTOM OBS (DEPLOYABLE IN RLBot) --------------------


class SharedObs(ObsBuilder):
    """
    Explicit observation designed to be easy to reproduce in RLBot.

    Contents (all normalized):
      Self car:
        - position (3)
        - linear velocity (3)
        - forward vector (3)
        - up vector (3)
        - angular velocity (3)
        - boost (1)
        - on_ground (1)
      Ball:
        - position (3)
        - linear velocity (3)
      Opponent (1v1):
        - position (3)
        - linear velocity (3)
        - boost (1)
        - on_ground (1)

    Total = 31 floats.

    Perspective handling:
    - We mimic DefaultObs behavior: orange sees an "inverted" world so both teams
      learn in the same coordinate system. That makes the policy symmetric.
    """

    def __init__(self):
        super().__init__()

        self.pos_coef = np.asarray(
            [
                1.0 / common_values.SIDE_WALL_X,
                1.0 / common_values.BACK_NET_Y,
                1.0 / common_values.CEILING_Z,
            ],
            dtype=np.float32,
        )
        self.lin_vel_coef = 1.0 / common_values.CAR_MAX_SPEED
        self.ang_vel_coef = 1.0 / common_values.CAR_MAX_ANG_VEL
        self.boost_coef = 1.0 / 100.0
        self.ball_vel_coef = 1.0 / common_values.BALL_MAX_SPEED

        self.max_dist = np.linalg.norm(
            np.array(
                [
                    common_values.SIDE_WALL_X,
                    common_values.BACK_NET_Y,
                    common_values.CEILING_Z,
                ],
                dtype=np.float32,
            )
        )
        self.dist_coef = 1.0 / self.max_dist

    def _dir_and_dist(self, src: np.ndarray, dst: np.ndarray):
        delta = dst - src
        dist = np.linalg.norm(delta)
        if dist > 1e-6:
            return delta / dist, dist
        return np.zeros(3, dtype=np.float32), 0.0

    def reset(self, agents, initial_state, shared_info):
        pass

    def get_obs_space(self, agent):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(31,),
            dtype=np.float32,
        ), None

    def build_obs(self, agents, state: GameState, shared_info):
        obs_dict = {}

        for agent in agents:
            car = state.cars[agent]

            # Perspective normalization
            car_phys = car.physics if not car.is_orange else car.inverted_physics
            ball_phys = state.ball if not car.is_orange else state.inverted_ball

            # Find opponent (1v1)
            opp = next(a for a in agents if a != agent)
            opp_car = state.cars[opp]
            opp_phys = (
                opp_car.physics if not car.is_orange else opp_car.inverted_physics
            )

            to_ball_dir, to_ball_dist = self._dir_and_dist(
                car_phys.position, ball_phys.position
            )

            to_opp_dir, to_opp_dist = self._dir_and_dist(
                car_phys.position, opp_phys.position
            )

            ball_to_opp_dir, ball_to_opp_dist = self._dir_and_dist(
                ball_phys.position, opp_phys.position
            )

            obs = np.concatenate(
                [
                    # self
                    car_phys.position * self.pos_coef,
                    car_phys.linear_velocity * self.lin_vel_coef,
                    car_phys.forward,
                    car_phys.up,
                    car_phys.angular_velocity * self.ang_vel_coef,
                    np.array([car.boost_amount * self.boost_coef], dtype=np.float32),
                    np.array([float(car.on_ground)], dtype=np.float32),
                    # ball
                    ball_phys.position * self.pos_coef,
                    ball_phys.linear_velocity * self.ball_vel_coef,
                    # opponent
                    opp_phys.position * self.pos_coef,
                    opp_phys.linear_velocity * self.lin_vel_coef,
                    np.array(
                        [opp_car.boost_amount * self.boost_coef], dtype=np.float32
                    ),
                    np.array([float(opp_car.on_ground)], dtype=np.float32),
                    # relative directions
                    to_ball_dir,
                    np.array([to_ball_dist * self.dist_coef], dtype=np.float32),
                    to_opp_dir,
                    np.array([to_opp_dist * self.dist_coef], dtype=np.float32),
                    ball_to_opp_dir,
                    np.array([ball_to_opp_dist * self.dist_coef], dtype=np.float32),
                ],
                axis=0,
            ).astype(np.float32)

            obs_dict[agent] = obs

        return obs_dict


# -------------------- ENV BUILDER --------------------


def build_rlgym_v2_env():
    action_parser = RepeatAction(LookupTableAction(), repeats=8)

    reward_fn = CombinedReward(
        (InAirReward(), 0.002),
        (SpeedTowardBallReward(), 0.01),
        (VelocityBallToGoalReward(), 0.1),
        (GoalReward(), 10.0),
    )

    # ✅ Replace DefaultObs with our deployable SharedObs
    obs_builder = SharedObs()

    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=1, orange_size=1),
        KickoffMutator(),
    )

    env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=GoalCondition(),
        truncation_cond=AnyCondition(
            NoTouchTimeoutCondition(30),
            TimeoutCondition(300),
        ),
        transition_engine=RocketSimEngine(),
    )

    return RLGymV2GymWrapper(env)


def main():
    n_proc = 16
    min_inference_size = max(1, int(n_proc * 0.8))

    learner = Learner(
        build_rlgym_v2_env,
        n_proc=n_proc,
        min_inference_size=min_inference_size,
        # NETWORK
        policy_layer_sizes=(2048, 2048, 1024, 1024),
        critic_layer_sizes=(2048, 2048, 1024, 1024),
        # PPO
        ppo_batch_size=200_000,
        ppo_minibatch_size=50_000,
        ppo_epochs=2,
        ppo_ent_coef=0.01,
        # LRs
        policy_lr=1e-4,
        critic_lr=1e-4,
        # TRAINING CONTROL
        ts_per_iteration=200_000,
        exp_buffer_size=400_000,
        timestep_limit=1_000_000_000,
        # LOGGING
        metrics_logger=None,
        log_to_wandb=False,
        save_every_ts=1_000_000,
    )

    learner.learn()


if __name__ == "__main__":
    main()
