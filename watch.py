from __future__ import annotations

import numpy as np
import torch

from rlgym_ppo import Learner

from rocket_league_bot_src.env import EnvBuilder

RUN_FOLDER = "data/checkpoints/rlgym-ppo-run-1769303493850384184/24801974"


def make_env():
    return EnvBuilder(iteration_timesteps=100_000)(process_id=0)


def main():
    env = make_env()

    learner = Learner(
        make_env,
        n_proc=1,
        min_inference_size=1,
        policy_layer_sizes=(512, 512, 256),
        critic_layer_sizes=(512, 512, 256),
    )

    learner.load(RUN_FOLDER, load_wandb=False)

    obs = env.reset()

    while True:
        actions = []
        with torch.no_grad():
            for agent_obs in obs:
                action, _ = learner.ppo_learner.policy.get_action(
                    agent_obs, deterministic=True
                )
                actions.append(action)

        actions = np.asarray(actions, dtype=np.int32).reshape(-1, 1)
        obs, rewards, terminated, truncated, info = env.step(actions)

        if terminated or truncated:
            obs = env.reset()


if __name__ == "__main__":
    main()
