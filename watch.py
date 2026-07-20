from __future__ import annotations

import numpy as np
import torch

from rlgym_ppo import Learner

from rocket_league_bot_src.checkpoints import (
    find_latest_checkpoint,
    load_obs_standardizer,
    standardize_obs,
)
from rocket_league_bot_src.config import (
    CRITIC_LAYER_SIZES,
    DEFAULT_CHECKPOINT_ROOT,
    POLICY_LAYER_SIZES,
)
from rocket_league_bot_src.env import EnvBuilder

RUN_FOLDER = find_latest_checkpoint(DEFAULT_CHECKPOINT_ROOT)


def make_env():
    return EnvBuilder(iteration_timesteps=100_000)(process_id=0)


def main():
    env = make_env()

    learner = Learner(
        make_env,
        n_proc=1,
        min_inference_size=1,
        policy_layer_sizes=POLICY_LAYER_SIZES,
        critic_layer_sizes=CRITIC_LAYER_SIZES,
    )

    if not RUN_FOLDER:
        raise RuntimeError("No checkpoint found under data/checkpoints")

    learner.load(RUN_FOLDER, load_wandb=False)

    # Checkpoints trained with rlgym-ppo's standardize_obs carry running stats
    # in their book; the policy must see the same transform at replay time.
    standardizer = load_obs_standardizer(RUN_FOLDER)

    obs = env.reset()

    while True:
        actions = []
        with torch.no_grad():
            for agent_obs in obs:
                agent_obs = standardize_obs(
                    np.asarray(agent_obs, dtype=np.float32), standardizer
                )
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
