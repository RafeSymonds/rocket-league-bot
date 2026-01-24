# watch.py
import numpy as np
import torch

from rlgym_ppo import Learner
import train

RUN_FOLDER = "data/checkpoints/rlgym-ppo-run-1769267321188290800/200012"


def main():
    # IMPORTANT: render=True so RLViser can show it
    env = train.build_rlgym_v2_env(render=True, spawn_opponents=True)

    # IMPORTANT: These layer sizes MUST match what you trained with,
    # otherwise you get the "size mismatch for model.0.weight" error.
    learner = Learner(
        lambda: train.build_rlgym_v2_env(render=True, spawn_opponents=True),
        n_proc=1,
        min_inference_size=1,
        policy_layer_sizes=[2048, 2048, 1024, 1024],
        critic_layer_sizes=[2048, 2048, 1024, 1024],
    )

    learner.load(RUN_FOLDER, load_wandb=False)

    obs = env.reset()

    while True:
        # obs is typically a batch (one row per agent) in this wrapper
        actions = []
        with torch.no_grad():
            for agent_obs in obs:
                action, _ = learner.ppo_learner.policy.get_action(
                    agent_obs, deterministic=True
                )
                actions.append(action)

        # shape (num_agents, 1)
        actions = np.asarray(actions, dtype=np.int32).reshape(-1, 1)

        # Gymnasium-style step: returns 5 values (not 4)
        obs, rewards, terminated, truncated, info = env.step(actions)

        # draw current state using the renderer (RLViserRenderer)
        env.render()

        if terminated or truncated:
            obs = env.reset()


if __name__ == "__main__":
    main()
