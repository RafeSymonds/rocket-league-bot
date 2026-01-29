from __future__ import annotations


from rlgym_ppo import Learner

from rocket_league_bot_src.env import EnvBuilder


# ==================================================
# Training
# ==================================================

_global_iteration_timesteps: int = 0


def _create_rlgym_env(process_id: int = 0):  # Provide a default value for process_id
    global _global_iteration_timesteps
    env_builder_instance = EnvBuilder(iteration_timesteps=_global_iteration_timesteps)
    return env_builder_instance(process_id)


def main():
    global _global_iteration_timesteps
    ts_per_iteration = 100_000
    _global_iteration_timesteps = ts_per_iteration

    learner = Learner(
        _create_rlgym_env,
        n_proc=1,
        min_inference_size=12,
        policy_layer_sizes=(512, 512, 256),
        critic_layer_sizes=(512, 512, 256),
        ppo_batch_size=100_000,
        ppo_minibatch_size=25_000,
        ppo_epochs=2,
        ppo_ent_coef=0.01,
        policy_lr=3e-4,
        critic_lr=3e-4,
        ts_per_iteration=ts_per_iteration,
        exp_buffer_size=400_000,
        timestep_limit=1_000_000_000,
        log_to_wandb=False,
        save_every_ts=5_000_000,
    )

    learner.learn()


if __name__ == "__main__":
    main()
