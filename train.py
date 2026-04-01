from __future__ import annotations

import argparse

from rlgym_ppo import Learner

from rocket_league_bot_src.checkpoints import find_latest_checkpoint
from rocket_league_bot_src.config import (
    CRITIC_LAYER_SIZES,
    DEFAULT_CHECKPOINT_ROOT,
    POLICY_LAYER_SIZES,
)
from rocket_league_bot_src.env import EnvBuilder


_global_iteration_timesteps: int = 0
_global_env_builder: EnvBuilder | None = None


def _create_rlgym_env(process_id: int = 0):
    if _global_env_builder is None:
        raise RuntimeError("Environment builder was not initialized")
    return _global_env_builder(process_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Rocket League bot with curriculum.")

    parser.add_argument("--n-proc", type=int, default=1)
    parser.add_argument("--ts-per-iteration", type=int, default=50_000)
    parser.add_argument("--ppo-batch-size", type=int, default=50_000)
    parser.add_argument("--ppo-minibatch-size", type=int, default=10_000)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--policy-lr", type=float, default=2.5e-4)
    parser.add_argument("--critic-lr", type=float, default=2.5e-4)
    parser.add_argument("--ent-coef", type=float, default=0.003)
    parser.add_argument("--exp-buffer-size", type=int, default=200_000)
    parser.add_argument("--save-every-ts", type=int, default=2_000_000)
    parser.add_argument("--timestep-limit", type=int, default=1_000_000_000)
    parser.add_argument("--min-inference-size", type=int, default=1)
    parser.add_argument("--load-path", type=str, default="")
    parser.add_argument("--checkpoint-root", type=str, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--resume-latest", dest="resume_latest", action="store_true")
    parser.add_argument("--no-resume-latest", dest="resume_latest", action="store_false")
    parser.set_defaults(resume_latest=True)

    return parser.parse_args()

def main():
    global _global_iteration_timesteps, _global_env_builder
    args = parse_args()

    _global_iteration_timesteps = int(args.ts_per_iteration)
    _global_env_builder = EnvBuilder(
        iteration_timesteps=_global_iteration_timesteps,
        checkpoint_root=args.checkpoint_root,
    )

    learner = Learner(
        _create_rlgym_env,
        n_proc=int(args.n_proc),
        min_inference_size=int(args.min_inference_size),
        policy_layer_sizes=POLICY_LAYER_SIZES,
        critic_layer_sizes=CRITIC_LAYER_SIZES,
        ppo_batch_size=int(args.ppo_batch_size),
        ppo_minibatch_size=int(args.ppo_minibatch_size),
        ppo_epochs=int(args.ppo_epochs),
        ppo_ent_coef=float(args.ent_coef),
        policy_lr=float(args.policy_lr),
        critic_lr=float(args.critic_lr),
        ts_per_iteration=int(args.ts_per_iteration),
        exp_buffer_size=int(args.exp_buffer_size),
        timestep_limit=int(args.timestep_limit),
        log_to_wandb=False,
        save_every_ts=int(args.save_every_ts),
    )

    load_path = args.load_path
    if not load_path and args.resume_latest:
        load_path = find_latest_checkpoint(args.checkpoint_root)

    if load_path:
        print(f"Resuming from checkpoint: {load_path}")
        learner.load(load_path, load_wandb=False)
    else:
        print("Starting fresh training run (no checkpoint loaded).")

    learner.learn()


if __name__ == "__main__":
    main()
