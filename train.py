from __future__ import annotations

import argparse

from rlgym_ppo import Learner

from rocket_league_bot_src.checkpoints import (
    find_latest_checkpoint,
    find_latest_compatible_checkpoint,
    load_curriculum_state_from_checkpoint,
    load_checkpoint_book,
)
from rocket_league_bot_src.config import (
    CRITIC_LAYER_SIZES,
    DEFAULT_CHECKPOINT_ROOT,
    OBS_DIM,
    POLICY_LAYER_SIZES,
)
from rocket_league_bot_src.env import EnvBuilder


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
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--load-path", type=str, default="")
    parser.add_argument("--checkpoint-root", type=str, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--resume-latest", dest="resume_latest", action="store_true")
    parser.add_argument("--no-resume-latest", dest="resume_latest", action="store_false")
    parser.set_defaults(resume_latest=True)

    return parser.parse_args()

def main():
    args = parse_args()

    load_path = args.load_path
    if not load_path and args.resume_latest:
        load_path = find_latest_compatible_checkpoint(args.checkpoint_root, expected_obs_dim=OBS_DIM)
        latest_checkpoint = find_latest_checkpoint(args.checkpoint_root)
        if latest_checkpoint and load_path and latest_checkpoint != load_path:
            latest_book = load_checkpoint_book(latest_checkpoint)
            latest_obs_shape = latest_book.get("obs_running_stats", {}).get("shape")
            print(
                "Skipping latest checkpoint due to observation mismatch: "
                f"{latest_checkpoint} has obs shape {latest_obs_shape}, expected {[OBS_DIM]}."
            )
        elif latest_checkpoint and not load_path:
            latest_book = load_checkpoint_book(latest_checkpoint)
            latest_obs_shape = latest_book.get("obs_running_stats", {}).get("shape")
            print(
                "No compatible checkpoint found to resume. "
                f"Latest checkpoint {latest_checkpoint} has obs shape {latest_obs_shape}, expected {[OBS_DIM]}."
            )

    initial_curriculum_state = load_curriculum_state_from_checkpoint(load_path) if load_path else {}
    env_builder = EnvBuilder(
        iteration_timesteps=int(args.ts_per_iteration),
        checkpoint_root=args.checkpoint_root,
        n_proc=int(args.n_proc),
        initial_curriculum_state=initial_curriculum_state,
    )

    learner = Learner(
        env_builder,
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
        checkpoint_load_folder=None,
        device=str(args.device),
    )

    if load_path:
        print(f"Resuming from checkpoint: {load_path}")
        learner.load(load_path, load_wandb=False)
    else:
        print("Starting fresh training run (no checkpoint loaded).")

    learner.learn()


if __name__ == "__main__":
    main()
