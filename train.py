from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rlgym_ppo import Learner

from rocket_league_bot_src.checkpoints import (
    find_latest_checkpoint,
    find_latest_compatible_checkpoint,
    find_opponent_checkpoint,
    load_curriculum_state_from_checkpoint,
    load_checkpoint_book,
)
from rocket_league_bot_src.config import (
    CRITIC_LAYER_SIZES,
    DEFAULT_CHECKPOINT_ROOT,
    OBS_DIM,
    POLICY_LAYER_SIZES,
    Stage,
)
from rocket_league_bot_src.env import EnvBuilder


def _disable_background_kbhit() -> None:
    if sys.stdin.isatty():
        return

    class _NoOpKBHit:
        def kbhit(self) -> bool:
            return False

        def getch(self) -> str:
            return ""

        def set_normal_term(self) -> None:
            return None

    try:
        import rlgym_ppo.learner as learner_module

        learner_module.KBHit = _NoOpKBHit
    except Exception:
        pass


def _attach_curriculum_checkpoint_hook(
    learner: Learner, env_builder: EnvBuilder
) -> None:
    original_save = learner.save

    def save_with_curriculum(cumulative_timesteps):
        original_save(cumulative_timesteps)
        checkpoint_dir = Path(learner.checkpoints_save_folder) / str(
            cumulative_timesteps
        )
        book_path = checkpoint_dir / "BOOK_KEEPING_VARS.json"
        if not book_path.exists():
            return
        try:
            book = json.loads(book_path.read_text())
        except Exception:
            return
        book["curriculum_state"] = env_builder.curriculum_manager.to_dict()
        book_path.write_text(json.dumps(book, indent=4))

    learner.save = save_with_curriculum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Rocket League bot with curriculum."
    )

    parser.add_argument("--n-proc", type=int, default=1)
    parser.add_argument("--ts-per-iteration", type=int, default=50_000)
    parser.add_argument("--ppo-batch-size", type=int, default=100_000)
    parser.add_argument("--ppo-minibatch-size", type=int, default=10_000)
    parser.add_argument("--ppo-epochs", type=int, default=25)
    parser.add_argument("--policy-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--exp-buffer-size", type=int, default=200_000)
    parser.add_argument("--save-every-ts", type=int, default=2_000_000)
    parser.add_argument("--timestep-limit", type=int, default=1_000_000_000)
    parser.add_argument("--min-inference-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--opponent-device", type=str, default="gpu")
    parser.add_argument(
        "--self-play-mode", choices=("current", "frozen"), default="current"
    )
    parser.add_argument("--load-path", type=str, default="")
    parser.add_argument("--checkpoint-root", type=str, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--force-stage", type=str, default="")
    parser.add_argument("--force-difficulty", type=float, default=None)
    parser.add_argument("--opponent-checkpoint", type=str, default="")
    parser.add_argument("--opponent-gap-ts", type=int, default=4_000_000)
    parser.add_argument("--resume-latest", dest="resume_latest", action="store_true")
    parser.add_argument(
        "--no-resume-latest", dest="resume_latest", action="store_false"
    )
    parser.add_argument(
        "--replay-folder",
        type=str,
        default="",
        help="Path to parsed replay data for replay-based resets",
    )
    parser.add_argument(
        "--use-continuous-actions",
        dest="use_discrete_actions",
        action="store_false",
        help="Use continuous actions instead of discrete",
    )
    parser.set_defaults(resume_latest=True)

    return parser.parse_args()


def main():
    args = parse_args()
    _disable_background_kbhit()

    load_path = args.load_path
    if not load_path and args.resume_latest:
        load_path = find_latest_compatible_checkpoint(
            args.checkpoint_root, expected_obs_dim=OBS_DIM
        )
        latest_checkpoint = find_latest_checkpoint(args.checkpoint_root)
        if latest_checkpoint and load_path and latest_checkpoint != load_path:
            latest_book = load_checkpoint_book(latest_checkpoint)
            latest_obs_shape = latest_book.get("obs_running_stats", {}).get("shape")
            selected_book = load_checkpoint_book(load_path)
            selected_obs_shape = selected_book.get("obs_running_stats", {}).get("shape")
            if latest_obs_shape != [OBS_DIM]:
                print(
                    "Skipping latest checkpoint due to observation mismatch: "
                    f"{latest_checkpoint} has obs shape {latest_obs_shape}, expected {[OBS_DIM]}."
                )
            else:
                latest_stage = latest_book.get("curriculum_state", {}).get("stage")
                selected_stage = selected_book.get("curriculum_state", {}).get("stage")
                print(
                    "Skipping raw-latest checkpoint in favor of a better resume candidate: "
                    f"selected {load_path} (stage={selected_stage}, obs={selected_obs_shape}) "
                    f"over {latest_checkpoint} (stage={latest_stage}, obs={latest_obs_shape})."
                )
        elif latest_checkpoint and not load_path:
            latest_book = load_checkpoint_book(latest_checkpoint)
            latest_obs_shape = latest_book.get("obs_running_stats", {}).get("shape")
            print(
                "No compatible checkpoint found to resume. "
                f"Latest checkpoint {latest_checkpoint} has obs shape {latest_obs_shape}, expected {[OBS_DIM]}."
            )

    initial_curriculum_state = (
        load_curriculum_state_from_checkpoint(load_path) if load_path else {}
    )
    opponent_checkpoint = args.opponent_checkpoint
    if args.self_play_mode == "frozen" and not opponent_checkpoint and load_path:
        current_book = load_checkpoint_book(load_path)
        current_ts = int(current_book.get("cumulative_timesteps", 0))
        opponent_checkpoint = find_opponent_checkpoint(
            args.checkpoint_root,
            current_ts=current_ts,
            gap_ts=int(args.opponent_gap_ts),
            exclude_checkpoint_dir=load_path,
        )
        if opponent_checkpoint:
            print(
                f"Using opponent checkpoint: {opponent_checkpoint} "
                f"(target gap {int(args.opponent_gap_ts)})"
            )
    elif args.self_play_mode == "frozen" and opponent_checkpoint:
        print(f"Using fixed opponent checkpoint: {opponent_checkpoint}")
    elif args.self_play_mode == "current" and opponent_checkpoint:
        print(
            "Ignoring --opponent-checkpoint because --self-play-mode current was selected."
        )

    if args.force_stage:
        stage = Stage[str(args.force_stage).upper()]
        initial_curriculum_state = {
            "stage": stage.value,
            "difficulty": float(args.force_difficulty or 0.0),
            "stage_iterations": 0,
            "ema_touch_rate": 0.0,
            "ema_goal_rate": 0.0,
        }
        print(
            f"Forcing curriculum stage to {stage.value} "
            f"at difficulty {float(args.force_difficulty or 0.0):.3f}."
        )
    env_builder = EnvBuilder(
        iteration_timesteps=int(args.ts_per_iteration),
        checkpoint_root=args.checkpoint_root,
        n_proc=int(args.n_proc),
        initial_curriculum_state=initial_curriculum_state,
        current_checkpoint_dir=load_path,
        self_play_mode=str(args.self_play_mode),
        fixed_opponent_checkpoint=opponent_checkpoint,
        opponent_gap_ts=int(args.opponent_gap_ts),
        opponent_device=str(args.opponent_device),
        replay_folder=args.replay_folder if args.replay_folder else None,
        use_discrete_actions=args.use_discrete_actions,
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
    _attach_curriculum_checkpoint_hook(learner, env_builder)

    if load_path:
        print(f"Resuming from checkpoint: {load_path}")
        learner.load(load_path, load_wandb=False)
    else:
        print("Starting fresh training run (no checkpoint loaded).")

    learner.learn()


if __name__ == "__main__":
    main()
