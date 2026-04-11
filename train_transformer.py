from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from rlgym.api import RLGym
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym_ppo.util.rlgym_v2_gym_wrapper import RLGymV2GymWrapper

from rocket_league_bot_src.action_parser import NectoAction
from rocket_league_bot_src.config import (
    ACTION_REPEAT,
    USE_DISCRETE_ACTIONS,
    EARL_EMBED_DIM,
    EARL_KV_FEATURES,
    EARL_NUM_HEADS,
    EARL_NUM_LAYERS,
    EARL_QUERY_FEATURES,
    NUM_DISCRETE_ACTIONS,
)
from rocket_league_bot_src.curriculum import CurriculumManager
from rocket_league_bot_src.obs_transformer import TransformerObs
from rocket_league_bot_src.rewards import CurriculumReward
from rocket_league_bot_src.transformer_policy import (
    TransformerPolicy,
    make_lookup_table,
)
from rocket_league_bot_src.conditions import (
    CurriculumDoneCondition,
    CurriculumTruncationCondition,
)
from rocket_league_bot_src.checkpoints import find_latest_checkpoint
from rocket_league_bot_src.opponent import SelfPlayOpponentGymWrapper


try:
    from rlgym_tools.rocket_league.shared_info_providers.scoreboard_provider import (
        ScoreboardProvider,
    )
except Exception:
    ScoreboardProvider = None

try:
    from rocket_league_bot_src.mutators_with_replay import DynamicMatchMutatorWithReplay
except ImportError:
    DynamicMatchMutatorWithReplay = None

try:
    from rocket_league_bot_src.mutators import DynamicMatchMutator
except ImportError:
    DynamicMatchMutator = None


ACTION_LOOKUP = make_lookup_table()
NUM_ACTIONS = len(ACTION_LOOKUP)


@dataclass
class RolloutData:
    observations: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    advantages: np.ndarray = None
    returns: np.ndarray = None


class TransformerPPO:
    def __init__(
        self,
        policy: TransformerPolicy,
        target_policy: TransformerPolicy,
        lr: float = 1e-4,
        ent_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        clip_eps: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
    ):
        self.policy = policy
        self.target_policy = target_policy
        self.lr = lr
        self.ent_coef = ent_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.lam = lam

        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    def predict(self, obs, deterministic=False):
        device = next(self.policy.parameters()).device
        q, kv, m = obs
        q_t = torch.as_tensor(q).float().to(device)
        kv_t = torch.as_tensor(kv).float().to(device).unsqueeze(0)
        m_t = torch.as_tensor(m).float().to(device).unsqueeze(0)

        with torch.no_grad():
            logits, _ = self.policy((q_t, kv_t, m_t))
            probs = F.softmax(logits, dim=-1)
            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
            log_prob = torch.log(probs + 1e-8)
            chosen_log_prob = torch.gather(log_prob, -1, action.unsqueeze(-1)).squeeze(
                -1
            )

        return action.item(), chosen_log_prob.item()

    def evaluate(self, obs, actions):
        device = next(self.policy.parameters()).device
        q, kv, m = obs
        q_t = torch.as_tensor(q).float().to(device)
        kv_t = torch.as_tensor(kv).float().to(device).unsqueeze(0)
        m_t = torch.as_tensor(m).float().to(device).unsqueeze(0)
        actions_t = torch.as_tensor(actions).long().to(device)

        logits, _ = self.policy((q_t, kv_t, m_t))
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        log_probs = torch.log(probs + 1e-8)
        action_log_probs = torch.gather(log_probs, -1, actions_t.unsqueeze(-1)).squeeze(
            -1
        )
        entropy = dist.entropy()

        with torch.no_grad():
            values = self.target_policy.critic((q_t, kv_t, m_t)).squeeze(-1)

        return action_log_probs, values, entropy

    def update(self, rollouts: RolloutData):
        device = next(self.policy.parameters()).device
        observations = rollouts.observations
        actions = torch.as_tensor(rollouts.actions).long().to(device)
        old_log_probs = torch.as_tensor(rollouts.log_probs).float().to(device)
        returns = torch.as_tensor(rollouts.returns).float().to(device)
        advantages = torch.as_tensor(rollouts.advantages).float().to(device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        entropy_loss_sum = 0.0
        count = 0

        for _ in range(4):
            indices = torch.randperm(len(observations))

            for idx in indices:
                obs = observations[idx]
                act = actions[idx]
                old_lp = old_log_probs[idx]
                ret = returns[idx]
                adv = advantages[idx]

                new_log_probs, values, entropy = self.evaluate(obs, act.unsqueeze(0))
                new_log_probs = new_log_probs.squeeze(-1)
                values = values.squeeze(-1)

                ratio = torch.exp(new_log_probs - old_lp)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                entropy_loss = -self.ent_coef * entropy.mean()
                value_loss = self.value_coef * F.mse_loss(values, ret)

                loss = policy_loss + entropy_loss + value_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                entropy_loss_sum += entropy_loss.item()
                count += 1

        self.target_policy.load_state_dict(self.policy.state_dict())

        return (
            policy_loss_sum / max(count, 1),
            value_loss_sum / max(count, 1),
            entropy_loss_sum / max(count, 1),
        )


def compute_gae(
    rewards: List[float],
    dones: List[bool],
    values: List[float],
    next_value: float,
    gamma: float,
    lam: float,
):
    advantages = []
    gae = 0
    values = list(values) + [next_value]

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    returns = np.array(advantages) + np.array(values[:-1])
    return np.array(advantages), returns


class TransformerEnvBuilder:
    def __init__(
        self,
        iteration_timesteps: int,
        checkpoint_root: str = "data/checkpoints_transformer",
        n_proc: int = 1,
        initial_curriculum_state: dict | None = None,
        current_checkpoint_dir: str = "",
        self_play_mode: str = "current",
        fixed_opponent_checkpoint: str = "",
        opponent_gap_ts: int = 4_000_000,
        opponent_device: str = "gpu",
        replay_folder: str = "",
        use_discrete_actions: bool = USE_DISCRETE_ACTIONS,
    ):
        self.iteration_timesteps = iteration_timesteps
        self.checkpoint_root = checkpoint_root
        self.n_proc = max(1, int(n_proc))
        self.curriculum_manager = CurriculumManager()
        self.curriculum_manager.load_dict(initial_curriculum_state or {})
        self.curriculum_state_path = str(
            Path("data") / "transformer_curriculum_state.json"
        )
        self.opponent_state_path = str(Path("data") / "transformer_opponent_state.json")
        self.current_checkpoint_dir = current_checkpoint_dir
        self.self_play_mode = str(self_play_mode)
        self.fixed_opponent_checkpoint = fixed_opponent_checkpoint
        self.opponent_gap_ts = int(opponent_gap_ts)
        self.opponent_device = opponent_device
        self.replay_folder = replay_folder or ""
        self.use_discrete_actions = use_discrete_actions
        Path(self.curriculum_state_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.curriculum_state_path).write_text(
            json.dumps(self.curriculum_manager.to_dict(), indent=2, sort_keys=True)
        )

    def __call__(self, process_id: int = None):
        if process_id is None:
            import multiprocessing as mp

            process = mp.current_process()
            process_id = int(process._identity[0] - 1) if process._identity else 0

        curriculum_manager = self.curriculum_manager

        if self.use_discrete_actions:
            action_parser = NectoAction()
        else:
            action_parser = RepeatAction(LookupTableAction(), repeats=ACTION_REPEAT)

        if self.replay_folder and DynamicMatchMutatorWithReplay is not None:
            state_mutator = DynamicMatchMutatorWithReplay(
                curriculum_manager=curriculum_manager,
                replay_folder=self.replay_folder,
                replay_reset_probability=0.7,
                use_lazy_loading=True,
            )
        elif DynamicMatchMutator is not None:
            state_mutator = DynamicMatchMutator(curriculum_manager)
        else:
            raise RuntimeError("No state mutator available. Install rlgym-tools.")

        env = RLGym(
            state_mutator=state_mutator,
            obs_builder=TransformerObs(),
            action_parser=action_parser,
            reward_fn=CurriculumReward(curriculum_manager),
            termination_cond=CurriculumDoneCondition(curriculum_manager),
            truncation_cond=CurriculumTruncationCondition(curriculum_manager),
            transition_engine=RocketSimEngine(),
            **(
                {"shared_info_provider": ScoreboardProvider()}
                if ScoreboardProvider is not None
                else {}
            ),
        )

        if self.self_play_mode == "frozen":
            gym_env = SelfPlayOpponentGymWrapper(
                env,
                opponent_state_path=self.opponent_state_path,
                deterministic_opponent=False,
                device=self.opponent_device,
            )
        else:
            gym_env = env

        return gym_env


def collect_rollout(env, policy: TransformerPPO, n_steps: int, device: str):
    observations = []
    actions = []
    rewards = []
    dones = []
    log_probs = []
    values = []

    obs_dict = env.reset()

    for _ in range(n_steps):
        for agent_id, obs in obs_dict.items():
            q, kv, m = obs

            q_t = torch.from_numpy(q).float().to(device)
            kv_t = torch.from_numpy(kv).float().to(device).unsqueeze(0)
            m_t = torch.from_numpy(m).float().to(device).unsqueeze(0)

            logits, _ = policy.policy((q_t, kv_t, m_t))
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = torch.log(probs + 1e-8)
            chosen_log_prob = torch.gather(log_prob, -1, action.unsqueeze(-1)).squeeze(
                -1
            )

            with torch.no_grad():
                value = policy.target_policy.critic((q_t, kv_t, m_t)).item()

            action_np = action.cpu().numpy()
            observations.append((q, kv, m))
            actions.append(action.item())
            log_probs.append(chosen_log_prob.item())
            values.append(value)

            action_dict = {agent_id: action_np}
            obs_dict, reward_dict, terminated_dict, truncated_dict = env.step(
                action_dict
            )

            for agent_id, reward in reward_dict.items():
                rewards.append(float(reward))
                done = terminated_dict[agent_id] or truncated_dict[agent_id]
                dones.append(done)

    return (
        observations,
        np.array(actions),
        np.array(rewards),
        np.array(dones),
        np.array(log_probs),
        np.array(values),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-proc", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--save-interval", type=int, default=50000)
    parser.add_argument("--max-steps", type=int, default=50000000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--checkpoint-root", type=str, default="data/checkpoints_transformer"
    )
    parser.add_argument("--resume-latest", action="store_true")
    parser.add_argument(
        "--self-play-mode", choices=("current", "frozen"), default="current"
    )
    parser.add_argument("--opponent-checkpoint", type=str, default="")
    parser.add_argument("--opponent-gap-ts", type=int, default=4_000_000)
    parser.add_argument("--replay-folder", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    resume_path = None
    if args.resume_latest:
        resume_path = find_latest_checkpoint(args.checkpoint_root)
        if resume_path:
            print(f"Resuming from {resume_path}")

    policy = TransformerPolicy().to(device)
    target_policy = TransformerPolicy().to(device)
    target_policy.load_state_dict(policy.state_dict())

    ppo = TransformerPPO(
        policy,
        target_policy,
        lr=args.lr,
        ent_coef=args.ent_coef,
    )

    if resume_path:
        policy_path = os.path.join(resume_path, "policy.pt")
        if os.path.exists(policy_path):
            state_dict = torch.load(policy_path, map_location=device)
            policy.load_state_dict(state_dict)
            target_policy.load_state_dict(state_dict)
            print(f"Loaded checkpoint from {resume_path}")

    opponent_checkpoint = args.opponent_checkpoint
    if args.self_play_mode == "frozen" and not opponent_checkpoint and resume_path:
        from rocket_league_bot_src.checkpoints import find_opponent_checkpoint
        from rocket_league_bot_src.checkpoints import load_checkpoint_book

        current_book = load_checkpoint_book(resume_path)
        current_ts = int(current_book.get("cumulative_timesteps", 0))
        opponent_checkpoint = find_opponent_checkpoint(
            args.checkpoint_root,
            current_ts=current_ts,
            gap_ts=int(args.opponent_gap_ts),
            exclude_checkpoint_dir=resume_path,
        )
        if opponent_checkpoint:
            print(f"Using opponent checkpoint: {opponent_checkpoint}")

    initial_curriculum_state = {}
    if resume_path:
        from rocket_league_bot_src.checkpoints import (
            load_curriculum_state_from_checkpoint,
        )

        initial_curriculum_state = load_curriculum_state_from_checkpoint(resume_path)

    env_builder = TransformerEnvBuilder(
        iteration_timesteps=100000,
        checkpoint_root=args.checkpoint_root,
        n_proc=args.n_proc,
        initial_curriculum_state=initial_curriculum_state,
        current_checkpoint_dir=resume_path or "",
        self_play_mode=args.self_play_mode,
        fixed_opponent_checkpoint=opponent_checkpoint,
        opponent_gap_ts=args.opponent_gap_ts,
        opponent_device="gpu",
        replay_folder=args.replay_folder if args.replay_folder else "",
        use_discrete_actions=USE_DISCRETE_ACTIONS,
    )

    env = env_builder(process_id=0)

    os.makedirs(args.checkpoint_root, exist_ok=True)
    writer = SummaryWriter(f"runs/transformer")

    print(f"Transformer training configuration:")
    print(
        f"  EARL: embed_dim={EARL_EMBED_DIM}, heads={EARL_NUM_HEADS}, layers={EARL_NUM_LAYERS}"
    )
    print(f"  Query features: {EARL_QUERY_FEATURES}, KV features: {EARL_KV_FEATURES}")
    print(f"  Device: {device}")
    print(f"  LR: {args.lr}, Ent Coef: {args.ent_coef}")
    print(f"  Self-play mode: {args.self_play_mode}")

    global_step = 0
    iteration = 0

    while global_step < args.max_steps:
        observations, actions, rewards, dones, log_probs, values = collect_rollout(
            env, ppo, args.n_steps, device
        )

        if len(observations) == 0:
            print("Warning: No observations collected, resetting environment")
            env = env_builder(process_id=0)
            continue

        with torch.no_grad():
            last_obs = observations[-1]
            q_t = torch.from_numpy(last_obs[0]).float().to(device)
            kv_t = torch.from_numpy(last_obs[1]).float().to(device).unsqueeze(0)
            m_t = torch.from_numpy(last_obs[2]).float().to(device).unsqueeze(0)
            next_value = ppo.target_policy.critic((q_t, kv_t, m_t)).item()

        advantages, returns = compute_gae(
            rewards.tolist(),
            dones.tolist(),
            values.tolist(),
            next_value,
            ppo.gamma,
            ppo.lam,
        )

        rollouts = RolloutData(
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            log_probs=log_probs,
            values=values,
            advantages=advantages,
            returns=returns,
        )

        policy_loss, value_loss, entropy_loss = ppo.update(rollouts)

        global_step += args.n_steps
        iteration += 1

        if iteration % 10 == 0:
            avg_reward = float(np.mean(rewards)) if len(rewards) > 0 else 0.0
            print(
                f"Iter {iteration}: step={global_step}, reward={avg_reward:.3f}, "
                f"policy_loss={policy_loss:.4f}, value_loss={value_loss:.4f}"
            )

            writer.add_scalar("train/policy_loss", policy_loss, global_step)
            writer.add_scalar("train/value_loss", value_loss, global_step)
            writer.add_scalar("train/entropy_loss", entropy_loss, global_step)
            writer.add_scalar("train/avg_reward", avg_reward, global_step)

        if global_step % args.save_interval == 0:
            save_path = os.path.join(args.checkpoint_root, f"step_{global_step}")
            os.makedirs(save_path, exist_ok=True)
            torch.save(policy.state_dict(), os.path.join(save_path, "policy.pt"))
            config = {
                "global_step": global_step,
                "iteration": iteration,
                "policy_type": "transformer",
                "earl_embed_dim": EARL_EMBED_DIM,
                "earl_num_heads": EARL_NUM_HEADS,
                "earl_num_layers": EARL_NUM_LAYERS,
                "earl_query_features": EARL_QUERY_FEATURES,
                "earl_kv_features": EARL_KV_FEATURES,
                "action_dim": NUM_ACTIONS,
            }
            with open(os.path.join(save_path, "BOOK_KEEPING_VARS.json"), "w") as f:
                json.dump(config, f, indent=2)
            print(f"Saved checkpoint to {save_path}")

    save_path = os.path.join(args.checkpoint_root, "final")
    os.makedirs(save_path, exist_ok=True)
    torch.save(policy.state_dict(), os.path.join(save_path, "policy.pt"))
    writer.close()


if __name__ == "__main__":
    main()
