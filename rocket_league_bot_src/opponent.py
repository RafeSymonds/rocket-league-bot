from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gym
import numpy as np
import torch

from rlgym_ppo.ppo.discrete_policy import DiscreteFF

from .checkpoints import load_checkpoint_book
from .config import OBS_DIM, POLICY_LAYER_SIZES


class FrozenOpponentPolicy:
    def __init__(self, device: str = "cpu", deterministic: bool = False):
        self.device = self._normalize_device(device)
        self.deterministic = deterministic
        self._checkpoint_dir = ""
        self._policy: DiscreteFF | None = None
        self._obs_mean = np.zeros(OBS_DIM, dtype=np.float32)
        self._obs_std = np.ones(OBS_DIM, dtype=np.float32)
        self._n_actions = 90

    @staticmethod
    def _normalize_device(device: str) -> str:
        device = str(device).strip().lower()
        if device in {"gpu", "cuda", "cuda:0"}:
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        if device == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        return device

    @property
    def checkpoint_dir(self) -> str:
        return self._checkpoint_dir

    def load(self, checkpoint_dir: str) -> None:
        checkpoint_dir = str(Path(checkpoint_dir))
        if checkpoint_dir == self._checkpoint_dir:
            return

        book = load_checkpoint_book(checkpoint_dir)
        obs_shape = book.get("obs_running_stats", {}).get("shape")
        if obs_shape and int(obs_shape[0]) != OBS_DIM:
            raise ValueError(
                f"Frozen opponent obs dim mismatch for {checkpoint_dir}: {obs_shape} vs {[OBS_DIM]}"
            )

        policy = DiscreteFF(OBS_DIM, self._n_actions, POLICY_LAYER_SIZES, self.device)
        state = torch.load(Path(checkpoint_dir) / "PPO_POLICY.pt", map_location=self.device)
        policy.load_state_dict(state)
        policy.eval()

        stats = book.get("obs_running_stats")
        if isinstance(stats, dict):
            mean = np.asarray(stats.get("mean", []), dtype=np.float32)
            var = np.asarray(stats.get("var", []), dtype=np.float32)
            count = int(stats.get("count", 0))
            if count >= 2 and mean.shape == (OBS_DIM,) and var.shape == (OBS_DIM,):
                variance = var / max(1, count - 1)
                variance = np.where(variance == 0, 1.0, variance)
                self._obs_mean = mean
                self._obs_std = np.sqrt(variance).astype(np.float32)
            else:
                self._obs_mean = np.zeros(OBS_DIM, dtype=np.float32)
                self._obs_std = np.ones(OBS_DIM, dtype=np.float32)
        else:
            self._obs_mean = np.zeros(OBS_DIM, dtype=np.float32)
            self._obs_std = np.ones(OBS_DIM, dtype=np.float32)

        self._policy = policy
        self._checkpoint_dir = checkpoint_dir

    def clear(self) -> None:
        self._checkpoint_dir = ""
        self._policy = None
        self._obs_mean = np.zeros(OBS_DIM, dtype=np.float32)
        self._obs_std = np.ones(OBS_DIM, dtype=np.float32)

    def act(self, obs: np.ndarray) -> int:
        if self._policy is None:
            raise RuntimeError("Frozen opponent policy was not loaded")
        obs = np.asarray(obs, dtype=np.float32)
        obs = (obs - self._obs_mean) / self._obs_std
        action, _ = self._policy.get_action(obs.reshape(1, -1), deterministic=self.deterministic)
        return int(np.asarray(action).reshape(-1)[0])


class SelfPlayOpponentGymWrapper:
    def __init__(
        self,
        rlgym_env,
        opponent_state_path: str,
        deterministic_opponent: bool = False,
        device: str = "cpu",
    ):
        self.rlgym_env = rlgym_env
        self.opponent_state_path = Path(opponent_state_path)
        self._opponent = FrozenOpponentPolicy(device=device, deterministic=deterministic_opponent)
        self._blue_agent = None
        self._orange_agent = None
        self._obs_buffer = np.zeros((1, OBS_DIM), dtype=np.float32)
        self._last_obs_dict: dict[Any, np.ndarray] = {}

        print("WARNING: CALLING ENV.RESET() ONE EXTRA TIME TO DETERMINE STATE AND ACTION SPACES")
        obs_dict = self.rlgym_env.reset()
        self._sync_opponent()
        self._refresh_agents(obs_dict)
        self._last_obs_dict = obs_dict

        act_space = list(self.rlgym_env.action_spaces.values())[0][1]
        obs_space = list(self.rlgym_env.observation_spaces.values())[0][1]
        self.is_discrete = False
        if type(act_space) == int:
            self.action_space = gym.spaces.Discrete(n=act_space)
            self.is_discrete = True
        else:
            self.action_space = None

        if type(obs_space) == int and obs_space > 0:
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_space,))
        else:
            blue_obs = obs_dict[self._blue_agent]
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=np.shape(blue_obs))

    def _sync_opponent(self) -> None:
        if not self.opponent_state_path.exists():
            self._opponent.clear()
            return
        try:
            payload = json.loads(self.opponent_state_path.read_text())
        except Exception:
            return
        checkpoint_dir = str(payload.get("checkpoint_dir", "")).strip()
        enabled = bool(payload.get("enabled", False))
        if not enabled or not checkpoint_dir:
            self._opponent.clear()
            return
        self._opponent.load(checkpoint_dir)

    def _refresh_agents(self, obs_dict: dict[Any, np.ndarray]) -> None:
        self._blue_agent = None
        self._orange_agent = None
        for agent_id in obs_dict:
            car = self.rlgym_env.state.cars[agent_id]
            if car.is_orange and self._orange_agent is None:
                self._orange_agent = agent_id
            elif not car.is_orange and self._blue_agent is None:
                self._blue_agent = agent_id
        if self._blue_agent is None:
            raise RuntimeError("Failed to locate blue agent in self-play wrapper")

    def reset(self):
        self._sync_opponent()
        obs_dict = self.rlgym_env.reset()
        self._refresh_agents(obs_dict)
        self._last_obs_dict = obs_dict
        self._obs_buffer[0] = np.asarray(obs_dict[self._blue_agent], dtype=np.float32)
        return self._obs_buffer

    def step(self, actions):
        if self.is_discrete:
            actions = np.asarray(actions).astype(np.int32)

        blue_action = int(np.asarray(actions).reshape(-1)[0])
        action_dict = {self._blue_agent: np.array([blue_action], dtype=np.int32)}

        if self._orange_agent is not None:
            if self._opponent.checkpoint_dir:
                orange_obs = np.asarray(self._last_obs_dict[self._orange_agent], dtype=np.float32)
                orange_action = self._opponent.act(orange_obs)
                action_dict[self._orange_agent] = np.array([orange_action], dtype=np.int32)
            else:
                random_action = int(np.random.randint(self.action_space.n))
                action_dict[self._orange_agent] = np.array([random_action], dtype=np.int32)

        obs_dict, reward_dict, terminated_dict, truncated_dict = self.rlgym_env.step(action_dict)
        self._refresh_agents(obs_dict)
        self._last_obs_dict = obs_dict
        self._obs_buffer[0] = np.asarray(obs_dict[self._blue_agent], dtype=np.float32)

        reward = float(reward_dict[self._blue_agent])
        done = bool(terminated_dict[self._blue_agent])
        truncated = bool(truncated_dict[self._blue_agent])
        info = {"state": self.rlgym_env.state}
        return self._obs_buffer, reward, done, truncated, info

    def render(self):
        self.rlgym_env.render()

    def seed(self, seed):
        pass

    def close(self):
        self.rlgym_env.close()
