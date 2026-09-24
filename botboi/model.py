"""Inference-only policy used by the RLBot runtime, evaluation, and export.

Training uses rlgym_learn_algos' DiscreteFF. Policy rebuilds the same
Linear/ReLU stack (same `model.<i>` parameter names, without the final
softmax) so exported weights load directly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .actions import N_ACTIONS, TICK_SKIP
from .obs import OBS_SIZE, OBS_VERSION


class Policy(nn.Module):
    def __init__(self, obs_size: int, n_actions: int, layer_sizes: list[int]):
        super().__init__()
        self.obs_size = obs_size
        self.n_actions = n_actions
        self.layer_sizes = list(layer_sizes)
        layers: list[nn.Module] = []
        prev = obs_size
        for size in layer_sizes:
            layers += [nn.Linear(prev, size), nn.ReLU()]
            prev = size
        layers.append(nn.Linear(prev, n_actions))
        self.model = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Return one action index per row of `obs` (shape [n, obs_size])."""
        logits = self.model(torch.as_tensor(obs, dtype=torch.float32))
        if deterministic:
            return logits.argmax(dim=-1).numpy()
        return torch.distributions.Categorical(logits=logits).sample().numpy()

    @classmethod
    def from_actor_state_dict(cls, state_dict: dict[str, torch.Tensor]) -> Policy:
        """Build from DiscreteFF weights (`model.<i>.weight` / `model.<i>.bias`)."""
        linear_indices = sorted(
            int(key.split(".")[1]) for key in state_dict if key.endswith(".weight")
        )
        shapes = [state_dict[f"model.{i}.weight"].shape for i in linear_indices]
        policy = cls(
            obs_size=shapes[0][1],
            n_actions=shapes[-1][0],
            layer_sizes=[shape[0] for shape in shapes[:-1]],
        )
        policy.load_state_dict(state_dict)
        return policy.eval()

    def save(self, path: str | Path, meta: dict) -> None:
        torch.save({"state_dict": self.state_dict(), "meta": meta}, path)

    @classmethod
    def load(cls, path: str | Path) -> tuple[Policy, dict]:
        data = torch.load(path, map_location="cpu", weights_only=True)
        return cls.from_actor_state_dict(data["state_dict"]), data["meta"]


def check_compatible(meta: dict, name: str) -> None:
    """Refuse to run a policy trained against a different obs or action setup."""
    if meta.get("obs_version") != OBS_VERSION or meta.get("obs_size") != OBS_SIZE:
        raise ValueError(
            f"{name} was trained with obs v{meta.get('obs_version')} "
            f"({meta.get('obs_size')} floats), but this code builds obs "
            f"v{OBS_VERSION} ({OBS_SIZE} floats)."
        )
    if meta.get("tick_skip") != TICK_SKIP or meta.get("n_actions") != N_ACTIONS:
        raise ValueError(f"{name} uses a different action setup: {meta}")
