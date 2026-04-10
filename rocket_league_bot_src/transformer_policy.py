from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

from earl_pytorch import EARLPerceiver

from .config import (
    EARL_EMBED_DIM,
    EARL_KV_FEATURES,
    EARL_NUM_HEADS,
    EARL_NUM_LAYERS,
    EARL_QUERY_FEATURES,
    NUM_DISCRETE_ACTIONS,
)


def make_lookup_table() -> np.ndarray:
    actions = []
    for throttle in (-1, 0, 1):
        for steer in (-1, 0, 1):
            for boost in (0, 1):
                for handbrake in (0, 1):
                    if boost == 1 and throttle != 1:
                        continue
                    actions.append(
                        [throttle or boost, steer, 0, steer, 0, 0, boost, handbrake]
                    )

    for pitch in (-1, 0, 1):
        for yaw in (-1, 0, 1):
            for roll in (-1, 0, 1):
                for jump in (0, 1):
                    for boost in (0, 1):
                        if jump == 1 and yaw != 0:
                            continue
                        if pitch == roll == jump == 0:
                            continue
                        handbrake = jump == 1 and (pitch != 0 or yaw != 0 or roll != 0)
                        actions.append(
                            [boost, yaw, pitch, yaw, roll, jump, boost, handbrake]
                        )

    return np.asarray(actions, dtype=np.float32)


class ControlsPredictorDot(nn.Module):
    def __init__(self, in_features, features=32, layers=1, actions=None):
        super().__init__()
        if actions is None:
            self.actions = torch.from_numpy(make_lookup_table()).float()
        else:
            self.actions = torch.from_numpy(actions).float()
        self.net = self._mlp(8, in_features, layers, features)
        self.emb_convertor = nn.Linear(in_features, features)

    @staticmethod
    def _mlp(input_dim, output_dim, layers, hidden_dim):
        if layers == 1:
            return nn.Linear(input_dim, output_dim)
        layers_list = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(layers - 2):
            layers_list.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers_list.append(nn.Linear(hidden_dim, output_dim))
        return nn.Sequential(*layers_list)

    def forward(self, player_emb, actions=None):
        if actions is None:
            actions = self.actions
        player_emb = self.emb_convertor(player_emb)
        act_emb = self.net(actions.to(player_emb.device))

        if act_emb.ndim == 2:
            return torch.einsum("ad,bpd->bpa", act_emb, player_emb)
        return torch.einsum("bad,bpd->bpa", act_emb, player_emb)


class TransformerActor(nn.Module):
    def __init__(self, earl=None, output=None):
        super().__init__()
        if earl is None:
            earl = EARLPerceiver(
                EARL_EMBED_DIM,
                EARL_NUM_HEADS,
                EARL_NUM_LAYERS,
                1,
                query_features=EARL_QUERY_FEATURES,
                key_value_features=EARL_KV_FEATURES,
            )
        if output is None:
            output = ControlsPredictorDot(EARL_EMBED_DIM)
        self.earl = earl
        self.relu = nn.ReLU()
        self.output = output
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, inp, actions=None):
        q, kv, m = inp
        res = self.earl(q, kv, m)
        weights = None
        if isinstance(res, tuple):
            res, weights = res
        res = self.output(self.relu(res), actions)
        if isinstance(res, tuple):
            res = tuple(r[:, 0, :] for r in res)
        else:
            res = res[:, 0, :]
        if weights is None:
            return res
        return res, weights


class TransformerCritic(nn.Module):
    def __init__(self, earl=None, output=None):
        super().__init__()
        if earl is None:
            earl = EARLPerceiver(
                EARL_EMBED_DIM,
                EARL_NUM_HEADS,
                EARL_NUM_LAYERS,
                1,
                query_features=EARL_QUERY_FEATURES,
                key_value_features=EARL_KV_FEATURES,
            )
        if output is None:
            output = nn.Linear(EARL_EMBED_DIM, 1)
        self.earl = earl
        self.relu = nn.ReLU()
        self.output = output
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, inp):
        q, kv, m = inp
        res = self.earl(q, kv, m)
        if isinstance(res, tuple):
            res = res[0]
        res = res[:, 0, :]
        res = self.output(self.relu(res))
        return res


class TransformerPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = TransformerActor()
        self.critic = TransformerCritic()

    def forward(self, obs, actions=None):
        return self.actor(obs, actions)

    def get_action(self, q, kv, m, deterministic=False):
        q_t = torch.from_numpy(q).float()
        kv_t = torch.from_numpy(kv).float()
        m_t = torch.from_numpy(m).float()

        with torch.no_grad():
            logits, weights = self.actor((q_t, kv_t, m_t))
            if weights is not None:
                weights = weights[0, 0].cpu().numpy()
            action = torch.argmax(logits, dim=-1)

        return action.item(), weights

    def evaluate(self, obs, actions):
        q, kv, m = obs
        q = torch.from_numpy(q).float()
        kv = torch.from_numpy(kv).float()
        m = torch.from_numpy(m).float()
        actions = torch.from_numpy(actions).long()

        logits, weights = self.actor((q, kv, m))
        log_probs = torch.log_softmax(logits, dim=-1)
        action_log_probs = torch.gather(log_probs, -1, actions.unsqueeze(-1)).squeeze(
            -1
        )

        values = self.critic((q, kv, m))

        return action_log_probs, values, weights
