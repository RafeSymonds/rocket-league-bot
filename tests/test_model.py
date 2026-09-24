from __future__ import annotations

import numpy as np
import pytest
import torch
from rlgym_learn_algos.ppo import DiscreteFF

from botboi.actions import ACTION_TABLE, N_ACTIONS
from botboi.checkpoints import policy_meta
from botboi.model import Policy, check_compatible
from botboi.obs import OBS_SIZE


def test_action_table_is_necto_90():
    assert ACTION_TABLE.shape == (90, 8) and N_ACTIONS == 90
    # Every row is a valid controller input.
    assert np.all(np.abs(ACTION_TABLE) <= 1)
    assert len({tuple(row) for row in ACTION_TABLE}) == 90


def test_policy_matches_training_actor():
    torch.manual_seed(0)
    actor = DiscreteFF(OBS_SIZE, N_ACTIONS, (64, 32), torch.float32, torch.device("cpu"))
    policy = Policy.from_actor_state_dict(actor.state_dict())
    obs = np.random.default_rng(0).normal(size=(256, OBS_SIZE)).astype(np.float32)
    with torch.no_grad():
        expected = actor.get_output(obs).argmax(dim=-1).numpy()
    np.testing.assert_array_equal(policy.act(obs, deterministic=True), expected)


def test_save_load_round_trip(tmp_path):
    policy = Policy(OBS_SIZE, N_ACTIONS, [32])
    meta = policy_meta(123, "test")
    policy.save(tmp_path / "policy.pt", meta)
    loaded, loaded_meta = Policy.load(tmp_path / "policy.pt")
    assert loaded_meta == meta
    obs = np.zeros((4, OBS_SIZE), dtype=np.float32)
    np.testing.assert_array_equal(loaded.act(obs), policy.act(obs))


def test_incompatible_policy_is_refused():
    meta = policy_meta(1, "test")
    check_compatible(meta, "ok")
    with pytest.raises(ValueError):
        check_compatible({**meta, "obs_version": meta["obs_version"] + 1}, "old")
    with pytest.raises(ValueError):
        check_compatible({**meta, "tick_skip": 4}, "other actions")
