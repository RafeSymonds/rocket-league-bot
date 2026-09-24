"""The in-game bot must see exactly what training saw.

Drives RocketSim with random actions, converts every tick into the RLBot v5
packet the game would send (boost pads in a shuffled RLBot order), feeds the
packets through RLBotObsAdapter, and compares its observations with BotObs
on the simulator state at every decision tick.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
from rlbot import flat
from rlgym.api import RLGym
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.done_conditions import GoalCondition, TimeoutCondition
from rlgym.rocket_league.reward_functions import GoalReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import KickoffMutator, MutatorSequence
from rlgym_tools.rocket_league.state_mutators.variable_team_size_mutator import (
    VariableTeamSizeMutator,
)

import rlbot_packets as rp
from botboi.actions import ACTION_TABLE, TICK_SKIP
from botboi.mutators import RandomGroundStateMutator
from botboi.obs import (
    CAR_BLOCK_SIZE,
    NUM_BOOST_PADS,
    OBS_SIZE,
    SELF_EXTRAS_SIZE,
    SLOT_SIZE,
    BotObs,
)
from botboi.rlbot_obs import RLBotObsAdapter

SELF_BLOCK_START = 9 + NUM_BOOST_PADS
# Car-block features that rlgym_compat can only estimate, so they may be off
# for a tick at transitions:
# - on_ground: RLBot has no wheel-contact data, so it is guessed during the
#   first ticks of a jump.
# - is_boosting: compat accumulates boost time in 1/120 s steps and float
#   rounding keeps a tapped boost alive one tick past the 0.1 s minimum.
ESTIMATED_OFFSETS = {"on_ground": 16, "is_boosting": 19}


def estimated_indices(feature: str) -> list[int]:
    offset = ESTIMATED_OFFSETS[feature]
    first_slot = SELF_BLOCK_START + CAR_BLOCK_SIZE + SELF_EXTRAS_SIZE + 6
    indices = [SELF_BLOCK_START + offset]
    for slot in range(3):
        indices.append(first_slot + slot * SLOT_SIZE + 1 + offset)
    return indices


def one_tick_env(team_size: int, mutator) -> RLGym:
    return RLGym(
        state_mutator=MutatorSequence(
            VariableTeamSizeMutator({(team_size, team_size): 1.0}), mutator
        ),
        obs_builder=BotObs(),
        action_parser=RepeatAction(LookupTableAction(), repeats=1),
        reward_fn=GoalReward(),
        termination_cond=GoalCondition(),
        truncation_cond=TimeoutCondition(timeout_seconds=60),
        transition_engine=RocketSimEngine(),
    )


def collect_differences(team_size: int, mutator, episodes: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)
    env = one_tick_env(team_size, mutator)
    obs_builder = BotObs()
    pad_order = rng.permutation(NUM_BOOST_PADS)
    diffs = []
    for _ in range(episodes):
        env.reset()
        agents = list(env.state.cars)
        adapter = RLBotObsAdapter(rp.field_info(pad_order), flat.MatchConfiguration())
        controls = {a: np.zeros(8, dtype=np.float32) for a in agents}
        adapter.update(rp.game_packet(env.state, agents, controls, pad_order))
        held = {}
        for tick in range(1, 900):
            if (tick - 1) % TICK_SKIP == 0:
                held = {a: int(rng.integers(len(ACTION_TABLE))) for a in agents}
            _, _, terminated, truncated = env.step({a: np.array([held[a]]) for a in agents})
            # With rlbot_delay (the RocketSimEngine default) the tick just
            # simulated used the controls chosen on the previous step.
            adapter.update(rp.game_packet(env.state, agents, controls, pad_order))
            controls = {a: ACTION_TABLE[held[a]] for a in agents}
            if tick % TICK_SKIP == 0:
                expected = obs_builder.build_obs(agents, env.state, {})
                for player_id, agent in enumerate(agents):
                    diffs.append(np.abs(expected[agent] - adapter.build_obs(player_id)))
            if any(terminated.values()) or any(truncated.values()):
                break
    env.close()
    return np.array(diffs)


@pytest.mark.parametrize("team_size", [1, 2])
@pytest.mark.parametrize("mutator", [KickoffMutator(), RandomGroundStateMutator()], ids=["kickoff", "random"])
def test_rlbot_obs_matches_training_obs(team_size, mutator):
    diffs = collect_differences(team_size, mutator, episodes=2, seed=team_size)
    assert diffs.shape[1] == OBS_SIZE and len(diffs) > 100

    loose = [i for feature in ESTIMATED_OFFSETS for i in estimated_indices(feature)]
    strict_indices = np.delete(np.arange(OBS_SIZE), loose)
    strict = diffs[:, strict_indices]
    worst = strict_indices[np.argmax(strict.max(axis=0))]
    assert strict.max() < 1e-3, f"obs index {worst} differs by {strict.max()}"

    for feature in ESTIMATED_OFFSETS:
        mismatch_rate = (diffs[:, estimated_indices(feature)] > 1e-3).mean()
        assert mismatch_rate < 0.08, f"{feature} disagrees on {mismatch_rate:.1%} of samples"
