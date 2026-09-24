"""The exported RLBot bot runs on its own and follows the action schedule."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import rlbot.utils.logging  # noqa: F401  (rlbot 2.0.0b55 hits a circular import if rlbot.config loads first)
from rlbot.config import load_match_config, load_player_config

from botboi.actions import N_ACTIONS
from botboi.checkpoints import policy_meta
from botboi.export import BOT_DIR, RUNTIME_MODULES, build_bot_folder
from botboi.model import Policy
from botboi.obs import OBS_SIZE

TESTS = Path(__file__).resolve().parent

# Runs inside the exported folder with a clean interpreter, so only the
# copied botboi modules are importable.
DRIVER = textwrap.dedent(
    """
    import sys
    import numpy as np
    from rlbot import flat
    sys.path.append(sys.argv[1])  # tests dir, for rlbot_packets only
    import rlbot_packets as rp
    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym.rocket_league.state_mutators import KickoffMutator, FixedTeamSizeMutator, MutatorSequence
    import bot

    import botboi
    assert botboi.__file__.startswith(str(bot.HERE)), botboi.__file__

    engine = RocketSimEngine()
    state = engine.create_base_state()
    MutatorSequence(FixedTeamSizeMutator(1, 1), KickoffMutator()).apply(state, {})
    state = engine.set_state(state, {})
    agents = list(state.cars)
    pad_order = np.arange(34)

    b = bot.BotBoi(bot.AGENT_ID)
    b.field_info = rp.field_info(pad_order)
    b.match_config = flat.MatchConfiguration()
    b.player_id, b.index, b.team = 1, 1, 1
    b.initialize()

    zero = {a: np.zeros(8, np.float32) for a in agents}
    decisions = 0
    previous = None
    for tick in range(40):
        packet = rp.game_packet(state, agents, zero, pad_order)
        controls = b.get_output(packet)
        if controls is not previous:
            decisions += 1
            previous = controls
        state = engine.step({a: np.zeros((1, 8), np.float32) for a in agents}, {})
    # 40 ticks with an action every 8 ticks = 5 decisions.
    print("decisions", decisions)
    """
)


def test_bot_folder_runs_standalone(tmp_path):
    policy = Policy(OBS_SIZE, N_ACTIONS, [32])
    build_bot_folder(policy, policy_meta(0, "test"), tmp_path)
    assert (tmp_path / "policy.pt").exists()
    for name in RUNTIME_MODULES:
        assert (tmp_path / "botboi" / name).exists()

    (tmp_path / "driver.py").write_text(DRIVER)
    result = subprocess.run(
        [sys.executable, "driver.py", str(TESTS)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "decisions 5" in result.stdout, result.stdout + result.stderr


def test_rlbot_configs_parse():
    bot = load_player_config(BOT_DIR / "bot.toml", team=1)
    assert bot.variety.agent_id == "rafe/botboi"
    one = load_match_config(BOT_DIR / "match_1v1.toml")
    two = load_match_config(BOT_DIR / "match_2v2.toml")
    assert len(one.player_configurations) == 2
    assert len(two.player_configurations) == 4
