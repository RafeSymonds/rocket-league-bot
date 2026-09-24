"""Action space shared by training and the RLBot runtime.

The policy picks one of the 90 rows of rlgym's lookup table (Necto's action
set) and the action is held for TICK_SKIP physics ticks (120 Hz / 8 = 15
decisions per second). Each row is
[throttle, steer, pitch, yaw, roll, jump, boost, handbrake].
"""

from __future__ import annotations

from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction

TICK_SKIP = 8
ACTION_TABLE = LookupTableAction.make_lookup_table()
N_ACTIONS = len(ACTION_TABLE)


def make_action_parser() -> RepeatAction:
    return RepeatAction(LookupTableAction(), repeats=TICK_SKIP)
