"""
Discrete action parser matching Necto's 124-action space.

Ground actions (54 combos): throttle {-1, 0, 1} x steer {-1, 0, 1} x boost {0, 1} x handbrake {0, 1}
Aerial actions (70 combos): pitch {-1, 0, 1} x yaw {-1, 0, 1} x roll {-1, 0, 1} x jump {0, 1} x boost {0, 1}

Invalid combos filtered (e.g., boost without throttle).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces
from rlgym.utils.action_parsers import ActionParser
from rlgym.utils.gamestates import GameState


class NectoAction(ActionParser):
    """
    Discrete action parser with 124 actions:
    - 54 ground actions (throttle/steer/boost/handbrake combos)
    - 70 aerial actions (pitch/yaw/roll/jump/boost combos)
    """

    def __init__(self):
        super().__init__()
        self._lookup_table = self.make_lookup_table()

    @staticmethod
    def make_lookup_table() -> np.ndarray:
        """Build the 124-action lookup table."""
        actions = []

        # Ground actions
        for throttle in (-1, 0, 1):
            for steer in (-1, 0, 1):
                for boost in (0, 1):
                    for handbrake in (0, 1):
                        if boost == 1 and throttle != 1:
                            continue
                        actions.append(
                            [throttle or boost, steer, 0, steer, 0, 0, boost, handbrake]
                        )

        # Aerial actions
        for pitch in (-1, 0, 1):
            for yaw in (-1, 0, 1):
                for roll in (-1, 0, 1):
                    for jump in (0, 1):
                        for boost in (0, 1):
                            if jump == 1 and yaw != 0:
                                continue
                            if pitch == roll == jump == 0:
                                continue
                            handbrake = int(
                                jump == 1 and (pitch != 0 or yaw != 0 or roll != 0)
                            )
                            actions.append(
                                [boost, yaw, pitch, yaw, roll, jump, boost, handbrake]
                            )

        return np.array(actions, dtype=np.float32)

    def get_action_space(self) -> spaces.Space:
        return spaces.Discrete(len(self._lookup_table))

    def get_lookup_table(self) -> np.ndarray:
        """Return the lookup table for policy head split."""
        return self._lookup_table

    def get_num_actions(self) -> int:
        """Return number of discrete actions."""
        return len(self._lookup_table)

    def parse_actions(self, actions: Any, state: GameState) -> np.ndarray:
        """
        Convert discrete action indices to continuous action vectors.

        Supports two modes:
        - Discrete indices: actions shape (N,) -> continuous (N, 8)
        - Continuous already: actions shape (N, 8) -> passed through
        """
        parsed_actions = []

        for action in actions:
            if action.size != 8:
                if action.shape == ():
                    action = np.expand_dims(action, axis=0)

                if np.isnan(action).any():
                    stripped = action[~np.isnan(action)].squeeze().astype(int)
                    parsed_actions.append(self._lookup_table[stripped])
                else:
                    padded = np.pad(
                        action.astype(float),
                        (0, 8 - action.size),
                        "constant",
                        constant_values=np.NAN,
                    )
                    stripped = padded[~np.isnan(padded)].squeeze().astype(int)
                    parsed_actions.append(self._lookup_table[stripped])
            else:
                parsed_actions.append(np.asarray(action, dtype=np.float32))

        return np.asarray(parsed_actions)


def test_action_parser():
    """Test the action parser."""
    ap = NectoAction()
    table = ap.get_lookup_table()
    print(f"Lookup table shape: {table.shape}")
    print(f"Action space: {ap.get_action_space()}")
    print(f"Ground actions: 54, Aerial actions: 70, Total: {len(table)}")

    test_action = np.array([0, 1, 2, 3])
    parsed = ap.parse_actions(test_action, None)
    print(f"Test parse: {parsed.shape}")


if __name__ == "__main__":
    test_action_parser()
