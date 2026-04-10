"""
Discrete action parser matching Necto's 124-action space.

Ground actions (54 combos): throttle {-1, 0, 1} x steer {-1, 0, 1} x boost {0, 1} x handbrake {0, 1}
Aerial actions (70 combos): pitch {-1, 0, 1} x yaw {-1, 0, 1} x roll {-1, 0, 1} x jump {0, 1} x boost {0, 1}

Invalid combos filtered (e.g., boost without throttle).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from rlgym.api import AgentID
from rlgym.api.config.action_parser import ActionParser
from rlgym.rocket_league.api import GameState


class NectoAction(ActionParser[AgentID, np.ndarray, np.ndarray, GameState, tuple[str, int]]):
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

    def get_action_space(self, agent: AgentID) -> tuple[str, int]:
        return ("discrete", len(self._lookup_table))

    def reset(self, agents, initial_state: GameState, shared_info: dict[str, Any]) -> None:
        pass

    def get_lookup_table(self) -> np.ndarray:
        """Return the lookup table for policy head split."""
        return self._lookup_table

    def get_num_actions(self) -> int:
        """Return number of discrete actions."""
        return len(self._lookup_table)

    def parse_actions(
        self,
        actions: dict[AgentID, np.ndarray],
        state: GameState,
        shared_info: dict[str, Any],
    ) -> dict[AgentID, np.ndarray]:
        """
        Convert discrete action indices to continuous action vectors.

        Supports two modes:
        - Discrete indices: action shape `(ticks,)` or `(ticks, 1)` -> continuous `(ticks, 8)`
        - Continuous already: action shape `(ticks, 8)` -> passed through
        """
        parsed_actions = {}

        for agent, action in actions.items():
            action = np.asarray(action)
            if action.shape == ():
                action = np.expand_dims(action, axis=0)

            if len(action.shape) == 2 and action.shape[1] == 1:
                action = action.squeeze(1)

            # PPO discrete outputs are integer indices over ticks.
            if np.issubdtype(action.dtype, np.integer):
                parsed_actions[agent] = self._lookup_table[action.astype(int)]
                continue

            # Legacy NaN-padded discrete indices.
            if len(action.shape) == 1 and action.size != 8:
                stripped = action[~np.isnan(action)] if np.isnan(action).any() else action
                parsed_actions[agent] = self._lookup_table[stripped.astype(int)]
                continue

            # True continuous controls must already be shaped (8,) or (ticks, 8).
            if len(action.shape) == 2 and action.shape[1] != 8:
                raise ValueError(f"Unexpected action shape {action.shape}")
            parsed = np.asarray(action, dtype=np.float32)
            if len(parsed.shape) == 1:
                parsed = np.expand_dims(parsed, axis=0)
            parsed_actions[agent] = parsed

        return parsed_actions


def test_action_parser():
    """Test the action parser."""
    ap = NectoAction()
    table = ap.get_lookup_table()
    print(f"Lookup table shape: {table.shape}")
    print(f"Action space: {ap.get_action_space('agent')}")
    print(f"Ground actions: 54, Aerial actions: 70, Total: {len(table)}")

    test_action = np.array([0, 1, 2, 3])
    parsed = ap.parse_actions({"agent": test_action}, None, {})
    print(f"Test parse: {parsed.shape}")


if __name__ == "__main__":
    test_action_parser()
