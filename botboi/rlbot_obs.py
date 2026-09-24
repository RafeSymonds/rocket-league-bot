"""Builds training-identical observations from RLBot v5 game packets.

rlgym_compat keeps a GameState with the same attributes as the simulator's,
and BotObs runs on it unchanged. One field needs fixing: RLBot reports boost
pad timers as seconds *since* pickup, while the simulator (and therefore
training) uses seconds *until* the pad is back.
"""

from __future__ import annotations

import numpy as np
from rlbot import flat
from rlgym.rocket_league.common_values import BOOST_LOCATIONS
from rlgym_compat import GameState

from .obs import BotObs

BIG_PAD_RESPAWN_SECONDS = 10.0
SMALL_PAD_RESPAWN_SECONDS = 4.0


class RLBotObsAdapter:
    def __init__(self, field_info: flat.FieldInfo, match_config: flat.MatchConfiguration):
        self.state = GameState.create_compat_game_state(field_info, match_config)
        self.obs_builder = BotObs()
        # RLBot pad i -> simulator pad index, matched by location.
        sim_xy = np.asarray(BOOST_LOCATIONS, dtype=np.float32)[:, :2]
        self.pad_index = np.array(
            [
                int(np.argmin(np.linalg.norm(sim_xy - [pad.location.x, pad.location.y], axis=1)))
                for pad in field_info.boost_pads
            ]
        )
        self.pad_respawn = np.array(
            [
                BIG_PAD_RESPAWN_SECONDS if pad.is_full_boost else SMALL_PAD_RESPAWN_SECONDS
                for pad in field_info.boost_pads
            ],
            dtype=np.float32,
        )

    def update(self, packet: flat.GamePacket) -> None:
        self.state.update(packet)
        timers = self.state.boost_pad_timers
        for i, pad in enumerate(packet.boost_pads):
            remaining = 0.0 if pad.is_active else max(self.pad_respawn[i] - pad.timer, 0.0)
            timers[self.pad_index[i]] = remaining

    def build_obs(self, player_id) -> np.ndarray:
        return self.obs_builder.build_obs([player_id], self.state, {})[player_id]
