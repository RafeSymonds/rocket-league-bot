"""
Replay-based state setter for curriculum training.

Loads parsed replay data and samples game states for training resets.
This provides the diverse, realistic game situations that Necto uses (70% of resets).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Iterator

import numpy as np
from rlgym.api import StateMutator
from rlgym.rocket_league.api import GameState
from rlgym.utils.gamestates import GameState as LegacyGameState
from rlgym.utils.state_setters import StateWrapper
from typing_extensions import override

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "reply-training"))

from replay_pretraining.replays.replays import to_rlgym_dfs, load_parsed_replay
from replay_pretraining.utils.util import make_lookup_table


class ReplayStateSetter(StateMutator):
    """
    State setter that loads parsed replay data and yields game states for training.

    Usage:
        setter = ReplayStateSetter("data/replays/ranked-duels")
        env = rlgym.make(state_setter=setter, ...)
    """

    def __init__(
        self,
        replay_folder: str,
        probability: float = 0.7,
        preload: bool = True,
        max_episodes: Optional[int] = None,
    ):
        """
        Args:
            replay_folder: Path to folder containing parsed replay subfolders
            probability: Probability of using replay reset vs other setters (not used here, for curriculum integration)
            preload: If True, load all replays at init. If False, load lazily.
            max_episodes: Maximum number of episodes to load per replay folder
        """
        super().__init__()
        self.replay_folder = Path(replay_folder)
        self.probability = probability
        self.preload = preload
        self.max_episodes = max_episodes

        self.lookup_table = make_lookup_table()
        self._episodes: List[tuple] = []
        self._episode_idx = 0
        self._current_df = None
        self._current_actions = None
        self._df_idx = 0

        if preload:
            self._load_replays()

    def _load_replays(self) -> None:
        """Load all replay episodes into memory."""
        replay_folders = []
        for item in self.replay_folder.iterdir():
            if item.is_dir() and not item.name.startswith("_"):
                for sub_item in item.iterdir():
                    if sub_item.is_dir() and not sub_item.name.startswith("_"):
                        replay_folders.append(sub_item)

        for replay_path in replay_folders:
            self._load_single_replay(replay_path)

            if self.max_episodes and len(self._episodes) >= self.max_episodes:
                break

        np.random.shuffle(self._episodes)
        print(
            f"Loaded {len(self._episodes)} replay episodes from {len(replay_folders)} replays"
        )

    def _load_single_replay(self, replay_path: Path) -> None:
        """Load episodes from a single parsed replay folder."""
        try:
            parsed_replay = load_parsed_replay(str(replay_path))
        except Exception:
            return

        if len(parsed_replay.metadata.get("players", [])) % 2 != 0:
            return

        try:
            episodes = list(to_rlgym_dfs(parsed_replay, self.lookup_table))
            for df, actions in episodes:
                self._episodes.append((df, actions))
        except Exception:
            return

    def _sample_episode(self) -> tuple:
        """Sample a random episode."""
        return self._episodes[self._episode_idx % len(self._episodes)]

    @override
    def apply(self, state: GameState, shared_info: dict) -> None:
        """Apply the next replay state to the game state."""
        if not self._episodes:
            return

        df, actions = self._sample_episode()
        self._episode_idx += 1

        if self._df_idx >= len(df):
            self._df_idx = 0

        row = df.iloc[self._df_idx]
        self._df_idx += 1

        game_state = LegacyGameState(row.tolist())

        game_state.players = sorted(game_state.players, key=lambda p: p.team_num)

        b = o = 0
        for player in game_state.players:
            if player.team_num == 0:
                player.car_id = StateWrapper.BLUE_ID1 + b
                b += 1
            else:
                player.car_id = StateWrapper.ORANGE_ID1 + o
                o += 1

        state._decode(list(game_state))


class ReplayStateSetterV2(StateMutator):
    """
    Version 2: More memory-efficient replay setter using lazy loading.

    Loads replays on-demand rather than preloading everything.
    Good for large replay collections that don't fit in memory.
    """

    def __init__(
        self,
        replay_folder: str,
        probability: float = 0.7,
        max_loaded_replays: int = 100,
        max_frames_between_resets: int = 30,
    ):
        """
        Args:
            replay_folder: Path to folder containing parsed replay subfolders
            probability: Probability of using replay reset (for curriculum integration)
            max_loaded_replays: Max number of replay folders to keep loaded at once
            max_frames_between_resets: Sample state every N frames during playback
        """
        super().__init__()
        self.replay_folder = Path(replay_folder)
        self.probability = probability
        self.max_loaded_replays = max_loaded_replays
        self.max_frames_between_resets = max_frames_between_resets

        self.lookup_table = make_lookup_table()
        self._loaded_replays: List[Path] = []
        self._current_replay_idx = 0
        self._current_episodes: List[tuple] = []
        self._episode_idx = 0
        self._frame_count = 0

        self._discover_replays()

    def _discover_replays(self) -> None:
        """Scan for available replay folders."""
        for item in self.replay_folder.iterdir():
            if item.is_dir() and not item.name.startswith("_"):
                for sub_item in item.iterdir():
                    if sub_item.is_dir() and not sub_item.name.startswith("_"):
                        self._loaded_replays.append(sub_item)

        np.random.shuffle(self._loaded_replays)
        print(f"Discovered {len(self._loaded_replays)} replay folders")

    def _load_replay_episodes(self, replay_path: Path) -> List[tuple]:
        """Load episodes from a single replay."""
        try:
            parsed_replay = load_parsed_replay(str(replay_path))
        except Exception:
            return []

        if len(parsed_replay.metadata.get("players", [])) % 2 != 0:
            return []

        try:
            return list(to_rlgym_dfs(parsed_replay, self.lookup_table))
        except Exception:
            return []

    def _get_next_replay(self) -> Optional[Path]:
        """Get the next replay to load (round-robin with shuffle)."""
        if not self._loaded_replays:
            return None
        replay = self._loaded_replays[self._current_replay_idx]
        self._current_replay_idx = (self._current_replay_idx + 1) % len(
            self._loaded_replays
        )
        return replay

    @override
    def apply(self, state: GameState, shared_info: dict) -> None:
        """Load next replay and apply a game state from it."""
        if self._frame_count < self.max_frames_between_resets:
            self._frame_count += 1
            return

        self._frame_count = 0

        if not self._current_episodes:
            replay_path = self._get_next_replay()
            if replay_path:
                self._current_episodes = self._load_replay_episodes(replay_path)
                self._episode_idx = 0
                np.random.shuffle(self._current_episodes)

        if not self._current_episodes:
            return

        df, actions = self._current_episodes[
            self._episode_idx % len(self._current_episodes)
        ]
        self._episode_idx += 1

        idx = np.random.randint(0, len(df))
        row = df.iloc[idx]

        game_state = LegacyGameState(row.tolist())
        game_state.players = sorted(game_state.players, key=lambda p: p.team_num)

        b = o = 0
        for player in game_state.players:
            if player.team_num == 0:
                player.car_id = StateWrapper.BLUE_ID1 + b
                b += 1
            else:
                player.car_id = StateWrapper.ORANGE_ID1 + o
                o += 1

        state._decode(list(game_state))
