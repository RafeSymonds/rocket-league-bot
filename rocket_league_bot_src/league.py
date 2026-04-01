from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class LeagueSnapshot:
    checkpoint_dir: str
    cumulative_timesteps: int
    stage: str
    difficulty: float


class SnapshotLeague:
    """
    Local snapshot registry for progressive self-play.

    This repo still trains with a single shared policy for all agents. That means
    older-checkpoint opponents are not wired into the live environment yet.
    This registry is the first piece needed for that league workflow: it tracks
    promotable checkpoints and exposes a stable snapshot list for future opponent
    sampling and evaluation scripts.
    """

    def __init__(self, root: str = "data/league"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "snapshots.json"

    def _load(self) -> list[LeagueSnapshot]:
        if not self.manifest_path.exists():
            return []
        raw = json.loads(self.manifest_path.read_text())
        return [LeagueSnapshot(**item) for item in raw]

    def _save(self, snapshots: list[LeagueSnapshot]) -> None:
        payload = [asdict(snapshot) for snapshot in snapshots]
        self.manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    def register_snapshot(
        self,
        checkpoint_dir: str,
        cumulative_timesteps: int,
        stage: str,
        difficulty: float,
    ) -> bool:
        snapshots = self._load()
        checkpoint_dir = str(Path(checkpoint_dir))
        if any(snapshot.checkpoint_dir == checkpoint_dir for snapshot in snapshots):
            return False

        snapshots.append(
            LeagueSnapshot(
                checkpoint_dir=checkpoint_dir,
                cumulative_timesteps=int(cumulative_timesteps),
                stage=str(stage),
                difficulty=float(difficulty),
            )
        )
        snapshots.sort(key=lambda snapshot: snapshot.cumulative_timesteps)
        self._save(snapshots)
        return True

    def list_snapshots(self) -> list[LeagueSnapshot]:
        return self._load()
