"""Train -> resume -> evaluate, end to end, with tiny smoke settings."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

from botboi.checkpoints import checkpoint_timesteps, find_latest_checkpoint, resolve_policy

REPO = Path(__file__).resolve().parents[1]


def run(args: list[str]) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [sys.executable, "-m", *args],
        cwd=REPO,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    return result


@pytest.mark.slow
def test_train_resume_evaluate(tmp_path):
    train = ["botboi.train", "--runs-dir", str(tmp_path), "--run", "t", "--preset", "smoke"]
    first = run([*train, "--timesteps", "20000"])
    assert "Starting a new run" in first.stdout
    run_dir = tmp_path / "t"
    assert checkpoint_timesteps(find_latest_checkpoint(run_dir)) >= 20000

    second = run([*train, "--timesteps", "40000", "--phase", "main"])
    assert "Resuming from" in second.stdout and "phase: main" in second.stdout
    assert checkpoint_timesteps(find_latest_checkpoint(run_dir)) >= 40000

    snapshots = sorted((run_dir / "policies").glob("*.pt"), key=lambda p: int(p.stem))
    assert len(snapshots) >= 2
    with open(run_dir / "metrics.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 6
    assert any(row.get("Game/Goals per game minute") for row in rows), "no game stats logged"

    _, meta = resolve_policy(str(run_dir))
    assert meta["timesteps"] >= 40000
    result = run(["botboi.evaluate", str(snapshots[0]), str(run_dir), "--games", "2", "--max-seconds", "5"])
    assert "1v1: A" in result.stdout and "2v2: A" in result.stdout
