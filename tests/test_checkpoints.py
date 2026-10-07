from __future__ import annotations

import json

from botboi.checkpoints import find_checkpoints, find_latest_checkpoint


def make_checkpoint(run_dir, name: str, complete: bool):
    folder = run_dir / "checkpoints" / "run-1" / name
    (folder / "ppo_learner").mkdir(parents=True)
    (folder / "ppo_learner" / "actor_critic.pt").write_bytes(b"partial")
    if complete:
        (folder / "ppo_agent.json").write_text(json.dumps({"cumulative_timesteps": 1}))
        (folder / "botboi.json").write_text("{}")
    return folder


def test_interrupted_save_is_skipped(tmp_path):
    # A crash mid-save leaves a newer folder without botboi.json.
    older = make_checkpoint(tmp_path, "100", complete=True)
    newer = make_checkpoint(tmp_path, "200", complete=True)
    make_checkpoint(tmp_path, "300", complete=False)
    assert find_checkpoints(tmp_path) == [older, newer]
    assert find_latest_checkpoint(tmp_path) == newer
