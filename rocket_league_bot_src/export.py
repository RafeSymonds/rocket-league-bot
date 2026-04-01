from __future__ import annotations

import json
import shutil
from pathlib import Path

from .checkpoints import build_runtime_config


def export_checkpoint_to_rlbot_package(
    checkpoint_dir: str,
    bot_dir: str = "BotBoi_v1/src",
) -> None:
    src = Path(checkpoint_dir)
    target = Path(bot_dir)
    target.mkdir(parents=True, exist_ok=True)

    for name in ("PPO_POLICY.pt", "BOOK_KEEPING_VARS.json"):
        source = src / name
        if not source.exists():
            raise FileNotFoundError(f"Missing expected checkpoint file: {source}")
        shutil.copy2(source, target / name)

    runtime_config = build_runtime_config(str(src))
    (target / "runtime_config.json").write_text(
        json.dumps(runtime_config, indent=2, sort_keys=True)
    )
