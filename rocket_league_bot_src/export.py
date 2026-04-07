from __future__ import annotations

import configparser
import json
import os
import shutil
from pathlib import Path
from typing import Any

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


def summarize_rlbot_package(bot_dir: str) -> dict[str, Any]:
    bot_path = Path(bot_dir)
    required = [
        bot_path / "bot.py",
        bot_path / "bot.cfg",
        bot_path / "appearance.cfg",
        bot_path / "PPO_POLICY.pt",
        bot_path / "BOOK_KEEPING_VARS.json",
        bot_path / "runtime_config.json",
    ]

    missing = [str(path) for path in required if not path.exists()]
    runtime: dict[str, Any] = {}
    book: dict[str, Any] = {}
    if not missing:
        runtime = json.loads((bot_path / "runtime_config.json").read_text())
        book = json.loads((bot_path / "BOOK_KEEPING_VARS.json").read_text())

    return {
        "bot_dir": str(bot_path),
        "missing": missing,
        "runtime": runtime,
        "book": book,
    }


def _iter_windows_home_candidates() -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        try:
            resolved = path.expanduser().resolve()
        except Exception:
            return
        if not resolved.exists():
            return
        key = str(resolved)
        if key in seen:
            return
        seen.add(key)
        candidates.append(resolved)

    userprofile = os.environ.get("USERPROFILE", "").strip()
    if userprofile:
        add(Path(userprofile))

    for name in ("Desktop", "Documents", "Downloads"):
        link_path = Path.home() / name
        try:
            resolved = link_path.resolve()
        except Exception:
            continue
        if resolved.exists():
            add(resolved.parent)

    users_root = Path("/mnt/c/Users")
    if users_root.exists():
        for child in sorted(users_root.iterdir()):
            if child.is_dir():
                add(child)

    return candidates


def detect_rlbot_botpack_dir() -> Path | None:
    env_path = os.environ.get("RLBOT_BOTPACK_DIR", "").strip()
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate.exists():
            return candidate.resolve()

    suffixes = (
        Path("AppData/Roaming/RLBotGUIX/botpacks"),
        Path("AppData/Roaming/RLBot/botpacks"),
        Path("Documents/RLBotGUIX/botpacks"),
        Path("Documents/RLBot/botpacks"),
        Path("RLBotGUIX/botpacks"),
        Path("RLBot/botpacks"),
    )
    for home in _iter_windows_home_candidates():
        for suffix in suffixes:
            candidate = home / suffix
            if candidate.exists():
                return candidate
    return None


def _read_bot_name(bot_cfg_path: Path) -> str:
    parser = configparser.ConfigParser()
    parser.read(bot_cfg_path)
    return parser.get("Locations", "name", fallback=bot_cfg_path.parent.name).strip() or bot_cfg_path.parent.name


def install_rlbot_package(package_root: str, botpack_dir: str) -> Path:
    source_root = Path(package_root)
    bot_cfg_path = source_root / "src" / "bot.cfg"
    if not bot_cfg_path.exists():
        raise FileNotFoundError(f"Missing RLBot config: {bot_cfg_path}")

    target_root = Path(botpack_dir)
    target_root.mkdir(parents=True, exist_ok=True)
    package_name = _read_bot_name(bot_cfg_path)
    install_path = target_root / package_name
    shutil.copytree(source_root, install_path, dirs_exist_ok=True)
    return install_path
