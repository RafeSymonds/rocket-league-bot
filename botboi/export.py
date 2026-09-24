"""Put a trained policy into the RLBot bot.

    python -m botboi.export                       latest checkpoint of run "botboi"
    python -m botboi.export runs/botboi/policies/500000000.pt
    python -m botboi.export botboi --dest /mnt/c/Users/<you>/Documents/BotBoi

Always writes bot/policy.pt. With --dest it also builds a self-contained bot
folder there (bot files, policy.pt, and the botboi runtime modules), which is
how you get the bot from WSL onto the Windows side that runs Rocket League.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .checkpoints import resolve_policy
from .model import Policy

REPO = Path(__file__).resolve().parent.parent
BOT_DIR = REPO / "bot"
# The only botboi modules the bot imports (directly or indirectly).
RUNTIME_MODULES = ["__init__.py", "actions.py", "model.py", "obs.py", "rlbot_obs.py"]
SKIP = shutil.ignore_patterns("__pycache__", "venv", "*.pyc")


def build_bot_folder(policy: Policy, meta: dict, dest: Path) -> None:
    """Copy the bot files and botboi runtime modules to dest, plus policy.pt."""
    dest.mkdir(parents=True, exist_ok=True)
    # dirs_exist_ok keeps an existing venv in dest, so re-exports don't force
    # a reinstall.
    shutil.copytree(BOT_DIR, dest, ignore=SKIP, dirs_exist_ok=True)
    (dest / "botboi").mkdir(exist_ok=True)
    for name in RUNTIME_MODULES:
        shutil.copy2(REPO / "botboi" / name, dest / "botboi" / name)
    policy.save(dest / "policy.pt", meta)


def export(spec: str, dest: Path | None) -> None:
    policy, meta = resolve_policy(spec)
    policy.save(BOT_DIR / "policy.pt", meta)
    print(f"Wrote {BOT_DIR / 'policy.pt'} ({meta['timesteps']:,} steps, from {meta['source']})")
    if dest is not None:
        build_bot_folder(policy, meta, dest)
        print(f"Built bot folder {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("policy", nargs="?", default="botboi",
                        help="policy .pt, checkpoint folder, run folder, or run name (default: botboi)")
    parser.add_argument("--dest", type=Path, default=None, help="also build a standalone bot folder here")
    args = parser.parse_args()
    export(args.policy, args.dest)


if __name__ == "__main__":
    main()
