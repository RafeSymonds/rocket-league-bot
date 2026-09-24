"""Start an RLBot v5 match from a match config.

    venv\Scripts\python run_match.py match_1v1.toml

Needs RLBot v5 installed (https://rlbot.org), which provides RLBotServer.
RLBot launches Rocket League without Easy Anti-Cheat, so this is offline only.
"""

import sys
from pathlib import Path
from time import sleep

from rlbot import flat
from rlbot.managers import MatchManager

if __name__ == "__main__":
    config = Path(sys.argv[1] if len(sys.argv) > 1 else "match_1v1.toml").resolve()
    manager = MatchManager()
    manager.start_match(config)
    try:
        while (
            manager.packet is None
            or manager.packet.match_info.match_phase != flat.MatchPhase.Ended
        ):
            sleep(0.5)
    finally:
        manager.shut_down()
