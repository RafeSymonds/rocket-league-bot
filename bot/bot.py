"""BotBoi for RLBot v5.

Runs the exported policy (policy.pt next to this file) with the same
observation and action code used in training. A new action is chosen every
TICK_SKIP physics ticks and held in between, as in training.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The botboi package sits next to this file in an exported bot folder and one
# level up when running from the repo.
sys.path[:0] = [str(HERE), str(HERE.parent)]

from rlbot import flat  # noqa: E402
from rlbot.managers import Bot  # noqa: E402

from botboi.actions import ACTION_TABLE, TICK_SKIP  # noqa: E402
from botboi.model import Policy, check_compatible  # noqa: E402
from botboi.rlbot_obs import RLBotObsAdapter  # noqa: E402

AGENT_ID = "rafe/botboi"
# argmax actions. Set False to sample from the policy like training does.
DETERMINISTIC = True
PLAY_PHASES = (flat.MatchPhase.Kickoff, flat.MatchPhase.Active)
TRACKED_PHASES = (flat.MatchPhase.Countdown, *PLAY_PHASES)


def to_controls(action_index: int) -> flat.ControllerState:
    throttle, steer, pitch, yaw, roll, jump, boost, handbrake = ACTION_TABLE[action_index]
    return flat.ControllerState(
        throttle=float(throttle),
        steer=float(steer),
        pitch=float(pitch),
        yaw=float(yaw),
        roll=float(roll),
        jump=bool(jump),
        boost=bool(boost),
        handbrake=bool(handbrake),
    )


class BotBoi(Bot):
    def initialize(self):
        self.policy, meta = Policy.load(HERE / "policy.pt")
        check_compatible(meta, "policy.pt")
        self.adapter = RLBotObsAdapter(self.field_info, self.match_config)
        self.controls = flat.ControllerState()
        self.ticks_until_action = 0
        self.last_frame: int | None = None
        self.logger.info(f"Loaded policy trained for {meta['timesteps']:,} steps")

    def get_output(self, packet: flat.GamePacket) -> flat.ControllerState:
        phase = packet.match_info.match_phase
        if phase not in TRACKED_PHASES or not packet.balls:
            self.last_frame = None
            return flat.ControllerState()

        self.adapter.update(packet)
        if phase not in PLAY_PHASES:
            # Countdown: cars are frozen. Act on the first kickoff tick.
            self.ticks_until_action = 0
            self.last_frame = None
            return flat.ControllerState()

        frame = packet.match_info.frame_num
        # Packets can skip ticks; count real ticks so actions last TICK_SKIP.
        self.ticks_until_action -= 1 if self.last_frame is None else frame - self.last_frame
        self.last_frame = frame
        if self.ticks_until_action <= 0:
            obs = self.adapter.build_obs(self.player_id)
            action = int(self.policy.act(obs[None], DETERMINISTIC)[0])
            self.controls = to_controls(action)
            self.ticks_until_action = TICK_SKIP
        return self.controls


if __name__ == "__main__":
    BotBoi(AGENT_ID).run()
