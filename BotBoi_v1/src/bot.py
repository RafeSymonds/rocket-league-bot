from __future__ import annotations

import json
import math
import os
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

# Must match training: LookupTableAction wrapped by RepeatAction(repeats=8)
from rlgym.rocket_league.action_parsers import LookupTableAction


# ----------------------------
# Constants (match rlgym.rocket_league.common_values)
# ----------------------------
SIDE_WALL_X = 4096.0
BACK_NET_Y = 5120.0
CEILING_Z = 2044.0

CAR_MAX_SPEED = 2300.0
CAR_MAX_ANG_VEL = 5.5
BALL_MAX_SPEED = 6000.0

POS_COEF = np.array(
    [1.0 / SIDE_WALL_X, 1.0 / BACK_NET_Y, 1.0 / CEILING_Z], dtype=np.float32
)
LIN_VEL_COEF = 1.0 / CAR_MAX_SPEED
ANG_VEL_COEF = 1.0 / CAR_MAX_ANG_VEL
BALL_VEL_COEF = 1.0 / BALL_MAX_SPEED
BOOST_COEF = 1.0 / 100.0

# Training SharedObs uses this:
# max_dist = norm([SIDE_WALL_X, BACK_NET_Y, CEILING_Z])
DIST_COEF = 1.0 / float(np.linalg.norm([SIDE_WALL_X, BACK_NET_Y, CEILING_Z]))


# ----------------------------
# Network (MUST match training Learner.policy_layer_sizes)
# policy_layer_sizes=(512, 512, 256)
# ----------------------------
class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, act_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs)


def _find_key(d: dict[str, Any], candidates: list[str]) -> Optional[str]:
    for k in candidates:
        if k in d:
            return k
    return None


# ----------------------------
# RL orientation helpers (packet gives pitch/yaw/roll)
# ----------------------------
def forward_vector(pitch: float, yaw: float) -> np.ndarray:
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    return np.array([cp * cy, cp * sy, sp], dtype=np.float32)


def up_vector(pitch: float, yaw: float, roll: float) -> np.ndarray:
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    cr = math.cos(roll)
    sr = math.sin(roll)
    return np.array(
        [
            cr * sp * cy + sr * sy,
            cr * sp * sy - sr * cy,
            cr * cp,
        ],
        dtype=np.float32,
    )


def invert_xy(v: np.ndarray) -> np.ndarray:
    # Match rlgym "inverted" perspective: x,y flip, z unchanged
    v2 = v.copy()
    v2[0] = -v2[0]
    v2[1] = -v2[1]
    return v2


# ----------------------------
# RLBot Agent
# ----------------------------
class BotBoi(BaseAgent):
    def initialize_agent(self):
        bot_dir = os.path.dirname(__file__)

        # Action lookup table (must match training LookupTableAction)
        self.action_parser = LookupTableAction()
        # rlgym stores the LUT internally; this is stable-enough in practice
        self.action_table = self.action_parser._lookup_table

        # Load bookkeeping if available (helps confirm act_dim)
        book_path = os.path.join(bot_dir, "BOOK_KEEPING_VARS.json")
        book: dict[str, Any] = {}
        if os.path.exists(book_path):
            with open(book_path, "r", encoding="utf-8") as f:
                book = json.load(f)

        # MUST match training SharedObs.get_obs_space(): shape=(48,)
        self.obs_dim = 47

        # Determine act_dim
        act_key = _find_key(book, ["action_dim", "action_size", "n_actions", "act_dim"])
        if act_key is not None:
            self.act_dim = int(book[act_key])
        else:
            self.act_dim = len(self.action_table)

        if self.act_dim != len(self.action_table):
            print(
                f"[BotBoi] WARNING: act_dim={self.act_dim} but LookupTableAction size={len(self.action_table)}. "
                "If these differ, your actions won't map correctly."
            )

        # MUST match training Learner.policy_layer_sizes=(512, 512, 256)
        hidden_sizes = [512, 512, 256]

        # Use CPU for RLBot by default (stable, no GPU dependency)
        self.device = torch.device("cpu")
        self.policy = MLPPolicy(self.obs_dim, self.act_dim, hidden_sizes).to(
            self.device
        )

        policy_path = os.path.join(bot_dir, "PPO_POLICY.pt")
        state = torch.load(policy_path, map_location=self.device)

        # If your training saved a dict with extra keys, this helps
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        self.policy.load_state_dict(state)
        self.policy.eval()

        # Mimic RepeatAction(LookupTableAction(), repeats=8)
        self.hold_ticks = 8
        self._hold_counter = 0
        self._held_action_index = 0

        print(f"[BotBoi] Loaded policy. obs_dim={self.obs_dim}, act_dim={self.act_dim}")

    # ----------------------------
    # Build EXACT SharedObs (training) observation in RLBot:
    # returns shape (48,)
    # ----------------------------
    def build_obs(self, packet: GameTickPacket) -> np.ndarray:
        me = packet.game_cars[self.index]
        ball = packet.game_ball

        # Find opponent (assumes 1v1)
        opp = None
        for i in range(packet.num_cars):
            if i != self.index:
                opp = packet.game_cars[i]
                break
        if opp is None:
            # fallback (shouldn't happen in 1v1)
            opp = me

        # --- self raw ---
        car_pos = np.array(
            [me.physics.location.x, me.physics.location.y, me.physics.location.z],
            dtype=np.float32,
        )
        car_vel = np.array(
            [me.physics.velocity.x, me.physics.velocity.y, me.physics.velocity.z],
            dtype=np.float32,
        )
        car_ang_vel = np.array(
            [
                me.physics.angular_velocity.x,
                me.physics.angular_velocity.y,
                me.physics.angular_velocity.z,
            ],
            dtype=np.float32,
        )

        pitch = float(me.physics.rotation.pitch)
        yaw = float(me.physics.rotation.yaw)
        roll = float(me.physics.rotation.roll)
        car_fwd = forward_vector(pitch, yaw)
        car_up = up_vector(pitch, yaw, roll)

        boost = float(me.boost)
        on_ground = 1.0 if me.has_wheel_contact else 0.0

        # --- ball raw ---
        ball_pos = np.array(
            [ball.physics.location.x, ball.physics.location.y, ball.physics.location.z],
            dtype=np.float32,
        )
        ball_vel = np.array(
            [ball.physics.velocity.x, ball.physics.velocity.y, ball.physics.velocity.z],
            dtype=np.float32,
        )

        # --- opp raw ---
        opp_pos = np.array(
            [opp.physics.location.x, opp.physics.location.y, opp.physics.location.z],
            dtype=np.float32,
        )
        opp_vel = np.array(
            [opp.physics.velocity.x, opp.physics.velocity.y, opp.physics.velocity.z],
            dtype=np.float32,
        )
        opp_boost = float(opp.boost)
        opp_ground = 1.0 if opp.has_wheel_contact else 0.0

        # --- invert perspective for orange to match training ---
        # Training uses inverted_physics/inverted_ball for orange.
        if self.team == 1:
            car_pos = invert_xy(car_pos)
            car_vel = invert_xy(car_vel)
            car_ang_vel = invert_xy(car_ang_vel)
            car_fwd = invert_xy(car_fwd)
            car_up = invert_xy(car_up)

            ball_pos = invert_xy(ball_pos)
            ball_vel = invert_xy(ball_vel)

            opp_pos = invert_xy(opp_pos)
            opp_vel = invert_xy(opp_vel)

        # --- relations (must match SharedObs) ---
        to_ball = ball_pos - car_pos
        to_ball_dist = float(np.linalg.norm(to_ball))
        to_ball_dir = (to_ball / (to_ball_dist + 1e-6)).astype(np.float32)

        to_opp = opp_pos - car_pos
        to_opp_dist = float(np.linalg.norm(to_opp))
        to_opp_dir = (to_opp / (to_opp_dist + 1e-6)).astype(np.float32)

        # IMPORTANT: training computes goal_y AFTER inversion:
        # goal_y = (-BACK_NET_Y if car.is_orange else BACK_NET_Y)
        # We already inverted world for orange, so goal_y is always +BACK_NET_Y here.
        goal_pos = np.array([0.0, BACK_NET_Y, 0.0], dtype=np.float32)

        ball_to_goal = goal_pos - ball_pos
        ball_to_goal_dist = float(np.linalg.norm(ball_to_goal))
        ball_to_goal_dir = (ball_to_goal / (ball_to_goal_dist + 1e-6)).astype(
            np.float32
        )

        rel_ball_vel = (ball_vel - car_vel) * BALL_VEL_COEF

        approach_speed = float(np.dot(car_vel, to_ball_dir) / CAR_MAX_SPEED)

        # --- normalize exactly like training SharedObs ---
        obs = np.concatenate(
            [
                # self
                car_pos * POS_COEF,
                car_vel * LIN_VEL_COEF,
                car_fwd,
                car_up,
                car_ang_vel * ANG_VEL_COEF,
                np.array([boost * BOOST_COEF], dtype=np.float32),
                np.array([on_ground], dtype=np.float32),
                # ball
                ball_pos * POS_COEF,
                ball_vel * BALL_VEL_COEF,
                rel_ball_vel.astype(np.float32),
                # opponent (relative)
                (opp_pos - car_pos) * POS_COEF,
                (opp_vel - car_vel) * LIN_VEL_COEF,
                np.array([opp_boost * BOOST_COEF], dtype=np.float32),
                np.array([opp_ground], dtype=np.float32),
                # relations
                to_ball_dir,
                np.array([to_ball_dist * DIST_COEF], dtype=np.float32),
                to_opp_dir,
                np.array([to_opp_dist * DIST_COEF], dtype=np.float32),
                ball_to_goal_dir,
                np.array([ball_to_goal_dist * DIST_COEF], dtype=np.float32),
                np.array([approach_speed], dtype=np.float32),
            ],
            axis=0,
        ).astype(np.float32)

        # Safety check
        if obs.shape[0] != self.obs_dim:
            print(f"[BotBoi] ERROR: obs_len={obs.shape[0]} expected={self.obs_dim}")
            return np.zeros((self.obs_dim,), dtype=np.float32)

        return obs

    # ----------------------------
    # Discrete action index -> RLBot controller
    # ----------------------------
    def action_index_to_controls(self, action_index: int) -> SimpleControllerState:
        a = self.action_table[action_index]
        c = SimpleControllerState()
        c.throttle = float(a[0])
        c.steer = float(a[1])
        c.pitch = float(a[2])
        c.yaw = float(a[3])
        c.roll = float(a[4])
        c.jump = bool(a[5])
        c.boost = bool(a[6])
        c.handbrake = bool(a[7])
        return c

    # ----------------------------
    # RLBot step
    # ----------------------------
    def get_output(self, game_tick_packet: GameTickPacket) -> SimpleControllerState:
        # Mimic RepeatAction(repeats=8) by holding chosen action for 8 ticks
        if self._hold_counter > 0:
            self._hold_counter -= 1
            return self.action_index_to_controls(self._held_action_index)

        obs = self.build_obs(game_tick_packet)
        obs_t = torch.from_numpy(obs).float().to(self.device)

        with torch.no_grad():
            logits = self.policy(obs_t)
            action_index = int(torch.argmax(logits).item())

        self._held_action_index = action_index
        self._hold_counter = self.hold_ticks - 1  # use immediately this tick
        return self.action_index_to_controls(action_index)
