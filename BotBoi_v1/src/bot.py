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

from rlgym.rocket_league.action_parsers import LookupTableAction


SIDE_WALL_X = 4096.0
BACK_NET_Y = 5120.0
CEILING_Z = 2044.0
CAR_MAX_SPEED = 2300.0
BALL_MAX_SPEED = 6000.0

POS_COEF = np.array([1.0 / SIDE_WALL_X, 1.0 / BACK_NET_Y, 1.0 / CEILING_Z], dtype=np.float32)
CAR_VEL_COEF = 1.0 / CAR_MAX_SPEED
BALL_VEL_COEF = 1.0 / BALL_MAX_SPEED
BOOST_COEF = 1.0 / 100.0
HEIGHT_COEF = 1.0 / CEILING_Z
DIST_COEF = 1.0 / float(np.linalg.norm([SIDE_WALL_X, BACK_NET_Y, CEILING_Z]))

OBS_DIM = 44
DEFAULT_HIDDEN_SIZES = [512, 512, 256]


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
    out = v.copy()
    out[0] = -out[0]
    out[1] = -out[1]
    return out


def dir_and_dist(vec: np.ndarray) -> tuple[np.ndarray, float]:
    dist = float(np.linalg.norm(vec))
    if dist > 1e-6:
        return (vec / dist).astype(np.float32), dist
    return np.zeros(3, dtype=np.float32), 0.0


class BotBoi(BaseAgent):
    def initialize_agent(self):
        bot_dir = os.path.dirname(__file__)

        self.action_parser = LookupTableAction()
        self.action_table = self.action_parser._lookup_table

        book_path = os.path.join(bot_dir, "BOOK_KEEPING_VARS.json")
        book: dict[str, Any] = {}
        if os.path.exists(book_path):
            with open(book_path, "r", encoding="utf-8") as f:
                book = json.load(f)

        runtime_config_path = os.path.join(bot_dir, "runtime_config.json")
        runtime_config: dict[str, Any] = {}
        if os.path.exists(runtime_config_path):
            with open(runtime_config_path, "r", encoding="utf-8") as f:
                runtime_config = json.load(f)

        self.obs_dim = int(runtime_config.get("obs_dim", OBS_DIM))

        act_dim_from_runtime = runtime_config.get("action_dim")
        act_key = _find_key(book, ["action_dim", "action_size", "n_actions", "act_dim"])
        if act_dim_from_runtime is not None:
            self.act_dim = int(act_dim_from_runtime)
        elif act_key is not None:
            self.act_dim = int(book[act_key])
        else:
            self.act_dim = len(self.action_table)
        if self.act_dim != len(self.action_table):
            print(
                f"[BotBoi] WARNING: act_dim={self.act_dim} but lookup table has {len(self.action_table)} actions"
            )

        hidden_sizes = list(runtime_config.get("policy_hidden_sizes", DEFAULT_HIDDEN_SIZES))
        self.device = torch.device("cpu")
        self.policy = MLPPolicy(self.obs_dim, self.act_dim, hidden_sizes).to(self.device)

        policy_path = os.path.join(bot_dir, "PPO_POLICY.pt")
        state = torch.load(policy_path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        self.policy.load_state_dict(state)
        self.policy.eval()

        self.hold_ticks = int(runtime_config.get("action_repeat", 8))
        self._hold_counter = 0
        self._held_action_index = 0

        checkpoint_dir = runtime_config.get("checkpoint_dir", "")
        cumulative_timesteps = runtime_config.get("cumulative_timesteps", "")
        print(
            f"[BotBoi] Loaded policy. obs_dim={self.obs_dim}, act_dim={self.act_dim}, "
            f"hold_ticks={self.hold_ticks}, checkpoint={checkpoint_dir}, ts={cumulative_timesteps}"
        )

    def build_obs(self, packet: GameTickPacket) -> np.ndarray:
        me = packet.game_cars[self.index]
        ball = packet.game_ball

        car_pos = np.array([me.physics.location.x, me.physics.location.y, me.physics.location.z], dtype=np.float32)
        car_vel = np.array([me.physics.velocity.x, me.physics.velocity.y, me.physics.velocity.z], dtype=np.float32)
        car_fwd = forward_vector(float(me.physics.rotation.pitch), float(me.physics.rotation.yaw))
        car_up = up_vector(
            float(me.physics.rotation.pitch),
            float(me.physics.rotation.yaw),
            float(me.physics.rotation.roll),
        )

        ball_pos = np.array([ball.physics.location.x, ball.physics.location.y, ball.physics.location.z], dtype=np.float32)
        ball_vel = np.array([ball.physics.velocity.x, ball.physics.velocity.y, ball.physics.velocity.z], dtype=np.float32)

        if self.team == 1:
            car_pos = invert_xy(car_pos)
            car_vel = invert_xy(car_vel)
            car_fwd = invert_xy(car_fwd)
            car_up = invert_xy(car_up)
            ball_pos = invert_xy(ball_pos)
            ball_vel = invert_xy(ball_vel)
            my_goal = np.array([0.0, BACK_NET_Y, 0.0], dtype=np.float32)
            enemy_goal = np.array([0.0, -BACK_NET_Y, 0.0], dtype=np.float32)
        else:
            my_goal = np.array([0.0, -BACK_NET_Y, 0.0], dtype=np.float32)
            enemy_goal = np.array([0.0, BACK_NET_Y, 0.0], dtype=np.float32)

        rel_ball_pos = ball_pos - car_pos
        rel_ball_vel = ball_vel - car_vel

        to_ball_dir, to_ball_dist = dir_and_dist(rel_ball_pos)

        ball_speed = float(np.linalg.norm(ball_vel)) * BALL_VEL_COEF
        ball_height = float(ball_pos[2]) * HEIGHT_COEF

        speed_toward_ball = float(np.dot(car_vel, to_ball_dir)) * CAR_VEL_COEF
        cos_forward_to_ball = float(np.dot(car_fwd, to_ball_dir))

        ball_to_goal_dir, _ = dir_and_dist(enemy_goal - ball_pos)
        cos_ball_to_goal = float(np.dot(to_ball_dir, ball_to_goal_dir))

        to_my_goal_dir, to_my_goal_dist = dir_and_dist(my_goal - car_pos)
        to_enemy_goal_dir, to_enemy_goal_dist = dir_and_dist(enemy_goal - car_pos)

        closest_dist = float("inf")
        opp_rel_pos = np.zeros(3, dtype=np.float32)
        opp_rel_vel = np.zeros(3, dtype=np.float32)
        to_opp_dir = np.zeros(3, dtype=np.float32)
        to_opp_dist = 0.0

        for i in range(packet.num_cars):
            if i == self.index:
                continue
            other = packet.game_cars[i]
            if other.team == me.team:
                continue

            other_pos = np.array(
                [other.physics.location.x, other.physics.location.y, other.physics.location.z],
                dtype=np.float32,
            )
            other_vel = np.array(
                [other.physics.velocity.x, other.physics.velocity.y, other.physics.velocity.z],
                dtype=np.float32,
            )

            if self.team == 1:
                other_pos = invert_xy(other_pos)
                other_vel = invert_xy(other_vel)

            rel_pos = other_pos - car_pos
            d = float(np.linalg.norm(rel_pos))
            if d < closest_dist:
                closest_dist = d
                opp_rel_pos = rel_pos
                opp_rel_vel = other_vel - car_vel
                to_opp_dir, to_opp_dist = dir_and_dist(rel_pos)

        obs = np.concatenate(
            [
                car_fwd,
                car_up,
                car_vel * CAR_VEL_COEF,
                np.array([float(me.boost) * BOOST_COEF], dtype=np.float32),
                np.array([1.0 if me.has_wheel_contact else 0.0], dtype=np.float32),
                rel_ball_pos * POS_COEF,
                rel_ball_vel * BALL_VEL_COEF,
                to_ball_dir,
                np.array([to_ball_dist * DIST_COEF], dtype=np.float32),
                np.array([ball_speed], dtype=np.float32),
                np.array([ball_height], dtype=np.float32),
                np.array([speed_toward_ball], dtype=np.float32),
                np.array([cos_forward_to_ball], dtype=np.float32),
                np.array([cos_ball_to_goal], dtype=np.float32),
                to_my_goal_dir,
                to_enemy_goal_dir,
                np.array([to_my_goal_dist * DIST_COEF], dtype=np.float32),
                np.array([to_enemy_goal_dist * DIST_COEF], dtype=np.float32),
                opp_rel_pos * POS_COEF,
                opp_rel_vel * CAR_VEL_COEF,
                to_opp_dir,
                np.array([to_opp_dist * DIST_COEF], dtype=np.float32),
            ],
            axis=0,
        ).astype(np.float32)

        if obs.shape[0] != self.obs_dim:
            print(f"[BotBoi] ERROR: obs_len={obs.shape[0]}, expected={self.obs_dim}")
            return np.zeros((self.obs_dim,), dtype=np.float32)

        return obs

    def action_index_to_controls(self, action_index: int) -> SimpleControllerState:
        idx = int(np.clip(action_index, 0, len(self.action_table) - 1))
        a = self.action_table[idx]
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

    def get_output(self, game_tick_packet: GameTickPacket) -> SimpleControllerState:
        if self._hold_counter > 0:
            self._hold_counter -= 1
            return self.action_index_to_controls(self._held_action_index)

        obs = self.build_obs(game_tick_packet)
        obs_t = torch.from_numpy(obs).float().to(self.device)

        with torch.no_grad():
            logits = self.policy(obs_t)
            action_index = int(torch.argmax(logits).item())

        self._held_action_index = action_index
        self._hold_counter = self.hold_ticks - 1
        return self.action_index_to_controls(action_index)
