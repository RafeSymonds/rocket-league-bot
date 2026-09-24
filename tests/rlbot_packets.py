"""Turn RocketSim states into the RLBot v5 packets the game would send.

Used by the parity test to check that the in-game bot builds exactly the
observations training produced. The field mapping follows the RLBot
flatbuffers schema (gamedata.fbs) and rlgym_compat's reading of it.
"""

from __future__ import annotations

import numpy as np
from rlbot import flat
from rlgym.rocket_league.common_values import BOOST_LOCATIONS

DOUBLEJUMP_MAX_DELAY = 1.25
TICK_TIME = 1 / 120
OCTANE_HITBOX = flat.BoxShape(length=118.00738, width=84.19941, height=36.159073)
OCTANE_OFFSET = flat.Vector3(x=13.87566, y=0, z=20.754988)


def vec(v) -> flat.Vector3:
    return flat.Vector3(float(v[0]), float(v[1]), float(v[2]))


def physics(phys) -> flat.Physics:
    pitch, yaw, roll = phys.euler_angles
    return flat.Physics(
        location=vec(phys.position),
        rotation=flat.Rotator(float(pitch), float(yaw), float(roll)),
        velocity=vec(phys.linear_velocity),
        angular_velocity=vec(phys.angular_velocity),
    )


def is_big_pad(sim_index: int) -> bool:
    return BOOST_LOCATIONS[sim_index][2] > 71.5


def field_info(pad_order: np.ndarray) -> flat.FieldInfo:
    """RLBot lists pads in its own order: RLBot pad i is simulator pad pad_order[i]."""
    pads = [
        flat.BoostPad(location=vec(BOOST_LOCATIONS[j]), is_full_boost=is_big_pad(j))
        for j in pad_order
    ]
    return flat.FieldInfo(boost_pads=pads, goals=[])


def air_state(car) -> flat.AirState:
    if car.on_ground and not car.is_jumping:
        return flat.AirState.OnGround
    if car.is_jumping:
        return flat.AirState.Jumping
    if car.is_flipping:
        return flat.AirState.Dodging
    if car.has_double_jumped and car.air_time_since_jump < 0.2:
        return flat.AirState.DoubleJumping
    return flat.AirState.InAir


def player(agent_index: int, car, controls: np.ndarray) -> flat.PlayerInfo:
    can_still_dodge = car.has_jumped and not car.has_flipped and not car.has_double_jumped
    dodge_timeout = (
        max(DOUBLEJUMP_MAX_DELAY - car.air_time_since_jump, 0.0)
        if can_still_dodge and not car.on_ground
        else -1.0
    )
    throttle, steer, pitch, yaw, roll, jump, boost, handbrake = controls
    return flat.PlayerInfo(
        physics=physics(car.physics),
        hitbox=OCTANE_HITBOX,
        hitbox_offset=OCTANE_OFFSET,
        air_state=air_state(car),
        dodge_timeout=float(dodge_timeout),
        demolished_timeout=float(car.demo_respawn_timer) if car.is_demoed else -1.0,
        is_supersonic=bool(car.is_supersonic),
        is_bot=True,
        name=f"car{agent_index}",
        team=int(car.team_num),
        boost=float(car.boost_amount),
        player_id=agent_index,
        last_input=flat.ControllerState(
            throttle=float(throttle),
            steer=float(steer),
            pitch=float(pitch),
            yaw=float(yaw),
            roll=float(roll),
            jump=bool(jump),
            boost=bool(boost),
            handbrake=bool(handbrake),
        ),
        has_jumped=bool(car.has_jumped),
        has_double_jumped=bool(car.has_double_jumped),
        has_dodged=bool(car.has_flipped),
        dodge_elapsed=float(car.flip_time),
    )


def game_packet(state, agent_ids: list, controls: dict, pad_order: np.ndarray) -> flat.GamePacket:
    """`agent_ids[i]` becomes RLBot player_id i. `controls` holds the controller
    input each car used on the last tick."""
    pads = []
    for j in pad_order:
        cooldown = float(state.boost_pad_timers[j])
        respawn = 10.0 if is_big_pad(j) else 4.0
        # RLBot counts seconds since pickup; the simulator counts down to respawn.
        pads.append(flat.BoostPadState(is_active=cooldown == 0, timer=0.0 if cooldown == 0 else respawn - cooldown))
    return flat.GamePacket(
        players=[player(i, state.cars[a], controls[a]) for i, a in enumerate(agent_ids)],
        boost_pads=pads,
        balls=[flat.BallInfo(physics=physics(state.ball), shape=flat.SphereShape(diameter=182.5))],
        match_info=flat.MatchInfo(
            seconds_elapsed=state.tick_count * TICK_TIME,
            match_phase=flat.MatchPhase.Active,
            frame_num=int(state.tick_count),
        ),
        teams=[flat.TeamInfo(0, 0), flat.TeamInfo(1, 0)],
    )
