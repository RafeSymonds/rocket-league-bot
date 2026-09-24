"""Play two policies against each other in RocketSim and report who wins.

Each game starts from a kickoff and ends at the first goal (a win) or after
the no-touch / length timeout (a draw). Policy A plays blue in half the games
and orange in the other half. "Own goals" counts goals where the conceding
team touched the ball last; early in training these are common, because the
bot hits the ball hard before it learns to aim.

Examples:
    python -m botboi.evaluate runs/botboi/policies/200000000.pt botboi
    python -m botboi.evaluate botboi runs/botboi/policies/100000000.pt --games 200 --modes 1
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
from rlgym.api import RLGym
from rlgym.rocket_league.done_conditions import (
    AnyCondition,
    GoalCondition,
    NoTouchTimeoutCondition,
    TimeoutCondition,
)
from rlgym.rocket_league.reward_functions import GoalReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import (
    FixedTeamSizeMutator,
    KickoffMutator,
    MutatorSequence,
)

from .actions import make_action_parser
from .checkpoints import resolve_policy
from .model import Policy
from .obs import BotObs


def make_eval_env(team_size: int, max_seconds: float) -> RLGym:
    return RLGym(
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=team_size, orange_size=team_size),
            KickoffMutator(),
        ),
        obs_builder=BotObs(),
        action_parser=make_action_parser(),
        reward_fn=GoalReward(),
        termination_cond=GoalCondition(),
        truncation_cond=AnyCondition(
            NoTouchTimeoutCondition(timeout_seconds=30),
            TimeoutCondition(timeout_seconds=max_seconds),
        ),
        transition_engine=RocketSimEngine(),
    )


def play_game(env: RLGym, blue: Policy, orange: Policy, deterministic: bool) -> tuple[int, bool]:
    """Return (+1 if blue scores first, -1 if orange does, 0 for a timeout,
    whether the goal was an own goal, i.e. the conceding team touched last)."""
    obs = env.reset()
    last_touch_orange = None
    while True:
        actions = {}
        for policy, is_orange in ((blue, False), (orange, True)):
            agents = [a for a in obs if env.state.cars[a].is_orange == is_orange]
            chosen = policy.act(np.stack([obs[a] for a in agents]), deterministic)
            actions.update({a: np.array([act]) for a, act in zip(agents, chosen)})
        obs, _, terminated, truncated = env.step(actions)
        for car in env.state.cars.values():
            if car.ball_touches > 0:
                last_touch_orange = car.is_orange
        if any(terminated.values()):
            blue_scored = env.state.scoring_team == 0
            return (1 if blue_scored else -1), last_touch_orange == blue_scored
        if any(truncated.values()):
            return 0, False


def wilson_interval(wins: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a win rate."""
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a", help="policy A: .pt snapshot, checkpoint folder, run folder, or run name")
    parser.add_argument("b", help="policy B, same forms as A")
    parser.add_argument("--games", type=int, default=100, help="games per mode")
    parser.add_argument("--modes", default="1,2", help='team sizes to test, e.g. "1" or "1,2"')
    parser.add_argument("--max-seconds", type=float, default=120.0, help="game length cap (draw)")
    parser.add_argument("--deterministic", action="store_true", help="argmax actions instead of sampling")
    args = parser.parse_args()

    policy_a, meta_a = resolve_policy(args.a)
    policy_b, meta_b = resolve_policy(args.b)
    print(f"A: {args.a} ({meta_a['timesteps']:,} steps)")
    print(f"B: {args.b} ({meta_b['timesteps']:,} steps)")

    for team_size in (int(m) for m in args.modes.split(",")):
        env = make_eval_env(team_size, args.max_seconds)
        a_wins = b_wins = draws = own_goals = 0
        start = time.time()
        for game in range(args.games):
            a_is_blue = game % 2 == 0
            blue, orange = (policy_a, policy_b) if a_is_blue else (policy_b, policy_a)
            result, own_goal = play_game(env, blue, orange, args.deterministic)
            own_goals += own_goal
            if result == 0:
                draws += 1
            elif (result == 1) == a_is_blue:
                a_wins += 1
            else:
                b_wins += 1
        env.close()
        # Draws count as half a win so the rate stays comparable across runs.
        score = a_wins + 0.5 * draws
        low, high = wilson_interval(score, args.games)
        print(
            f"{team_size}v{team_size}: A {a_wins} - B {b_wins} - draws {draws} "
            f"(own goals {own_goals}) | "
            f"A score {score / args.games:.1%} (95% CI {low:.1%}-{high:.1%}) "
            f"[{time.time() - start:.0f}s]"
        )


if __name__ == "__main__":
    main()
