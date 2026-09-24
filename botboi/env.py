"""RLGym environment used for training and evaluation."""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from rlgym.api import RLGym, SharedInfoProvider
from rlgym.rocket_league.done_conditions import (
    AnyCondition,
    GoalCondition,
    NoTouchTimeoutCondition,
    TimeoutCondition,
)
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import KickoffMutator, MutatorSequence
from rlgym_tools.rocket_league.state_mutators.variable_team_size_mutator import (
    VariableTeamSizeMutator,
)
from rlgym_tools.rocket_league.state_mutators.weighted_sample_mutator import (
    WeightedSampleMutator,
)

from .actions import make_action_parser
from .config import PHASES, EnvConfig
from .mutators import RandomGroundStateMutator
from .obs import BotObs
from .rewards import build_reward

AERIAL_TOUCH_HEIGHT = 300.0


class StatsProvider(SharedInfoProvider):
    """Sums game stats in each env process and twice a second writes the
    running totals to <stats_dir>/<pid>.json. BotMetricsLogger in the learner
    turns them into per-iteration numbers such as goals per game minute.

    rlgym-learn can ship shared_info to the learner itself, but in 2.0.0
    setting shared_info_serde_type corrupts its startup message parsing (the
    reader skips the shared-info offset), so stats go through files instead.
    """

    FLUSH_SECONDS = 0.5

    def __init__(self, stats_dir: str | None):
        self.stats_dir = Path(stats_dir) if stats_dir else None
        self.totals: dict[str, float] = defaultdict(float)
        self._demoed: dict[Any, bool] = {}
        self._last_flush = time.monotonic()

    def create(self, shared_info: dict[str, Any]) -> dict[str, Any]:
        return shared_info

    def set_state(self, agents, initial_state, shared_info):
        self._demoed = {agent: car.is_demoed for agent, car in initial_state.cars.items()}
        return shared_info

    def step(self, agents, state, shared_info):
        cars = list(state.cars.items())
        touches = sum(car.ball_touches > 0 for _, car in cars)
        totals = self.totals
        totals["steps"] += 1
        totals["cars"] += len(cars)
        totals["goals"] += state.goal_scored
        totals["touches"] += touches
        if state.ball.position[2] > AERIAL_TOUCH_HEIGHT:
            totals["aerial_touches"] += touches
        totals["demos"] += sum(
            car.is_demoed and not self._demoed.get(agent, False) for agent, car in cars
        )
        totals["ball_speed"] += float(np.linalg.norm(state.ball.linear_velocity))
        totals["car_speed"] += float(np.mean([np.linalg.norm(car.physics.linear_velocity) for _, car in cars]))
        totals["in_air"] += float(np.mean([not car.on_ground for _, car in cars]))
        totals["boost"] += float(np.mean([car.boost_amount for _, car in cars]))
        self._demoed = {agent: car.is_demoed for agent, car in cars}

        if self.stats_dir is not None and time.monotonic() - self._last_flush > self.FLUSH_SECONDS:
            self._flush()
        return shared_info

    def _flush(self) -> None:
        self._last_flush = time.monotonic()
        path = self.stats_dir / f"{os.getpid()}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.totals))
        os.replace(tmp, path)  # atomic, so the learner never reads half a file


def build_env(cfg: EnvConfig) -> RLGym:
    phase = PHASES[cfg.phase]
    modes = {(size, size): weight for size, weight in cfg.team_size_weights.items() if weight > 0}
    state_mutator = MutatorSequence(
        VariableTeamSizeMutator(modes),
        WeightedSampleMutator(
            [KickoffMutator(), RandomGroundStateMutator()],
            [cfg.kickoff_prob, 1.0 - cfg.kickoff_prob],
        ),
    )
    return RLGym(
        state_mutator=state_mutator,
        obs_builder=BotObs(),
        action_parser=make_action_parser(),
        reward_fn=build_reward(phase["rewards"], phase["team_spirit"]),
        termination_cond=GoalCondition(),
        truncation_cond=AnyCondition(
            NoTouchTimeoutCondition(timeout_seconds=cfg.no_touch_timeout_seconds),
            TimeoutCondition(timeout_seconds=cfg.episode_timeout_seconds),
        ),
        transition_engine=RocketSimEngine(),
        shared_info_provider=StatsProvider(cfg.stats_dir),
    )
