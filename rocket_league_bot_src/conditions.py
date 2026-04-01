from __future__ import annotations

from typing import Any, Dict, List

from rlgym.api import DoneCondition
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.done_conditions import (
    AnyCondition,
    GoalCondition,
    NoTouchTimeoutCondition,
    TimeoutCondition,
)

from .config import Stage


class TouchDoneCondition(DoneCondition[str, GameState]):
    def reset(
        self,
        agents: List[str],
        initial_state: GameState,
        shared_info: Dict[str, Any],
    ) -> None:
        if initial_state is None:
            self.prev_touches = {a: 0 for a in agents}
            return
        self.prev_touches = {a: int(initial_state.cars[a].ball_touches) for a in agents}

    def is_done(
        self, agents: List[str], state: GameState, shared_info: Dict[str, Any]
    ) -> Dict[str, bool]:
        touched = False
        for a in agents:
            cur = int(state.cars[a].ball_touches)
            if cur > self.prev_touches.get(a, cur):
                touched = True
            self.prev_touches[a] = cur
        return {a: touched for a in agents}


class CurriculumDoneCondition(DoneCondition[str, GameState]):
    def __init__(self, curriculum_manager):
        self.curriculum_manager = curriculum_manager
        self.touch_done = TouchDoneCondition()
        self.goal_done = GoalCondition()

    def reset(self, agents, initial_state, shared_info):
        self.touch_done.reset(agents, initial_state, shared_info)
        self.goal_done.reset(agents, initial_state, shared_info)

    def is_done(self, agents, state, shared_info):
        cfg = self.curriculum_manager.current_config()
        if cfg.end_on_touch:
            return self.touch_done.is_done(agents, state, shared_info)
        return self.goal_done.is_done(agents, state, shared_info)


class CurriculumTruncationCondition(DoneCondition[str, GameState]):
    def __init__(self, curriculum_manager):
        self.curriculum_manager = curriculum_manager
        self._conditions: dict[Stage, DoneCondition[str, GameState]] = {}
        for stage in Stage:
            cfg = self.curriculum_manager.current_config() if stage == self.curriculum_manager.stage else None
            if cfg is None or cfg.stage != stage:
                from .config import build_stage_config

                cfg = build_stage_config(stage, difficulty=1.0)
            self._conditions[stage] = AnyCondition(
                NoTouchTimeoutCondition(cfg.no_touch_timeout_s),
                TimeoutCondition(cfg.timeout_s),
            )

    def reset(self, agents, initial_state, shared_info):
        for cond in self._conditions.values():
            cond.reset(agents, initial_state, shared_info)

    def is_done(self, agents, state, shared_info):
        stage = self.curriculum_manager.stage
        return self._conditions[stage].is_done(agents, state, shared_info)
