from __future__ import annotations

from typing import Any, Dict, List

from rlgym.api import DoneCondition
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.done_conditions import GoalCondition

from .config import make_stage_config


class TouchDoneCondition(DoneCondition[str, GameState]):
    def reset(
        self,
        agents: List[str],
        initial_state: GameState,
        shared_info: Dict[str, Any],
    ) -> None:
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
    def __init__(self, stage_ref: "EnvBuilder"):
        self.stage_ref = stage_ref
        self.touch_done = TouchDoneCondition()
        self.goal_done = GoalCondition()

    def reset(self, agents, initial_state, shared_info):
        self.touch_done.reset(agents, initial_state, shared_info)
        self.goal_done.reset(agents, initial_state, shared_info)

    def is_done(self, agents, state, shared_info):
        cfg = make_stage_config(self.stage_ref.stage)
        if cfg.end_on_touch:
            return self.touch_done.is_done(agents, state, shared_info)
        return self.goal_done.is_done(agents, state, shared_info)
