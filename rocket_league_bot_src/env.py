from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from rlgym.api import RLGym
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym_ppo.util.rlgym_v2_gym_wrapper import RLGymV2GymWrapper

from .conditions import CurriculumDoneCondition, CurriculumTruncationCondition
from .config import ACTION_REPEAT
from .curriculum import CurriculumManager
from .league import SnapshotLeague
from .mutators import DynamicMatchMutator
from .obs import SharedObs
from .opponent import SelfPlayOpponentGymWrapper
from .reporting import write_training_report
from .rewards import CurriculumReward
from .utils import Stats
from .checkpoints import find_opponent_checkpoint, load_checkpoint_book

try:
    from rlgym_tools.rocket_league.shared_info_providers.scoreboard_provider import (
        ScoreboardProvider,
    )
except Exception:  # pragma: no cover - optional dependency until installed
    ScoreboardProvider = None


class ProcessIterationLogger:
    _METRICS_COLUMNS = [
        "unix_time",
        "stage",
        "difficulty",
        "sps",
        "episodes",
        "avg_return",
        "touch_rate",
        "goal_rate",
        "blue_goal_rate",
        "median_t_first",
        "median_t_goal",
        "ema_touch",
        "ema_goal",
    ]

    def __init__(
        self,
        env,
        process_id: int,
        iteration_timesteps: int,
        curriculum_manager: CurriculumManager,
        checkpoint_root: str,
        curriculum_state_path: str,
        opponent_state_path: str,
        self_play_mode: str,
        opponent_gap_ts: int,
        current_checkpoint_dir: str,
        fixed_opponent_checkpoint: str,
    ):
        self.env = env
        self.pid = process_id
        self.iteration_ts = iteration_timesteps
        self.cm = curriculum_manager
        self.checkpoint_root = checkpoint_root
        self.curriculum_state_path = curriculum_state_path
        self.opponent_state_path = opponent_state_path
        self.self_play_mode = str(self_play_mode)
        self.opponent_gap_ts = int(opponent_gap_ts)
        self.current_checkpoint_dir = current_checkpoint_dir
        self.fixed_opponent_checkpoint = fixed_opponent_checkpoint
        self.league = SnapshotLeague()
        self._last_exported_checkpoint = ""
        self.log_counter = 0

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self._reset_iteration_stats()
        self._reset_episode_stats()
        self._init_metrics_file()
        self._sync_curriculum_state()
        self._write_opponent_state()

    def _init_metrics_file(self):
        self._metrics_path = None
        if self.pid != 0:
            return
        os.makedirs("data", exist_ok=True)
        self._metrics_path = os.path.join("data", "training_metrics.csv")
        if os.path.exists(self._metrics_path):
            self._migrate_metrics_file_if_needed()
            return
        with open(self._metrics_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(self._METRICS_COLUMNS)

    def _migrate_metrics_file_if_needed(self) -> None:
        if self._metrics_path is None:
            return
        try:
            with open(self._metrics_path, "r", newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
        except Exception:
            return
        if not rows:
            return

        header = rows[0]
        if header == self._METRICS_COLUMNS:
            return

        migrated_rows: list[list[str]] = [self._METRICS_COLUMNS]
        if header == [c for c in self._METRICS_COLUMNS if c != "blue_goal_rate"]:
            for row in rows[1:]:
                if not row:
                    continue
                padded = row[:8] + [""] + row[8:]
                if len(padded) < len(self._METRICS_COLUMNS):
                    padded.extend([""] * (len(self._METRICS_COLUMNS) - len(padded)))
                migrated_rows.append(padded[: len(self._METRICS_COLUMNS)])
        else:
            return

        with open(self._metrics_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerows(migrated_rows)

    def _reset_iteration_stats(self):
        self.iteration_start_time = time.time()
        self.iteration_steps = 0
        self.iteration_episodes = 0
        self.iteration_return = 0.0
        self.iteration_goals = 0
        self.iteration_blue_goals = 0
        self.iteration_success_eps = 0
        self.iteration_median_t_first = []
        self.iteration_median_t_goal = []

    def _reset_episode_stats(self):
        self.ep_return = 0.0
        self.ep_steps = 0
        self.ep_ball_touches = 0
        self.ep_first_touch_step = -1
        self.ep_goal_step = -1
        self._prev_touches = {}

    def _append_metrics_row(
        self,
        stage: str,
        difficulty: float,
        sps: float,
        avg_return: float,
        touch_rate: float,
        goal_rate: float,
        blue_goal_rate: float,
        median_t_first: float,
        median_t_goal: float,
    ) -> None:
        if self._metrics_path is None:
            return
        snap = self.cm.snapshot()
        with open(self._metrics_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    f"{time.time():.3f}",
                    stage,
                    f"{difficulty:.6f}",
                    f"{sps:.3f}",
                    int(self.iteration_episodes),
                    f"{avg_return:.6f}",
                    f"{touch_rate:.6f}",
                    f"{goal_rate:.6f}",
                    f"{blue_goal_rate:.6f}",
                    f"{median_t_first:.3f}",
                    f"{median_t_goal:.3f}",
                    f"{snap.ema_touch:.6f}",
                    f"{snap.ema_goal:.6f}",
                ]
            )
        try:
            write_training_report(metrics_path=self._metrics_path)
        except Exception as exc:
            print(f"[report] failed to update training report: {exc}")

    def close(self, **kwargs):
        pass

    def reset(self, **kwargs):
        self._sync_curriculum_state()
        self._reset_episode_stats()
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}

        state = info.get("state")
        if state is not None:
            for agent, car in state.cars.items():
                self._prev_touches[agent] = int(car.ball_touches)
        return obs

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        else:
            obs, reward, done, info = result
            terminated = done
            truncated = False

        self.iteration_steps += 1
        self.ep_steps += 1
        self.ep_return += float(np.mean(reward))

        state = info.get("state")
        if state is not None:
            for agent, car in state.cars.items():
                prev = self._prev_touches.get(agent, int(car.ball_touches))
                cur = int(car.ball_touches)
                self._prev_touches[agent] = cur
                if cur > prev:
                    self.ep_ball_touches += 1
                    if self.ep_first_touch_step == -1:
                        self.ep_first_touch_step = self.ep_steps
            if state.goal_scored and self.ep_goal_step == -1:
                self.ep_goal_step = self.ep_steps

        if terminated or truncated:
            self.iteration_episodes += 1
            self.iteration_return += self.ep_return

            if self.ep_ball_touches > 0:
                self.iteration_success_eps += 1
                self.iteration_median_t_first.append(self.ep_first_touch_step)

            if self.ep_goal_step != -1:
                self.iteration_goals += 1
                self.iteration_median_t_goal.append(self.ep_goal_step)
                if state is not None and int(state.scoring_team) == 0:
                    self.iteration_blue_goals += 1

            self._reset_episode_stats()

        if self.iteration_steps >= self.iteration_ts:
            self._report_and_reset_iteration()

        return obs, reward, terminated, truncated, info

    def _report_and_reset_iteration(self):
        avg_return = self.iteration_return / self.iteration_episodes if self.iteration_episodes > 0 else 0.0
        touch_rate = self.iteration_success_eps / self.iteration_episodes if self.iteration_episodes > 0 else 0.0
        goal_rate = self.iteration_goals / self.iteration_episodes if self.iteration_episodes > 0 else 0.0
        blue_goal_rate = self.iteration_blue_goals / self.iteration_episodes if self.iteration_episodes > 0 else 0.0
        median_t_first = np.median(self.iteration_median_t_first) if self.iteration_median_t_first else -1.0
        median_t_goal = np.median(self.iteration_median_t_goal) if self.iteration_median_t_goal else -1.0

        duration = max(time.time() - self.iteration_start_time, 1e-6)
        sps = self.iteration_steps / duration
        cfg = self.cm.current_config()
        snap = self.cm.snapshot()

        self._append_metrics_row(
            stage=cfg.stage.value,
            difficulty=snap.difficulty,
            sps=sps,
            avg_return=avg_return,
            touch_rate=touch_rate,
            goal_rate=goal_rate,
            blue_goal_rate=blue_goal_rate,
            median_t_first=float(median_t_first),
            median_t_goal=float(median_t_goal),
        )

        if self.pid == 0 and self.log_counter % 3 == 0:
            print(
                f"[P-{self.pid:02d} | {cfg.stage.value:<9}] "
                f"diff={snap.difficulty:0.2f} | "
                f"SPS={sps:7.1f} | "
                f"Eps={self.iteration_episodes:4d} | "
                f"AvgRet={avg_return:7.3f} | "
                f"Touch={touch_rate:0.2f} | "
                f"Goal={goal_rate:0.2f} | "
                f"BlueGoal={blue_goal_rate:0.2f}"
            )
        self.log_counter += 1

        stats = Stats(
            touch_rate=touch_rate,
            goal_rate=goal_rate,
            median_t_first=float(median_t_first if median_t_first != -1 else 9999.0),
            median_t_goal=float(median_t_goal if median_t_goal != -1 else 9999.0),
        )
        if self.pid == 0:
            self.cm.maybe_advance(stats)
            self._write_curriculum_state()
            self._write_opponent_state(
                blue_goal_rate=blue_goal_rate,
                goal_rate=goal_rate,
            )
        else:
            self._sync_curriculum_state()
        self._maybe_register_league_snapshot()
        self._maybe_auto_export_latest_checkpoint()
        self._reset_iteration_stats()

    def _sync_curriculum_state(self) -> None:
        path = Path(self.curriculum_state_path)
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text())
        except Exception:
            return
        if isinstance(payload, dict):
            self.cm.load_dict(payload)

    def _write_curriculum_state(self) -> None:
        if self.pid != 0:
            return
        path = Path(self.curriculum_state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.cm.to_dict(), indent=2, sort_keys=True))

    def _persist_curriculum_state_to_checkpoint(self, checkpoint_dir: Path) -> None:
        book = checkpoint_dir / "BOOK_KEEPING_VARS.json"
        if not book.exists():
            return
        try:
            data = json.loads(book.read_text())
        except Exception:
            return
        data["curriculum_state"] = self.cm.to_dict()
        book.write_text(json.dumps(data, indent=4))

    def _maybe_register_league_snapshot(self) -> None:
        if self.pid != 0:
            return

        latest = self._find_latest_checkpoint()
        if latest is None:
            return

        book = latest / "BOOK_KEEPING_VARS.json"
        if not book.exists():
            return
        try:
            data = json.loads(book.read_text())
        except Exception:
            return

        ts = int(data.get("cumulative_timesteps", 0))
        if ts <= 0 or ts % 10_000_000 != 0:
            return

        cfg = self.cm.current_config()
        self.league.register_snapshot(
            checkpoint_dir=str(latest),
            cumulative_timesteps=ts,
            stage=cfg.stage.value,
            difficulty=self.cm.snapshot().difficulty,
        )

    def _maybe_auto_export_latest_checkpoint(self) -> None:
        if self.pid != 0:
            return
        latest = self._find_latest_checkpoint()
        if latest is None:
            return
        self.current_checkpoint_dir = str(latest)
        self._write_opponent_state()
        self._persist_curriculum_state_to_checkpoint(latest)
        latest_str = str(latest)
        if latest_str == self._last_exported_checkpoint:
            return
        self._last_exported_checkpoint = latest_str

        try:
            from .export import export_checkpoint_to_rlbot_package

            export_checkpoint_to_rlbot_package(latest_str)
        except Exception as exc:
            print(f"[export] failed to export latest checkpoint: {exc}")

    def _find_latest_checkpoint(self):
        root = Path(self.checkpoint_root)
        if not root.exists():
            return None

        candidates: list[tuple[int, float, Path]] = []
        for book in root.rglob("BOOK_KEEPING_VARS.json"):
            try:
                data = json.loads(book.read_text())
                ts = int(data.get("cumulative_timesteps", 0))
            except Exception:
                ts = 0
            candidates.append((ts, book.stat().st_mtime, book.parent))

        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates[-1][2]

    def _write_opponent_state(
        self,
        blue_goal_rate: float | None = None,
        goal_rate: float | None = None,
    ) -> None:
        if self.pid != 0:
            return
        path = Path(self.opponent_state_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.self_play_mode != "frozen":
            payload = {
                "enabled": False,
                "mode": self.self_play_mode,
                "checkpoint_dir": "",
                "gap_ts": 0,
                "base_gap_ts": int(self.opponent_gap_ts),
                "blue_goal_rate": None if blue_goal_rate is None else float(blue_goal_rate),
            }
            path.write_text(json.dumps(payload, indent=2, sort_keys=True))
            return

        checkpoint_dir = ""
        effective_gap_ts = int(self.opponent_gap_ts)
        if blue_goal_rate is not None and goal_rate is not None and self.cm.current_config().full_match:
            if blue_goal_rate >= 0.90 and goal_rate >= 0.90:
                effective_gap_ts = max(500_000, self.opponent_gap_ts // 4)
            elif blue_goal_rate >= 0.75 and goal_rate >= 0.80:
                effective_gap_ts = max(1_000_000, self.opponent_gap_ts // 2)

        if self.fixed_opponent_checkpoint:
            checkpoint_dir = self.fixed_opponent_checkpoint
        elif self.current_checkpoint_dir:
            book = load_checkpoint_book(self.current_checkpoint_dir)
            current_ts = int(book.get("cumulative_timesteps", 0))
            checkpoint_dir = find_opponent_checkpoint(
                self.checkpoint_root,
                current_ts=current_ts,
                gap_ts=effective_gap_ts,
                exclude_checkpoint_dir=self.current_checkpoint_dir,
            )

        payload = {
            "enabled": bool(checkpoint_dir),
            "mode": self.self_play_mode,
            "checkpoint_dir": str(checkpoint_dir),
            "gap_ts": int(effective_gap_ts),
            "base_gap_ts": int(self.opponent_gap_ts),
            "blue_goal_rate": None if blue_goal_rate is None else float(blue_goal_rate),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))


class EnvBuilder:
    def __init__(
        self,
        iteration_timesteps: int,
        checkpoint_root: str = "data/checkpoints",
        n_proc: int = 1,
        initial_curriculum_state: dict[str, object] | None = None,
        current_checkpoint_dir: str = "",
        self_play_mode: str = "current",
        fixed_opponent_checkpoint: str = "",
        opponent_gap_ts: int = 4_000_000,
        opponent_device: str = "gpu",
    ):
        self.iteration_timesteps = iteration_timesteps
        self.checkpoint_root = checkpoint_root
        self.n_proc = max(1, int(n_proc))
        self.curriculum_manager = CurriculumManager()
        self.curriculum_manager.load_dict(initial_curriculum_state)
        self.curriculum_state_path = str(Path("data") / "curriculum_state.json")
        self.opponent_state_path = str(Path("data") / "opponent_state.json")
        self.current_checkpoint_dir = current_checkpoint_dir
        self.self_play_mode = str(self_play_mode)
        self.fixed_opponent_checkpoint = fixed_opponent_checkpoint
        self.opponent_gap_ts = int(opponent_gap_ts)
        self.opponent_device = opponent_device
        Path(self.curriculum_state_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.curriculum_state_path).write_text(
            json.dumps(self.curriculum_manager.to_dict(), indent=2, sort_keys=True)
        )

    def __call__(self, process_id: int | None = None):
        if process_id is None:
            process = mp.current_process()
            process_id = int(process._identity[0] - 1) if process._identity else 0

        curriculum_manager = self.curriculum_manager
        action_parser = RepeatAction(LookupTableAction(), repeats=ACTION_REPEAT)

        env = RLGym(
            state_mutator=DynamicMatchMutator(curriculum_manager),
            obs_builder=SharedObs(),
            action_parser=action_parser,
            reward_fn=CurriculumReward(curriculum_manager),
            termination_cond=CurriculumDoneCondition(curriculum_manager),
            truncation_cond=CurriculumTruncationCondition(curriculum_manager),
            transition_engine=RocketSimEngine(),
            **(
                {"shared_info_provider": ScoreboardProvider()}
                if ScoreboardProvider is not None
                else {}
            ),
        )

        if self.self_play_mode == "frozen":
            gym_env = SelfPlayOpponentGymWrapper(
                env,
                opponent_state_path=self.opponent_state_path,
                deterministic_opponent=False,
                device=self.opponent_device,
            )
        else:
            gym_env = RLGymV2GymWrapper(env)

        wrapped = ProcessIterationLogger(
            gym_env,
            process_id=process_id,
            iteration_timesteps=max(1, self.iteration_timesteps // self.n_proc),
            curriculum_manager=curriculum_manager,
            checkpoint_root=self.checkpoint_root,
            curriculum_state_path=self.curriculum_state_path,
            opponent_state_path=self.opponent_state_path,
            self_play_mode=self.self_play_mode,
            opponent_gap_ts=self.opponent_gap_ts,
            current_checkpoint_dir=self.current_checkpoint_dir,
            fixed_opponent_checkpoint=self.fixed_opponent_checkpoint,
        )
        return wrapped
