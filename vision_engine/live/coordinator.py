from __future__ import annotations

from dataclasses import replace
from threading import RLock

from .state import GameState
from vision_engine.solver.protocol import SolverResponse


class LiveCoordinator:
    """Single owner of the state version used by live UI/solver synchronization."""

    def __init__(self, initial_state: GameState):
        self._lock = RLock()
        self._state = initial_state

    @property
    def state(self) -> GameState:
        with self._lock:
            return self._state

    def update_board(self, *, accepted_mask: int, phase: str | None = None) -> GameState:
        with self._lock:
            if accepted_mask == self._state.accepted_mask and (phase is None or phase == self._state.phase):
                return self._state
            self._state = replace(
                self._state,
                accepted_mask=int(accepted_mask),
                state_version=self._state.state_version + 1,
                current_jewel=None,
                phase=self._state.phase if phase is None else phase,
            )
            return self._state

    def update_jewel(self, jewel: str | None) -> GameState:
        with self._lock:
            self._state = replace(self._state, current_jewel=jewel)
            return self._state

    def update_strategy(self, strategy: str) -> GameState:
        with self._lock:
            if strategy == self._state.strategy:
                return self._state
            self._state = replace(
                self._state,
                strategy=strategy,
                strategy_version=self._state.strategy_version + 1,
            )
            return self._state

    def update_plan(
        self, *, target: str, lucky_lines: tuple[str, ...] = (), normal_lines: tuple[str, ...] = ()
    ) -> GameState:
        """Persist the automatic strategy plan for the current episode.

        A plan change is strategy state, so bump ``strategy_version``. This makes
        every in-flight result from the previous route automatically stale.
        """
        with self._lock:
            lucky_lines = tuple(lucky_lines)
            normal_lines = tuple(normal_lines)
            if (
                target == self._state.plan_target
                and lucky_lines == self._state.plan_lucky_lines
                and normal_lines == self._state.plan_normal_lines
            ):
                return self._state
            self._state = replace(
                self._state,
                plan_target=str(target),
                plan_lucky_lines=lucky_lines,
                plan_normal_lines=normal_lines,
                strategy_version=self._state.strategy_version + 1,
            )
            return self._state

    def start_episode(self, *, episode_id: str, accepted_mask: int = 0, phase: str = "playing") -> GameState:
        with self._lock:
            self._state = replace(
                self._state,
                episode_id=str(episode_id),
                accepted_mask=int(accepted_mask),
                current_jewel=None,
                phase=phase,
                state_version=0,
                strategy_version=0,
                plan_target="",
                plan_lucky_lines=(),
                plan_normal_lines=(),
            )
            return self._state

    def accepts(self, response: SolverResponse) -> bool:
        with self._lock:
            return response.identity == self._state.identity and self._state.phase != "complete"
