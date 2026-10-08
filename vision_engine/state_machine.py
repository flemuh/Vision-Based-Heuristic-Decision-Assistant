from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GamePhase(str, Enum):
    NO_BOARD = "no-board"
    READY = "ready"
    PLAYING = "playing"
    ONE_LEFT = "one-left"
    COMPLETE_PENDING = "complete-pending"
    COMPLETE = "complete"
    RESULT = "result"
    DESYNC = "desync"


@dataclass(frozen=True)
class PhaseState:
    phase: GamePhase
    reason: str


class GameStateMachine:
    """Small deterministic phase detector used for UI and data hygiene.

    It does not control the game. It only prevents the collector from learning
    across implausible phase transitions.
    """

    def __init__(self):
        self.phase = GamePhase.NO_BOARD

    def reset(self) -> None:
        self.phase = GamePhase.NO_BOARD

    def update(
        self,
        *,
        has_board: bool,
        selected_count: int,
        current_jewel: str | None,
        result_visible: bool = False,
        desync: bool = False,
        complete_pending: bool = False,
    ) -> PhaseState:
        if desync:
            self.phase = GamePhase.DESYNC
            return PhaseState(self.phase, "screen state failed monotonic/consensus checks")
        if result_visible:
            self.phase = GamePhase.RESULT
            return PhaseState(self.phase, "result detected")
        if complete_pending:
            self.phase = GamePhase.COMPLETE_PENDING
            return PhaseState(self.phase, "result screen detected; final board verification pending")
        if not has_board:
            self.phase = GamePhase.NO_BOARD
            return PhaseState(self.phase, "waiting for a stable board")
        if selected_count > 14:
            self.phase = GamePhase.DESYNC
            return PhaseState(self.phase, f"invalid selected count {selected_count}/14")
        if selected_count == 14:
            self.phase = GamePhase.COMPLETE
            return PhaseState(self.phase, "14 stones placed")
        if selected_count == 13:
            self.phase = GamePhase.ONE_LEFT
            return PhaseState(self.phase, "one stone remains")
        if selected_count == 0 and current_jewel is None:
            self.phase = GamePhase.READY
            return PhaseState(self.phase, "board ready; waiting for current jewel")
        self.phase = GamePhase.PLAYING
        return PhaseState(self.phase, "active round")
