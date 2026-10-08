from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable


@dataclass(frozen=True, slots=True)
class GameState:
    """Authoritative immutable live-play state passed to the solver.

    ``episode_id`` identifies a round. ``state_version`` changes whenever the
    accepted board mask changes. ``strategy_version`` changes whenever the user
    changes strategy. Solver results are publishable only when all three match.
    """

    episode_id: str
    state_version: int
    strategy_version: int
    board_hash: str
    board_cells: tuple[str, ...]
    accepted_mask: int
    current_jewel: str | None
    strategy: str
    phase: str = "playing"
    plan_target: str = ""
    plan_lucky_lines: tuple[str, ...] = ()
    plan_normal_lines: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.board_cells) != 25:
            raise ValueError("board_cells must contain exactly 25 entries")
        if self.state_version < 0 or self.strategy_version < 0:
            raise ValueError("versions must be non-negative")
        if self.accepted_mask < 0:
            raise ValueError("accepted_mask must be non-negative")

    @property
    def placed_count(self) -> int:
        return int(self.accepted_mask).bit_count()

    @property
    def identity(self) -> tuple[str, int, int]:
        return self.episode_id, self.state_version, self.strategy_version

    def with_jewel(self, jewel: str | None) -> "GameState":
        return replace(self, current_jewel=jewel)


@dataclass(frozen=True, slots=True)
class StateVersions:
    episode_id: str
    state_version: int
    strategy_version: int

    @classmethod
    def from_state(cls, state: GameState) -> "StateVersions":
        return cls(*state.identity)
