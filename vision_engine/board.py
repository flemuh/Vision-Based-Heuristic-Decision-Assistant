from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from .constants import CENTER, JEWELS, JEWEL_INDEX


@lru_cache(maxsize=256)
def _board_lookup(cells: tuple[str | None, ...]) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Compile immutable four-copies-per-jewel board into positions and bitmasks.

    Works across new Board objects for the same screen layout. The cache is
    bounded and shared only within a process; no mutable board state is stored.
    """
    positions = tuple(tuple(i for i, cell in enumerate(cells) if cell == j) for j in JEWELS)
    masks = tuple(sum(1 << i for i in found) for found in positions)
    return positions, masks


@dataclass(frozen=True)
class Board:
    """5x5 board. Index 12 is MU and represented as None."""

    cells: tuple[str | None, ...]

    def __post_init__(self) -> None:
        if len(self.cells) != 25:
            raise ValueError("Board must contain 25 cells")
        if self.cells[CENTER] is not None:
            raise ValueError("Center cell must be MU/None")
        counts = {j: 0 for j in JEWELS}
        for i, jewel in enumerate(self.cells):
            if i == CENTER:
                continue
            if jewel not in JEWELS:
                raise ValueError(f"Invalid jewel at {i}: {jewel}")
            counts[jewel] += 1
        if any(v != 4 for v in counts.values()):
            raise ValueError(f"A valid board must contain exactly 4 of each jewel: {counts}")

    @classmethod
    def from_rows(cls, rows: Iterable[Iterable[str | None]]) -> "Board":
        cells = tuple(x for row in rows for x in row)
        return cls(cells)

    def positions(self, jewel: str) -> tuple[int, ...]:
        return _board_lookup(self.cells)[0][JEWEL_INDEX[jewel]]

    def jewel_counts_in_mask(self, mask: int) -> tuple[int, ...]:
        return tuple((mask & m).bit_count() for m in _board_lookup(self.cells)[1])

    def legal_positions(self, jewel: str, mask: int) -> tuple[int, ...]:
        return tuple(i for i in self.positions(jewel) if not ((mask >> i) & 1))

    @staticmethod
    def rc(index: int) -> tuple[int, int]:
        return index // 5 + 1, index % 5 + 1

    @staticmethod
    def index(row: int, col: int) -> int:
        return (row - 1) * 5 + (col - 1)

    def pretty(self, mask: int = 0) -> str:
        rows = []
        for r in range(5):
            parts = []
            for c in range(5):
                i = r * 5 + c
                if i == CENTER:
                    parts.append(" MU ")
                else:
                    j = self.cells[i]
                    mark = "*" if (mask >> i) & 1 else " "
                    parts.append(f"{j}{mark}")
            rows.append(" ".join(parts))
        return "\n".join(rows)
