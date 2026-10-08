from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Protocol


class ScreenRegionLike(Protocol):
    left: int
    top: int


@dataclass(frozen=True, slots=True)
class AutoPlayPoint:
    x: int
    y: int


def _cell_geometry(
    region: ScreenRegionLike,
    board_roi: tuple[int, int, int, int],
    position: int,
) -> tuple[float, float, float, float]:
    if not 0 <= int(position) < 25:
        raise ValueError(f"Invalid board position: {position}")
    bx, by, bw, bh = board_roi
    row, col = divmod(int(position), 5)
    cell_w = bw / 5.0
    cell_h = bh / 5.0
    cx = region.left + bx + (col + 0.5) * cell_w
    cy = region.top + by + (row + 0.5) * cell_h
    return cx, cy, cell_w, cell_h


def cell_center(
    region: ScreenRegionLike,
    board_roi: tuple[int, int, int, int],
    position: int,
) -> AutoPlayPoint:
    cx, cy, _cw, _ch = _cell_geometry(region, board_roi, position)
    return AutoPlayPoint(x=int(round(cx)), y=int(round(cy)))


def safe_cell_point(
    region: ScreenRegionLike,
    board_roi: tuple[int, int, int, int],
    position: int,
    *,
    jitter_fraction: float = 0.18,
    rng: random.Random | None = None,
) -> AutoPlayPoint:
    cx, cy, cell_w, cell_h = _cell_geometry(region, board_roi, position)
    r = rng or random.SystemRandom()
    frac = max(0.0, min(float(jitter_fraction), 0.30))
    dx = r.uniform(-frac * cell_w, frac * cell_w)
    dy = r.uniform(-frac * cell_h, frac * cell_h)
    return AutoPlayPoint(x=int(round(cx + dx)), y=int(round(cy + dy)))
