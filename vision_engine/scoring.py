from __future__ import annotations

from dataclasses import dataclass

from .constants import JEWEL_SCORE, LUCKY_LINES, LUCKY_SCORE, NORMAL_LINES, NORMAL_SCORE


@dataclass(frozen=True)
class ScoreBreakdown:
    lucky: int
    normal: int
    jewel_cells: int
    total: int
    lucky_names: tuple[str, ...]
    normal_names: tuple[str, ...]


def line_mask(cells: tuple[int, ...]) -> int:
    m = 0
    for i in cells:
        m |= 1 << i
    return m


LUCKY_MASKS = {name: line_mask(cells) for name, cells in LUCKY_LINES.items()}
NORMAL_MASKS = {name: line_mask(cells) for name, cells in NORMAL_LINES.items()}


def score_mask(mask: int) -> ScoreBreakdown:
    lucky_names = tuple(name for name, lm in LUCKY_MASKS.items() if (mask & lm) == lm)
    normal_names = tuple(name for name, lm in NORMAL_MASKS.items() if (mask & lm) == lm)

    cleared = 0
    for name in lucky_names:
        cleared |= LUCKY_MASKS[name]
    for name in normal_names:
        cleared |= NORMAL_MASKS[name]

    selected = mask.bit_count()
    jewel_cells = (mask & ~cleared).bit_count()
    total = len(lucky_names) * LUCKY_SCORE + len(normal_names) * NORMAL_SCORE + jewel_cells * JEWEL_SCORE
    return ScoreBreakdown(
        lucky=len(lucky_names),
        normal=len(normal_names),
        jewel_cells=jewel_cells,
        total=total,
        lucky_names=lucky_names,
        normal_names=normal_names,
    )
