"""Geometry-proven routes that guarantee at least 999 final points.

The goal is *not* a Monte Carlo win probability. A route contains the exact
cells that must be selected; its completion probability is the probability of
seeing at least the required jewel inventory in the remaining draws assuming
unbiased physical draws. Alternative routes overlap and must never have their
individual probabilities added together.

Only static geometry is enumerated here. This is intentionally inexpensive
and never searches the 1,961,256 completed fourteen-cell boards during play.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

from .board import Board
from .constants import LUCKY_LINES, LUCKY_SCORE, MAX_DRAWS, NORMAL_LINES, NORMAL_SCORE, JEWEL_SCORE
from .route_math import RouteEstimate, cells_feasibility, format_needs
from .scoring import line_mask


@dataclass(frozen=True)
class ScoreRoute:
    lucky_lines: tuple[str, ...]
    normal_lines: tuple[str, ...]
    required_mask: int
    required_cells: tuple[int, ...]
    lower_bound_score: int
    slack: int


@lru_cache(maxsize=1)
def winning_route_catalog() -> tuple[ScoreRoute, ...]:
    """All minimal line groups that force >=999 on fourteen occupied cells.

    More completed lines cannot reduce score: the line award is greater than
    the largest possible loss of loose-cell points (312>4*45, 240>5*45).
    Consequently a group with n required cells and guaranteed line awards
    has a valid *lower bound* of awards + (14-n)*45.
    """
    lucky = tuple(LUCKY_LINES)
    normal = tuple(NORMAL_LINES)
    seen: dict[int, ScoreRoute] = {}
    for lc in range(5):
        for nc in range(9):
            # No need to enumerate impossible 4-Lucky / 8-Normal groups.
            if (lc, nc) not in {(3, 0), (2, 1), (2, 2), (1, 3), (3, 1)}:
                continue
            for ls in combinations(lucky, lc):
                lucky_mask = 0
                for name in ls:
                    lucky_mask |= line_mask(LUCKY_LINES[name])
                for ns in combinations(normal, nc):
                    required = lucky_mask
                    for name in ns:
                        required |= line_mask(NORMAL_LINES[name])
                    required_count = required.bit_count()
                    if required_count > MAX_DRAWS:
                        continue
                    guaranteed = lc * LUCKY_SCORE + nc * NORMAL_SCORE + (MAX_DRAWS - required_count) * JEWEL_SCORE
                    if guaranteed < 999:
                        continue
                    route = ScoreRoute(ls, ns, required, tuple(i for i in range(25) if (required >> i) & 1), guaranteed, MAX_DRAWS - required_count)
                    old = seen.get(required)
                    if old is None or (guaranteed, -len(ls) - len(ns)) > (old.lower_bound_score, -len(old.lucky_lines) - len(old.normal_lines)):
                        seen[required] = route
    # Keep the higher-scoring superset routes, too: 2 Lucky + 2 Normal can
    # genuinely exceed 1000 after 3 Lucky has become impossible.
    return tuple(sorted(seen.values(), key=lambda r: (-r.lower_bound_score, r.required_mask.bit_count(), r.lucky_lines, r.normal_lines)))


def is_score_route(lucky_lines: tuple[str, ...], normal_lines: tuple[str, ...]) -> bool:
    return any(r.lucky_lines == lucky_lines and r.normal_lines == normal_lines for r in winning_route_catalog())


def best_score_route(board: Board, mask: int) -> RouteEstimate:
    """Highest-probability fixed witness of >=999, not policy win chance."""
    best: RouteEstimate | None = None
    for route in winning_route_catalog():
        outside = (mask & ~route.required_mask).bit_count()
        if outside > route.slack:
            continue
        # Fast structural checks before the more costly exact hypergeometric DP.
        p, need = cells_feasibility(board, mask, route.required_cells)
        if p <= 0:
            continue
        est = RouteEstimate(
            target=">=999", probability=p,
            lucky_lines=route.lucky_lines, normal_lines=route.normal_lines,
            needs=format_needs(need), missing_cells=sum(need),
        )
        if best is None or (est.probability, -est.missing_cells, route.lower_bound_score, est.lines) > (
            best.probability, -best.missing_cells,
            guaranteed_score_floor(best.lucky_lines, best.normal_lines), best.lines,
        ):
            best = est
    return best if best is not None else RouteEstimate(">=999", 0.0)


def guaranteed_score_floor(lucky_lines: tuple[str, ...], normal_lines: tuple[str, ...]) -> int:
    cells = 0
    for name in lucky_lines:
        cells |= line_mask(LUCKY_LINES[name])
    for name in normal_lines:
        cells |= line_mask(NORMAL_LINES[name])
    return (
        len(lucky_lines) * LUCKY_SCORE + len(normal_lines) * NORMAL_SCORE
        + (MAX_DRAWS - cells.bit_count()) * JEWEL_SCORE
    )
