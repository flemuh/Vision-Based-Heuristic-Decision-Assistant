from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, product
from math import comb

from .board import Board
from .constants import COPIES_PER_JEWEL, JEWELS, JEWEL_INDEX, LUCKY_LINES, MAX_DRAWS, NORMAL_LINES
from .scoring import score_mask


ROUTE_TARGETS: tuple[tuple[str, int, int], ...] = (
    ("3 Lucky + 1 Normal", 3, 1),
    ("3 Lucky", 3, 0),
    ("2 Lucky + 2 Normal", 2, 2),
    ("2 Lucky + 1 Normal", 2, 1),
    ("1 Lucky + 3 Normal", 1, 3),
    ("1 Lucky + 2 Normal", 1, 2),
    ("1 Lucky + 1 Normal", 1, 1),
)

# Nominal milestone values are only for Adaptive route selection / display. The
# authoritative final score still comes from score_mask because line overlap
# changes how many 45-point loose jewel cells remain.
ROUTE_NOMINAL_SCORE = {
    "3 Lucky + 1 Normal": 1176,
    "3 Lucky": 1026,
    "2 Lucky + 2 Normal": 1104,
    "2 Lucky + 1 Normal": 999,
    "1 Lucky + 3 Normal": 1032,
    "1 Lucky + 2 Normal": 954,
    "1 Lucky + 1 Normal": 762,
}


@dataclass(frozen=True)
class RouteEstimate:
    target: str
    probability: float
    lucky_lines: tuple[str, ...] = ()
    normal_lines: tuple[str, ...] = ()
    needs: tuple[tuple[str, int], ...] = ()
    missing_cells: int = 0

    @property
    def lines(self) -> tuple[str, ...]:
        return tuple([*(f"Lucky {x}" for x in self.lucky_lines), *(f"Normal {x}" for x in self.normal_lines)])


@dataclass(frozen=True)
class LineFeasibility:
    kind: str
    name: str
    probability: float
    needs: tuple[tuple[str, int], ...]
    selected: int
    size: int

    @property
    def missing(self) -> int:
        return max(0, self.size - self.selected)


@dataclass(frozen=True)
class LuckySlackEstimate:
    """How much of the two-placement 3-Lucky slack budget remains.

    Any concrete trio of Lucky lines uses 12 jewel cells. Because a round has
    14 jewels, that trio can tolerate at most two placements outside its union.
    This is a structural constraint, not a Monte-Carlo probability.
    """

    lucky_lines: tuple[str, ...]
    off_route_placements: int
    slack_used: int
    slack_remaining: int
    viable: bool
    probability: float
    needs: tuple[tuple[str, int], ...] = ()
    missing_cells: int = 0


def lucky_slack_portfolio(board: Board, mask: int) -> tuple[LuckySlackEstimate, ...]:
    """Return all four possible 3-Lucky trios with exact feasibility.

    The four Lucky lines are disjoint outside MU, so each trio occupies exactly
    12 jewel cells and leaves exactly two slack cells. A trio remains structurally
    viable while no more than two already-selected jewels lie outside it.
    """

    out: list[LuckySlackEstimate] = []
    for lucky_names in combinations(tuple(LUCKY_LINES), 3):
        cells: set[int] = set()
        for name in lucky_names:
            cells.update(LUCKY_LINES[name])
        route_mask = 0
        for pos in cells:
            route_mask |= 1 << pos
        off_route = (int(mask) & ~route_mask).bit_count()
        viable = off_route <= 2
        p, need = cells_feasibility(board, mask, tuple(sorted(cells))) if viable else (0.0, (0,) * len(JEWELS))
        out.append(LuckySlackEstimate(
            lucky_lines=tuple(lucky_names),
            off_route_placements=off_route,
            slack_used=min(2, off_route),
            slack_remaining=max(0, 2 - off_route),
            viable=viable,
            probability=float(p) if viable else 0.0,
            needs=format_needs(need) if viable else (),
            missing_cells=sum(need) if viable else 0,
        ))
    return tuple(out)


def best_lucky_slack(board: Board, mask: int) -> LuckySlackEstimate:
    """Best still-viable 3-Lucky trio for the current state.

    Preserve remaining slack first, then prefer the trio whose remaining jewel
    requirements are most feasible. This is used only as a guardrail when the
    primary score objective is unresolved; it never overrides a clear P(>1000)
    winner.
    """

    portfolio = lucky_slack_portfolio(board, mask)
    return max(
        portfolio,
        key=lambda x: (
            1 if x.viable else 0,
            x.slack_remaining if x.viable else -1,
            x.probability,
            -x.missing_cells,
            x.lucky_lines,
        ),
    )


@lru_cache(maxsize=32768)
def multivariate_at_least_probability(
    available: tuple[int, ...], need: tuple[int, ...], draws: int
) -> float:
    """Exact multivariate-hypergeometric P(X_i >= need_i for every jewel i).

    Uses a generating-function dynamic program: for each jewel type the
    coefficient of z^k is C(available_i, k), restricted to k >= need_i. The
    coefficient of z^draws after multiplying the six tiny polynomials is the
    exact favorable draw count. This is mathematically identical to exhaustive
    multivariate enumeration but runs in O(types * draws * copies).
    """
    available = tuple(max(0, int(x)) for x in available)
    need = tuple(max(0, int(x)) for x in need)
    draws = max(0, int(draws))
    if len(available) != len(JEWELS) or len(need) != len(JEWELS):
        raise ValueError("available/need must have one count per jewel type")
    total_available = sum(available)
    if draws > total_available:
        draws = total_available
    if not any(need):
        return 1.0
    if draws <= 0 or sum(need) > draws or any(n > a for n, a in zip(need, available)):
        return 0.0
    denominator = comb(total_available, draws)
    if denominator <= 0:
        return 0.0

    dp = [0] * (draws + 1)
    dp[0] = 1
    for have, req in zip(available, need):
        nxt = [0] * (draws + 1)
        for used, ways_so_far in enumerate(dp):
            if not ways_so_far:
                continue
            lo = req
            hi = min(have, draws - used)
            for take in range(lo, hi + 1):
                nxt[used + take] += ways_so_far * comb(have, take)
        dp = nxt
    return min(1.0, max(0.0, dp[draws] / denominator))


def _selected_counts(board: Board, mask: int) -> tuple[int, ...]:
    return board.jewel_counts_in_mask(mask)


def remaining_inventory(board: Board, mask: int) -> tuple[int, ...]:
    selected = _selected_counts(board, mask)
    return tuple(max(0, COPIES_PER_JEWEL - selected[i]) for i in range(len(JEWELS)))


def requirements_for_cells(board: Board, mask: int, cells: tuple[int, ...]) -> tuple[int, ...]:
    req = [0] * len(JEWELS)
    for pos in cells:
        if (mask >> pos) & 1:
            continue
        jewel = board.cells[pos]
        if jewel is not None:
            req[JEWEL_INDEX[jewel]] += 1
    return tuple(req)


def format_needs(need: tuple[int, ...]) -> tuple[tuple[str, int], ...]:
    return tuple((JEWELS[i], int(n)) for i, n in enumerate(need) if n > 0)


def cells_feasibility(board: Board, mask: int, cells: tuple[int, ...]) -> tuple[float, tuple[int, ...]]:
    need = requirements_for_cells(board, mask, cells)
    remaining_draws = max(0, MAX_DRAWS - mask.bit_count())
    available = remaining_inventory(board, mask)
    return multivariate_at_least_probability(available, need, remaining_draws), need


def line_feasibilities(board: Board, mask: int) -> tuple[LineFeasibility, ...]:
    out: list[LineFeasibility] = []
    for kind, mapping in (("Lucky", LUCKY_LINES), ("Normal", NORMAL_LINES)):
        for name, cells in mapping.items():
            p, need = cells_feasibility(board, mask, cells)
            selected = sum(1 for pos in cells if (mask >> pos) & 1)
            out.append(LineFeasibility(kind, name, p, format_needs(need), selected, len(cells)))
    return tuple(out)


def _canonical_line_choices(completed: set[str], all_names: tuple[str, ...], target_count: int):
    """Yield full, target-sized line identities for an at-least line target.

    Older planner versions represented a route only by the lines that were still
    missing.  Once one candidate completed a line, e.g. H in a 3-Lucky target,
    that candidate's route identity collapsed from ``H+V+\\`` to ``V+\\``.
    Reusing that shortened identity to compare another candidate incorrectly
    assumed H was already complete there too.

    Route identities are now canonical: they always carry exactly the number of
    Lucky/Normal lines required by the target, including lines that are already
    complete in the state being evaluated.  Completed lines contribute zero
    requirements, so feasibility math is unchanged; only route identity becomes
    safe to compare across sibling candidate moves.
    """
    target_count = max(0, int(target_count))
    if target_count == 0:
        yield ()
        return

    completed_ordered = tuple(name for name in all_names if name in completed)
    if len(completed_ordered) >= target_count:
        # The target is already met for this line type.  Any target-sized subset
        # of completed lines is a valid canonical identity.
        yield from combinations(completed_ordered, target_count)
        return

    needed = target_count - len(completed_ordered)
    open_names = tuple(name for name in all_names if name not in completed)
    if needed > len(open_names):
        return
    for extra in combinations(open_names, needed):
        yield tuple((*completed_ordered, *extra))


def _target_candidates(mask: int, lucky_target: int, normal_target: int):
    s = score_mask(mask)
    completed_lucky = set(s.lucky_names)
    completed_normal = set(s.normal_names)

    lucky_choices = tuple(_canonical_line_choices(
        completed_lucky, tuple(LUCKY_LINES), lucky_target
    ))
    normal_choices = tuple(_canonical_line_choices(
        completed_normal, tuple(NORMAL_LINES), normal_target
    ))
    if not lucky_choices or not normal_choices:
        return
    for lucky_names in lucky_choices:
        for normal_names in normal_choices:
            yield tuple(lucky_names), tuple(normal_names)


def best_route_feasibility(
    board: Board, mask: int, target: str, lucky_target: int, normal_target: int
) -> RouteEstimate:
    """Best fixed route feasibility for a line-count target from this exact state.

    This is deliberately policy-independent: it answers whether the remaining
    physical jewel inventory can supply the cells needed by the best concrete
    route. It is therefore not the same quantity as P(target | current policy).
    """
    best: RouteEstimate | None = None
    for lucky_names, normal_names in _target_candidates(mask, lucky_target, normal_target) or ():
        cell_set: set[int] = set()
        for name in lucky_names:
            cell_set.update(LUCKY_LINES[name])
        for name in normal_names:
            cell_set.update(NORMAL_LINES[name])
        cells = tuple(sorted(cell_set))
        p, need = cells_feasibility(board, mask, cells)
        est = RouteEstimate(
            target=target,
            probability=p,
            lucky_lines=lucky_names,
            normal_lines=normal_names,
            needs=format_needs(need),
            missing_cells=sum(need),
        )
        if best is None:
            best = est
            continue
        # Prefer the more feasible route, then one requiring fewer future cells,
        # then deterministic lexicographic line names for reproducibility.
        if (est.probability, -est.missing_cells, est.lines) > (best.probability, -best.missing_cells, best.lines):
            best = est

    if best is None:
        return RouteEstimate(target, 0.0)
    return best



def specific_route_feasibility(
    board: Board, mask: int, target: str, lucky_lines: tuple[str, ...], normal_lines: tuple[str, ...]
) -> RouteEstimate:
    """Exact feasibility for one already-selected concrete route.

    Unlike :func:`best_route_feasibility`, this never changes line identities.
    It is used by the live strategy planner to keep a route locked across moves
    and only switch when the locked route becomes impossible or materially worse.
    """
    cell_set: set[int] = set()
    for name in lucky_lines:
        if name not in LUCKY_LINES:
            raise ValueError(f"unknown Lucky line: {name}")
        cell_set.update(LUCKY_LINES[name])
    for name in normal_lines:
        if name not in NORMAL_LINES:
            raise ValueError(f"unknown Normal line: {name}")
        cell_set.update(NORMAL_LINES[name])
    cells = tuple(sorted(cell_set))
    p, need = cells_feasibility(board, mask, cells) if cells else (0.0, (0,) * len(JEWELS))
    return RouteEstimate(
        target=target,
        probability=float(p),
        lucky_lines=tuple(lucky_lines),
        normal_lines=tuple(normal_lines),
        needs=format_needs(need),
        missing_cells=sum(need),
    )

def route_portfolio(board: Board, mask: int) -> tuple[RouteEstimate, ...]:
    return tuple(
        best_route_feasibility(board, mask, name, lucky, normal)
        for name, lucky, normal in ROUTE_TARGETS
    )


def route_probability(routes: tuple[RouteEstimate, ...], target: str) -> float:
    for route in routes:
        if route.target == target:
            return route.probability
    return 0.0


def route_by_name(routes: tuple[RouteEstimate, ...], target: str) -> RouteEstimate | None:
    for route in routes:
        if route.target == target:
            return route
    return None
