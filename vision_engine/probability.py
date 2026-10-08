from __future__ import annotations

from functools import lru_cache
from math import comb
from typing import Iterable

import numpy as np

from .board import Board
from .constants import COPIES_PER_JEWEL, JEWELS, LUCKY_LINES, MAX_DRAWS


def all_count_states(draws: int = MAX_DRAWS):
    """Yield (counts, exact hypergeometric probability) under 4 copies of each jewel."""
    total = comb(COPIES_PER_JEWEL * len(JEWELS), draws)
    for a in range(5):
        for b in range(5):
            for c in range(5):
                for d in range(5):
                    for e in range(5):
                        f = draws - a - b - c - d - e
                        if not 0 <= f <= 4:
                            continue
                        vals = (a, b, c, d, e, f)
                        w = 1
                        for k in vals:
                            w *= comb(4, k)
                        yield vals, w / total


def lucky_line_counts(board: Board) -> dict[str, tuple[int, ...]]:
    out: dict[str, tuple[int, ...]] = {}
    for name, cells in LUCKY_LINES.items():
        arr = [0] * len(JEWELS)
        for i in cells:
            arr[JEWELS.index(board.cells[i])] += 1
        out[name] = tuple(arr)
    return out


def triple_requirements(board: Board) -> list[tuple[tuple[str, ...], tuple[int, ...]]]:
    import itertools

    lines = lucky_line_counts(board)
    out = []
    for names in itertools.combinations(lines.keys(), 3):
        req = tuple(sum(lines[n][j] for n in names) for j in range(len(JEWELS)))
        out.append((names, req))
    return out


def p_three_lucky(board: Board) -> float:
    triples = triple_requirements(board)
    p = 0.0
    for counts, weight in all_count_states():
        if any(all(counts[j] >= req[j] for j in range(6)) for _, req in triples):
            p += weight
    return p


def p_at_least_two_lucky(board: Board) -> float:
    import itertools

    lines = lucky_line_counts(board)
    reqs = []
    for names in itertools.combinations(lines.keys(), 2):
        reqs.append(tuple(sum(lines[n][j] for n in names) for j in range(6)))
    p = 0.0
    for counts, weight in all_count_states():
        if any(all(counts[j] >= req[j] for j in range(6)) for req in reqs):
            p += weight
    return p
