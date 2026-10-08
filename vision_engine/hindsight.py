from __future__ import annotations

from itertools import combinations
from math import comb

from .board import Board
from .constants import JEWELS
from .probability import all_count_states
from .scoring import score_mask


def best_scores_by_count(board: Board) -> dict[tuple[int, ...], int]:
    """Exact terminal upper bound for every 14-jewel count vector.

    Enumerates all C(24,14)=1,961,256 possible selected-cell sets. This is an
    offline research calculation, not the online policy.
    """
    jewel_idx = [JEWELS.index(j) if j is not None else -1 for j in board.cells]
    non_center = [i for i, j in enumerate(board.cells) if j is not None]
    best: dict[tuple[int, ...], int] = {}
    for selected in combinations(non_center, 14):
        mask = 0
        counts = [0] * 6
        for i in selected:
            mask |= 1 << i
            counts[jewel_idx[i]] += 1
        key = tuple(counts)
        value = score_mask(mask).total
        if value > best.get(key, -1):
            best[key] = value
    return best


def hindsight_metrics(board: Board) -> dict[str, float]:
    best = best_scores_by_count(board)
    p_gt1000 = 0.0
    p_ge999 = 0.0
    expected = 0.0
    for counts, p in all_count_states():
        s = best[counts]
        expected += p * s
        if s > 1000:
            p_gt1000 += p
        if s >= 999:
            p_ge999 += p
    return {"p_gt1000": p_gt1000, "p_ge999": p_ge999, "expected_best_score": expected}
