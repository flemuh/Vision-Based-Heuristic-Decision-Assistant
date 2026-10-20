"""R3.9 experimental exact plan-feature kernel.

Mirrors Advisor._plan_features without changing policy or reward logic.
Only optional: the caller must verify parity before enabling for live play.
"""
from __future__ import annotations

from itertools import combinations
from functools import lru_cache

import numpy as np
from vision_engine.constants import JEWELS
from vision_engine.scoring import LUCKY_MASKS, NORMAL_MASKS

LUCKY = np.asarray(list(LUCKY_MASKS.values()), dtype=np.int64)
NORMAL = np.asarray(list(NORMAL_MASKS.values()), dtype=np.int64)
THREES = np.asarray([a | b | c for a, b, c in combinations(LUCKY, 3)], dtype=np.int64)
BINOMIAL = np.zeros((25, 25), dtype=np.int64)
BINOMIAL[0, 0] = 1
for n in range(1, 25):
    BINOMIAL[n, 0] = 1
    BINOMIAL[n, n] = 1
    for r in range(1, n):
        BINOMIAL[n, r] = BINOMIAL[n - 1, r - 1] + BINOMIAL[n - 1, r]


def board_bitmasks(board):
    return np.asarray([sum(1 << i for i, cell in enumerate(board.cells) if cell == j) for j in JEWELS], dtype=np.int64)


@lru_cache(maxsize=1)
def kernel():
    from numba import njit

    @njit(cache=True)
    def bitcount(bits):
        val = 0
        while bits:
            bits &= bits - 1
            val += 1
        return val

    @njit(cache=True)
    def score(mask, lucky_masks, normal_masks):
        lucky, normal, covered = 0, 0, 0
        for lm in lucky_masks:
            if mask & lm == lm:
                lucky += 1
                covered |= lm
        for nm in normal_masks:
            if mask & nm == nm:
                normal += 1
                covered |= nm
        return lucky * 312 + normal * 240 + bitcount(mask & ~covered) * 45

    @njit(cache=True)
    def probability(mask, route, board_bits, choose, draws):
        available = np.empty(6, dtype=np.int64)
        needs = np.empty(6, dtype=np.int64)
        total_available = 0
        needed = 0
        for i in range(6):
            selected = bitcount(mask & board_bits[i])
            available[i] = 4 - selected
            needs[i] = bitcount(route & ~mask & board_bits[i])
            total_available += available[i]
            needed += needs[i]
            if needs[i] > available[i]:
                return 0.0
        if needed == 0:
            return 1.0
        if draws <= 0 or needed > draws:
            return 0.0
        draws = min(draws, total_available)
        denom = choose[total_available, draws]
        if denom <= 0:
            return 0.0
        dp = np.zeros(15, dtype=np.int64)
        dp[0] = 1
        for i in range(6):
            next_dp = np.zeros(15, dtype=np.int64)
            for used in range(draws + 1):
                ways = dp[used]
                if ways == 0:
                    continue
                hi = min(available[i], draws - used)
                for x in range(needs[i], hi + 1):
                    next_dp[used + x] += ways * choose[available[i], x]
            dp = next_dp
        p = dp[draws] / denom
        return max(0.0, min(1.0, p))

    @njit(cache=True)
    def calculate(mask, board_bits, lucky_masks, normal_masks, threes, choose):
        remaining = 14 - bitcount(mask)
        current_score = score(mask, lucky_masks, normal_masks)
        lp = np.empty(len(lucky_masks), dtype=np.float64)
        np_ = np.empty(len(normal_masks), dtype=np.float64)
        lucky_sum = 0.0
        normal_sum = 0.0
        best_lucky = 0.0
        best_normal = 0.0
        score_proxy = float(current_score)
        urgency = 0.0
        for i in range(len(lucky_masks)):
            lm = lucky_masks[i]
            p = probability(mask, lm, board_bits, choose, remaining)
            lp[i] = p
            lucky_sum += p
            best_lucky = max(best_lucky, p)
            if mask & lm != lm:
                projected = score(mask | lm, lucky_masks, normal_masks)
                score_proxy += p * max(0.0, float(projected - current_score))
            missing = bitcount(lm & ~mask)
            if 0 < missing <= remaining:
                urgency += p * 132.0 / missing
        for i in range(len(normal_masks)):
            nm = normal_masks[i]
            p = probability(mask, nm, board_bits, choose, remaining)
            np_[i] = p
            normal_sum += p
            best_normal = max(best_normal, p)
            if mask & nm != nm:
                projected = score(mask | nm, lucky_masks, normal_masks)
                score_proxy += p * max(0.0, float(projected - current_score))
            missing = bitcount(nm & ~mask)
            if 0 < missing <= remaining:
                urgency += p * 35.0 / missing
        p3 = 0.0
        for union in threes:
            p3 = max(p3, probability(mask, union, board_bits, choose, remaining))
        score_proxy += 180.0 * p3
        score_proxy += urgency
        return np.asarray((p3, lucky_sum, normal_sum, best_lucky, best_normal, score_proxy, float(remaining)), dtype=np.float64)

    return calculate


NAMES = ('p3_route','lucky_sum','normal_sum','best_lucky','best_normal','score_proxy','remaining')


def compiled_features(mask: int, board_bits, compiled):
    vector = compiled(mask, board_bits, LUCKY, NORMAL, THREES, BINOMIAL)
    return dict(zip(NAMES, (float(x) for x in vector)))
