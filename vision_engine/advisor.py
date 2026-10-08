from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, replace
from itertools import combinations
from typing import Callable

from .board import Board
from .constants import JEWELS, JEWEL_INDEX, LUCKY_LINES, MAX_DRAWS, NORMAL_LINES
from .scoring import LUCKY_MASKS, NORMAL_MASKS, score_mask
from .strategy import move_line_impacts
from .route_math import (
    ROUTE_NOMINAL_SCORE, ROUTE_TARGETS, LuckySlackEstimate, RouteEstimate,
    best_lucky_slack, line_feasibilities, multivariate_at_least_probability, route_portfolio, route_probability,
    route_by_name, specific_route_feasibility,
)


@dataclass(frozen=True)
class Utility:
    p3: float
    p2l1n: float
    p1l2n: float
    p1l1n: float
    p_gt1000: float
    p_ge999: float
    expected_score: float

    def __add__(self, other: "Utility") -> "Utility":
        return Utility(*(a + b for a, b in zip(self.as_tuple(), other.as_tuple())))

    def scale(self, x: float) -> "Utility":
        return Utility(*(a * x for a in self.as_tuple()))

    def as_tuple(self) -> tuple[float, float, float, float, float, float, float]:
        return (
            self.p3, self.p2l1n, self.p1l2n, self.p1l1n,
            self.p_gt1000, self.p_ge999, self.expected_score,
        )


@dataclass(frozen=True)
class Recommendation:
    position: int
    utility: Utility
    method: str
    learned_score: float = 0.0
    active_goal: str = ""
    samples: int = 0
    stderr_p3: float = 0.0
    stderr_gt1000: float = 0.0
    stderr_score: float = 0.0
    solver_regret: float = 0.0
    # V5.6.1: make uncertainty/ranking explicit instead of presenting every
    # Monte-Carlo ordering as a certain winner.
    tie_kind: str = ""          # "exact", "target", "statistical", or ""
    tied_with: tuple[int, ...] = ()
    confidence: str = "CLEAR"
    primary_margin: float = 0.0
    stderr_goal: float = 0.0
    # V5.6.2: policy-independent exact physical route feasibility. These are
    # intentionally separate from Utility probabilities, which describe outcomes
    # under the selected solver policy.
    routes: tuple[RouteEstimate, ...] = ()
    primary_stat_tie: bool = False
    tie_break_metric: str = ""
    tie_break_margin: float = 0.0
    # V5.6.3: structural two-placement budget for the best concrete trio of
    # Lucky lines. Used as a guardrail only after the primary score objective
    # is unresolved; it does not turn Score >1000 into a 3-Lucky strategy.
    lucky_slack: LuckySlackEstimate | None = None
    # V7.0.4: deterministic, policy-independent score-structure tie-breaks.
    # These never override a statistically clear P(>1000) winner; they are used
    # only after the primary Monte-Carlo criterion is unresolved.
    score_route_potential: float = 0.0
    normal_leverage: float = 0.0
    # V7.0.5: episode-level automatic strategy plan. All recommendations from a
    # decision carry the same locked plan; candidate-specific ``plan_probability``
    # says how healthy that exact route remains after the candidate move.
    plan_target: str = ""
    plan_lucky_lines: tuple[str, ...] = ()
    plan_normal_lines: tuple[str, ...] = ()
    plan_probability: float = 0.0
    # V18.0.1: probability of the *best route for the target* after this exact
    # current-jewel placement.  plan_probability remains the probability of the
    # concrete locked route.  Keeping both prevents a dead locked route from
    # being confused with a dead target.
    target_probability: float = 0.0
    plan_on_route: bool = False
    plan_reason: str = ""
    route_switched: bool = False
    forced_slack: bool = False
    lucky_leverage: float = 0.0
    # V18.0.4: immediate line completion is a deterministic tie-break after
    # target/route feasibility. It must never override a materially better plan,
    # but when the plan is equally healthy, banking a Lucky now beats merely
    # preparing another line; a Normal clear is the next-best immediate gain.
    lucky_clears_gained: int = 0
    normal_clears_gained: int = 0
    endgame_value: float = 0.0
    terminal_move: bool = False


def utility_learning_target(u: Utility) -> float:
    """Counterfactual scalar retained for exports/backwards compatibility."""
    return (
        5000.0 * u.p3
        + 2200.0 * u.p2l1n
        + 1400.0 * u.p1l2n
        + 700.0 * u.p1l1n
        + 1800.0 * u.p_gt1000
        + 700.0 * u.p_ge999
        + u.expected_score
    )


class Advisor:
    """Hybrid online advisor with policy-consistent Monte Carlo + exact endgame.

    V5.6.1 fixes three sources of misleading early/mid-game rankings:

    * all candidate actions are compared against the same future draw scenarios
      (common random numbers), so one action is not rewarded for simply receiving
      an easier random sample;
    * rollout placements follow the selected strategy instead of one generic
      greedy policy;
    * the rollout policy is line/time aware: it estimates whether missing jewels
      can still arrive in the moves remaining and preserves near-complete Lucky
      and Normal routes accordingly.

    The last ``exact_horizon`` moves still use exact expectimax.
    """

    def __init__(
        self,
        board: Board,
        mode: str = "adaptive",
        exact_horizon: int = 6,
        rollout_exact_horizon: int = 3,
        rollouts: int = 2400,
        bias_weights: tuple[float, ...] | None = None,
        rng_seed: int = 1337,
        learned_action_scores: Callable[[int, str, tuple[int, ...]], dict[int, float]] | None = None,
        p3_min: float = 0.08,
        p2l1n_min: float = 0.12,
        p1l2n_min: float = 0.16,
        p1l1n_min: float = 0.22,
        fallback_override_pp: float = 0.32,
        adaptive_score_min: float = 0.08,
        p3_near_best_tolerance: float = 0.025,
        ml_near_tie_probability: float = 0.012,
        ml_near_tie_score: float = 15.0,
        plan_target: str = "",
        plan_lucky_lines: tuple[str, ...] = (),
        plan_normal_lines: tuple[str, ...] = (),
        auto_p3_floor: float = 0.04,
        auto_fallback_ratio: float = 3.0,
        route_switch_ratio: float = 1.35,
    ):
        self.board = board
        self.mode = mode
        self.exact_horizon = int(exact_horizon)
        # Root decisions can afford deeper exact search. Early-game Monte Carlo
        # must not invoke that full horizon inside every rollout scenario or the
        # FAST pass becomes unusably expensive.
        self.rollout_exact_horizon = max(0, min(int(rollout_exact_horizon), self.exact_horizon))
        self.rollouts = int(rollouts)
        self.bias_weights = tuple(bias_weights or (1.0,) * 6)
        self.rng_seed = int(rng_seed)
        self._cache: dict[int, Utility] = {}
        # V18.0.12: policy-independent exact score cache used only for the
        # final two decisions when every >1000 finish is already impossible.
        # This prevents a stale structured fallback route from preferring a
        # lower-scoring Normal clear over an immediately banked Lucky line.
        self._score_exact_cache: dict[int, Utility] = {}
        self.learned_action_scores = learned_action_scores
        self.thresholds = {
            "3 Lucky": float(p3_min),
            "2 Lucky + 1 Normal": float(p2l1n_min),
            "1 Lucky + 2 Normal": float(p1l2n_min),
            "1 Lucky + 1 Normal": float(p1l1n_min),
        }
        self.fallback_override_pp = float(fallback_override_pp)
        self.adaptive_score_min = float(adaptive_score_min)
        self.p3_near_best_tolerance = float(p3_near_best_tolerance)
        self.ml_near_tie_probability = float(ml_near_tie_probability)
        self.ml_near_tie_score = float(ml_near_tie_score)
        self.plan_target = str(plan_target or "")
        self.plan_lucky_lines = tuple(plan_lucky_lines or ())
        self.plan_normal_lines = tuple(plan_normal_lines or ())
        self.auto_p3_floor = max(0.0, float(auto_p3_floor))
        self.auto_fallback_ratio = max(1.0, float(auto_fallback_ratio))
        self.route_switch_ratio = max(1.0, float(route_switch_ratio))

        # Caches used by the rollout planning policy. They are intentionally
        # local to an Advisor instance/state request.
        self._requirement_prob_cache: dict[tuple, float] = {}
        self._line_prob_cache: dict[tuple[int, int], float] = {}
        self._plan_cache: dict[int, dict[str, float]] = {}
        self._p3_route_cache: dict[int, float] = {}
        self._route_portfolio_cache: dict[int, tuple[RouteEstimate, ...]] = {}
        self._lucky_slack_cache: dict[int, LuckySlackEstimate] = {}
        self._normal_leverage_cache: dict[int, float] = {}
        self._lucky_leverage_cache: dict[int, float] = {}
        self._rollout_goal: str = ">1000" if self.mode == "score" else ("3 Lucky" if self.mode in {"lucky", "auto"} else "")
        self._active_plan_target: str = self.plan_target
        self._active_plan_lucky_lines: tuple[str, ...] = self.plan_lucky_lines
        self._active_plan_normal_lines: tuple[str, ...] = self.plan_normal_lines

        # Per-recommendation common scenarios + paired samples for confidence.
        self._active_common_key: tuple[int, str, int] | None = None
        self._active_common_sequences: list[tuple[str, ...]] = []
        self._last_mc_samples: dict[int, list[Utility]] = {}

    def _effective_route_switch_ratio(self, mask: int) -> float:
        """Return route-lock hysteresis appropriate for the remaining horizon.

        Early/mid game benefits from a sticky concrete route so tiny probability
        changes do not rotate H/V/\\// every draw.  In the final turns there is
        no time for a temporarily weaker route to recover, so hysteresis is
        deliberately relaxed while keeping the configured ratio as the upper
        bound.
        """
        placed = int(mask).bit_count()
        if placed >= 13:
            cap = 1.00
        elif placed >= 12:
            cap = 1.10
        elif placed >= 10:
            cap = 1.20
        else:
            cap = self.route_switch_ratio
        return max(1.0, min(self.route_switch_ratio, cap))

    def _terminal(self, mask: int) -> Utility:
        s = score_mask(mask)
        return Utility(
            1.0 if s.lucky >= 3 else 0.0,
            1.0 if s.lucky >= 2 and s.normal >= 1 else 0.0,
            1.0 if s.lucky >= 1 and s.normal >= 2 else 0.0,
            1.0 if s.lucky >= 1 and s.normal >= 1 else 0.0,
            1.0 if s.total > 1000 else 0.0,
            1.0 if s.total >= 999 else 0.0,
            float(s.total),
        )

    def _exact_score_state(self, mask: int) -> Utility:
        """Exact score-optimal continuation, independent of the locked route.

        The automatic planner normally follows a structured line target.  In the
        last two decisions that can become counterproductive after every >1000
        finish is mathematically dead: e.g. it may bank a Normal line merely to
        preserve ``1 Lucky + 1 Normal`` even though closing a Lucky immediately
        has a strictly higher final expected score.  This helper computes the
        authoritative score-salvage continuation without mutating the live plan.
        """
        cached = self._score_exact_cache.get(mask)
        if cached is not None:
            return cached
        if mask.bit_count() >= MAX_DRAWS:
            u = self._terminal(mask)
            self._score_exact_cache[mask] = u
            return u
        probs = self._draw_probabilities(mask)
        if not probs:
            u = self._terminal(mask)
            self._score_exact_cache[mask] = u
            return u
        total = Utility(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        for jewel, p in probs:
            positions = self.board.legal_positions(jewel, mask)
            if not positions:
                continue
            best: Utility | None = None
            best_key: tuple[float, ...] | None = None
            for pos in positions:
                u = self._exact_score_state(mask | (1 << pos))
                key = (
                    u.p_gt1000, u.p_ge999, u.expected_score,
                    u.p3, u.p2l1n, u.p1l2n, u.p1l1n,
                )
                if best is None or best_key is None or key > best_key:
                    best, best_key = u, key
            if best is not None:
                total = total + best.scale(p)
        self._score_exact_cache[mask] = total
        return total

    def _selected_counts(self, mask: int) -> tuple[int, ...]:
        return self.board.jewel_counts_in_mask(mask)

    def _draw_probabilities(self, mask: int) -> list[tuple[str, float]]:
        counts = self._selected_counts(mask)
        weights = []
        for j_idx, jewel in enumerate(JEWELS):
            remaining = 4 - counts[j_idx]
            if remaining <= 0:
                continue
            weights.append((jewel, remaining * self.bias_weights[j_idx]))
        z = sum(w for _, w in weights)
        if z <= 0:
            return []
        return [(j, w / z) for j, w in weights]

    def _goal_metric(self, u: Utility, goal: str) -> float:
        return {
            "3 Lucky": u.p3,
            "2 Lucky + 1 Normal": u.p2l1n,
            "1 Lucky + 2 Normal": u.p1l2n,
            "1 Lucky + 1 Normal": u.p1l1n,
            ">1000": u.p_gt1000,
            ">=999": u.p_ge999,
            "expected score": u.expected_score,
        }[goal]

    def _active_goal_single(self, u: Utility) -> tuple[str, float]:
        if self.mode == "score":
            return ">1000", u.p_gt1000
        if self.mode == "lucky":
            return "3 Lucky", u.p3
        for name, value in (
            ("3 Lucky", u.p3),
            ("2 Lucky + 1 Normal", u.p2l1n),
            ("1 Lucky + 2 Normal", u.p1l2n),
            ("1 Lucky + 1 Normal", u.p1l1n),
        ):
            if value >= self.thresholds[name]:
                return name, value
        if u.p_gt1000 > 0:
            return ">1000", u.p_gt1000
        if u.p_ge999 > 0:
            return ">=999", u.p_ge999
        return "expected score", u.expected_score

    def _p3_is_overridden(self, u: Utility) -> bool:
        return (
            u.p3 < max(0.15, self.thresholds["3 Lucky"] * 1.5)
            and (u.p_gt1000 - u.p3) >= self.fallback_override_pp
        )

    def _key(self, u: Utility) -> tuple[float, ...]:
        """Per-state ordering used inside exact expectimax."""
        if self.mode == "score":
            return (u.p_gt1000, u.p_ge999, u.expected_score, u.p3, u.p2l1n, u.p1l2n, u.p1l1n)
        if self.mode == "lucky":
            return (u.p3, u.p_gt1000, u.p2l1n, u.p1l2n, u.p1l1n, u.expected_score, u.p_ge999)

        p3 = u.p3 if u.p3 >= self.thresholds["3 Lucky"] and not self._p3_is_overridden(u) else 0.0
        pgt = u.p_gt1000 if u.p_gt1000 >= self.adaptive_score_min else 0.0
        p2 = u.p2l1n if u.p2l1n >= self.thresholds["2 Lucky + 1 Normal"] else 0.0
        p12 = u.p1l2n if u.p1l2n >= self.thresholds["1 Lucky + 2 Normal"] else 0.0
        p11 = u.p1l1n if u.p1l1n >= self.thresholds["1 Lucky + 1 Normal"] else 0.0
        return (p3, pgt, p2, p12, p11, u.expected_score, u.p_ge999)

    def _exact_state(self, mask: int) -> Utility:
        if mask in self._cache:
            return self._cache[mask]
        if mask.bit_count() >= MAX_DRAWS:
            u = self._terminal(mask)
            self._cache[mask] = u
            return u
        probs = self._draw_probabilities(mask)
        if not probs:
            return self._terminal(mask)
        total = Utility(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        for jewel, p in probs:
            positions = self.board.legal_positions(jewel, mask)
            if not positions:
                continue
            best: Utility | None = None
            best_key: tuple[float, ...] | None = None
            for pos in positions:
                child_mask = mask | (1 << pos)
                u = self._exact_state(child_mask)
                if self._rollout_goal in {name for name, _l, _n in ROUTE_TARGETS}:
                    k = (*self._policy_key_for_mask(child_mask), *self._key(u))
                else:
                    k = self._key(u)
                if best is None or best_key is None or k > best_key:
                    best, best_key = u, k
            if best is not None:
                total = total + best.scale(p)
        self._cache[mask] = total
        return total

    # ------------------------------------------------------------------
    # Line/time-aware rollout policy
    # ------------------------------------------------------------------
    def _requirement_probability(
        self, available: tuple[int, ...], need: tuple[int, ...], draws: int
    ) -> float:
        """Exact multivariate-hypergeometric route feasibility.

        V5.6.1 multiplied marginal tails, which is not exact for draws without
        replacement (e.g. needing BL×2 + HA×1). V5.6.2 enumerates the tiny
        six-jewel state space exactly. Empirical RNG weights remain a separate
        policy model and never change this physical-inventory feasibility value.
        """
        need = tuple(max(0, int(x)) for x in need)
        available = tuple(max(0, int(x)) for x in available)
        draws = max(0, int(draws))
        key = (available, need, draws)
        cached = self._requirement_prob_cache.get(key)
        if cached is not None:
            return cached
        score = multivariate_at_least_probability(available, need, draws)
        self._requirement_prob_cache[key] = score
        return score

    def _cells_completion_probability(self, mask: int, cells: tuple[int, ...]) -> float:
        target_mask = 0
        for i in cells:
            target_mask |= 1 << i
        cache_key = (mask, target_mask)
        cached = self._line_prob_cache.get(cache_key)
        if cached is not None:
            return cached
        missing = [i for i in cells if not ((mask >> i) & 1)]
        if not missing:
            self._line_prob_cache[cache_key] = 1.0
            return 1.0
        draws = MAX_DRAWS - mask.bit_count()
        if len(missing) > draws:
            self._line_prob_cache[cache_key] = 0.0
            return 0.0
        req = [0] * len(JEWELS)
        for i in missing:
            jewel = self.board.cells[i]
            if jewel is None:
                continue
            req[JEWEL_INDEX[jewel]] += 1
        selected = self._selected_counts(mask)
        available = tuple(4 - selected[i] for i in range(len(JEWELS)))
        p = self._requirement_probability(available, tuple(req), draws)
        self._line_prob_cache[cache_key] = p
        return p

    def _best_three_lucky_route_probability(self, mask: int) -> float:
        cached = self._p3_route_cache.get(mask)
        if cached is not None:
            return cached
        masks = list(LUCKY_MASKS.values())
        best = 0.0
        for combo in combinations(masks, 3):
            union = combo[0] | combo[1] | combo[2]
            cells = tuple(i for i in range(25) if (union >> i) & 1)
            best = max(best, self._cells_completion_probability(mask, cells))
        self._p3_route_cache[mask] = best
        return best

    def _plan_features(self, mask: int) -> dict[str, float]:
        cached = self._plan_cache.get(mask)
        if cached is not None:
            return cached
        s = score_mask(mask)
        remaining = MAX_DRAWS - mask.bit_count()
        lucky_probs = {
            name: self._cells_completion_probability(mask, cells)
            for name, cells in LUCKY_LINES.items()
        }
        normal_probs = {
            name: self._cells_completion_probability(mask, cells)
            for name, cells in NORMAL_LINES.items()
        }

        # Expected line value proxy. For each incomplete line, ask how much the
        # score would improve if the missing cells were eventually completed,
        # then weight that marginal value by its draw feasibility. This naturally
        # values Normal lines that intersect already-completed Lucky lines and
        # penalizes spending a needed jewel copy in an unrelated cell.
        score_proxy = float(s.total)
        for name, lm in LUCKY_MASKS.items():
            if (mask & lm) == lm:
                continue
            projected = score_mask(mask | lm).total
            score_proxy += lucky_probs[name] * max(0.0, float(projected - s.total))
        for name, nm in NORMAL_MASKS.items():
            if (mask & nm) == nm:
                continue
            projected = score_mask(mask | nm).total
            score_proxy += normal_probs[name] * max(0.0, float(projected - s.total))

        # A 3-Lucky route has combination value beyond three independent line
        # marginals (>1000 is normally born here), so preserve a modest synergy
        # bonus in score/adaptive rollout policies.
        p3_route = self._best_three_lucky_route_probability(mask)
        score_proxy += 180.0 * p3_route

        # Urgency boosts near-complete lines when there are few draws left. It is
        # small enough not to replace the probability model, but stops a generic
        # score proxy from throwing away a 4/5 Normal fallback one move too soon.
        urgency = 0.0
        for cells, p, line_value in [
            *[(LUCKY_LINES[n], lucky_probs[n], 132.0) for n in LUCKY_LINES],
            *[(NORMAL_LINES[n], normal_probs[n], 35.0) for n in NORMAL_LINES],
        ]:
            missing = sum(1 for i in cells if not ((mask >> i) & 1))
            if 0 < missing <= remaining:
                urgency += p * line_value / missing
        score_proxy += urgency

        out = {
            "p3_route": p3_route,
            "lucky_sum": sum(lucky_probs.values()),
            "normal_sum": sum(normal_probs.values()),
            "best_lucky": max(lucky_probs.values(), default=0.0),
            "best_normal": max(normal_probs.values(), default=0.0),
            "score_proxy": score_proxy,
            "remaining": float(remaining),
        }
        self._plan_cache[mask] = out
        return out

    def _routes_for_mask(self, mask: int) -> tuple[RouteEstimate, ...]:
        cached = self._route_portfolio_cache.get(mask)
        if cached is None:
            cached = route_portfolio(self.board, mask)
            self._route_portfolio_cache[mask] = cached
        return cached

    def _route_value_for_mask(self, mask: int, target: str) -> float:
        return route_probability(self._routes_for_mask(mask), target)

    def _slack_for_mask(self, mask: int) -> LuckySlackEstimate:
        cached = self._lucky_slack_cache.get(mask)
        if cached is None:
            cached = best_lucky_slack(self.board, mask)
            self._lucky_slack_cache[mask] = cached
        return cached

    @staticmethod
    def _score_route_potential(routes: tuple[RouteEstimate, ...]) -> float:
        """Policy-independent structural potential for outcomes above 1000.

        Route probabilities overlap, so this is deliberately *not* presented as
        another probability.  It is a deterministic tie-break score used only
        when P(>1000) is statistically unresolved.  Weighting by the nominal
        score excess rewards routes such as 2L+2N/3L+1N without pretending their
        overlapping probabilities can simply be added.
        """
        return float(sum(
            route.probability * max(0, ROUTE_NOMINAL_SCORE.get(route.target, 0) - 1000)
            for route in routes
            if ROUTE_NOMINAL_SCORE.get(route.target, 0) > 1000
        ))

    def _normal_leverage_for_mask(self, mask: int) -> float:
        """Exact Normal-line progress/feasibility tie-break.

        A line with existing selected cells deserves more weight, but only when
        the remaining inventory can actually supply its missing jewels.  The
        squared progress term makes a 2/5 line meaningfully stronger than two
        unrelated 1/5 lines while the exact hypergeometric feasibility prevents
        impossible/rare jewel requirements from being overvalued.
        """
        cached = self._normal_leverage_cache.get(mask)
        if cached is not None:
            return cached
        value = 0.0
        for line in line_feasibilities(self.board, mask):
            if line.kind != "Normal" or line.selected >= line.size:
                continue
            progress = line.selected / max(1, line.size)
            value += float(line.probability) * progress * progress
        self._normal_leverage_cache[mask] = float(value)
        return float(value)


    def _lucky_leverage_for_mask(self, mask: int) -> float:
        """Exact contingency value of incomplete Lucky lines.

        This is deliberately a *tie-break only*.  It never overrides target or
        locked-route feasibility.  When two moves preserve the main plan equally,
        it prefers the one that keeps a useful MU-crossing Lucky line alive over
        incidental Normal progress.
        """
        cached = self._lucky_leverage_cache.get(mask)
        if cached is not None:
            return cached
        value = 0.0
        for line in line_feasibilities(self.board, mask):
            if line.kind != "Lucky" or line.selected >= line.size:
                continue
            progress = line.selected / max(1, line.size)
            value += float(line.probability) * progress * progress
        self._lucky_leverage_cache[mask] = float(value)
        return float(value)

    @staticmethod
    def _route_identity(route: RouteEstimate | None) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if route is None:
            return (), ()
        return tuple(route.lucky_lines), tuple(route.normal_lines)

    def _auto_target_from_candidates(
        self, candidate_routes: dict[int, tuple[RouteEstimate, ...]]
    ) -> tuple[str, str, dict[str, float]]:
        """Choose the automatic target *after conditioning on the current jewel*.

        The old V18 planner selected the target from the pre-draw mask.  That
        allowed the UI to keep showing e.g. ``3 Lucky`` even when the jewel
        already visible on screen made every 3-Lucky continuation impossible.
        """
        maxima = {
            name: max((route_probability(routes, name) for routes in candidate_routes.values()), default=0.0)
            for name in self._AUTO_LADDER
        }
        current = self.plan_target if self.plan_target in self._AUTO_LADDER else ""

        # V18.0.13 primary-goal guard.  In Win >1000 / Adaptive, 3 Lucky is
        # the primary route and must not be abandoned merely because its exact
        # feasibility is lower than a 999-style fallback.  A mixed fallback can
        # be healthier numerically while still giving up the only direct 3-Lucky
        # win route.  Keep/recover 3 Lucky for as long as at least one concrete
        # 3-Lucky continuation survives the *already visible* current jewel.
        # Only descend the fallback ladder when 3 Lucky is physically impossible.
        p3 = maxima["3 Lucky"]
        if p3 > 1e-12:
            if current == "3 Lucky":
                reason = "keep 3-Lucky primary target; a concrete 3-Lucky route is still feasible"
            elif current:
                reason = "recover 3-Lucky primary target; previous fallback was premature while 3 Lucky remained feasible"
            else:
                reason = "start with 3-Lucky primary target; a concrete 3-Lucky route is feasible"
            return "3 Lucky", reason, maxima

        start_i = max(1, self._AUTO_LADDER.index(current) if current else 1)
        for i in range(start_i, len(self._AUTO_LADDER)):
            target = self._AUTO_LADDER[i]
            p = maxima[target]
            threshold = self.thresholds.get(target, 0.0)
            if p >= threshold or (p > 1e-12 and i == len(self._AUTO_LADDER) - 1):
                return target, "previous target weak/impossible after current jewel; automatic fallback", maxima
            if p > 1e-12 and i + 1 < len(self._AUTO_LADDER):
                next_p = maxima[self._AUTO_LADDER[i + 1]]
                if next_p < max(self.thresholds.get(self._AUTO_LADDER[i + 1], 0.0), p * self.auto_fallback_ratio):
                    return target, "keep current fallback after current-jewel check", maxima
        return "expected score", "all structured targets impossible after current jewel; best-score salvage", maxima

    def _select_locked_plan_for_current_jewel(
        self,
        mask: int,
        positions: tuple[int, ...],
        candidate_routes: dict[int, tuple[RouteEstimate, ...]],
    ) -> tuple[str, RouteEstimate | None, str, bool]:
        """Select target/route using every legal placement of the known jewel.

        Hysteresis is retained while the old route survives.  If the current
        jewel kills that concrete route, hysteresis is bypassed and another
        route for the *same target* is searched before the fallback ladder is
        allowed to descend.
        """
        target, reason, _maxima = self._auto_target_from_candidates(candidate_routes)
        if target == "expected score":
            return target, None, reason, False

        child_masks = {p: mask | (1 << p) for p in positions}
        identities: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
        for routes in candidate_routes.values():
            route = route_by_name(routes, target)
            if route is not None and route.probability > 1e-12:
                identities.add(self._route_identity(route))
        if (
            target == self.plan_target
            and (self.plan_lucky_lines or self.plan_normal_lines)
            and self._route_identity_matches_target(
                target, tuple(self.plan_lucky_lines), tuple(self.plan_normal_lines)
            )
        ):
            identities.add((tuple(self.plan_lucky_lines), tuple(self.plan_normal_lines)))

        scored: list[tuple[float, int, tuple[str, ...], tuple[str, ...]]] = []
        for lucky_lines, normal_lines in identities:
            best_p = 0.0
            missing = 999
            for child in child_masks.values():
                try:
                    est = specific_route_feasibility(self.board, child, target, lucky_lines, normal_lines)
                except ValueError:
                    continue
                if est.probability > best_p + 1e-15 or (abs(est.probability - best_p) <= 1e-15 and est.missing_cells < missing):
                    best_p = est.probability
                    missing = est.missing_cells
            scored.append((best_p, -missing, lucky_lines, normal_lines))
        scored.sort(reverse=True)
        if not scored or scored[0][0] <= 1e-12:
            # Defensive fallback.  In normal operation the target selector above
            # already prevents this, but never emit an impossible locked plan.
            return "expected score", None, "selected target has no current-jewel continuation; best-score salvage", False

        best_p, _neg_missing, best_lucky, best_normal = scored[0]
        best_route = RouteEstimate(target, best_p, lucky_lines=best_lucky, normal_lines=best_normal)

        old_identity = (tuple(self.plan_lucky_lines), tuple(self.plan_normal_lines))
        old_score = next((x for x in scored if (x[2], x[3]) == old_identity), None) if target == self.plan_target else None
        if old_score is not None and old_score[0] > 1e-12:
            switch_ratio = self._effective_route_switch_ratio(mask)
            if best_p <= old_score[0] * switch_ratio:
                old = RouteEstimate(target, old_score[0], lucky_lines=old_identity[0], normal_lines=old_identity[1])
                return target, old, reason + "; locked route retained after current-jewel check", False
            return target, best_route, reason + "; route switched after material feasibility gain", True

        switched = bool(target == self.plan_target and (self.plan_lucky_lines or self.plan_normal_lines))
        if switched:
            reason += "; previous locked route impossible with current jewel; emergency same-target route switch"
        elif self.plan_target and target != self.plan_target:
            reason += "; target changed after current-jewel feasibility check"
        return target, best_route, reason, switched

    @staticmethod
    def _route_target_counts(target: str) -> tuple[int, int]:
        for name, lucky, normal in ROUTE_TARGETS:
            if name == target:
                return lucky, normal
        return 0, 0

    @classmethod
    def _route_identity_matches_target(
        cls, target: str, lucky_lines: tuple[str, ...], normal_lines: tuple[str, ...]
    ) -> bool:
        """Reject shortened/legacy route identities for a structured target.

        V18.0.3 could persist a route after a just-completed line had been
        dropped from its identity (for example ``V+\\`` for a 3-Lucky target
        where H happened to be complete in that candidate). Such a shortened
        route is easier than the actual target and must never receive
        hysteresis protection in a later decision.
        """
        lucky_target, normal_target = cls._route_target_counts(target)
        return (
            len(tuple(dict.fromkeys(lucky_lines))) == lucky_target
            and len(tuple(dict.fromkeys(normal_lines))) == normal_target
        )

    def _adaptive_goal_from_routes(
        self, candidate_routes: list[tuple[RouteEstimate, ...]], remaining_after_move: int | None = None
    ) -> str:
        """Choose a concrete route using exact policy-independent feasibility.

        Prefer credible >1000-capable line combinations; once those routes are
        no longer credible, preserve the strongest mixed Lucky/Normal fallback.
        The route probabilities already include remaining moves and remaining
        jewel copies, so this cannot wait until 13/14 to notice a 4/5 Normal.
        """
        maxima = {name: max((route_probability(routes, name) for routes in candidate_routes), default=0.0)
                  for name, _l, _n in ROUTE_TARGETS}
        credible = max(0.02, min(0.12, self.adaptive_score_min))
        winning = [name for name, _l, _n in ROUTE_TARGETS if ROUTE_NOMINAL_SCORE[name] > 1000]

        # Early game retains too much route flexibility for "best fixed route"
        # feasibility to choose one concrete line set. Keep the broad >1000
        # objective *only while at least one >1000-capable route is physically
        # still possible*. If all such route feasibility is exactly zero, switch
        # immediately to a mixed-line fallback even with many moves remaining.
        if (
            remaining_after_move is not None
            and remaining_after_move >= 7
            and any(maxima[name] > 1e-12 for name in winning)
        ):
            return ">1000"

        viable_winning = [name for name in winning if maxima[name] >= credible]
        if viable_winning:
            return max(
                viable_winning,
                key=lambda name: (maxima[name] * ROUTE_NOMINAL_SCORE[name], maxima[name], ROUTE_NOMINAL_SCORE[name]),
            )
        fallbacks = ["2 Lucky + 1 Normal", "1 Lucky + 2 Normal", "1 Lucky + 1 Normal"]
        viable_fallbacks = [name for name in fallbacks if maxima.get(name, 0.0) > 1e-12]
        if viable_fallbacks:
            return max(
                viable_fallbacks,
                key=lambda name: (maxima[name] * ROUTE_NOMINAL_SCORE[name], maxima[name], ROUTE_NOMINAL_SCORE[name]),
            )
        return "expected score"

    _AUTO_LADDER = ("3 Lucky", "2 Lucky + 1 Normal", "1 Lucky + 2 Normal", "1 Lucky + 1 Normal")

    def _best_route_for_target(self, mask: int, target: str) -> RouteEstimate:
        route = route_by_name(self._routes_for_mask(mask), target)
        return route if route is not None else RouteEstimate(target, 0.0)

    def _locked_route_estimate(self, mask: int, target: str) -> RouteEstimate | None:
        if not target or target != self.plan_target:
            return None
        if not self.plan_lucky_lines and not self.plan_normal_lines:
            return None
        if not self._route_identity_matches_target(
            target, tuple(self.plan_lucky_lines), tuple(self.plan_normal_lines)
        ):
            return None
        try:
            return specific_route_feasibility(
                self.board, mask, target, self.plan_lucky_lines, self.plan_normal_lines
            )
        except ValueError:
            return None

    def _auto_target(self, mask: int) -> tuple[str, str]:
        """Choose the automatic target using a stable fallback ladder.

        3 Lucky remains the live win plan while it is healthy. It is abandoned
        before literal zero only when it is below ``auto_p3_floor`` *and* the
        next fallback is clearly stronger. Subsequent fallbacks use their old
        viability thresholds and only move downward; the planner never jumps
        back up during the same episode unless a new episode clears the plan.
        """
        probs = {name: self._best_route_for_target(mask, name).probability for name in self._AUTO_LADDER}
        current = self.plan_target if self.plan_target in self._AUTO_LADDER else ""

        # Keep the rollout/live policy aligned with the current-jewel selector:
        # 3 Lucky remains the primary goal until it is physically impossible.
        if probs["3 Lucky"] > 1e-12:
            if current == "3 Lucky":
                return "3 Lucky", "keep 3-Lucky primary route while feasible"
            if current:
                return "3 Lucky", "recover 3-Lucky primary route from premature fallback"
            return "3 Lucky", "start with feasible 3-Lucky primary route"

        start_i = max(1, self._AUTO_LADDER.index(current) if current else 1)
        for i in range(start_i, len(self._AUTO_LADDER)):
            target = self._AUTO_LADDER[i]
            p = probs[target]
            threshold = self.thresholds.get(target, 0.0)
            if p >= threshold or (p > 1e-12 and i == len(self._AUTO_LADDER) - 1):
                reason = "previous target weak/impossible; automatic fallback"
                return target, reason
            # If this target is still physically possible but below threshold,
            # keep it unless the next fallback is clearly healthier.
            if p > 1e-12 and i + 1 < len(self._AUTO_LADDER):
                next_p = probs[self._AUTO_LADDER[i + 1]]
                if next_p < max(self.thresholds.get(self._AUTO_LADDER[i + 1], 0.0), p * self.auto_fallback_ratio):
                    return target, "keep current fallback route"
        return "expected score", "all structured win/fallback routes exhausted"

    def _select_locked_plan(self, mask: int) -> tuple[str, RouteEstimate | None, str]:
        target, reason = self._auto_target(mask)
        if target == "expected score":
            return target, None, reason
        best = self._best_route_for_target(mask, target)
        locked = self._locked_route_estimate(mask, target)
        if locked is not None and locked.probability > 1e-12:
            # Hysteresis: keep the existing concrete line set unless another
            # route is materially stronger. This prevents H/V/\/ from rotating
            # every move because of tiny probability differences.
            switch_ratio = self._effective_route_switch_ratio(mask)
            if best.probability <= locked.probability * switch_ratio:
                return target, locked, reason + "; locked route retained"
        if best.probability > 0:
            if locked is not None and locked.probability > 0:
                reason += "; route switched after material feasibility gain"
            elif self.plan_target == target:
                reason += "; previous locked route became impossible"
            return target, best, reason
        return "expected score", None, "target route impossible; best-score salvage"

    @staticmethod
    def _position_in_plan(position: int, route: RouteEstimate | None) -> bool:
        if route is None:
            return False
        for name in route.lucky_lines:
            if position in LUCKY_LINES[name]:
                return True
        for name in route.normal_lines:
            if position in NORMAL_LINES[name]:
                return True
        return False

    def _policy_key_for_mask(self, mask: int) -> tuple[float, ...]:
        plan = self._plan_features(mask)
        s = score_mask(mask)
        slack = self._slack_for_mask(mask)
        slack_key = (
            float(slack.slack_remaining) if slack.viable else -1.0,
            float(slack.probability) if slack.viable else 0.0,
        )
        lucky_key = (
            plan["p3_route"], plan["lucky_sum"], plan["score_proxy"],
            plan["best_normal"], float(s.total),
        )
        score_key = (
            plan["score_proxy"], *slack_key, plan["best_normal"], plan["normal_sum"],
            plan["p3_route"], plan["best_lucky"], float(s.total),
        )
        goal = self._rollout_goal
        if self.mode == "auto" and self._active_plan_target in {name for name, _l, _n in ROUTE_TARGETS}:
            locked = specific_route_feasibility(
                self.board, mask, self._active_plan_target,
                self._active_plan_lucky_lines, self._active_plan_normal_lines,
            )
            return (
                locked.probability,
                -float(locked.missing_cells),
                plan["score_proxy"], plan["best_normal"], plan["p3_route"], float(s.total),
            )
        if goal in {name for name, _l, _n in ROUTE_TARGETS}:
            lucky_target, normal_target = self._route_target_counts(goal)
            # Keep rollout policy target-specific without running the much more
            # expensive full route-portfolio enumerator at every simulated node.
            # Exact route probabilities are still computed for the real candidate
            # actions and are the primary ranking metric.
            lucky_progress = min(s.lucky, lucky_target) / lucky_target if lucky_target else 1.0
            normal_progress = min(s.normal, normal_target) / normal_target if normal_target else 1.0
            return (
                lucky_progress + normal_progress,
                float(s.lucky), float(s.normal),
                plan["lucky_sum"] if lucky_target else 0.0,
                plan["normal_sum"] if normal_target else 0.0,
                plan["best_lucky"], plan["best_normal"], plan["score_proxy"], float(s.total),
            )
        if goal in {">1000", ">=999", "expected score"}:
            return score_key
        if self.mode == "lucky":
            return lucky_key
        if self.mode == "score":
            return score_key
        if plan["p3_route"] >= self.thresholds["3 Lucky"]:
            return lucky_key
        return (
            plan["best_normal"], plan["normal_sum"], plan["score_proxy"],
            plan["best_lucky"], float(s.total),
        )

    def _policy_position(self, mask: int, jewel: str) -> int:
        positions = self.board.legal_positions(jewel, mask)
        if not positions:
            raise ValueError(f"No legal positions left for {jewel}")
        if self.mode == "auto" and self._active_plan_target in {name for name, _l, _n in ROUTE_TARGETS}:
            route = RouteEstimate(
                self._active_plan_target, 0.0,
                lucky_lines=self._active_plan_lucky_lines,
                normal_lines=self._active_plan_normal_lines,
            )
            return max(
                positions,
                key=lambda p: (
                    specific_route_feasibility(
                        self.board, mask | (1 << p), self._active_plan_target,
                        self._active_plan_lucky_lines, self._active_plan_normal_lines,
                    ).probability,
                    1 if self._position_in_plan(p, route) else 0,
                    self._policy_key_for_mask(mask | (1 << p)),
                ),
            )
        return max(positions, key=lambda p: self._policy_key_for_mask(mask | (1 << p)))

    # Backwards-compatible name used by older tools/tests.
    def _greedy_position(self, mask: int, jewel: str) -> int:
        return self._policy_position(mask, jewel)

    # ------------------------------------------------------------------
    # Common-random-number Monte Carlo
    # ------------------------------------------------------------------
    def _scenario_rng(self, mask: int, current_jewel: str, scenario_index: int) -> random.Random:
        raw = f"{self.rng_seed}|{mask}|{current_jewel}|scenario|{scenario_index}".encode()
        seed = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")
        return random.Random(seed)

    def _future_sequence(self, mask: int, current_jewel: str, scenario_index: int) -> tuple[str, ...]:
        """One future jewel sequence shared by *all* candidate positions."""
        counts = list(self._selected_counts(mask))
        counts[JEWEL_INDEX[current_jewel]] += 1  # current action consumes one copy regardless of cell
        available = [4 - x for x in counts]
        future_moves = max(0, MAX_DRAWS - (mask.bit_count() + 1))
        rng = self._scenario_rng(mask, current_jewel, scenario_index)
        out: list[str] = []
        for _ in range(future_moves):
            weighted = [available[i] * self.bias_weights[i] for i in range(len(JEWELS))]
            z = sum(weighted)
            if z <= 0:
                break
            x = rng.random() * z
            acc = 0.0
            chosen = len(JEWELS) - 1
            for i, w in enumerate(weighted):
                acc += w
                if x <= acc:
                    chosen = i
                    break
            out.append(JEWELS[chosen])
            available[chosen] -= 1
        return tuple(out)

    def _prepare_common_sequences(self, mask: int, current_jewel: str, n: int) -> None:
        self._active_common_key = (mask, current_jewel, n)
        self._active_common_sequences = [
            self._future_sequence(mask, current_jewel, i) for i in range(n)
        ]

    def _rollout_after_action(self, mask: int, sequence: tuple[str, ...]) -> Utility:
        cur = mask
        seq_i = 0
        while cur.bit_count() < MAX_DRAWS:
            remaining_moves = MAX_DRAWS - cur.bit_count()
            if remaining_moves <= self.rollout_exact_horizon:
                return self._exact_state(cur)
            if seq_i >= len(sequence):
                return self._terminal(cur)
            jewel = sequence[seq_i]
            seq_i += 1
            positions = self.board.legal_positions(jewel, cur)
            if not positions:
                # This should not happen with a valid common sequence, but keep
                # the child solver safe if imported data is inconsistent.
                continue
            pos = self._policy_position(cur, jewel)
            cur |= 1 << pos
        return self._terminal(cur)

    @staticmethod
    def _stderr(values: list[float]) -> float:
        n = len(values)
        if n <= 1:
            return 0.0
        mean = sum(values) / n
        var = sum((x - mean) ** 2 for x in values) / (n - 1)
        return math.sqrt(max(0.0, var) / n)

    def _monte_carlo_action(self, mask: int, current_jewel: str, pos: int, n: int) -> tuple[Utility, float, float, float]:
        key = (mask, current_jewel, n)
        if self._active_common_key != key or len(self._active_common_sequences) != n:
            self._prepare_common_sequences(mask, current_jewel, n)
        vals = [
            self._rollout_after_action(mask | (1 << pos), sequence)
            for sequence in self._active_common_sequences
        ]
        self._last_mc_samples[pos] = vals
        total = Utility(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        for u in vals:
            total = total + u
        mean = total.scale(1.0 / max(1, n))
        return (
            mean,
            self._stderr([u.p3 for u in vals]),
            self._stderr([u.p_gt1000 for u in vals]),
            self._stderr([u.expected_score for u in vals]),
        )

    def _choose_global_goal(self, recs, remaining_after_move: int | None = None) -> str:
        # Backwards-compatible helper: older analysis/tests pass Utilities
        # directly. Live V5.6.2 recommendations pass Recommendation objects with
        # exact route portfolios attached.
        if recs and isinstance(recs[0], Utility):
            utilities = recs
            if self.mode == "score":
                max_gt = max(u.p_gt1000 for u in utilities)
                if max_gt > 1e-12:
                    return ">1000"
                if max(u.p_ge999 for u in utilities) > 1e-12:
                    return ">=999"
                return "expected score"
            if self.mode == "lucky":
                return "3 Lucky"
            max_p3 = max(u.p3 for u in utilities)
            if max_p3 >= self.thresholds["3 Lucky"]:
                return "3 Lucky"
            max_gt = max(u.p_gt1000 for u in utilities)
            if max_gt >= self.adaptive_score_min:
                return ">1000"
            for name, getter in (
                ("2 Lucky + 1 Normal", lambda u: u.p2l1n),
                ("1 Lucky + 2 Normal", lambda u: u.p1l2n),
                ("1 Lucky + 1 Normal", lambda u: u.p1l1n),
            ):
                if max(getter(u) for u in utilities) > 0:
                    return name
            return "expected score"

        if self.mode == "auto":
            # Live auto mode selects its plan before simulation in recommend().
            return self._rollout_goal or "expected score"
        if self.mode == "score":
            max_gt = max(r.utility.p_gt1000 for r in recs)
            if max_gt > 1e-12:
                return ">1000"
            max_ge = max(r.utility.p_ge999 for r in recs)
            if max_ge > 1e-12:
                return ">=999"
            return "expected score"
        if self.mode == "lucky":
            return "3 Lucky"
        return self._adaptive_goal_from_routes([r.routes for r in recs], remaining_after_move)

    def _sort_tuple_for_goal(self, r: Recommendation, goal: str) -> tuple[float, ...]:
        u = r.utility
        if goal in {name for name, _l, _n in ROUTE_TARGETS}:
            # V7.0.5 automatic live play follows one concrete locked route.
            # When feasibility is equal, an on-route cell always beats spending
            # a placement on a fourth/unrelated line.
            if self.mode == "auto" and r.plan_target == goal:
                # With the current jewel being the final placement, the game has
                # no future route to preserve. Rank by achieved target and exact
                # final score; equivalent finishes are true ties regardless of
                # which previously locked line name they happened to complete.
                if r.terminal_move:
                    return (
                        r.target_probability, r.endgame_value,
                        self._goal_metric(u, goal), u.p_gt1000, u.p_ge999, u.p3,
                    )
                slack = r.lucky_slack
                slack_remaining = float(slack.slack_remaining) if slack is not None and slack.viable else -1.0
                return (
                    r.plan_probability,
                    r.target_probability,
                    1.0 if r.plan_on_route else 0.0,
                    slack_remaining if goal == "3 Lucky" else 0.0,
                    r.endgame_value,
                    float(r.lucky_clears_gained),
                    float(r.normal_clears_gained),
                    r.lucky_leverage,
                    r.normal_leverage,
                    self._goal_metric(u, goal), u.p_gt1000, u.p_ge999, u.expected_score,
                )
            # Legacy/synthetic Recommendation objects may not carry route data.
            if not r.routes:
                if goal == "3 Lucky":
                    return (u.p3, u.p_gt1000, u.p2l1n, u.p1l2n, u.p1l1n, u.expected_score, u.p_ge999)
                return (self._goal_metric(u, goal), u.p_gt1000, u.p_ge999, u.expected_score)
            rp = route_probability(r.routes, goal)
            if goal == "3 Lucky":
                return (rp, u.p3, u.p_gt1000, u.p2l1n, u.p1l2n, u.p1l1n, u.expected_score)
            return (rp, u.p_gt1000, u.p_ge999, u.expected_score, u.p3)
        if goal == ">1000":
            slack = r.lucky_slack
            slack_remaining = float(slack.slack_remaining) if slack is not None and slack.viable else -1.0
            slack_feasibility = float(slack.probability) if slack is not None and slack.viable else 0.0
            return (
                u.p_gt1000, slack_remaining, r.score_route_potential, r.normal_leverage,
                slack_feasibility, u.p_ge999, u.expected_score, u.p3,
            )
        if goal == ">=999":
            return (
                u.p_ge999, u.expected_score, u.p_gt1000,
                float(r.lucky_clears_gained), float(r.normal_clears_gained),
                r.lucky_leverage, r.normal_leverage, u.p3,
            )
        if goal == "expected score":
            return (
                u.expected_score, u.p_gt1000, u.p_ge999,
                float(r.lucky_clears_gained), float(r.normal_clears_gained),
                r.lucky_leverage, r.normal_leverage, u.p3,
            )
        return (self._goal_metric(u, goal), u.p_gt1000, u.p_ge999, u.expected_score)

    def _sort_for_goal(self, recs: list[Recommendation], goal: str) -> list[Recommendation]:
        return sorted(recs, key=lambda r: self._sort_tuple_for_goal(r, goal), reverse=True)

    def _criteria_for_goal(self, r: Recommendation, goal: str) -> list[tuple[str, float, str | None]]:
        u = r.utility
        if goal in {name for name, _l, _n in ROUTE_TARGETS}:
            if self.mode == "auto" and r.plan_target == goal:
                if r.terminal_move:
                    return [
                        (f"target {goal}", r.target_probability, None),
                        ("final exact score", r.endgame_value, None),
                        (f"P({goal} | policy)", self._goal_metric(u, goal), None),
                        ("P(>1000)", u.p_gt1000, "p_gt1000"),
                        ("P(>=999)", u.p_ge999, "p_ge999"),
                        ("P(3 Lucky | policy)", u.p3, "p3"),
                    ]
                criteria = [
                    (f"locked route {goal}", r.plan_probability, None),
                    (f"target {goal}", r.target_probability, None),
                    ("on locked route", 1.0 if r.plan_on_route else 0.0, None),
                ]
                if goal == "3 Lucky":
                    slack = r.lucky_slack
                    criteria.append((
                        "3L slack remaining",
                        float(slack.slack_remaining) if slack is not None and slack.viable else -1.0,
                        None,
                    ))
                if r.endgame_value:
                    criteria.append(("endgame exact expected score", r.endgame_value, None))
                criteria.extend([
                    ("Immediate Lucky clears", float(r.lucky_clears_gained), None),
                    ("Immediate Normal clears", float(r.normal_clears_gained), None),
                    ("Lucky contingency leverage", r.lucky_leverage, None),
                    ("Normal-line leverage", r.normal_leverage, None),
                    (f"P({goal} | policy)", self._goal_metric(u, goal), None),
                    ("P(>1000)", u.p_gt1000, "p_gt1000"),
                    ("P(>=999)", u.p_ge999, "p_ge999"),
                    ("E[score]", u.expected_score, "expected_score"),
                ])
                return criteria
            return [
                (f"route {goal}", route_probability(r.routes, goal), None),
                ("P(>1000)", u.p_gt1000, "p_gt1000"),
                ("P(>=999)", u.p_ge999, "p_ge999"),
                ("E[score]", u.expected_score, "expected_score"),
                ("P(3 Lucky | policy)", u.p3, "p3"),
            ]
        if goal == ">1000":
            slack = r.lucky_slack
            slack_remaining = float(slack.slack_remaining) if slack is not None and slack.viable else -1.0
            slack_feasibility = float(slack.probability) if slack is not None and slack.viable else 0.0
            criteria = [("P(>1000)", u.p_gt1000, "p_gt1000")]
            # When the Monte-Carlo primary objective overlaps statistically,
            # preserve 3-Lucky flexibility first, then use exact structural
            # >1000 route quality and exact Normal-line leverage before falling
            # back to incidental policy outcomes such as >=999.  Synthetic/old
            # Recommendation objects without route data keep the legacy order.
            criteria.append(("3L slack remaining", slack_remaining, None))
            if r.routes:
                criteria.append((">1000 structural route potential", r.score_route_potential, None))
                criteria.append(("Normal-line leverage", r.normal_leverage, None))
            criteria.extend([
                ("3L structural feasibility", slack_feasibility, None),
                ("P(>=999)", u.p_ge999, "p_ge999"),
                ("E[score]", u.expected_score, "expected_score"),
                ("P(3 Lucky | policy)", u.p3, "p3"),
            ])
            return criteria
        if goal == ">=999":
            return [
                ("P(>=999)", u.p_ge999, "p_ge999"),
                ("E[score]", u.expected_score, "expected_score"),
                ("P(>1000)", u.p_gt1000, "p_gt1000"),
                ("Immediate Lucky clears", float(r.lucky_clears_gained), None),
                ("Immediate Normal clears", float(r.normal_clears_gained), None),
                ("Lucky contingency leverage", r.lucky_leverage, None),
                ("Normal-line leverage", r.normal_leverage, None),
            ]
        return [
            ("E[score]", u.expected_score, "expected_score"),
            ("P(>1000)", u.p_gt1000, "p_gt1000"),
            ("P(>=999)", u.p_ge999, "p_ge999"),
            ("Immediate Lucky clears", float(r.lucky_clears_gained), None),
            ("Immediate Normal clears", float(r.normal_clears_gained), None),
            ("Lucky contingency leverage", r.lucky_leverage, None),
            ("Normal-line leverage", r.normal_leverage, None),
        ]

    @staticmethod
    def _sample_attr(u: Utility, metric: str) -> float:
        return {
            "p3": u.p3, "p2l1n": u.p2l1n, "p1l2n": u.p1l2n,
            "p1l1n": u.p1l1n, "p_gt1000": u.p_gt1000,
            "p_ge999": u.p_ge999, "expected_score": u.expected_score,
        }[metric]

    def _criterion_compare(self, a: Recommendation, b: Recommendation, criterion: tuple[str, float, str | None], exact: bool) -> int:
        label, av, sample_metric = criterion
        # Find the matching b value by label/order to avoid coupling callers to
        # Utility field names.
        b_map = {x[0]: x for x in self._criteria_for_goal(b, a.active_goal or b.active_goal)}
        bv = b_map[label][1]
        tol = 0.01 if label == "E[score]" else 1e-12
        if sample_metric is None or exact:
            if abs(av - bv) <= tol:
                return 0
            return 1 if av > bv else -1
        aa = [self._sample_attr(u, sample_metric) for u in self._last_mc_samples.get(a.position, [])]
        bb = [self._sample_attr(u, sample_metric) for u in self._last_mc_samples.get(b.position, [])]
        if len(aa) != len(bb) or not aa:
            if abs(av - bv) <= tol:
                return 0
            return 1 if av > bv else -1
        diffs = [x - y for x, y in zip(aa, bb)]
        mean = sum(diffs) / len(diffs)
        se = self._stderr(diffs)
        if se <= 1e-15:
            if abs(mean) <= tol:
                return 0
            return 1 if mean > 0 else -1
        if abs(mean) <= 1.96 * se:
            return 0
        return 1 if mean > 0 else -1

    def _hierarchical_compare(self, a: Recommendation, b: Recommendation, goal: str, exact: bool) -> tuple[int, str, bool]:
        ca = self._criteria_for_goal(a, goal)
        primary_tied = False
        for i, criterion in enumerate(ca):
            # active_goal is used by _criterion_compare to retrieve b's same label.
            aa = replace(a, active_goal=goal)
            bb = replace(b, active_goal=goal)
            cmp = self._criterion_compare(aa, bb, criterion, exact)
            if i == 0 and cmp == 0:
                primary_tied = True
            if cmp != 0:
                return cmp, criterion[0], primary_tied
        return 0, "", primary_tied

    def _hierarchical_sort(self, recs: list[Recommendation], goal: str, exact: bool) -> list[Recommendation]:
        remaining = list(recs)
        ordered: list[Recommendation] = []
        while remaining:
            best = remaining[0]
            for candidate in remaining[1:]:
                cmp, _metric, _primary_tied = self._hierarchical_compare(candidate, best, goal, exact)
                if cmp > 0 or (cmp == 0 and self._sort_tuple_for_goal(candidate, goal) > self._sort_tuple_for_goal(best, goal)):
                    best = candidate
            ordered.append(best)
            remaining.remove(best)
        return ordered

    def _ml_tie_break(self, recs: list[Recommendation], goal: str) -> list[Recommendation]:
        if len(recs) <= 1 or self.learned_action_scores is None:
            return recs
        top = recs[0]
        top_metric = self._goal_metric(top.utility, goal)
        near, rest = [], []
        for r in recs:
            metric = self._goal_metric(r.utility, goal)
            prob_close = abs(metric - top_metric) <= (
                self.ml_near_tie_score if goal == "expected score" else self.ml_near_tie_probability
            )
            score_close = abs(r.utility.expected_score - top.utility.expected_score) <= self.ml_near_tie_score
            if prob_close and score_close:
                near.append(r)
            else:
                rest.append(r)
        near.sort(key=lambda r: (r.learned_score, self._goal_metric(r.utility, goal), r.utility.expected_score), reverse=True)
        return near + rest

    def _metric_samples(self, pos: int, goal: str) -> list[float]:
        return [self._goal_metric(u, goal) for u in self._last_mc_samples.get(pos, [])]

    def _annotate_ties(self, recs: list[Recommendation], goal: str, exact: bool) -> list[Recommendation]:
        if not recs:
            return recs
        top = recs[0]
        actual_ties: list[int] = []
        primary_tied_any = False
        tie_break_metric = ""
        tie_break_margin = 0.0
        top_criteria = self._criteria_for_goal(top, goal)
        for other in recs[1:]:
            cmp, metric, primary_tied = self._hierarchical_compare(top, other, goal, exact)
            if primary_tied:
                primary_tied_any = True
                if cmp != 0 and not tie_break_metric:
                    tie_break_metric = metric
                    a_map = {x[0]: x[1] for x in self._criteria_for_goal(top, goal)}
                    b_map = {x[0]: x[1] for x in self._criteria_for_goal(other, goal)}
                    tie_break_margin = a_map.get(metric, 0.0) - b_map.get(metric, 0.0)
            if cmp == 0:
                actual_ties.append(other.position)

        primary_margin = 0.0
        if len(recs) > 1 and top_criteria:
            other_criteria = {x[0]: x[1] for x in self._criteria_for_goal(recs[1], goal)}
            primary_margin = top_criteria[0][1] - other_criteria.get(top_criteria[0][0], 0.0)

        tie_set = {top.position, *actual_ties}
        kind = "exact" if exact and actual_ties else ("statistical" if actual_ties else "")
        out: list[Recommendation] = []
        for r in recs:
            if r.position in tie_set and actual_ties:
                others = tuple(p for p in tie_set if p != r.position)
                out.append(replace(
                    r, tie_kind=kind, tied_with=others, confidence="TIE",
                    primary_margin=primary_margin, primary_stat_tie=primary_tied_any,
                    tie_break_metric=tie_break_metric, tie_break_margin=tie_break_margin,
                ))
            else:
                top_confidence = (
                    "LOW_SAMPLE"
                    if r.position == top.position and not exact and 0 < r.samples < 80
                    else ("CLEAR" if r.position == top.position else "RANKED")
                )
                out.append(replace(
                    r, confidence=top_confidence,
                    primary_margin=primary_margin if r.position == top.position else r.primary_margin,
                    primary_stat_tie=primary_tied_any if r.position == top.position else False,
                    tie_break_metric=tie_break_metric if r.position == top.position else "",
                    tie_break_margin=tie_break_margin if r.position == top.position else 0.0,
                ))
        return out

    def recommend(self, mask: int, current_jewel: str) -> list[Recommendation]:
        positions = self.board.legal_positions(current_jewel, mask)
        if not positions:
            raise ValueError(f"No legal positions left for {current_jewel}")

        learned = (
            self.learned_action_scores(mask, current_jewel, positions)
            if self.learned_action_scores is not None
            else {p: 0.0 for p in positions}
        )
        candidate_routes = {p: self._routes_for_mask(mask | (1 << p)) for p in positions}
        candidate_slack = {p: self._slack_for_mask(mask | (1 << p)) for p in positions}
        active_plan: RouteEstimate | None = None
        plan_reason = ""
        route_switched = False
        if self.mode == "auto":
            self._rollout_goal, active_plan, plan_reason, route_switched = self._select_locked_plan_for_current_jewel(
                mask, positions, candidate_routes
            )
            self._active_plan_target = self._rollout_goal
            self._active_plan_lucky_lines = active_plan.lucky_lines if active_plan is not None else ()
            self._active_plan_normal_lines = active_plan.normal_lines if active_plan is not None else ()
        elif self.mode == "lucky":
            self._rollout_goal = "3 Lucky"
        elif self.mode == "score":
            self._rollout_goal = ">1000"
        else:
            self._rollout_goal = self._adaptive_goal_from_routes(
                list(candidate_routes.values()), MAX_DRAWS - (mask.bit_count() + 1)
            )

        future_after_move = MAX_DRAWS - (mask.bit_count() + 1)
        raw: list[tuple[int, Utility, str, int, float, float, float]] = []
        self._last_mc_samples = {}
        exact_now = future_after_move <= self.exact_horizon
        if exact_now:
            # Exact-state values depend only on board/strategy/config, not on
            # which root candidate asked for them. V7 keeps this transposition
            # table hot across sibling moves and later requests.
            for pos in positions:
                u = self._exact_state(mask | (1 << pos))
                raw.append((pos, u, "exact-expectimax", 0, 0.0, 0.0, 0.0))
        else:
            per_action = max(1, math.ceil(self.rollouts / max(1, len(positions))))
            self._prepare_common_sequences(mask, current_jewel, per_action)
            for pos in positions:
                u, se3, segt, ses = self._monte_carlo_action(mask, current_jewel, pos, per_action)
                raw.append((pos, u, f"monte-carlo:{per_action}", per_action, se3, segt, ses))

        # V18.0.12 endgame score salvage.  When only this move plus at most one
        # future jewel remain, first evaluate every candidate with a route-neutral
        # exact score policy.  If *all* >1000 finishes are dead, the structured
        # fallback ladder must no longer force a lower-scoring line composition.
        # Preserve >=999 when it is still reachable; otherwise maximize exact
        # expected final points.  This is intentionally limited to the last two
        # decisions so normal route lock/hysteresis remains unchanged earlier.
        endgame_salvage_goal = ""
        if self.mode == "auto" and future_after_move <= 1:
            score_exact = {
                pos: self._exact_score_state(mask | (1 << pos))
                for pos in positions
            }
            if max((u.p_gt1000 for u in score_exact.values()), default=0.0) <= 1e-12:
                if max((u.p_ge999 for u in score_exact.values()), default=0.0) > 1e-12:
                    endgame_salvage_goal = ">=999"
                else:
                    endgame_salvage_goal = "expected score"
                raw = [
                    (pos, score_exact[pos], "exact-score-salvage", 0, 0.0, 0.0, 0.0)
                    for pos, _u, _method, _samples, _se3, _segt, _ses in raw
                ]

        # Build recommendation objects before choosing the display/ranking goal so
        # Adaptive can use exact route portfolios rather than policy-contaminated
        # outcome probabilities.
        preliminary: list[Recommendation] = []
        for pos, u, method, samples, se3, segt, ses in raw:
            child_mask = mask | (1 << pos)
            child_plan = None
            plan_probability = 0.0
            target_probability = route_probability(candidate_routes[pos], self._rollout_goal)
            plan_on_route = False
            if self.mode == "auto" and active_plan is not None:
                child_plan = specific_route_feasibility(
                    self.board, child_mask, self._rollout_goal,
                    active_plan.lucky_lines, active_plan.normal_lines,
                )
                plan_probability = child_plan.probability
                plan_on_route = self._position_in_plan(pos, active_plan)
            salvage = bool(endgame_salvage_goal)
            preliminary.append(Recommendation(
                pos, u, method, learned.get(pos, 0.0), "",
                samples, se3, segt, ses, 0.0,
                routes=candidate_routes[pos], lucky_slack=candidate_slack[pos],
                score_route_potential=self._score_route_potential(candidate_routes[pos]),
                normal_leverage=self._normal_leverage_for_mask(child_mask),
                plan_target="" if salvage else (self._rollout_goal if self.mode == "auto" else ""),
                plan_lucky_lines=() if salvage else (active_plan.lucky_lines if active_plan is not None else ()),
                plan_normal_lines=() if salvage else (active_plan.normal_lines if active_plan is not None else ()),
                plan_probability=0.0 if salvage else plan_probability,
                target_probability=(u.p_ge999 if endgame_salvage_goal == ">=999" else 0.0) if salvage else target_probability,
                plan_on_route=False if salvage else plan_on_route,
                plan_reason=(
                    "exact endgame salvage: >1000 is impossible; preserve >=999, then maximize score"
                    if endgame_salvage_goal == ">=999"
                    else "exact endgame salvage: >1000 and >=999 are impossible; maximize final score"
                ) if salvage else plan_reason,
                route_switched=False if salvage else route_switched,
                lucky_leverage=self._lucky_leverage_for_mask(child_mask),
                lucky_clears_gained=sum(
                    1 for impact in move_line_impacts(mask, pos)
                    if impact.kind == "Lucky" and impact.completed and impact.before < impact.size
                ),
                normal_clears_gained=sum(
                    1 for impact in move_line_impacts(mask, pos)
                    if impact.kind == "Normal" and impact.completed and impact.before < impact.size
                ),
                endgame_value=u.expected_score if future_after_move <= 1 else 0.0,
                terminal_move=future_after_move == 0,
            ))
        if not endgame_salvage_goal and self.mode == "auto" and active_plan is not None and self._rollout_goal == "3 Lucky":
            forced_slack = not any(r.plan_on_route for r in preliminary)
            if forced_slack:
                preliminary = [replace(r, forced_slack=True) for r in preliminary]

        goal = (
            endgame_salvage_goal
            if endgame_salvage_goal
            else (self._rollout_goal if self.mode == "auto" else self._choose_global_goal(preliminary, future_after_move))
        )
        recs: list[Recommendation] = []
        for r in preliminary:
            goal_samples = self._metric_samples(r.position, goal) if r.samples and goal not in {name for name, _l, _n in ROUTE_TARGETS} else []
            se_goal = self._stderr(goal_samples) if goal_samples else 0.0
            recs.append(replace(r, active_goal=goal, stderr_goal=se_goal))

        recs = self._sort_for_goal(recs, goal)
        recs = self._hierarchical_sort(recs, goal, exact_now)
        recs = self._annotate_ties(recs, goal, exact_now)

        # ML may break only a true mathematical/statistical tie; it may never
        # override a hierarchy criterion that already separated the moves.
        if recs and recs[0].confidence == "TIE" and self.learned_action_scores is not None:
            tie_positions = {recs[0].position, *recs[0].tied_with}
            tie_group = [r for r in recs if r.position in tie_positions]
            rest = [r for r in recs if r.position not in tie_positions]
            tie_group.sort(key=lambda r: r.learned_score, reverse=True)
            recs = tie_group + rest

        if recs:
            best_target = utility_learning_target(recs[0].utility)
            recs = [
                replace(r, solver_regret=max(0.0, best_target - utility_learning_target(r.utility)))
                for r in recs
            ]
        return recs

