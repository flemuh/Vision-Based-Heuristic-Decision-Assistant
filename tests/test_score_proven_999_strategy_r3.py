"""Golden checks for the score-proven >=999 fallback (strategy R3)."""
from itertools import combinations

from vision_engine.advisor import Advisor
from vision_engine.board import Board
from vision_engine.constants import LUCKY_LINES, NORMAL_LINES
from vision_engine.score_routes import (
    best_score_route, guaranteed_score_floor, is_score_route, winning_route_catalog,
)
from vision_engine.scoring import score_mask


def user_like_board() -> Board:
    return Board.from_rows([
        ["CR", "HA", "LI", "CR", "CR"],
        ["BL", "CH", "HA", "LI", "CH"],
        ["BL", "CR", None, "SO", "CH"],
        ["CH", "LI", "SO", "SO", "HA"],
        ["LI", "SO", "BL", "BL", "HA"],
    ])


def mask(*positions: int) -> int:
    return sum(1 << pos for pos in positions)


def test_catalog_proves_every_999_and_higher_layout_by_exhaustive_completion():
    """Geometry union + final line scoring agree for all 7,020 winning masks."""
    route_masks = set()
    for route in winning_route_catalog():
        assert route.lower_bound_score >= 999
        assert route.slack >= 0
        free = [i for i in range(25) if i != 12 and not (route.required_mask >> i & 1)]
        for extras in combinations(free, route.slack):
            final = route.required_mask | mask(*extras)
            breakdown = score_mask(final)
            assert breakdown.total >= route.lower_bound_score
            route_masks.add(final)
    assert len(winning_route_catalog()) == 184
    assert len(route_masks) == 7020
    assert sum(score_mask(m).total > 1000 for m in route_masks) == 492
    assert sum(score_mask(m).total == 999 for m in route_masks) == 6528


def test_legacy_two_lucky_one_normal_954_cannot_be_promoted_to_999():
    # One Normal may overlap only one of the two Lucky lines. The 12 needed
    # cells then leave two loose 45-point cells: 624 + 240 + 90 = 954.
    invalid = []
    for ls in combinations(LUCKY_LINES, 2):
        for ns in NORMAL_LINES:
            if guaranteed_score_floor(ls, (ns,)) == 954:
                invalid.append((ls, ns))
    assert invalid
    assert all(not is_score_route(ls, (ns,)) for ls, ns in invalid)
    valid = [
        (ls, ns)
        for ls in combinations(LUCKY_LINES, 2)
        for ns in NORMAL_LINES
        if guaranteed_score_floor(ls, (ns,)) == 999
    ]
    assert len(valid) == 24
    assert all(is_score_route(ls, (ns,)) for ls, ns in valid)


def test_switches_to_proven_999_only_after_current_cr_destroys_three_lucky():
    b = user_like_board()
    selected = mask(0, 1, 2, 3, 4, 6, 7, 8, 16, 20)
    advisor = Advisor(b, mode="auto", rollouts=30, plan_target="3 Lucky",
                      plan_lucky_lines=("H", "V", "\\"), route_switch_ratio=100.0)
    recs = advisor.recommend(selected, "CR")
    top = recs[0]
    assert top.plan_target == ">=999"
    assert top.target_probability > 0
    assert top.plan_probability > 0
    assert top.route_score_floor >= 999
    assert is_score_route(top.plan_lucky_lines, top.plan_normal_lines)
    assert top.route_slack_remaining >= 0
    assert top.route_off_route <= 14 - sum(1 for i in range(25) if any(
        i in LUCKY_LINES[l] for l in top.plan_lucky_lines
    ) or any(i in NORMAL_LINES[n] for n in top.plan_normal_lines))
    assert "3 Lucky impossible" in top.plan_reason


def test_score_witness_is_not_sum_of_correlated_routes():
    b = user_like_board()
    selected = mask(0, 1, 2, 3, 4, 6, 7, 8, 16, 20, 11)
    route = best_score_route(b, selected)
    assert route.target == ">=999"
    assert 0 < route.probability <= 1
    assert guaranteed_score_floor(route.lucky_lines, route.normal_lines) >= 999
    # Tests the geometrical witness, NOT the overall chance of reaching >=999.
    assert route.probability != 0


def test_off_route_moves_spend_nonrenewable_slack():
    # Score-specific 999 target via 2 Lucky + 1 Normal uses 11 cells: three
    # placements outside its mask are possible; a fourth kills that route.
    route = next(r for r in winning_route_catalog() if r.lower_bound_score == 999)
    outside = [i for i in range(25) if i != 12 and not (route.required_mask >> i & 1)]
    b = user_like_board()
    advisor = Advisor(b, mode="auto", rollouts=1)
    first = mask(*outside[:2])
    used, remaining, floor = advisor._plan_budget(
        first, __import__("vision_engine.route_math", fromlist=["RouteEstimate"]).RouteEstimate(
            ">=999", 1.0, route.lucky_lines, route.normal_lines
        ),
    )
    assert (used, remaining, floor) == (2, 1, 999)
    over = mask(*outside[:4])
    assert (over & ~route.required_mask).bit_count() > route.slack


def test_real_954_fallback_is_not_mislabeled_a_999_chance():
    b = Board((
        'LI', 'LI', 'SO', 'LI', 'CH', 'BL', 'SO', 'HA', 'CH', 'CR',
        'SO', 'HA', None, 'CR', 'CR', 'HA', 'CH', 'HA', 'BL', 'SO',
        'CH', 'BL', 'BL', 'LI', 'CR',
    ))
    selected = mask(0, 5, 6, 11, 14, 18, 19, 20, 21, 22, 24)
    # The old 2L + 1N target gives a nominal 999 but this best route has a
    # provable terminal floor of just 954, not a guaranteed 999.
    from vision_engine.route_math import route_by_name, route_portfolio
    legacy = route_by_name(route_portfolio(b, selected), "2 Lucky + 1 Normal")
    assert legacy.probability > 0
    assert guaranteed_score_floor(legacy.lucky_lines, legacy.normal_lines) == 954
    # A CR can still be placed on the 999-proven route.
    good = Advisor(b, mode="auto", rollouts=20).recommend(selected, "CR")
    assert good[0].plan_target == ">=999"
    assert good[0].position == 13
    assert good[0].route_score_floor >= 999
    # An LI draw removes every viable 999 continuation. Never pretend that
    # the 954 nominal '2 Lucky + 1 Normal' fallback is a victory route.
    lost = Advisor(b, mode="auto", rollouts=20).recommend(selected, "LI")
    assert lost[0].plan_target == "expected score"
    assert lost[0].active_goal == "expected score"
    assert "impossible" in lost[0].plan_reason


def test_score_fallback_survives_persistent_worker_wire_roundtrip():
    from vision_engine.solver.protocol import recommendation_from_wire, recommendation_to_wire
    from vision_engine.solver.worker import solve_hot_worker
    b = user_like_board()
    selected = mask(0, 1, 2, 3, 4, 6, 7, 8, 16, 20)
    req = {
        "board_cells": b.cells, "mask": selected, "jewel": "CR",
        "mode": "auto", "rollouts": 30, "exact_horizon": 6,
        "plan_target": "3 Lucky", "plan_lucky_lines": ("H", "V", "\\"),
    }
    results = solve_hot_worker(req)
    restored = recommendation_from_wire(recommendation_to_wire(results[0]))
    assert restored.plan_target == ">=999"
    assert restored.route_score_floor >= 999
    assert restored.route_off_route == results[0].route_off_route
    assert restored.route_slack_remaining == results[0].route_slack_remaining
