from math import comb

from vision_engine.advisor import Advisor
from vision_engine.board import Board
from vision_engine.route_math import (
    best_route_feasibility,
    multivariate_at_least_probability,
    route_portfolio,
)


def live_board():
    # Board reconstructed from the user's live screenshots.
    return Board.from_rows([
        ["BL", "CR", "CH", "CH", "CR"],
        ["CR", "SO", "BL", "LI", "CR"],
        ["CH", "HA", None, "SO", "LI"],
        ["SO", "LI", "HA", "BL", "LI"],
        ["BL", "CH", "SO", "HA", "HA"],
    ])


def brute_probability(available, need, draws):
    # Independent reference enumerator for regression testing the production DP.
    from itertools import product
    den = comb(sum(available), draws)
    num = 0
    for xs in product(*[range(a + 1) for a in available]):
        if sum(xs) != draws or any(xs[i] < need[i] for i in range(6)):
            continue
        ways = 1
        for a, x in zip(available, xs):
            ways *= comb(a, x)
        num += ways
    return num / den


def test_multivariate_hypergeometric_matches_independent_bruteforce():
    available = (3, 4, 4, 3, 4, 4)
    need = (2, 0, 0, 0, 1, 0)  # BL x2 + HA x1
    draws = 10
    got = multivariate_at_least_probability(available, need, draws)
    want = brute_probability(available, need, draws)
    assert abs(got - want) < 1e-12


def test_duplicate_requirement_is_not_treated_like_distinct_singletons():
    available = (3, 4, 4, 4, 4, 4)
    draws = 10
    duplicate = multivariate_at_least_probability(available, (2, 0, 0, 0, 1, 0), draws)
    distinct = multivariate_at_least_probability(available, (1, 0, 0, 0, 1, 1), draws)
    assert duplicate != distinct


def test_route_portfolio_is_policy_independent_and_contains_all_targets():
    b = live_board()
    mask = (1 << 4) | (1 << 13) | (1 << 14)
    routes = route_portfolio(b, mask)
    names = {r.target for r in routes}
    assert names == {
        "3 Lucky + 1 Normal", "3 Lucky", "2 Lucky + 2 Normal",
        "2 Lucky + 1 Normal", "1 Lucky + 3 Normal",
        "1 Lucky + 2 Normal", "1 Lucky + 1 Normal",
    }
    for route in routes:
        assert 0.0 <= route.probability <= 1.0


def test_best_three_lucky_route_reports_physical_needs():
    b = live_board()
    mask = (1 << 4) | (1 << 13) | (1 << 14)
    r = best_route_feasibility(b, mask, "3 Lucky", 3, 0)
    assert r.target == "3 Lucky"
    assert 0.0 <= r.probability <= 1.0
    assert sum(n for _, n in r.needs) == r.missing_cells


def test_user_8_of_14_regression_uses_exact_horizon_six_and_picks_l2c2():
    b = live_board()
    # Screenshot state: selected L1C5, L2C3,L2C4,L2C5, L3C2,L3C4,L3C5,L5C3.
    selected = [4, 7, 8, 9, 11, 13, 14, 22]
    mask = sum(1 << i for i in selected)
    recs = Advisor(b, mode="score", exact_horizon=6, rollouts=60).recommend(mask, "SO")
    assert recs[0].method == "exact-expectimax"
    assert recs[0].position == 6  # L2C2, not L4C1
    assert abs(recs[0].utility.p_gt1000 - recs[1].utility.p_gt1000) < 1e-12
    assert recs[0].utility.p_ge999 > recs[1].utility.p_ge999


def test_routes_are_embedded_separately_from_policy_utility():
    b = live_board()
    rec = Advisor(b, mode="score", exact_horizon=6, rollouts=20).recommend(0, "CR")[0]
    assert rec.routes
    assert any(r.target == "3 Lucky" for r in rec.routes)
