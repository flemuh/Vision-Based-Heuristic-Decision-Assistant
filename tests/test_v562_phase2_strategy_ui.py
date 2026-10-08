from pathlib import Path

from vision_engine.advisor import Advisor, Recommendation, Utility
from vision_engine.board import Board
from vision_engine.route_math import RouteEstimate


def board():
    return Board.from_rows([
        ["BL", "CR", "CH", "CH", "CR"],
        ["CR", "SO", "BL", "LI", "CR"],
        ["CH", "HA", None, "SO", "LI"],
        ["SO", "LI", "HA", "BL", "LI"],
        ["BL", "CH", "SO", "HA", "HA"],
    ])


def U(p3=0, p2=0, p12=0, p11=0, gt=0, ge=0, score=0):
    return Utility(p3, p2, p12, p11, gt, ge, score)


def test_score_primary_statistical_tie_uses_ge999_as_hierarchical_tiebreak():
    adv = Advisor(board(), mode="score", exact_horizon=0, rollouts=100)
    a_samples = []
    b_samples = []
    # Primary >1000: 4% vs 3%, with noisy paired differences -> overlap.
    # Secondary >=999: 14% vs 9%, with b successes a subset of a -> clear +5pp.
    for i in range(100):
        a_gt = 1.0 if i in {0, 1, 2, 3} else 0.0
        b_gt = 1.0 if i in {1, 2, 4} else 0.0
        a_ge = 1.0 if i < 14 else 0.0
        b_ge = 1.0 if i < 9 else 0.0
        a_samples.append(U(p3=.03, gt=a_gt, ge=a_ge, score=830))
        b_samples.append(U(p3=.03, gt=b_gt, ge=b_ge, score=825))
    adv._last_mc_samples = {6: a_samples, 22: b_samples}
    a = Recommendation(6, U(p3=.03, gt=.04, ge=.14, score=830), "mc", active_goal=">1000", samples=100)
    b = Recommendation(22, U(p3=.03, gt=.03, ge=.09, score=825), "mc", active_goal=">1000", samples=100)
    ranked = adv._hierarchical_sort([b, a], ">1000", exact=False)
    annotated = adv._annotate_ties(ranked, ">1000", exact=False)
    assert annotated[0].position == 6
    assert annotated[0].confidence == "CLEAR"
    assert annotated[0].primary_stat_tie is True
    assert annotated[0].tie_break_metric == "P(>=999)"
    assert annotated[1].confidence == "RANKED"


def test_true_full_hierarchy_tie_still_shares_first_place():
    adv = Advisor(board(), mode="score")
    u = U(p3=.1, gt=.2, ge=.3, score=900)
    a = Recommendation(1, u, "exact", active_goal=">1000")
    b = Recommendation(5, u, "exact", active_goal=">1000")
    ranked = adv._hierarchical_sort([b, a], ">1000", exact=True)
    out = adv._annotate_ties(ranked, ">1000", exact=True)
    assert out[0].confidence == "TIE"
    assert out[0].tie_kind == "exact"


def test_adaptive_uses_exact_route_portfolio_instead_of_policy_p3():
    adv = Advisor(board(), mode="adaptive", adaptive_score_min=.08)
    routes_a = (
        RouteEstimate("3 Lucky + 1 Normal", .01),
        RouteEstimate("3 Lucky", .03),
        RouteEstimate("2 Lucky + 2 Normal", .12),
        RouteEstimate("2 Lucky + 1 Normal", .50),
        RouteEstimate("1 Lucky + 3 Normal", .02),
        RouteEstimate("1 Lucky + 2 Normal", .60),
        RouteEstimate("1 Lucky + 1 Normal", .80),
    )
    routes_b = tuple(RouteEstimate(r.target, max(0.0, r.probability - .02)) for r in routes_a)
    # Policy P3 is deliberately misleading/high; the exact route portfolio says
    # 2L+2N is the strongest credible >1000-capable route.
    recs = [
        Recommendation(0, U(p3=.80, gt=.02, ge=.1, score=850), "test", routes=routes_a),
        Recommendation(1, U(p3=.75, gt=.03, ge=.1, score=860), "test", routes=routes_b),
    ]
    assert adv._choose_global_goal(recs) == "2 Lucky + 2 Normal"


def test_gui_has_immediate_persistent_ranking_and_clear_strategy_name():
    source = (Path(__file__).parents[1] / "vision_engine" / "gui.py").read_text(encoding="utf-8")
    assert "Score >1000" in source
    assert "ROUTE FEASIBILITY" in source
    assert "HIERARCHICAL TIE-BREAK" in source
    assert "receive-feasibility" in source
    assert "self.update_idletasks()" in source
    assert "self._ranked_grid_mask" in source
    assert "self.after(25, self._start_solver_stage" in source


def test_adaptive_keeps_broad_score_goal_while_early_game_route_flexibility_is_high():
    adv = Advisor(board(), mode="adaptive", exact_horizon=6, rollouts=12)
    recs = adv.recommend(0, "CR")
    assert recs[0].active_goal == ">1000"


def test_adaptive_does_not_keep_score_goal_when_all_over1000_routes_are_impossible_even_early():
    adv = Advisor(board(), mode="adaptive", exact_horizon=6, rollouts=12)
    routes = (
        RouteEstimate("3 Lucky + 1 Normal", 0.0),
        RouteEstimate("3 Lucky", 0.0),
        RouteEstimate("2 Lucky + 2 Normal", 0.0),
        RouteEstimate("2 Lucky + 1 Normal", .45),
        RouteEstimate("1 Lucky + 3 Normal", 0.0),
        RouteEstimate("1 Lucky + 2 Normal", .60),
        RouteEstimate("1 Lucky + 1 Normal", .85),
    )
    assert adv._adaptive_goal_from_routes([routes], remaining_after_move=9) != ">1000"
