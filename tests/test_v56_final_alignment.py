from vision_engine.advisor import Advisor, Recommendation, Utility
from vision_engine.board import Board
from vision_engine.config import AppConfig
from vision_engine.strategy import viability_label


def base_board():
    return Board.from_rows([
        ["CH", "HA", "SO", "CH", "HA"],
        ["SO", "BL", "BL", "CR", "LI"],
        ["BL", "CH", None, "CR", "HA"],
        ["BL", "SO", "CH", "LI", "SO"],
        ["LI", "CR", "CR", "LI", "HA"],
    ])


def U(p3=0.0, p2=0.0, p12=0.0, p11=0.0, gt=0.0, ge=0.0, score=0.0):
    return Utility(p3, p2, p12, p11, gt, ge, score)


def R(pos, u, goal):
    return Recommendation(pos, u, "test", active_goal=goal)


def test_max_score_priority_is_gt1000_then_ge999_then_expected_score():
    adv = Advisor(base_board(), mode="score")
    # Same P>1000: the higher P>=999 must win even with a lower expected score.
    a = R(0, U(p3=.1, gt=.50, ge=.90, score=980), ">1000")
    b = R(1, U(p3=.3, gt=.50, ge=.80, score=1200), ">1000")
    ranked = adv._sort_for_goal([b, a], ">1000")
    assert ranked[0].position == 0


def test_three_lucky_priority_keeps_p3_first_then_mixed_fallbacks():
    adv = Advisor(base_board(), mode="lucky")
    # Higher P3 wins even if another action has a much higher expected score.
    a = R(0, U(p3=.31, p2=.20, gt=.40, score=900), "3 Lucky")
    b = R(1, U(p3=.30, p2=.80, gt=.80, score=1400), "3 Lucky")
    assert adv._sort_for_goal([b, a], "3 Lucky")[0].position == 0
    # With equal P3 and >1000, mixed-line probability precedes expected score.
    c = R(2, U(p3=.30, p2=.60, gt=.50, score=900), "3 Lucky")
    d = R(3, U(p3=.30, p2=.20, gt=.50, score=1300), "3 Lucky")
    assert adv._sort_for_goal([d, c], "3 Lucky")[0].position == 2


def test_adaptive_switches_to_score_before_salvage_when_score_is_viable():
    adv = Advisor(base_board(), mode="adaptive", p3_min=.08, adaptive_score_min=.08)
    utilities = [
        U(p3=.03, p2=.70, gt=.20, ge=.50, score=1000),
        U(p3=.02, p2=.80, gt=.15, ge=.60, score=1010),
    ]
    assert adv._choose_global_goal(utilities) == ">1000"


def test_adaptive_enters_salvage_when_score_viability_is_low():
    adv = Advisor(base_board(), mode="adaptive", p3_min=.08, adaptive_score_min=.08)
    utilities = [
        U(p3=.03, p2=.30, gt=.05, score=920),
        U(p3=.02, p2=.20, gt=.04, score=950),
    ]
    assert adv._choose_global_goal(utilities) == "2 Lucky + 1 Normal"


def test_fast_rollouts_are_a_total_budget_not_a_per_action_floor(monkeypatch):
    adv = Advisor(base_board(), mode="score", exact_horizon=0, rollouts=60)
    observed = []

    def fake_mc(mask, jewel, pos, n):
        observed.append(n)
        return U(gt=.1, ge=.2, score=900 + pos), 0.0, 0.0, 0.0

    monkeypatch.setattr(adv, "_monte_carlo_action", fake_mc)
    recs = adv.recommend(0, "BL")
    assert len(recs) == 4
    assert observed == [15, 15, 15, 15]
    assert all(r.method == "monte-carlo:15" for r in recs)


def test_transition_poll_and_viability_defaults_match_v56_contract():
    cfg = AppConfig()
    assert cfg.fast_rollouts == 60
    assert cfg.transition_poll_ms == 120
    assert cfg.adaptive_score_min == .08
    assert viability_label(0.0, .08, .25) == "IMPOSSIBLE"
    assert viability_label(.04, .08, .25) == "LOW"
    assert viability_label(.10, .08, .25) == "MEDIUM"
    assert viability_label(.30, .08, .25) == "HIGH"
