from pathlib import Path

from vision_engine.advisor import Advisor, Recommendation, Utility
from vision_engine.board import Board
from vision_engine.strategy import compact_move_line_summary, move_line_impacts


def base_board():
    return Board.from_rows([
        ["CH", "HA", "SO", "CH", "HA"],
        ["SO", "BL", "BL", "CR", "LI"],
        ["BL", "CH", None, "CR", "HA"],
        ["BL", "SO", "CH", "LI", "SO"],
        ["LI", "CR", "CR", "LI", "HA"],
    ])


def screenshot_board():
    return Board.from_rows([
        ["HA", "CR", "HA", "HA", "SO"],
        ["LI", "BL", "BL", "LI", "LI"],
        ["CH", "BL", None, "CH", "CH"],
        ["CR", "CR", "SO", "CH", "CR"],
        ["LI", "SO", "HA", "SO", "BL"],
    ])


def test_common_random_scenarios_are_action_independent():
    adv = Advisor(base_board(), mode="score", exact_horizon=0, rollouts=20)
    adv._prepare_common_sequences(0, "LI", 5)
    first = list(adv._active_common_sequences)
    # Preparing the same state again must produce exactly the same futures; no
    # candidate position is part of the seed anymore.
    adv._prepare_common_sequences(0, "LI", 5)
    assert adv._active_common_sequences == first
    assert len(first) == 5


def test_rollout_policy_really_depends_on_selected_strategy():
    b = base_board()
    mask = 1 << 0
    lucky = Advisor(b, mode="lucky", exact_horizon=0, rollouts=1)
    score = Advisor(b, mode="score", exact_horizon=0, rollouts=1)
    assert lucky._policy_position(mask, "BL") != score._policy_position(mask, "BL")


def test_adaptive_line_planner_can_preserve_a_near_complete_normal_fallback():
    b = base_board()
    # R5 has four selected cells; HA at index 24 completes it. 3-Lucky is not
    # structurally viable in this state, so Adaptive should value the Normal
    # fallback instead of using the last HA elsewhere.
    selected = [1, 7, 13, 19, 20, 21, 22, 23]
    mask = sum(1 << i for i in selected)
    adv = Advisor(b, mode="adaptive", exact_horizon=0, rollouts=1)
    pick = adv._policy_position(mask, "HA")
    assert pick == 24
    impacts = move_line_impacts(mask, pick)
    assert any(x.kind == "Normal" and x.completed for x in impacts)


def test_exact_equal_actions_are_explicitly_marked_as_tied():
    adv = Advisor(base_board(), mode="score")
    u = Utility(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 894.0)
    recs = [
        Recommendation(13, u, "exact-expectimax", active_goal="expected score"),
        Recommendation(14, u, "exact-expectimax", active_goal="expected score"),
    ]
    out = adv._annotate_ties(recs, "expected score", exact=True)
    assert out[0].tie_kind == "exact"
    assert out[1].confidence == "TIE"


def test_paired_common_random_uncertainty_marks_statistical_tie():
    adv = Advisor(base_board(), mode="score")
    # Alternating paired differences have zero mean and non-zero paired error.
    a_samples = [Utility(0, 0, 0, 0, x, x, 900) for x in (1, 0, 1, 0)]
    b_samples = [Utility(0, 0, 0, 0, x, x, 900) for x in (0, 1, 0, 1)]
    adv._last_mc_samples = {0: a_samples, 1: b_samples}
    ua = Utility(0, 0, 0, 0, .5, .5, 900)
    ub = Utility(0, 0, 0, 0, .5, .5, 900)
    out = adv._annotate_ties([
        Recommendation(0, ua, "monte-carlo:4", active_goal=">1000", samples=4),
        Recommendation(1, ub, "monte-carlo:4", active_goal=">1000", samples=4),
    ], ">1000", exact=False)
    assert out[0].tie_kind == "statistical"
    assert out[1].confidence == "TIE"


def test_line_summary_explicitly_says_when_move_is_outside_lucky_lines():
    # Base-board index 5 (L2C1) is not on H/V/diagonals through MU.
    text = compact_move_line_summary(0, 5)
    assert "Lucky: none" in text
    assert "Normal:" in text


def test_user_screenshot_board_keeps_same_lucky_route_p3_equal():
    # L2C4 (8) and L5C1 (20) are both on the same '/' Lucky line. The prior
    # position-seeded Monte Carlo produced materially different P(3L) values.
    # With common futures + a 3-Lucky rollout policy they must receive the same
    # P(3L) estimate for this empty-board state.
    adv = Advisor(screenshot_board(), mode="lucky", exact_horizon=3, rollouts=60)
    recs = {r.position: r for r in adv.recommend(0, "LI")}
    assert abs(recs[8].utility.p3 - recs[20].utility.p3) < 1e-12


def test_live_ui_contains_provisional_ranked_probability_sections():
    gui_source = (Path(__file__).parents[1] / "vision_engine" / "gui.py").read_text(encoding="utf-8")
    assert "FAST / PROVISIONAL" in gui_source
    assert "KEY PROBABILITIES" in gui_source
    assert "RANKED OPTIONS" in gui_source
    assert "WHY THIS CELL" in gui_source


def test_adaptive_exact_endgame_selects_normal_clear_when_it_is_still_possible():
    # Regression from the user's 2-Lucky / 0-Normal / 849-point endgame shape.
    # Here C4 is one CH away from a Normal Clear. Adaptive must take L3C4 and
    # reach 999 instead of treating all remaining CH cells as equivalent.
    b = screenshot_board()
    selected = [0, 3, 4, 6, 7, 8, 11, 15, 16, 18, 20, 23, 24]
    mask = sum(1 << i for i in selected)
    recs = Advisor(b, mode="adaptive", exact_horizon=3, rollouts=60).recommend(mask, "CH")
    assert recs[0].position == 13  # L3C4
    assert recs[0].active_goal == "2 Lucky + 1 Normal"
    assert recs[0].utility.p2l1n == 1.0
    assert recs[0].utility.expected_score == 999.0
