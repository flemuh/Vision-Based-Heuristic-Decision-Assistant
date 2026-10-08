from __future__ import annotations

from vision_engine.advisor import Advisor
from vision_engine.board import Board


def bless_screenshot_board() -> Board:
    return Board.from_rows([
        ["SO", "CH", "CR", "BL", "LI"],
        ["BL", "HA", "CR", "LI", "BL"],
        ["HA", "CH", None, "LI", "LI"],
        ["CH", "HA", "HA", "SO", "CR"],
        ["SO", "BL", "CR", "CH", "SO"],
    ])


def life_four_screenshot_board() -> Board:
    return Board.from_rows([
        ["BL", "SO", "LI", "LI", "HA"],
        ["CR", "SO", "HA", "CH", "LI"],
        ["CH", "SO", None, "HA", "CR"],
        ["LI", "BL", "BL", "CH", "CR"],
        ["CR", "SO", "CH", "BL", "HA"],
    ])


def life_six_screenshot_board() -> Board:
    return Board.from_rows([
        ["BL", "CH", "LI", "CR", "HA"],
        ["LI", "HA", "LI", "SO", "SO"],
        ["HA", "CR", None, "LI", "CR"],
        ["CH", "SO", "SO", "BL", "CH"],
        ["BL", "CH", "CR", "HA", "BL"],
    ])


def mask(*positions: int) -> int:
    return sum(1 << p for p in positions)


def test_bless_screenshot_statistical_tie_uses_exact_score_structure():
    b = bless_screenshot_board()
    # User screenshot: CR L1C3, CH L3C2 and LI L3C4 are already selected.
    adv = Advisor(b, mode="score", rollouts=120, exact_horizon=6)
    recs = adv.recommend(mask(2, 11, 13), "BL")
    # All four BL actions have the same low-sample P(>1000) in this deterministic
    # scenario set. L1C4 should win the tie because it has the strongest exact
    # >1000 structural route portfolio and the strongest Normal-line leverage.
    assert b.rc(recs[0].position) == (1, 4)
    assert recs[0].primary_stat_tie is True
    assert recs[0].tie_break_metric == ">1000 structural route potential"
    assert recs[0].score_route_potential > recs[1].score_route_potential
    assert recs[0].normal_leverage > 0.0


def test_life_six_screenshot_does_not_spend_score_tie_on_off_lucky_cell():
    b = life_six_screenshot_board()
    # Six selected cells from the user's earlier live screenshot.
    adv = Advisor(b, mode="score", rollouts=120, exact_horizon=6)
    recs = adv.recommend(mask(4, 6, 8, 10, 13, 14), "LI")
    # The off-Lucky L2C1 has a slightly higher raw 120-rollout estimate, but the
    # paired primary comparison overlaps. The exact structural tie-break must
    # keep one of the Lucky-V cells ahead.
    assert b.rc(recs[0].position) in {(1, 3), (2, 3)}
    assert b.rc(recs[0].position) != (2, 1)


def test_life_four_live_winner_stays_l1c4_but_is_labeled_low_sample():
    b = life_four_screenshot_board()
    adv = Advisor(b, mode="score", rollouts=120, exact_horizon=6)
    recs = adv.recommend(mask(0, 6, 18, 24), "LI")
    assert b.rc(recs[0].position) == (1, 4)
    assert recs[0].confidence == "LOW_SAMPLE"
    assert recs[0].samples == 30
    # L1C3 is structurally better for a pure 3-Lucky route, which is why the UI
    # must explain the trade-off instead of pretending the small live MC sample
    # is an exact proof.
    by_rc = {b.rc(r.position): r for r in recs}
    assert by_rc[(1, 3)].lucky_slack.slack_remaining > by_rc[(1, 4)].lucky_slack.slack_remaining
    assert by_rc[(1, 4)].normal_leverage > by_rc[(1, 3)].normal_leverage
