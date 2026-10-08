from __future__ import annotations

from vision_engine.advisor import Advisor
from vision_engine.board import Board


def m(*positions: int) -> int:
    return sum(1 << p for p in positions)


def user_like_board() -> Board:
    return Board.from_rows([
        ["CR", "HA", "LI", "CR", "CR"],
        ["BL", "CH", "HA", "LI", "CH"],
        ["BL", "CR", None, "SO", "CH"],
        ["CH", "LI", "SO", "SO", "HA"],
        ["LI", "SO", "BL", "BL", "HA"],
    ])


def base_board() -> Board:
    return Board.from_rows([
        ["CH", "HA", "SO", "CH", "HA"],
        ["SO", "BL", "BL", "CR", "LI"],
        ["BL", "CH", None, "CR", "HA"],
        ["BL", "SO", "CH", "LI", "SO"],
        ["LI", "CR", "CR", "LI", "HA"],
    ])


def test_current_jewel_can_force_emergency_route_switch_without_dropping_target():
    b = user_like_board()
    selected = m(0, 1, 2, 3, 4, 6, 7, 8, 16, 20)
    recs = Advisor(
        b, mode="auto", rollouts=30,
        plan_target="3 Lucky", plan_lucky_lines=("H", "V", "\\"),
        route_switch_ratio=100.0,
    ).recommend(selected, "BL")
    top = recs[0]
    assert top.plan_target == "3 Lucky"
    assert top.route_switched is True
    assert top.plan_probability > 0
    assert "emergency same-target route switch" in top.plan_reason


def test_current_jewel_falls_back_only_when_no_three_lucky_route_survives():
    b = user_like_board()
    selected = m(0, 1, 2, 3, 4, 6, 7, 8, 16, 20)
    recs = Advisor(
        b, mode="auto", rollouts=30,
        plan_target="3 Lucky", plan_lucky_lines=("H", "V", "\\"),
        route_switch_ratio=100.0,
    ).recommend(selected, "CR")
    top = recs[0]
    assert top.plan_target == "2 Lucky + 1 Normal"
    assert top.target_probability > 0
    assert "after current jewel" in top.plan_reason


def test_true_plan_tie_prefers_lucky_contingency_before_normal_or_average_points():
    b = base_board()
    selected = m(0, 1, 2)
    recs = Advisor(
        b, mode="auto", rollouts=30,
        plan_target="3 Lucky", plan_lucky_lines=("H", "V", "\\"),
    ).recommend(selected, "LI")
    assert b.rc(recs[0].position) == (4, 4)
    assert abs(recs[0].plan_probability - recs[1].plan_probability) < 1e-12
    assert abs(recs[0].target_probability - recs[1].target_probability) < 1e-12
    assert recs[0].lucky_leverage > recs[1].lucky_leverage
    assert recs[0].tie_break_metric == "Lucky contingency leverage"


def test_forced_slack_is_explicit_when_current_jewel_has_no_locked_route_cell():
    # HA exists only outside H+V+\\, so a HA draw must consume slack while the
    # main 3-Lucky target can remain feasible early in the game.
    rows = [
        ["BL", "HA", "SO", "HA", "CR"],
        ["HA", "LI", "BL", "HA", "SO"],
        ["CR", "CH", None, "LI", "BL"],
        ["SO", "CR", "CH", "LI", "BL"],
        ["LI", "SO", "CR", "CH", "CH"],
    ]
    b = Board.from_rows(rows)
    recs = Advisor(
        b, mode="auto", rollouts=30,
        plan_target="3 Lucky", plan_lucky_lines=("H", "V", "\\"),
        route_switch_ratio=100.0, auto_p3_floor=0.0,
    ).recommend(0, "HA")
    assert recs[0].plan_target == "3 Lucky"
    assert recs[0].forced_slack is True
    assert not any(r.plan_on_route for r in recs)


def test_one_left_exact_equal_finishes_are_reported_as_true_tie():
    b = Board.from_rows([
        ["CR", "BL", "CR", "HA", "HA"],
        ["CR", "CH", "SO", "CH", "CH"],
        ["HA", "LI", None, "CH", "SO"],
        ["BL", "HA", "LI", "LI", "SO"],
        ["BL", "BL", "LI", "CR", "SO"],
    ])
    # Two Lucky lines already complete. R1 lacks L1C1 and R5 lacks L5C4;
    # current CR can close either, and both final boards have the same score.
    selected = m(1,2,3,4, 6,7,8, 16,17,20,21,22,24)
    recs = Advisor(
        b, mode="auto", rollouts=30,
        plan_target="2 Lucky + 1 Normal", plan_lucky_lines=("V", "/"), plan_normal_lines=("R1",),
    ).recommend(selected, "CR")
    top_positions = {recs[0].position, *recs[0].tied_with}
    assert 0 in top_positions and 23 in top_positions
    assert recs[0].tie_kind == "exact"
    finals = {r.position: r.utility.expected_score for r in recs if r.position in {0,23}}
    assert finals[0] == finals[23]


def test_recommendation_wire_round_trip_preserves_new_planner_fields():
    from vision_engine.advisor import Recommendation, Utility
    from vision_engine.solver.protocol import recommendation_from_wire, recommendation_to_wire

    rec = Recommendation(
        position=3,
        utility=Utility(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 777.0),
        method="test",
        plan_target="3 Lucky",
        plan_lucky_lines=("H", "V", "/"),
        plan_probability=0.12,
        target_probability=0.18,
        route_switched=True,
        forced_slack=True,
        lucky_leverage=0.44,
        endgame_value=888.0,
        terminal_move=True,
    )
    got = recommendation_from_wire(recommendation_to_wire(rec))
    assert got.target_probability == 0.18
    assert got.route_switched is True
    assert got.forced_slack is True
    assert got.lucky_leverage == 0.44
    assert got.terminal_move is True


def test_live_ui_separates_target_route_and_hides_average_points_from_primary_table():
    from pathlib import Path
    source = Path(__file__).parents[1].joinpath("vision_engine", "gui.py").read_text(encoding="utf-8")
    assert '"Goal"' in source
    assert '"Route"' in source
    assert 'columns=("move", "status", "target", "route", "win", "three")' in source
    assert 'FORCED SLACK:' in source
    assert 'ROUTE CHANGED' in source
    assert '_set_live_compact_mode(True)' in source
