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


