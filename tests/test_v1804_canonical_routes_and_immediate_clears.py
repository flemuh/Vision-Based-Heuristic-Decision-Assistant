from __future__ import annotations

from vision_engine.advisor import Advisor
from vision_engine.board import Board
from vision_engine.route_math import best_route_feasibility


def m(*positions: int) -> int:
    return sum(1 << p for p in positions)


def harmony_live_board() -> Board:
    """Board reconstructed from the 7/14 Harmony live screenshot."""
    return Board.from_rows([
        ["SO", "BL", "BL", "BL", "CH"],
        ["CR", "LI", "CR", "SO", "CR"],
        ["CR", "HA", None, "LI", "BL"],
        ["CH", "HA", "CH", "SO", "LI"],
        ["LI", "HA", "HA", "CH", "SO"],
    ])


def test_completed_line_remains_in_canonical_three_lucky_route_identity():
    b = harmony_live_board()
    before = m(0, 2, 5, 7, 10, 13, 14)
    # HA -> L3C2 completes Lucky H immediately.
    after = before | (1 << 11)
    route = best_route_feasibility(b, after, "3 Lucky", 3, 0)

    assert len(route.lucky_lines) == 3
    assert "H" in route.lucky_lines
    # A completed line contributes no remaining jewel requirement, but it must
    # remain part of the full route identity so sibling candidates cannot
    # incorrectly inherit it as already complete.
    assert route.missing_cells == sum(n for _j, n in route.needs)


def test_mixed_target_route_identity_keeps_completed_lucky_and_normal_lines():
    b = harmony_live_board()
    # H and R1 are complete. A 2L+1N route must still identify both of those
    # completed lines plus the second Lucky it intends to finish.
    selected = m(0, 1, 2, 3, 4, 10, 11, 13, 14)
    route = best_route_feasibility(b, selected, "2 Lucky + 1 Normal", 2, 1)

    assert len(route.lucky_lines) == 2
    assert len(route.normal_lines) == 1
    assert "H" in route.lucky_lines
    assert route.normal_lines == ("R1",)


def test_real_harmony_7_of_14_closes_lucky_h_before_equally_feasible_l5c3():
    b = harmony_live_board()
    selected = m(0, 2, 5, 7, 10, 13, 14)
    recs = Advisor(
        b,
        mode="auto",
        rollouts=120,
        plan_target="3 Lucky",
        plan_lucky_lines=("H", "V", "\\"),
    ).recommend(selected, "HA")

    by_pos = {r.position: r for r in recs}
    l3c2 = by_pos[11]
    l5c3 = by_pos[22]

    assert b.rc(recs[0].position) == (3, 2)
    assert recs[0].plan_lucky_lines == ("H", "V", "\\")
    assert abs(l3c2.plan_probability - l5c3.plan_probability) < 1e-12
    assert abs(l3c2.target_probability - l5c3.target_probability) < 1e-12
    assert l3c2.lucky_clears_gained == 1
    assert l5c3.lucky_clears_gained == 0
    assert recs[0].tie_break_metric == "Immediate Lucky clears"
    assert l3c2.utility.p3 >= l5c3.utility.p3
    assert l3c2.utility.expected_score > l5c3.utility.expected_score


def test_immediate_clear_is_only_a_tiebreak_not_permission_to_sacrifice_target():
    b = harmony_live_board()
    selected = m(0, 2, 5, 7, 10, 13, 14)
    recs = Advisor(
        b,
        mode="auto",
        rollouts=120,
        plan_target="3 Lucky",
        plan_lucky_lines=("H", "V", "\\"),
    ).recommend(selected, "HA")

    top = recs[0]
    # The winning immediate clear is safe specifically because it preserves the
    # same locked-route and global-target feasibility as L5C3. This protects the
    # intended rule: close a line on a plan tie, never at the cost of the target.
    l5c3 = next(r for r in recs if r.position == 22)
    assert abs(top.plan_probability - l5c3.plan_probability) < 1e-12
    assert abs(top.target_probability - l5c3.target_probability) < 1e-12


def test_legacy_shortened_locked_route_is_not_given_hysteresis_protection():
    b = harmony_live_board()
    # H is already complete in this state. V18.0.3 could persist the buggy
    # shortened identity (V, \\) for target 3 Lucky. The new planner must reject
    # that as incomplete and rebuild a full 3-line identity.
    selected = m(0, 2, 5, 7, 10, 11, 13, 14)
    recs = Advisor(
        b,
        mode="auto",
        rollouts=60,
        plan_target="3 Lucky",
        plan_lucky_lines=("V", "\\"),
        route_switch_ratio=100.0,
    ).recommend(selected, "HA")

    assert recs[0].plan_target == "3 Lucky"
    assert len(recs[0].plan_lucky_lines) == 3
    assert "H" in recs[0].plan_lucky_lines


def test_live_ui_explains_immediate_lucky_clear_priority():
    from pathlib import Path
    source = Path(__file__).parents[1].joinpath("vision_engine", "gui.py").read_text(encoding="utf-8")
    assert "CLOSES LUCKY NOW" in source
    assert "banking the Lucky line gets priority" in source
