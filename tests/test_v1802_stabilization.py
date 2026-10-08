from __future__ import annotations

from vision_engine.advisor import Advisor
from vision_engine.board import Board
from vision_engine.ui_semantics import move_visual_status, visual_status_style


def m(*positions: int) -> int:
    return sum(1 << p for p in positions)


def late_game_board() -> Board:
    return Board.from_rows([
        ["CR", "BL", "CR", "HA", "HA"],
        ["CR", "CH", "SO", "CH", "CH"],
        ["HA", "LI", None, "CH", "SO"],
        ["BL", "HA", "LI", "LI", "SO"],
        ["BL", "BL", "LI", "CR", "SO"],
    ])


def test_real_so_state_prefers_l5c5_over_equally_feasible_but_weaker_cells():
    """Golden regression from the 12/14 live screenshot.

    L5C5, L3C5 and L4C5 preserve the same primary target/locked-route
    feasibility, but L5C5 creates a much stronger one-away Normal contingency.
    It must be the unique #1 rather than another green primary-only tie.
    """
    b = late_game_board()
    selected = m(1, 2, 3, 4, 6, 7, 8, 16, 17, 20, 21, 22)
    recs = Advisor(
        b, mode="auto", rollouts=30,
        plan_target="2 Lucky + 1 Normal",
        plan_lucky_lines=("V", "/"),
        plan_normal_lines=("R1",),
    ).recommend(selected, "SO")

    assert b.rc(recs[0].position) == (5, 5)
    assert [b.rc(r.position) for r in recs[:3]] == [(5, 5), (3, 5), (4, 5)]
    assert abs(recs[0].target_probability - recs[1].target_probability) < 1e-12
    assert abs(recs[0].plan_probability - recs[1].plan_probability) < 1e-12
    assert recs[0].normal_leverage > recs[1].normal_leverage
    assert recs[0].tie_break_metric == "Normal-line leverage"

    assert move_visual_status(recs[0], recs[0], 0) == "BEST"
    assert move_visual_status(recs[0], recs[1], 1) == "CLOSE"
    assert move_visual_status(recs[0], recs[2], 2) == "CLOSE"


def test_exact_final_equal_finishes_share_green_semantics():
    b = late_game_board()
    selected = m(1,2,3,4, 6,7,8, 16,17,20,21,22,24)
    recs = Advisor(
        b, mode="auto", rollouts=30,
        plan_target="2 Lucky + 1 Normal",
        plan_lucky_lines=("V", "/"),
        plan_normal_lines=("R1",),
    ).recommend(selected, "CR")
    assert recs[0].tie_kind == "exact"
    tie_positions = {recs[0].position, *recs[0].tied_with}
    assert {0, 23}.issubset(tie_positions)
    second_equal = next(r for r in recs if r.position in recs[0].tied_with)
    assert move_visual_status(recs[0], second_equal, 1) == "EXACT_TIE"
    assert visual_status_style("EXACT_TIE")[1] == visual_status_style("BEST")[1]


def test_endgame_route_hysteresis_relaxes_without_changing_early_game_default():
    b = late_game_board()
    a = Advisor(b, mode="auto", route_switch_ratio=1.35)
    assert a._effective_route_switch_ratio(m(*range(9))) == 1.35
    assert a._effective_route_switch_ratio(m(*range(10))) == 1.20
    assert a._effective_route_switch_ratio(m(*range(12))) == 1.10
    assert a._effective_route_switch_ratio(m(*range(13))) == 1.00


def test_v1802_live_ui_contract_keeps_four_options_visible_and_logs_reachable():
    from pathlib import Path
    source = Path(__file__).parents[1].joinpath("vision_engine", "gui.py").read_text(encoding="utf-8")
    assert 'show="headings", height=4' in source
    assert 'columns=("move", "status", "target", "route", "win", "three")' in source
    assert 'Goal possible:' in source
    assert 'Route possible:' in source
    assert 'Recent technical log' in source
    assert 'self._refresh_system_logs' in source
    assert 'analysis_scroll = ttk.Scrollbar' in source
    assert 'learning_tree_scroll = ttk.Scrollbar' in source
    semantics = Path(__file__).parents[1].joinpath('vision_engine', 'ui_semantics.py').read_text(encoding='utf-8')
    assert 'FORCED_SLACK' in semantics
    assert 'EXACT_TIE' in semantics
