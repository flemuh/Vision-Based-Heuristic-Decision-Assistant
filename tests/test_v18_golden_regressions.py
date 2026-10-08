from __future__ import annotations

from vision_engine.advisor import Advisor
from vision_engine.board import Board
from vision_engine.round_lifecycle import CompletionDetector
from vision_engine.temporal import TemporalStateFilter


def m(*positions: int) -> int:
    return sum(1 << p for p in positions)


def test_golden_false_plus_one_rolls_back():
    base = m(1,3,7,9,10,12)
    false = base | (1 << 20)
    f = TemporalStateFilter(frames=3, required=2, max_mask_jump=1,
        catchup_required_frames=5, single_step_confirm_frames=4,
        single_step_rollback_frames=3, single_step_rollback_window_frames=10)
    f.reset(base)
    for _ in range(4): state = f.observe(false, "SO", .95)
    assert state.mask == false
    for _ in range(3): state = f.observe(base, "SO", .95)
    assert state.mask == base and state.reason.startswith("rollback-false-plus-one")


def test_golden_result_screen_recovers_recent_fourteen():
    d = CompletionDetector(min_accepted=12, disappear_required=2, recent_14_window_frames=10)
    a13 = sum(1 << i for i in range(13)); r14 = a13 | (1 << 14)
    d.observe(accepted_mask=a13, raw_mask=r14, jewel_visible=True)
    d.observe(accepted_mask=a13, raw_mask=0, jewel_visible=False)
    end = d.observe(accepted_mask=a13, raw_mask=0, jewel_visible=False)
    assert end.confirmed and end.final_mask == r14


def test_golden_bless_structural_tie_prefers_l1c4():
    b = Board.from_rows([
        ["SO","CH","CR","BL","LI"], ["BL","HA","CR","LI","BL"],
        ["HA","CH",None,"LI","LI"], ["CH","HA","HA","SO","CR"],
        ["SO","BL","CR","CH","SO"],
    ])
    recs = Advisor(b, mode="score", rollouts=120, exact_horizon=6).recommend(m(2,11,13), "BL")
    assert b.rc(recs[0].position) == (1,4)


def test_golden_auto_plan_rejects_off_route_fourth_lucky():
    b = Board.from_rows([
        ["CH","HA","SO","CH","HA"], ["SO","BL","BL","CR","LI"],
        ["BL","CH",None,"CR","HA"], ["BL","SO","CH","LI","SO"],
        ["LI","CR","CR","LI","HA"],
    ])
    recs = Advisor(b, mode="auto", rollouts=120,
        plan_target="3 Lucky", plan_lucky_lines=("H","V","\\"), route_switch_ratio=100.0
    ).recommend(m(6,13,22), "CH")
    assert recs[0].plan_target == "3 Lucky"
    assert recs[0].plan_on_route
    assert recs[0].position != 3


def test_golden_three_lucky_zero_falls_back():
    b = Board.from_rows([
        ["CH","HA","SO","CH","HA"], ["SO","BL","BL","CR","LI"],
        ["BL","CH",None,"CR","HA"], ["BL","SO","CH","LI","SO"],
        ["LI","CR","CR","LI","HA"],
    ])
    recs = Advisor(b, mode="auto", rollouts=120,
        plan_target="3 Lucky", plan_lucky_lines=("H","V","/")
    ).recommend(m(1,3,5), "CH")
    assert recs[0].plan_target != "3 Lucky"
