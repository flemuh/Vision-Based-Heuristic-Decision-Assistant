from pathlib import Path

from vision_engine.advisor import Advisor
from vision_engine.board import Board

ROOT = Path(__file__).parents[1]


def test_live_surface_uses_plain_confidence_language():
    source = (ROOT / "vision_engine" / "gui.py").read_text(encoding="utf-8")
    for label in ("EQUAL OPTIONS", "CLOSE CALL", "LOW SAMPLE", "HIGH"):
        assert label in source
    assert 'self._show_compact_decision(mask, jewel, recs, refined)' in source


def test_phase2_does_not_change_solver_choice_for_locked_route_golden_state():
    b = Board.from_rows([
        ["CH","HA","SO","CH","HA"], ["SO","BL","BL","CR","LI"],
        ["BL","CH",None,"CR","HA"], ["BL","SO","CH","LI","SO"],
        ["LI","CR","CR","LI","HA"],
    ])
    mask = (1 << 6) | (1 << 13) | (1 << 22)
    recs = Advisor(b, mode="auto", rollouts=120, plan_target="3 Lucky",
                   plan_lucky_lines=("H","V","\\"), route_switch_ratio=100.0).recommend(mask, "CH")
    assert recs[0].plan_target == "3 Lucky"
    assert recs[0].plan_on_route is True
    assert recs[0].position != 3

