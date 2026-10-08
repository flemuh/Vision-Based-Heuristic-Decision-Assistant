"""Pure UI semantics shared by Tk rendering and regression tests.

This module deliberately has no Tk/capture dependencies, so ranking color meaning
can be tested in headless CI.  Colors describe *why* a move is ranked where it is,
not just its ordinal position.
"""
from __future__ import annotations

from .advisor import Recommendation


def move_visual_status(top: Recommendation, rec: Recommendation, rank: int) -> str:
    eps = 1e-12
    if rec.position == top.position:
        return "FORCED_SLACK" if rec.forced_slack else "BEST"
    if top.tie_kind == "exact" and rec.position in set(top.tied_with):
        return "EXACT_TIE"
    same_target = abs(rec.target_probability - top.target_probability) <= eps
    same_route = abs(rec.plan_probability - top.plan_probability) <= eps
    if same_target and same_route:
        return "CLOSE"
    if same_target:
        return "LESS_FLEXIBILITY"
    return "FALLBACK"


def visual_status_style(status: str) -> tuple[str, str, str]:
    """Return (short label, button background, text/tree tag)."""
    return {
        "BEST": ("BEST", "#63d47b", "green"),
        "EXACT_TIE": ("EQUAL", "#63d47b", "green"),
        "CLOSE": ("CLOSE", "#f5dd62", "yellow"),
        "LESS_FLEXIBILITY": ("LESS FLEX", "#f2a65a", "orange"),
        "FORCED_SLACK": ("FORCED SLACK", "#e5b95c", "amber"),
        "FALLBACK": ("FALLBACK", "#b996e8", "purple"),
    }[status]
