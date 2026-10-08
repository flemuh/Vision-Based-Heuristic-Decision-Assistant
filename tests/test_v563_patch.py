from __future__ import annotations

from pathlib import Path

import pytest

from vision_engine.advisor import Advisor, Recommendation, Utility
from vision_engine.board import Board
from vision_engine.route_math import LuckySlackEstimate, best_lucky_slack
from vision_engine.solver_process import (
    read_solver_subprocess_result,
    start_solver_process,
    start_solver_subprocess,
)


def board() -> Board:
    return Board.from_rows([
        ["BL", "CR", "CH", "CH", "CR"],
        ["CR", "SO", "BL", "LI", "CR"],
        ["CH", "HA", None, "SO", "LI"],
        ["SO", "LI", "HA", "BL", "LI"],
        ["BL", "CH", "SO", "HA", "HA"],
    ])


def U(*, gt=0.10, ge=0.10, score=800.0, p3=0.02) -> Utility:
    return Utility(p3, 0.1, 0.1, 0.2, gt, ge, score)


def slack(remaining: int, p: float = 0.2) -> LuckySlackEstimate:
    used = max(0, 2 - remaining)
    return LuckySlackEstimate(("H", "V", "/"), used, used, remaining, True, p)


def test_three_lucky_slack_budget_is_exactly_two_off_route_placements():
    b = board()
    # Positions 1, 5 and 9 are outside every Lucky line.
    two = (1 << 1) | (1 << 5)
    three = two | (1 << 9)
    s2 = best_lucky_slack(b, two)
    s3 = best_lucky_slack(b, three)
    assert s2.viable and s2.slack_used == 2 and s2.slack_remaining == 0
    assert not s3.viable and s3.slack_remaining == 0


def test_score_guardrail_preserves_slack_when_primary_probability_is_tied():
    adv = Advisor(board(), mode="score", rollouts=4)
    keep = Recommendation(0, U(gt=.10, ge=.12, score=810), "mc", active_goal=">1000", lucky_slack=slack(2, .20))
    spend = Recommendation(1, U(gt=.10, ge=.40, score=950), "mc", active_goal=">1000", lucky_slack=slack(1, .40))
    adv._last_mc_samples = {
        0: [U(gt=.10), U(gt=.10), U(gt=.10), U(gt=.10)],
        1: [U(gt=.10), U(gt=.10), U(gt=.10), U(gt=.10)],
    }
    cmp, metric, primary_tied = adv._hierarchical_compare(keep, spend, ">1000", exact=False)
    assert primary_tied is True
    assert cmp > 0
    assert metric == "3L slack remaining"


def test_score_guardrail_never_overrides_a_clear_primary_score_win():
    adv = Advisor(board(), mode="score")
    keep = Recommendation(0, U(gt=.05, ge=.80, score=1000), "exact", active_goal=">1000", lucky_slack=slack(2, .90))
    spend = Recommendation(1, U(gt=.20, ge=.10, score=700), "exact", active_goal=">1000", lucky_slack=slack(0, .01))
    cmp, metric, primary_tied = adv._hierarchical_compare(spend, keep, ">1000", exact=True)
    assert primary_tied is False
    assert cmp > 0
    assert metric == "P(>1000)"


class _FakeQueue:
    def __init__(self):
        self.closed = False
        self.cancelled = False

    def cancel_join_thread(self):
        self.cancelled = True

    def close(self):
        self.closed = True


class _FailingProcess:
    def start(self):
        raise ValueError("Cannot open console input buffer for writing")

    def is_alive(self):
        return False


class _FailingContext:
    def __init__(self):
        self.q = _FakeQueue()

    def Queue(self, maxsize=1):
        return self.q

    def Process(self, **kwargs):
        return _FailingProcess()


def test_failed_windows_spawn_cleans_queue_before_propagating():
    ctx = _FailingContext()
    with pytest.raises(ValueError, match="console input buffer"):
        start_solver_process(ctx, {"board_cells": []}, "fast", 3)
    assert ctx.q.cancelled and ctx.q.closed


def test_plain_subprocess_fallback_returns_solver_result(tmp_path):
    b = board()
    mask13 = sum(1 << i for i in [2, 4, 6, 7, 8, 9, 10, 11, 13, 14, 16, 19, 22])
    request = {
        "board_cells": list(b.cells), "mask": mask13, "jewel": "CH", "mode": "score",
        "exact_horizon": 6, "rollout_exact_horizon": 3, "rollouts": 20,
        "rng_seed": 1337, "bias_weights": [1.0] * 6, "learned_scores": {},
    }
    # The child must import the real package, so use the repository root while
    # still verifying that request/result files are self-cleanable artifacts.
    root = Path(__file__).parents[1]
    proc, req_path, out_path = start_solver_subprocess(request, root, "test", 13)
    assert proc.wait(timeout=20) == 0
    kind, payload = read_solver_subprocess_result(out_path)
    assert kind == "ok"
    assert payload and payload[0].utility.expected_score == 762.0
    req_path.unlink(missing_ok=True)
    out_path.unlink(missing_ok=True)


def test_gui_patch_hides_legacy_troubleshooting_and_never_paints_fast_ranking():
    source = (Path(__file__).parents[1] / "vision_engine" / "gui.py").read_text(encoding="utf-8")
    assert 'text="Troubleshooting' not in source
    assert 'text="Save game manually"' not in source
    assert "FAST complete for {jewel}; refining final ranking" in source
    assert "_render_recommendations(mask, jewel, recs, refined=False)" not in source
    assert "self.phase == GamePhase.COMPLETE" in source
    assert "_show_game_complete_summary" in source

