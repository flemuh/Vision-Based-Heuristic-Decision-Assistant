from __future__ import annotations

from pathlib import Path

import pytest

from vision_engine.diagnostics import resource_trend as rt
from vision_engine.diagnostics.input_matrix import MatrixConfig, MatrixRunner
from vision_engine.diagnostics.waiting import WaitingDiagnostics

ROOT = Path(__file__).resolve().parents[1]


def _rows(series: dict[str, list[float]], step_s: float = 10.0) -> list[dict]:
    n = max(len(v) for v in series.values())
    return [
        {"monotonic": i * step_s, **{k: v[i] for k, v in series.items() if i < len(v)}}
        for i in range(n)
    ]


def test_trend_flags_monotonic_gdi_growth():
    rows = _rows({"gdi": [100 + 10 * i for i in range(20)], "user": [50] * 20})
    out = rt.trend_summary(rows)
    assert out["leak_suspected"] == ["gdi"]
    assert out["metrics"]["gdi"]["delta"] == 190
    assert out["metrics"]["user"]["growing"] is False


def test_trend_ignores_sawtooth_that_returns_to_baseline():
    saw = [100, 400, 120, 410, 110, 420, 105, 100]
    out = rt.trend_summary(_rows({"gdi": saw}))
    assert out["leak_suspected"] == []


def test_trend_is_safe_with_too_few_samples_or_missing_metrics():
    assert rt.trend_summary([])["metrics"] == {}
    assert rt.trend_summary(_rows({"gdi": [5]}))["leak_suspected"] == []
    assert rt.trend_summary(_rows({"handles": [1, None, 3]}))["samples"] == 3


def test_sample_resources_never_raises_and_records_to_own_ring():
    rt.RESOURCE_TREND.clear()
    row = rt.sample_resources("unit", games=2)
    assert row["label"] == "unit" and row["games"] == 2
    rows = rt.trend_rows()
    assert len(rows) == 1 and rows[0]["label"] == "unit"


def test_trend_rows_keep_first_sample_and_most_recent():
    rt.RESOURCE_TREND.clear()
    for i in range(50):
        rt.sample_resources(f"s{i}")
    rows = rt.trend_rows(max_rows=10)
    assert len(rows) == 10
    assert rows[0]["label"] == "s0" and rows[-1]["label"] == "s49"


def test_probe_snapshot_carries_resource_trend():
    rt.RESOURCE_TREND.clear()
    rt.sample_resources("snap")
    snap = WaitingDiagnostics().snapshot()
    assert isinstance(snap["resource_trend"], list) and snap["resource_trend"][0]["label"] == "snap"


def _runner(games=3, clicks=14, fail_at=None, avoid=None, seed=7):
    calls = {"n": 0}
    labels: list[str] = []

    def click(point, move_ms):
        calls["n"] += 1
        if fail_at is not None and calls["n"] == fail_at:
            raise RuntimeError("Windows blocked cursor movement")
        return "SetCursorPos"

    cfg = MatrixConfig(games=games, clicks_per_game=clicks, region=(100, 100, 1000, 800), seed=seed)
    runner = MatrixRunner(cfg, click_fn=click, sample_fn=lambda label, **kw: labels.append(label),
                          avoid_fn=avoid)
    return runner, labels


def _drive(runner):
    delays = []
    while True:
        d = runner.step()
        if d is None:
            return delays
        delays.append(d)


def test_matrix_completes_all_games_with_bounded_delays():
    runner, labels = _runner(games=3)
    delays = _drive(runner)
    s = runner.summary()
    assert s["completed"] and not s["failed"] and s["clicks_ok"] == 42
    assert len(delays) == 41 and all(900 <= d <= 1600 for d in delays)
    assert labels.count("game-end") == 3 and labels.count("click") == 42


def test_matrix_uses_distinct_cells_inside_each_game_and_stays_in_region():
    runner, _ = _runner(games=2)
    _drive(runner)
    for game in (1, 2):
        cells = [r["position"] for r in runner.rows if r["game"] == game]
        assert len(cells) == len(set(cells)) == 14
    # jitter is bounded to the cell, so every point stays inside the region
    assert all(100 <= r["x"] <= 1100 and 100 <= r["y"] <= 900 for r in runner.rows)


def test_matrix_records_failure_location_and_stops():
    runner, labels = _runner(games=3, fail_at=20)
    _drive(runner)
    s = runner.summary()
    assert s["failed"] and not s["completed"] and s["clicks_ok"] == 19
    assert (s["failure"]["game"], s["failure"]["click"]) == (2, 6)
    assert "failure" in labels
    assert runner.step() is None  # does not keep clicking after a failure


def test_matrix_stop_by_user_is_not_reported_as_a_failure():
    runner, _ = _runner(games=3)
    runner.step()
    runner.stop("closed")
    s = runner.summary()
    assert s["failed"] is False and s["completed"] is False


def test_matrix_avoids_its_own_window():
    rect = (100, 100, 600, 500)  # covers a big part of the region
    runner, _ = _runner(games=2, avoid=lambda: rect)
    _drive(runner)
    inside = [r for r in runner.rows if rect[0] <= r["x"] < rect[2] and rect[1] <= r["y"] < rect[3]]
    assert len(inside) < len(runner.rows) * 0.05
    assert runner.avoided > 0


def test_matrix_rejects_invalid_clicks_per_game():
    with pytest.raises(ValueError):
        _runner(clicks=26)


def test_gui_hooks_are_passive_samplers_only():
    gui = (ROOT / "vision_engine" / "gui.py").read_text(encoding="utf-8")
    assert 'sample_resources(\n            "board-advanced"' in gui
    assert 'sample_resources("autoplay-armed"' in gui
    # the movement path must stay untouched by the sampler
    start = gui.index("def _execute_autoplay_click")
    end = gui.index("def _autoplay_transition_watchdog")
    assert "sample_resources" not in gui[start:end]


def test_matrix_tool_sets_dpi_before_tk():
    text = (ROOT / "tools" / "win32_input_matrix.py").read_text(encoding="utf-8")
    assert text.index("enable_dpi_awareness()") < text.index("import tkinter")


def test_trend_tracks_private_memory_growth():
    rows = _rows({"private_mb": [100.0, 170.0, 260.0]})
    out = rt.trend_summary(rows)
    assert "private_mb" in out["leak_suspected"]
