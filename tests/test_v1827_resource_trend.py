from __future__ import annotations

from pathlib import Path


from vision_engine.diagnostics import resource_trend as rt
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


def test_trend_tracks_private_memory_growth():
    rows = _rows({"private_mb": [100.0, 170.0, 260.0]})
    out = rt.trend_summary(rows)
    assert "private_mb" in out["leak_suspected"]
