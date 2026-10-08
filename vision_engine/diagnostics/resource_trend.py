"""Passive resource-trend sampler (V18.0.27 diagnostics).

A leak shows up as a *curve*, not as one number at the moment of failure. The
sampler records GDI/USER objects, kernel handles and memory of this process
into its own bounded in-memory ring (no disk I/O, never raises, microseconds
per call) so a failure snapshot can show how the counters evolved.
"""
from __future__ import annotations

from typing import Any

from .ring_buffer import InMemoryDiagnostics

RESOURCE_TREND = InMemoryDiagnostics(capacity=4096)

METRICS = ("gdi", "user", "handles", "working_set_mb", "private_mb")
# Minimum growth over the observed window to call a metric "growing".
GROWTH_THRESHOLDS = {"gdi": 50.0, "user": 20.0, "handles": 200.0, "working_set_mb": 150.0, "private_mb": 150.0}
MIN_NONDECREASING_FRACTION = 0.70


def sample_resources(label: str, **extra: object) -> dict[str, Any]:
    """Record one sample; safe to call from any thread, never raises."""
    row: dict[str, Any] = {"label": str(label)}
    try:
        from .win32_resources import collect_gui_resources, collect_process_vitals

        gui = collect_gui_resources()
        vit = collect_process_vitals()
        row.update(
            gdi=gui.get("gdi"), user=gui.get("user"),
            gdi_peak=gui.get("gdi_peak"), user_peak=gui.get("user_peak"),
            handles=vit.get("handles"), working_set_mb=vit.get("working_set_mb"),
            private_mb=vit.get("private_mb"),
        )
    except BaseException as exc:  # pragma: no cover - defensive
        row["error"] = f"{type(exc).__name__}: {exc}"
    row.update(extra)
    try:
        RESOURCE_TREND.record("resource-sample", **row)
    except BaseException:  # pragma: no cover
        pass
    return row


def trend_rows(max_rows: int = 512) -> list[dict[str, Any]]:
    """Flat rows (first sample + the most recent ``max_rows - 1``)."""
    events = RESOURCE_TREND.snapshot()
    if len(events) > max_rows:
        events = (events[0],) + events[-(max_rows - 1):]
    return [{"sequence": e.sequence, "monotonic": e.monotonic, **e.data} for e in events]


def trend_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Growth per metric; flags a probable leak when it keeps rising."""
    out: dict[str, Any] = {"samples": len(rows), "metrics": {}, "leak_suspected": []}
    if len(rows) < 2:
        return out
    t0, t1 = float(rows[0].get("monotonic", 0.0)), float(rows[-1].get("monotonic", 0.0))
    minutes = max(1e-9, (t1 - t0) / 60.0)
    out["duration_s"] = round(t1 - t0, 2)
    for name in METRICS:
        series = [float(r[name]) for r in rows if isinstance(r.get(name), (int, float))]
        if len(series) < 2:
            continue
        steps = [b - a for a, b in zip(series, series[1:])]
        nondecreasing = sum(1 for s in steps if s >= 0) / len(steps)
        delta = series[-1] - series[0]
        info = {
            "first": series[0], "last": series[-1], "delta": round(delta, 2),
            "min": min(series), "max": max(series),
            "per_minute": round(delta / minutes, 2),
            "nondecreasing_fraction": round(nondecreasing, 3),
        }
        info["growing"] = bool(
            delta >= GROWTH_THRESHOLDS[name] and nondecreasing >= MIN_NONDECREASING_FRACTION
        )
        out["metrics"][name] = info
        if info["growing"]:
            out["leak_suspected"].append(name)
    return out
