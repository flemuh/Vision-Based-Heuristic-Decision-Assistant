from __future__ import annotations

import math
from pathlib import Path

from vision_engine.diagnostics.ring_buffer import InMemoryDiagnostics
from vision_engine.input.geometry import AutoPlayPoint
from vision_engine.input.motion import bezier_motion_points, linear_motion_points, motion_points
from vision_engine.live.input_exclusive import InputExclusiveCoordinator

ROOT = Path(__file__).parents[1]


def _progress_and_lateral(start: AutoPlayPoint, target: AutoPlayPoint, point: AutoPlayPoint):
    dx, dy = target.x - start.x, target.y - start.y
    length = math.hypot(dx, dy)
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux
    rx, ry = point.x - start.x, point.y - start.y
    return rx * ux + ry * uy, abs(rx * nx + ry * ny)


def test_v1826_bezier_is_conservative_progressive_and_exact():
    start = AutoPlayPoint(900, 250)
    target = AutoPlayPoint(420, 510)
    points = bezier_motion_points(start, target, 24)
    assert points[-1] == target
    assert 12 <= len(points) <= 25
    previous = -1.0
    lateral_seen = 0.0
    for point in points:
        progress, lateral = _progress_and_lateral(start, target, point)
        assert progress + 1.5 >= previous
        assert lateral <= 31.0
        previous = max(previous, progress)
        lateral_seen = max(lateral_seen, lateral)
    assert lateral_seen >= 2.0


def test_integrity_patch_motion_default_uses_v11_linear_smoothstep():
    start = AutoPlayPoint(10, 10)
    target = AutoPlayPoint(410, 110)
    assert motion_points(start, target, 20) == linear_motion_points(start, target, 20)
    assert bezier_motion_points(start, target, 20)[-1] == target


def test_v1826_win32_diagnostics_are_memory_only_and_bounded():
    diagnostics = InMemoryDiagnostics(capacity=16)
    for i in range(30):
        diagnostics.record("step", i=i)
    events = diagnostics.snapshot()
    assert len(events) == 16
    assert events[-1].data["i"] == 29
    src = (ROOT / "vision_engine" / "input" / "win32.py").read_text(encoding="utf-8")
    assert ".open(" not in src
    assert "cv2.imwrite" not in src
    assert "record_win32" in src


class FakeBridge:
    def __init__(self, calls):
        self.calls = calls
        self.admission = type("A", (), {"open": lambda _self: None})()

    def prepare_input_exclusive(self, timeout):
        self.calls.append(("solver-freeze", timeout))
        return {
            "ok": True, "idle": True, "exclusive_token": "token",
            "foreground": 0, "background": 0, "bridge_active": 0,
        }

    def release_input_exclusive(self, token):
        self.calls.append(("solver-release", token))
        return {"ok": True}


class FakeMonitor:
    def __init__(self, calls, ok=True):
        self.calls = calls
        self.ok = ok

    def request_quiesce(self, timeout):
        self.calls.append(("monitor-quiesce", timeout))
        return self.ok

    def release(self):
        self.calls.append(("monitor-release",))

    def force_release(self):
        self.calls.append(("monitor-force-release",))


def test_v1826_input_exclusive_release_order_is_monitor_server_bridge():
    calls = []
    coordinator = InputExclusiveCoordinator(FakeBridge(calls), FakeMonitor(calls), remote_solver=True)
    lease = coordinator.acquire(solver_timeout=5.0, monitor_timeout=2.0)
    assert calls == [("solver-freeze", 5.0), ("monitor-quiesce", 2.0)]
    lease.release()
    assert calls[-2:] == [("monitor-release",), ("solver-release", "token")]


def test_v1826_monitor_quiesce_failure_releases_solver_token():
    calls = []
    coordinator = InputExclusiveCoordinator(FakeBridge(calls), FakeMonitor(calls, ok=False), remote_solver=True)
    try:
        coordinator.acquire(solver_timeout=5.0, monitor_timeout=2.0)
    except RuntimeError as exc:
        assert "monitor" in str(exc)
    else:
        raise AssertionError("expected monitor quiesce failure")
    assert calls[-1] == ("solver-release", "token")
