from __future__ import annotations

import ctypes
import sys
from unittest import mock

import pytest

from vision_engine import dpi
from vision_engine.diagnostics import win32_probe, win32_resources
from vision_engine.diagnostics.win32_boundary import classify_boundary_snapshot
from vision_engine.input import win32 as input_win32
from vision_engine.input.geometry import AutoPlayPoint


class _FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(0.0, seconds)


def _probe(results: list[bool], errors: list[int] | None = None, physical: list[bool] | None = None):
    fake = _FakeClock()
    calls = iter(results)
    err_iter = iter(errors or [5] * len(results))
    physical_iter = iter(physical if physical is not None else results)
    last = {"err": 0, "cursor": (900, 367)}

    def set_cursor(x, y):
        ok = next(calls)
        reaches = next(physical_iter)
        last["err"] = 0 if ok else next(err_iter)
        if reaches:
            last["cursor"] = (x, y)
        else:
            last["cursor"] = (900, 367)
        return ok

    return win32_probe.run_rejection_probe(
        target=(803, 367),
        set_cursor=set_cursor,
        get_cursor=lambda: last["cursor"],
        get_last_error=lambda: last["err"],
        sleep=fake.sleep,
        clock=fake.clock,
    ), fake


def test_probe_transient_recovers_and_reports_first_recovery():
    result, fake = _probe([False, False, True, True])
    assert result["classification"] == win32_probe.TRANSIENT
    assert result["first_recovery_ms"] == 200
    assert [a["planned_delay_ms"] for a in result["attempts"]] == [10, 50, 200, 1000]
    assert result["failure_error_codes"] == [5]


def test_probe_persistent_when_nothing_recovers():
    result, _ = _probe([False] * 4, errors=[5, 5, 1400, 5])
    assert result["classification"] == win32_probe.PERSISTENT
    assert result["first_recovery_ms"] is None
    assert result["failure_error_codes"] == [5, 1400]


def test_probe_intermittent_when_last_attempt_fails_again():
    result, _ = _probe([False, True, False, False])
    assert result["classification"] == win32_probe.PARTIAL



def test_probe_uses_physical_cursor_not_boolean_return():
    result, _ = _probe([False, False, False, False], errors=[5, 5, 5, 5], physical=[False, True, True, True])
    assert result["classification"] == win32_probe.TRANSIENT
    assert result["first_recovery_ms"] == 50


def test_probe_delays_are_absolute_from_failure_not_cumulative():
    result, fake = _probe([False] * 4)
    assert fake.now - 100.0 == pytest.approx(1.0, abs=1e-6)


def test_gui_pressure_threshold():
    ok = win32_resources.gui_pressure(2000, 900)
    assert not ok["pressure"] and ok["gdi_fraction"] == 0.2
    hot = win32_resources.gui_pressure(8500, 100)
    assert hot["pressure"] and hot["pressured_pools"] == ["GDI"]
    both = win32_resources.gui_pressure(9000, 9900, gdi_quota=10000, user_quota=10000)
    assert both["pressured_pools"] == ["GDI", "USER"]


def test_process_summary_filters_related_and_ranks_by_handles():
    rows = [
        {"pid": 10, "ppid": 1, "name": "python.exe", "session": 1, "gdi": 120, "user": 60},
        {"pid": 11, "ppid": 1, "name": "python.exe", "session": 1, "gdi": 9000, "user": 70},
        {"pid": 12, "ppid": 1, "name": "wslhost.exe", "session": 1, "gdi": None, "user": None},
        {"pid": 13, "ppid": 1, "name": "explorer.exe", "session": 1, "gdi": 3000, "user": 1800},
        {"pid": 14, "ppid": 1, "name": "python.exe", "session": 0, "gdi": 99999, "user": 99999},
    ]
    out = win32_resources.summarize_processes(rows, own_pid=10, session_id=1)
    assert out["other_python_pids"] == [11, 14]
    assert {r["pid"] for r in out["related"]} == {10, 11, 12, 14}
    assert [r["pid"] for r in out["top_gdi"]][:2] == [11, 13]  # session 0 excluded
    assert out["top_user"][0]["pid"] == 13


def _failed_snapshot(**extra):
    snap = {
        "clip_rect": [0, 0, 1920, 1080],
        "target_xy": [803, 367],
        "actual_xy": [903, 367],
        "process_desktop": "Default",
        "input_desktop": "Default",
        "assistant": {"integrity": "medium"},
        "points": {"target": {"pid": 2, "root_hwnd": 20, "process": {"integrity": "medium"}}},
        "set_cursor_ok": False,
        "sendinput_ok": False,
    }
    snap.update(extra)
    return snap


def test_classifies_gui_resource_pressure_when_both_methods_fail():
    gui = win32_resources.gui_pressure(9500, 200)
    kind, details = classify_boundary_snapshot(_failed_snapshot(gui_resources=gui))
    assert kind == "GUI_RESOURCE_PRESSURE"
    assert any("GDI 9500/10000" in line for line in details)


def test_resource_pressure_does_not_override_working_sendinput():
    gui = win32_resources.gui_pressure(9500, 200)
    kind, _ = classify_boundary_snapshot(_failed_snapshot(gui_resources=gui, sendinput_ok=True))
    assert kind == "SETCURSORPOS_REJECTED_SENDINPUT_WORKED"


def test_desktop_mismatch_still_wins_over_resource_pressure():
    gui = win32_resources.gui_pressure(9500, 200)
    kind, _ = classify_boundary_snapshot(
        _failed_snapshot(gui_resources=gui, input_desktop="Winlogon")
    )
    assert kind == "INPUT_DESKTOP_MISMATCH"


def test_unaware_thread_is_called_out_in_fallthrough():
    kind, details = classify_boundary_snapshot(_failed_snapshot(dpi={"thread_awareness": 0}))
    assert kind == "WIN32_INPUT_REJECTED"
    assert any("DPI-UNAWARE" in line for line in details)


def test_dpi_bootstrap_is_safe_and_idempotent_off_windows():
    first = dpi.enable_dpi_awareness()
    second = dpi.enable_dpi_awareness()
    assert first == second
    if not sys.platform.startswith("win"):
        assert first["method"] == "not-windows"
        assert dpi.thread_dpi_report()["error"] == "not-windows"


def test_app_bootstraps_dpi_before_any_other_import():
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    body = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    assert body[0] == "from vision_engine.dpi import enable_dpi_awareness"
    assert body[1] == "enable_dpi_awareness()"


class _RejectingUser32:
    def SetCursorPos(self, x, y):  # noqa: N802 - Win32 name
        return False


def test_move_loop_tolerates_intermediate_rejections_and_only_aborts_at_final_target():
    start = AutoPlayPoint(100, 100)
    target = AutoPlayPoint(400, 300)
    with mock.patch.object(input_win32.sys, "platform", "win32"), \
         mock.patch.object(input_win32, "_windows_user32", return_value=_RejectingUser32()), \
         mock.patch.object(input_win32, "_cursor_position", return_value=start), \
         mock.patch.object(input_win32, "_sendinput_move", return_value=False), \
         mock.patch.object(input_win32, "capture_boundary_snapshot", return_value={}), \
         mock.patch.object(input_win32.time, "sleep"), \
         mock.patch.object(ctypes, "set_last_error", create=True), \
         mock.patch.object(ctypes, "get_last_error", create=True, return_value=5):
        from vision_engine.diagnostics.win32 import WIN32_DIAGNOSTICS

        WIN32_DIAGNOSTICS.clear()
        with pytest.raises(RuntimeError) as exc:
            input_win32.windows_move_and_left_click(target, duration_ms=64)
    assert "final click target" in str(exc.value)
    assert "PERSISTENT_REJECTION" in str(exc.value)
    events = WIN32_DIAGNOSTICS.snapshot()
    kinds = [e.kind for e in events]
    # Intermediate failures are recorded but are explicitly non-fatal.
    rejected = [e for e in events if e.kind == "move-step-api-rejected"]
    assert len(rejected) >= 2
    assert all(e.data["fatal"] is False for e in rejected)
    assert kinds.index("move-final-check") < kinds.index("move-recovery-probe") < kinds.index("move-abort")
    probe_event = next(e for e in events if e.kind == "move-recovery-probe")
    assert len(probe_event.data["attempts"]) == 4


def test_move_loop_does_not_physically_verify_each_intermediate_waypoint():
    start = AutoPlayPoint(100, 100)
    target = AutoPlayPoint(400, 300)
    fake_user32 = type("U", (), {"SetCursorPos": lambda self, x, y: True})()
    positions = [start, target]  # start read + one final verification read
    with mock.patch.object(input_win32.sys, "platform", "win32"), \
         mock.patch.object(input_win32, "_windows_user32", return_value=fake_user32), \
         mock.patch.object(input_win32, "_cursor_position", side_effect=positions) as cursor_pos, \
         mock.patch.object(input_win32, "_sendinput_left_click", return_value=True), \
         mock.patch.object(input_win32.time, "sleep"), \
         mock.patch.object(ctypes, "set_last_error", create=True), \
         mock.patch.object(ctypes, "get_last_error", create=True, return_value=0):
        backend = input_win32.windows_move_and_left_click(target, duration_ms=64)
    assert backend == "SetCursorPos"
    assert cursor_pos.call_count == 2


def test_move_loop_allows_final_probe_to_recover_before_click():
    start = AutoPlayPoint(100, 100)
    target = AutoPlayPoint(400, 300)
    wrong = AutoPlayPoint(330, 260)
    fake_user32 = type("U", (), {"SetCursorPos": lambda self, x, y: True})()
    with mock.patch.object(input_win32.sys, "platform", "win32"), \
         mock.patch.object(input_win32, "_windows_user32", return_value=fake_user32), \
         mock.patch.object(input_win32, "_cursor_position", side_effect=[start, wrong, target]), \
         mock.patch.object(input_win32, "_probe_rejection", return_value={"classification": "TRANSIENT_REJECTION"}), \
         mock.patch.object(input_win32, "_sendinput_left_click", return_value=True) as click, \
         mock.patch.object(input_win32.time, "sleep"), \
         mock.patch.object(ctypes, "set_last_error", create=True), \
         mock.patch.object(ctypes, "get_last_error", create=True, return_value=0):
        backend = input_win32.windows_move_and_left_click(target, duration_ms=64)
    assert backend == "delayed final recovery"
    click.assert_called_once()


def test_process_summary_reports_mu_aggregate_handles():
    rows = [
        {"pid": 101, "ppid": 1, "name": "main.exe", "session": 1, "gdi": 120, "user": 80, "handles": 500, "path": r"C:\\MU\\main.exe"},
        {"pid": 102, "ppid": 1, "name": "main.exe", "session": 1, "gdi": 130, "user": 90, "handles": 550, "path": r"C:\\MU2\\main.exe"},
        {"pid": 103, "ppid": 1, "name": "explorer.exe", "session": 1, "gdi": 300, "user": 200, "handles": 900, "path": r"C:\\Windows\\explorer.exe"},
    ]
    out = win32_resources.summarize_processes(rows, own_pid=999, session_id=1)
    assert out["mu_like_count"] == 2
    assert out["mu_like_totals"] == {"gdi": 250, "user": 170, "handles": 1050}
    assert out["top_handles"][0]["pid"] == 103
