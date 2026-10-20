from __future__ import annotations

import sys


from vision_engine import dpi
from vision_engine.diagnostics import win32_resources


class _FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(0.0, seconds)


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
