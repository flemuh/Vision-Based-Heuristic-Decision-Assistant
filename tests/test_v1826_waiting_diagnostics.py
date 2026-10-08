from __future__ import annotations

import json
from pathlib import Path
import socket

from vision_engine.diagnostics.waiting import WaitingDiagnostics
from tools.capture_waiting_diagnostics import _classify


def test_waiting_probe_serves_runtime_events_and_deep_threads(tmp_path: Path):
    diag = WaitingDiagnostics(capacity=32)
    diag.set_provider(lambda: {"commit_phase": "await-confirm", "mask": 3, "mask_count": 2})
    diag.record("click-complete", generation=7)
    marker = tmp_path / "probe.json"
    port = diag.start(marker, preferred_port=57920)
    try:
        assert marker.exists()
        info = json.loads(marker.read_text(encoding="utf-8"))
        assert info["port"] == port
        with socket.create_connection(("127.0.0.1", port), timeout=1.0) as sock:
            sock.sendall(b'{"command":"deep_snapshot"}\n')
            data = b""
            while not data.endswith(b"\n"):
                data += sock.recv(65536)
        payload = json.loads(data.decode("utf-8"))
        assert payload["runtime"]["commit_phase"] == "await-confirm"
        assert any(e["kind"] == "click-complete" for e in payload["events"])
        assert payload["threads"]
    finally:
        diag.stop(marker)
    assert not marker.exists()


def test_classifier_identifies_clean_await_confirm_as_vision_stall():
    samples = []
    for i in range(4):
        samples.append({
            "probe": {"runtime": {
                "commit_phase": "await-confirm",
                "mask": 3,
                "monitor_heartbeat": float(i + 1),
                "monitor_quiesced": False,
                "monitor_requested": False,
                "bridge_admission": {"open": True, "active": 0},
                "autoplay_enabled": True,
                "current_jewel": "HA",
            }},
            "solver": {
                "frozen": False, "freeze_pending": False,
                "foreground": 0, "background": 0,
            },
        })
    result, details = _classify(samples)
    assert result == "AWAIT_CONFIRM_VISION_STALL"
    assert any("vision" in line.lower() for line in details)


def test_classifier_identifies_solver_fence_release_stuck():
    samples = []
    for i in range(4):
        samples.append({
            "probe": {"runtime": {
                "commit_phase": "await-confirm",
                "mask": 3,
                "monitor_heartbeat": float(i + 1),
                "monitor_quiesced": False,
                "monitor_requested": False,
                "bridge_admission": {"open": False, "active": 0},
                "autoplay_enabled": True,
                "current_jewel": "HA",
            }},
            "solver": {
                "frozen": True, "freeze_pending": False,
                "foreground": 0, "background": 0,
            },
        })
    result, _ = _classify(samples)
    assert result == "SOLVER_FENCE_RELEASE_STUCK"


def test_release_contains_one_click_waiting_diagnostic_script():
    root = Path(__file__).resolve().parents[1]
    assert (root / "diagnose_waiting.bat").exists()
    gui = (root / "vision_engine" / "gui.py").read_text(encoding="utf-8")
    assert "WaitingDiagnostics" in gui
    assert "transition-watchdog-armed" in gui
    assert "click-confirmed" in gui
