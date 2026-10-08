from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

import vision_engine.live.solver_bridge as solver_bridge_module
from vision_engine.input_gate import MonitorInputGate
from vision_engine.live.input_exclusive import InputExclusiveCoordinator
from vision_engine.live.solver_bridge import LiveSolverBridge
from vision_engine.solver.client import SolverClient
from vision_engine.solver.service import SolverEngine, SolverTCPServer


def _serve_engine():
    engine = SolverEngine(workers=2, cache_entries=32)
    server = SolverTCPServer(("127.0.0.1", 0), engine, socket_timeout=10.0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    return engine, server, thread


def _start_bridge(port: int, session_id: str) -> LiveSolverBridge:
    bridge = LiveSolverBridge(
        Path("."), host="127.0.0.1", port=port, auto_start_wsl=False,
        session_id=session_id, session_timeout=1.0,
    )
    ok, status = bridge._start_sync()
    bridge.available = ok
    bridge.status = status
    assert ok, status
    return bridge


def test_v1826_complete_input_exclusive_sequence_real_server_and_monitor():
    engine, server, server_thread = _serve_engine()
    host, port = server.server_address
    observer = SolverClient(host, port, timeout=3)
    bridge = _start_bridge(port, "e2e-session")
    monitor = MonitorInputGate()
    coordinator = InputExclusiveCoordinator(bridge, monitor, remote_solver=True)
    monitor_done = threading.Event()
    events: list[str] = []

    def monitor_worker() -> None:
        deadline = time.monotonic() + 2.0
        while not monitor.requested and time.monotonic() < deadline:
            time.sleep(0.002)
        if monitor.requested:
            events.append("monitor-safe-point")
            monitor.monitor_safe_point(running=True)
        monitor_done.set()

    worker = threading.Thread(target=monitor_worker, daemon=True)
    worker.start()
    try:
        lease = coordinator.acquire(solver_timeout=2.0, monitor_timeout=1.0)
        status = observer.status()
        assert bridge.admission.snapshot()["open"] is False
        assert status["frozen"] is True
        assert status["foreground"] == 0
        assert status["background"] == 0
        assert monitor.quiesced is True

        # Atomic validation -> trajectory -> click are represented explicitly;
        # no solver admission is allowed inside this physical-input window.
        events.extend(["atomic-validation", "bezier-move", "click"])
        assert bridge.admission.ticket("forbidden-during-input") is None

        released = lease.release()
        assert released["ok"] is True
        assert monitor_done.wait(1.0)
        final = observer.status()
        assert final["frozen"] is False
        assert final["freeze_pending"] is False
        assert final["foreground"] == 0
        assert final["background"] == 0
        assert final["admission_open"] is True
        assert bridge.admission.snapshot()["open"] is True
        assert monitor.requested is False
        assert monitor.quiesced is False
        assert events == ["monitor-safe-point", "atomic-validation", "bezier-move", "click"]
    finally:
        bridge.close()
        observer.close()
        server.shutdown(); server.server_close(); engine.close()
        server_thread.join(timeout=2.0)


class RecordingBridge:
    def __init__(self, events: list[str], *, fail_release_once: bool = False):
        self.events = events
        self.open = True
        self.fail_release_once = fail_release_once
        self.release_calls = 0

    def prepare_input_exclusive(self, timeout: float):
        self.events.append("solver-freeze")
        self.open = False
        return {
            "ok": True, "idle": True, "exclusive_token": "tok",
            "foreground": 0, "background": 0, "bridge_active": 0,
        }

    def release_input_exclusive(self, token: str):
        self.release_calls += 1
        self.events.append("solver-release")
        if self.fail_release_once and self.release_calls == 1:
            raise RuntimeError("transient release failure")
        assert token == "tok"
        self.open = True
        return {"ok": True}


class RecordingMonitor:
    def __init__(self, events: list[str], *, fail_release_once: bool = False):
        self.events = events
        self.quiesced = False
        self.fail_release_once = fail_release_once
        self.release_calls = 0

    def request_quiesce(self, timeout: float):
        self.events.append("monitor-quiesce")
        self.quiesced = True
        return True

    def release(self):
        self.release_calls += 1
        self.events.append("monitor-release")
        if self.fail_release_once and self.release_calls == 1:
            raise RuntimeError("transient monitor release failure")
        self.quiesced = False

    def force_release(self):
        self.events.append("monitor-force-release")
        self.quiesced = False


@pytest.mark.parametrize("failure_stage", ["atomic-validation", "bezier-move", "click"])
def test_v1826_failure_inside_input_window_always_releases_both_fences(failure_stage: str):
    events: list[str] = []
    bridge = RecordingBridge(events)
    monitor = RecordingMonitor(events)
    coordinator = InputExclusiveCoordinator(bridge, monitor, remote_solver=True)  # type: ignore[arg-type]
    lease = coordinator.acquire(solver_timeout=1.0, monitor_timeout=1.0)

    try:
        for stage in ("atomic-validation", "bezier-move", "click"):
            events.append(stage)
            if stage == failure_stage:
                raise RuntimeError(f"injected:{stage}")
    except RuntimeError:
        pass
    finally:
        lease.release()

    assert monitor.quiesced is False
    assert bridge.open is True
    assert events[-2:] == ["monitor-release", "solver-release"]


def test_v1826_monitor_release_error_still_releases_solver_and_force_releases_monitor():
    events: list[str] = []
    bridge = RecordingBridge(events)
    monitor = RecordingMonitor(events, fail_release_once=True)
    coordinator = InputExclusiveCoordinator(bridge, monitor, remote_solver=True)  # type: ignore[arg-type]
    lease = coordinator.acquire(solver_timeout=1.0, monitor_timeout=1.0)

    with pytest.raises(RuntimeError, match="monitor-release"):
        lease.release()

    assert monitor.quiesced is False
    assert bridge.open is True
    assert "monitor-force-release" in events
    assert events[-1] == "solver-release"
    # The resource release completed despite the diagnostic exception; a retry
    # simply finalizes the lease bookkeeping and is idempotent afterwards.
    assert lease.release()["ok"] is True
    assert lease.release()["reason"] == "already-released"


def test_v1826_failed_solver_release_keeps_token_retryable_until_success():
    events: list[str] = []
    bridge = RecordingBridge(events, fail_release_once=True)
    monitor = RecordingMonitor(events)
    coordinator = InputExclusiveCoordinator(bridge, monitor, remote_solver=True)  # type: ignore[arg-type]
    lease = coordinator.acquire(solver_timeout=1.0, monitor_timeout=1.0)

    with pytest.raises(RuntimeError, match="solver-release"):
        lease.release()
    assert bridge.open is False
    assert lease.solver_token == "tok"
    assert lease.released is False

    assert lease.release()["ok"] is True
    assert bridge.open is True
    assert lease.solver_token is None
    assert lease.released is True


def test_v1826_bridge_release_uses_independent_control_connection_before_reopening(monkeypatch):
    events: list[str] = []

    class BrokenWorkerClient:
        def release(self, session_id: str, token: str):
            events.append("worker-release-failed")
            raise OSError("worker socket lost")

    class ControlClient:
        def __init__(self, *_args, **_kwargs):
            events.append("control-open")

        def release(self, session_id: str, token: str):
            events.append(f"control-release:{session_id}:{token}")
            return {"ok": True, "admission_open": True}

        def close(self):
            events.append("control-close")

    bridge = LiveSolverBridge(Path("."), auto_start_wsl=False, session_id="fallback-session")
    monkeypatch.setattr(solver_bridge_module, "SolverClient", ControlClient)
    events.clear()
    bridge.client = BrokenWorkerClient()  # type: ignore[assignment]
    bridge.admission.close()
    bridge._exclusive_token = "tok"
    try:
        result = bridge.release_input_exclusive("tok")
        assert result["ok"] is True
        assert events == [
            "worker-release-failed", "control-open",
            "control-release:fallback-session:tok", "control-close",
        ]
        assert bridge._exclusive_token is None
        assert bridge.admission.snapshot()["open"] is True
    finally:
        bridge.executor.shutdown(wait=False, cancel_futures=True)


def test_v1826_server_release_rejection_is_fail_closed():
    class RejectingClient:
        def release(self, _session_id: str, _token: str):
            return {"ok": False, "reason": "token-mismatch"}

    bridge = LiveSolverBridge(Path("."), auto_start_wsl=False, session_id="reject-session")
    bridge.client = RejectingClient()  # type: ignore[assignment]
    bridge.admission.close()
    bridge._exclusive_token = "tok"
    try:
        result = bridge.release_input_exclusive("tok")
        assert result["ok"] is False
        assert bridge._exclusive_token == "tok"
        assert bridge.admission.snapshot()["open"] is False
    finally:
        bridge.executor.shutdown(wait=False, cancel_futures=True)
