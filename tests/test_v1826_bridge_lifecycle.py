from __future__ import annotations

from concurrent.futures import Future
import threading
from pathlib import Path
import time

import pytest

import vision_engine.live.solver_bridge as solver_bridge_module
from vision_engine.live.solver_bridge import LiveSolverBridge
from vision_engine.live.state import GameState
from vision_engine.solver.client import SolverClient
from vision_engine.solver.protocol import SolverRequest, SolverSettings
from vision_engine.solver.service import SolverEngine, SolverTCPServer


BOARD = (
    "BL", "CR", "CH", "CH", "CR",
    "CR", "SO", "BL", "LI", "CR",
    "CH", "HA", None, "SO", "LI",
    "SO", "LI", "HA", "BL", "LI",
    "BL", "CH", "SO", "HA", "HA",
)


class ControlledRunningExecutor:
    def __init__(self) -> None:
        self.futures: list[Future] = []
        self._lock = threading.Lock()

    def submit(self, fn, *args, **kwargs):
        future = Future()
        assert future.set_running_or_notify_cancel() is True
        with self._lock:
            self.futures.append(future)
        return future

    def complete_one(self, result=()) -> None:
        with self._lock:
            future = next(f for f in self.futures if not f.done())
        future.set_result(result)

    def shutdown(self, wait=True, cancel_futures=False):
        return None


def _state(jewel: str = "BL") -> GameState:
    return GameState(
        episode_id="ep-lifecycle",
        state_version=1,
        strategy_version=0,
        board_hash="board-lifecycle",
        board_cells=BOARD,
        accepted_mask=(1 << 4) | (1 << 13),
        current_jewel=jewel,
        strategy="score",
    )


def _request(request_id: int, session_id: str, jewel: str = "BL") -> SolverRequest:
    return SolverRequest(
        request_id=request_id,
        state=_state(jewel),
        settings=SolverSettings(rollouts=1, exact_horizon=1, rollout_exact_horizon=1),
        purpose="bridge-lifecycle",
        session_id=session_id,
    )


def _wait_until(predicate, timeout: float = 1.0, message: str = "condition not reached"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.005)
    raise AssertionError(message)


def _serve_engine(engine: SolverEngine, port: int = 0):
    server = SolverTCPServer(("127.0.0.1", port), engine, socket_timeout=10.0)

    def run() -> None:
        try:
            server.serve_forever(poll_interval=0.02)
        finally:
            server.server_close()
            engine.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return server, thread


def _start_bridge(port: int, session_id: str, *, session_timeout: float = 0.5) -> LiveSolverBridge:
    bridge = LiveSolverBridge(
        Path("."), host="127.0.0.1", port=port, auto_start_wsl=False,
        session_id=session_id, session_timeout=session_timeout,
    )
    ok, status = bridge._start_sync()
    bridge.available = ok
    bridge.status = status
    assert ok, status
    return bridge


def test_v1826_close_detaches_session_even_when_bridge_marked_unavailable():
    engine = SolverEngine(workers=3, cache_entries=32)
    server, thread = _serve_engine(engine)
    host, port = server.server_address
    observer = SolverClient(host, port, timeout=3)
    old = _start_bridge(port, "old-session")
    new: LiveSolverBridge | None = None
    try:
        assert observer.status()["session_id"] == "old-session"
        old.available = False  # transient worker failure must not skip detach
        old.close()

        detached = observer.status()
        assert detached["session_id"] == "old-session"
        assert detached["admission_open"] is False
        assert detached["frozen"] is False
        assert detached["freeze_pending"] is False
        assert detached["idle"] is True

        new = _start_bridge(port, "new-session")
        rotated = observer.status()
        assert rotated["session_id"] == "new-session"
        assert rotated["admission_open"] is True
        assert rotated["foreground"] == 0
        assert rotated["background"] == 0
    finally:
        if new is not None:
            new.close()
        observer.close()
        server.shutdown()
        thread.join(timeout=2.0)


def test_v1826_close_uses_control_connection_while_worker_ipc_is_busy():
    engine = SolverEngine(workers=3, cache_entries=32)
    engine.foreground.shutdown(wait=True, cancel_futures=True)
    controlled = ControlledRunningExecutor()
    engine.foreground = controlled  # type: ignore[assignment]
    server, thread = _serve_engine(engine)
    host, port = server.server_address
    observer = SolverClient(host, port, timeout=3)
    bridge = _start_bridge(port, "closing-session", session_timeout=1.0)
    close_done = threading.Event()
    try:
        bridge.solve_async(_request(1, "closing-session"), lambda _r: None, lambda _e: None)
        _wait_until(lambda: observer.status()["foreground"] == 1, message="foreground never became active")

        def close_bridge() -> None:
            bridge.close()
            close_done.set()

        closer = threading.Thread(target=close_bridge, daemon=True)
        closer.start()

        # The lifecycle control connection must reach the server even though
        # the worker SolverClient socket is still blocked on the old solve.
        _wait_until(lambda: observer.status()["freeze_pending"] is True, timeout=2.0,
                    message="end_session never closed server admission")
        held = observer.status()
        assert held["admission_open"] is False
        assert held["foreground"] == 1
        assert close_done.is_set() is False

        controlled.complete_one(())
        assert close_done.wait(2.0)
        final = observer.status()
        assert final["foreground"] == 0
        assert final["background"] == 0
        assert final["freeze_pending"] is False
        assert final["frozen"] is False
        assert final["admission_open"] is False
    finally:
        observer.close()
        server.shutdown()
        thread.join(timeout=2.0)


def test_v1826_graceful_close_revokes_its_own_exclusive_token():
    engine = SolverEngine(workers=3, cache_entries=32)
    server, thread = _serve_engine(engine)
    host, port = server.server_address
    observer = SolverClient(host, port, timeout=3)
    bridge = _start_bridge(port, "owner-session")
    successor: LiveSolverBridge | None = None
    try:
        frozen = bridge.prepare_input_exclusive(timeout=1.0)
        assert frozen["ok"] is True
        assert observer.status()["frozen"] is True

        # Closing the GUI while it owns the token is a graceful detach, not a
        # permanent server freeze. end_session is allowed to revoke own token.
        bridge.close()
        after = observer.status()
        assert after["frozen"] is False
        assert after["freeze_pending"] is False
        assert after["idle"] is True
        assert after["admission_open"] is False

        successor = _start_bridge(port, "successor-session")
        assert observer.status()["session_id"] == "successor-session"
        assert observer.status()["admission_open"] is True
    finally:
        if successor is not None:
            successor.close()
        observer.close()
        server.shutdown()
        thread.join(timeout=2.0)


def test_v1826_crashed_owner_triggers_controlled_service_recovery(monkeypatch):
    old_engine = SolverEngine(workers=3, cache_entries=32)
    old_server, old_thread = _serve_engine(old_engine)
    host, port = old_server.server_address
    old = SolverClient(host, port, timeout=3)
    replacement: dict[str, object] = {}

    assert old.begin_session("crashed-session", 1.0)["ok"]
    frozen = old.freeze("crashed-session", 1.0)
    assert frozen["ok"] is True
    assert frozen["frozen"] is True

    def fake_ensure(root_dir, host_arg="127.0.0.1", port_arg=57641, workers=6, wait_seconds=5.0, distro="Ubuntu"):
        assert host_arg == host
        assert port_arg == port
        # _start_sync has already requested shutdown of the stuck service.
        old_thread.join(timeout=2.0)
        assert old_thread.is_alive() is False
        new_engine = SolverEngine(workers=3, cache_entries=32)
        new_server, new_thread = _serve_engine(new_engine, port=port)
        replacement.update(engine=new_engine, server=new_server, thread=new_thread)
        return True, None, "test-restarted-clean"

    monkeypatch.setattr(solver_bridge_module, "ensure_wsl_solver", fake_ensure)
    newcomer = LiveSolverBridge(
        Path("."), host=host, port=port, auto_start_wsl=False,
        session_id="new-session", session_timeout=0.10,
    )
    try:
        ok, status = newcomer._start_sync()
        newcomer.available = ok
        newcomer.status = status
        assert ok is True
        assert status == "session-recovered:test-restarted-clean"

        observer = SolverClient(host, port, timeout=3)
        try:
            state = observer.status()
            assert state["session_id"] == "new-session"
            assert state["frozen"] is False
            assert state["freeze_pending"] is False
            assert state["foreground"] == 0
            assert state["background"] == 0
            assert state["admission_open"] is True
        finally:
            observer.close()
    finally:
        old.close()
        newcomer.close()
        new_server = replacement.get("server")
        new_thread = replacement.get("thread")
        if new_server is not None:
            new_server.shutdown()  # type: ignore[attr-defined]
        if new_thread is not None:
            new_thread.join(timeout=2.0)  # type: ignore[attr-defined]
