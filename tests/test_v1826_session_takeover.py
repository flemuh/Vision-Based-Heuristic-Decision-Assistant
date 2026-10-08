from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import threading
from pathlib import Path
import time

import pytest

from vision_engine.live.solver_bridge import LiveSolverBridge
from vision_engine.live.state import GameState
from vision_engine.solver.client import SolverClient
from vision_engine.solver.protocol import SolverRequest, SolverResponse, SolverSettings
from vision_engine.solver.service import SolverEngine, SolverTCPServer


BOARD = (
    "BL", "CR", "CH", "CH", "CR",
    "CR", "SO", "BL", "LI", "CR",
    "CH", "HA", None, "SO", "LI",
    "SO", "LI", "HA", "BL", "LI",
    "BL", "CH", "SO", "HA", "HA",
)


class ControlledRunningExecutor:
    """Executor whose submitted work is physically RUNNING until the test releases it."""

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

    def running_count(self) -> int:
        with self._lock:
            return sum(1 for f in self.futures if not f.done())

    def shutdown(self, wait=True, cancel_futures=False):
        if cancel_futures:
            with self._lock:
                for future in self.futures:
                    future.cancel()


class ImmediateExecutor:
    def submit(self, fn, *args, **kwargs):
        future = Future()
        future.set_result(())
        return future

    def shutdown(self, wait=True, cancel_futures=False):
        return None


def _state(jewel: str) -> GameState:
    return GameState(
        episode_id="ep-session-takeover",
        state_version=3,
        strategy_version=0,
        board_hash="board-session-takeover",
        board_cells=BOARD,
        accepted_mask=(1 << 4) | (1 << 13) | (1 << 17),
        current_jewel=jewel,
        strategy="score",
    )


def _request(request_id: int, session_id: str, jewel: str = "BL") -> SolverRequest:
    return SolverRequest(
        request_id=request_id,
        state=_state(jewel),
        settings=SolverSettings(rollouts=1, exact_horizon=1, rollout_exact_horizon=1),
        purpose="session-takeover",
        session_id=session_id,
    )


def _server_with_controlled_pools():
    engine = SolverEngine(workers=3, cache_entries=64)
    engine.foreground.shutdown(wait=True, cancel_futures=True)
    engine.background.shutdown(wait=True, cancel_futures=True)
    foreground = ControlledRunningExecutor()
    background = ControlledRunningExecutor()
    engine.foreground = foreground  # type: ignore[assignment]
    engine.background = background  # type: ignore[assignment]
    server = SolverTCPServer(("127.0.0.1", 0), engine, socket_timeout=20.0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return engine, foreground, background, server


def _wait_until(predicate, timeout: float = 1.0, message: str = "condition not reached"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.005)
    raise AssertionError(message)


def test_v1826_new_session_waits_for_all_old_work_before_rotation():
    engine, foreground, background, server = _server_with_controlled_pools()
    host, port = server.server_address
    old = SolverClient(host, port, timeout=20)
    old_solve_client = SolverClient(host, port, timeout=20)
    newcomer = SolverClient(host, port, timeout=20)
    observer = SolverClient(host, port, timeout=20)

    old_result: dict[str, object] = {}
    rotate_result: dict[str, object] = {}
    old_solve_done = threading.Event()
    rotate_done = threading.Event()

    try:
        assert old.begin_session("old-session", 2.0)["ok"]

        # One old foreground request is physically alive.
        def run_old_solve() -> None:
            try:
                old_result["response"] = old_solve_client.solve(_request(1, "old-session", "BL"))
            except BaseException as exc:  # pragma: no cover - assertion below surfaces it
                old_result["error"] = exc
            finally:
                old_solve_done.set()

        solve_thread = threading.Thread(target=run_old_solve, daemon=True)
        solve_thread.start()
        _wait_until(lambda: observer.status()["foreground"] == 1, message="old foreground never became active")

        # One old background request is also physically alive.
        ack = old.precompute_requests([_request(-1, "old-session", "SO")])
        assert ack["scheduled"] == 1
        _wait_until(lambda: observer.status()["background"] == 1, message="old background never became active")

        # New app/session attempts to take ownership while both old jobs live.
        def rotate() -> None:
            try:
                rotate_result.update(newcomer.begin_session("new-session", timeout=5.0))
            except BaseException as exc:  # pragma: no cover
                rotate_result["error"] = exc
            finally:
                rotate_done.set()

        rotate_thread = threading.Thread(target=rotate, daemon=True)
        rotate_thread.start()

        _wait_until(lambda: observer.status()["freeze_pending"] is True, message="session rotation never closed admission")
        during = observer.status()
        assert during["session_id"] == "old-session"
        assert during["admission_open"] is False
        assert during["foreground"] == 1
        assert during["background"] == 1
        assert rotate_done.is_set() is False

        # Neither old nor new work may sneak in while takeover is pending.
        with pytest.raises(RuntimeError, match="admission closed|stale session"):
            observer.solve(_request(2, "old-session", "CR"))
        with pytest.raises(RuntimeError, match="admission closed|stale session"):
            observer.solve(_request(3, "new-session", "HA"))

        # Releasing only one class of work must not rotate the session.
        background.complete_one(())
        _wait_until(lambda: observer.status()["background"] == 0)
        assert observer.status()["foreground"] == 1
        assert rotate_done.is_set() is False
        assert observer.status()["session_id"] == "old-session"

        foreground.complete_one(())
        assert old_solve_done.wait(1.0)
        rotate_thread.join(timeout=1.0)
        assert rotate_thread.is_alive() is False
        assert rotate_result.get("error") is None
        assert rotate_result["ok"] is True
        assert rotate_result["session_id"] == "new-session"
        assert rotate_result["foreground"] == 0
        assert rotate_result["background"] == 0
        assert rotate_result["admission_open"] is True

        # Once rotated, old-session traffic is permanently stale.
        with pytest.raises(RuntimeError, match="stale session"):
            observer.solve(_request(4, "old-session", "LI"))

        # New-session foreground is admitted cleanly. Use an immediate pool so
        # this assertion cannot hang on the deterministic old-work executor.
        engine.foreground = ImmediateExecutor()  # type: ignore[assignment]
        response = newcomer.solve(_request(5, "new-session", "CH"))
        assert response.session_id == "new-session"
        final = observer.status()
        assert final["session_id"] == "new-session"
        assert final["foreground"] == 0
        assert final["background"] == 0
        assert final["idle"] is True
    finally:
        old.close()
        old_solve_client.close()
        newcomer.close()
        observer.close()
        server.shutdown()
        server.server_close()
        engine.close()


def test_v1826_new_session_cannot_steal_old_exclusive_token():
    engine = SolverEngine(workers=3, cache_entries=64)
    server = SolverTCPServer(("127.0.0.1", 0), engine, socket_timeout=20.0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address
    old = SolverClient(host, port, timeout=20)
    newcomer = SolverClient(host, port, timeout=20)
    observer = SolverClient(host, port, timeout=20)

    takeover: dict[str, object] = {}
    takeover_done = threading.Event()

    try:
        assert old.begin_session("old-session", 2.0)["ok"]
        frozen = old.freeze("old-session", 2.0)
        assert frozen["ok"] is True
        token = str(frozen["exclusive_token"])

        def begin_new() -> None:
            try:
                takeover.update(newcomer.begin_session("new-session", timeout=3.0))
            except BaseException as exc:  # pragma: no cover
                takeover["error"] = exc
            finally:
                takeover_done.set()

        thread = threading.Thread(target=begin_new, daemon=True)
        thread.start()
        time.sleep(0.08)

        held = observer.status()
        assert held["session_id"] == "old-session"
        assert held["frozen"] is True
        assert held["admission_open"] is False
        assert takeover_done.is_set() is False

        # Wrong session/token cannot release or steal ownership.
        with pytest.raises(RuntimeError, match="invalid-exclusive-token"):
            observer.release("new-session", token)
        assert observer.status()["session_id"] == "old-session"
        assert observer.status()["frozen"] is True

        released = old.release("old-session", token)
        assert released["ok"] is True
        thread.join(timeout=1.0)
        assert thread.is_alive() is False
        assert takeover.get("error") is None
        assert takeover["ok"] is True
        assert takeover["session_id"] == "new-session"
        assert takeover["frozen"] is False
        assert takeover["admission_open"] is True
    finally:
        old.close()
        newcomer.close()
        observer.close()
        server.shutdown()
        server.server_close()
        engine.close()


class StaleSessionClient:
    def __init__(self, stale_session: str) -> None:
        self.stale_session = stale_session
        self.done = threading.Event()

    def solve(self, request: SolverRequest) -> SolverResponse:
        try:
            stale_request = SolverRequest(
                request_id=request.request_id,
                state=request.state,
                settings=request.settings,
                learned_scores=request.learned_scores,
                purpose=request.purpose,
                session_id=self.stale_session,
            )
            return SolverResponse.for_request(stale_request, (), source="stale-session-test")
        finally:
            self.done.set()

    def close(self) -> None:
        return None


def test_v1826_windows_bridge_drops_callback_from_old_session_after_rotation():
    bridge = LiveSolverBridge(Path("."), auto_start_wsl=False, session_id="new-session")
    bridge.executor.shutdown(wait=False, cancel_futures=True)
    bridge.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="session-takeover-ipc")
    client = StaleSessionClient("old-session")
    bridge.client = client  # type: ignore[assignment]
    bridge.available = True

    results: list[str] = []
    errors: list[str] = []
    try:
        bridge.solve_async(
            _request(77, "new-session", "BL"),
            lambda response: results.append(response.session_id),
            lambda exc: errors.append(str(exc)),
        )
        assert client.done.wait(1.0)
        _wait_until(lambda: bridge.admission.snapshot()["active"] == 0)
        assert results == []
        assert errors == []
    finally:
        bridge.admission.close()
        bridge.executor.shutdown(wait=True, cancel_futures=True)
