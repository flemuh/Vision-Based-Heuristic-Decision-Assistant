from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import time

from vision_engine.live.solver_bridge import LiveSolverBridge
from vision_engine.live.state import GameState
from vision_engine.solver.protocol import SolverRequest, SolverResponse, SolverSettings


def _request(request_id: int) -> SolverRequest:
    state = GameState(
        episode_id="ep", state_version=0, strategy_version=0,
        board_hash="h", board_cells=("BL",) * 25,
        accepted_mask=0, current_jewel="BL", strategy="score",
    )
    return SolverRequest(request_id, state, SolverSettings(rollouts=1), session_id="session-x")


class BlockingClient:
    def __init__(self):
        self.solve_started = threading.Event()
        self.solve_release = threading.Event()
        self.solve_ids: list[int] = []
        self.calls: list[tuple] = []

    def solve(self, request: SolverRequest) -> SolverResponse:
        self.solve_ids.append(request.request_id)
        self.solve_started.set()
        self.solve_release.wait(1.0)
        return SolverResponse.for_request(request, (), source="fake")

    def freeze(self, session_id: str, timeout: float):
        self.calls.append(("freeze", session_id))
        return {
            "ok": True, "idle": True, "foreground": 0, "background": 0,
            "exclusive_token": "tok", "session_id": session_id,
        }

    def release(self, session_id: str, token: str):
        self.calls.append(("release", session_id, token))
        return {"ok": True, "admission_open": True}

    def close(self):
        pass


def _bridge(client: BlockingClient) -> LiveSolverBridge:
    bridge = LiveSolverBridge(Path("."), auto_start_wsl=False, session_id="session-x")
    bridge.executor.shutdown(wait=False, cancel_futures=True)
    bridge.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-ipc")
    bridge.client = client  # type: ignore[assignment]
    bridge.available = True
    return bridge


def test_v1826_close_admission_waits_active_and_drops_queued_solve():
    client = BlockingClient()
    bridge = _bridge(client)
    results: list[int] = []
    errors: list[str] = []

    bridge.solve_async(_request(1), lambda r: results.append(r.request_id), lambda e: errors.append(str(e)))
    assert client.solve_started.wait(0.5)
    bridge.solve_async(_request(2), lambda r: results.append(r.request_id), lambda e: errors.append(str(e)))

    holder: dict = {}
    t = threading.Thread(target=lambda: holder.setdefault("barrier", bridge.prepare_input_exclusive(1.0)))
    t.start()
    time.sleep(0.04)
    assert client.calls == []  # freeze must not race the active Windows IPC
    client.solve_release.set()
    t.join(1.0)

    barrier = holder["barrier"]
    assert barrier["ok"] is True
    assert barrier["bridge_active"] == 0
    assert client.solve_ids == [1]  # queued request 2 never reached the server
    assert results == []  # request 1 became stale while admission was closing
    assert client.calls == [("freeze", "session-x")]

    released = bridge.release_input_exclusive("tok")
    assert released["ok"] is True
    assert client.calls[-1] == ("release", "session-x", "tok")
    assert bridge.admission.snapshot()["open"] is True
    bridge.close()


def test_v1826_bridge_rejects_new_solve_while_exclusive():
    client = BlockingClient()
    client.solve_release.set()
    bridge = _bridge(client)
    barrier = bridge.prepare_input_exclusive(0.5)
    assert barrier["ok"] is True
    errors: list[str] = []
    bridge.solve_async(_request(3), lambda _r: None, lambda e: errors.append(str(e)))
    assert errors and "admission closed" in errors[0]
    assert 3 not in client.solve_ids
    bridge.release_input_exclusive(barrier["exclusive_token"])
    bridge.close()


def test_v1826_startup_recovers_crashed_same_version_session(monkeypatch):
    calls = []

    class RecoveryClient:
        def __init__(self):
            self.attempt = 0

        def begin_session(self, session_id, timeout):
            self.attempt += 1
            calls.append(("begin", self.attempt, session_id))
            if self.attempt == 1:
                raise RuntimeError("exclusive-owner-still-active")
            return {"ok": True}

        def shutdown_service(self):
            calls.append(("shutdown",))
            return True

        def close(self):
            calls.append(("close",))

    launches = []

    def fake_ensure(*_args, **_kwargs):
        launches.append(1)
        return True, object(), "ready"

    monkeypatch.setattr("vision_engine.live.solver_bridge.ensure_wsl_solver", fake_ensure)
    bridge = LiveSolverBridge(Path("."), auto_start_wsl=True, session_id="fresh")
    bridge.client = RecoveryClient()  # type: ignore[assignment]
    ok, status = bridge._start_sync()
    assert ok is True
    assert status.startswith("session-recovered:")
    assert len(launches) == 2
    assert calls == [("begin", 1, "fresh"), ("shutdown",), ("begin", 2, "fresh")]
    bridge.executor.shutdown(wait=False, cancel_futures=True)
