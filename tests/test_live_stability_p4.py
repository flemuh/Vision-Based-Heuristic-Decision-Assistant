from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import time

from vision_engine.live.solver_bridge import LiveSolverBridge
from vision_engine.live.state import GameState
from vision_engine.solver.protocol import SolverRequest, SolverResponse, SolverSettings


def _request(request_id: int, jewel: str = "BL", mask: int = 0) -> SolverRequest:
    state = GameState(
        episode_id="ep", state_version=request_id, strategy_version=0,
        board_hash="h", board_cells=("BL",) * 25,
        accepted_mask=mask, current_jewel=jewel, strategy="score",
    )
    return SolverRequest(request_id, state, SolverSettings(rollouts=1), session_id="session-x")


class BlockingClient:
    def __init__(self):
        self.first_started = threading.Event()
        self.release_first = threading.Event()
        self.solve_ids: list[int] = []

    def solve(self, request: SolverRequest) -> SolverResponse:
        self.solve_ids.append(request.request_id)
        if len(self.solve_ids) == 1:
            self.first_started.set()
            self.release_first.wait(2.0)
        return SolverResponse.for_request(request, (), source="fake")

    def freeze(self, session_id: str, timeout: float):
        return {
            "ok": True, "idle": True, "foreground": 0, "background": 0,
            "exclusive_token": "tok", "session_id": session_id,
        }

    def release(self, session_id: str, token: str):
        return {"ok": True, "admission_open": True}

    def close(self):
        pass


def _bridge(client: BlockingClient) -> LiveSolverBridge:
    bridge = LiveSolverBridge(Path("."), auto_start_wsl=False, session_id="session-x")
    bridge.executor.shutdown(wait=False, cancel_futures=True)
    bridge.executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="test-ipc")
    bridge.client = client  # type: ignore[assignment]
    bridge.available = True
    return bridge


def _wait_until(pred, timeout=1.5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return bool(pred())


def test_p4_remote_decisions_collapse_to_latest_state_only():
    client = BlockingClient()
    bridge = _bridge(client)
    results: list[int] = []
    errors: list[str] = []
    try:
        a = bridge.solve_async(_request(1, "BL", 0), lambda r: results.append(r.request_id), lambda e: errors.append(str(e)))
        assert a["accepted"] is True
        assert client.first_started.wait(0.5)

        # Two newer visual states arrive while request 1 is physically running.
        # Request 2 must never reach the server; request 3 replaces it.
        b = bridge.solve_async(_request(2, "SO", 1), lambda r: results.append(r.request_id), lambda e: errors.append(str(e)))
        c = bridge.solve_async(_request(3, "LI", 3), lambda r: results.append(r.request_id), lambda e: errors.append(str(e)))
        assert b["superseded"] is True
        assert c["superseded"] is True
        snap = bridge.decision_snapshot()
        assert snap["pending_latest"] is True
        assert snap["superseded"] >= 2

        client.release_first.set()
        assert _wait_until(lambda: client.solve_ids == [1, 3])
        assert _wait_until(lambda: results == [3])
        assert errors == []
    finally:
        client.release_first.set()
        bridge.close()


def test_p4_input_fence_drops_pending_latest_decision():
    client = BlockingClient()
    bridge = _bridge(client)
    results: list[int] = []
    errors: list[str] = []
    try:
        bridge.solve_async(_request(10), lambda r: results.append(r.request_id), lambda e: errors.append(str(e)))
        assert client.first_started.wait(0.5)
        bridge.solve_async(_request(11), lambda r: results.append(r.request_id), lambda e: errors.append(str(e)))

        holder: dict = {}
        t = threading.Thread(target=lambda: holder.setdefault("barrier", bridge.prepare_input_exclusive(1.5)))
        t.start()
        time.sleep(0.05)
        client.release_first.set()
        t.join(2.0)
        assert holder["barrier"]["ok"] is True
        assert client.solve_ids == [10]
        assert results == []
        assert bridge.decision_snapshot()["pending_latest"] is False
        bridge.release_input_exclusive(holder["barrier"]["exclusive_token"])
    finally:
        client.release_first.set()
        bridge.close()
