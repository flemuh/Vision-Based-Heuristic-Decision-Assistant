from __future__ import annotations

from concurrent.futures import Future
import threading
import time

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
    """Deterministic test executor: every submitted Future is already RUNNING.

    Future.cancel() therefore returns False, reproducing the important case from
    the Windows failure: work is physically alive and cannot merely be dropped
    from a queue. The test controls exactly when each job finishes.
    """

    def __init__(self) -> None:
        self.futures: list[Future] = []
        self._lock = threading.Lock()

    def submit(self, fn, *args, **kwargs):
        future = Future()
        assert future.set_running_or_notify_cancel() is True
        with self._lock:
            self.futures.append(future)
        return future

    def complete_one(self) -> None:
        with self._lock:
            future = next(f for f in self.futures if not f.done())
        # An empty recommendation tuple is sufficient for the background cache
        # callback; this test is about lifecycle/fencing, not advisor policy.
        future.set_result(())

    def shutdown(self, wait=True, cancel_futures=False):
        if cancel_futures:
            with self._lock:
                for future in self.futures:
                    future.cancel()


def _state(jewel: str) -> GameState:
    return GameState(
        episode_id="ep-pending6",
        state_version=3,
        strategy_version=0,
        board_hash="board-pending6",
        board_cells=BOARD,
        accepted_mask=(1 << 4) | (1 << 13) | (1 << 17),
        current_jewel=jewel,
        strategy="score",
    )


def _requests(session_id: str) -> list[SolverRequest]:
    settings = SolverSettings(rollouts=2, exact_horizon=2, rollout_exact_horizon=1)
    return [
        SolverRequest(
            request_id=-(i + 1),
            state=_state(jewel),
            settings=settings,
            purpose="pending6-deterministic",
            session_id=session_id,
        )
        for i, jewel in enumerate(("BL", "SO", "LI", "CR", "HA", "CH"))
    ]


def test_v1826_pending6_running_jobs_block_global_freeze_until_all_six_leave():
    engine = SolverEngine(workers=3, cache_entries=64)
    # No work has been submitted to the real pool yet, so it can be replaced
    # safely by a deterministic executor for this integration test.
    engine.background.shutdown(wait=True, cancel_futures=True)
    controlled = ControlledRunningExecutor()
    engine.background = controlled  # type: ignore[assignment]

    server = SolverTCPServer(("127.0.0.1", 0), engine, socket_timeout=20.0)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    host, port = server.server_address
    control = SolverClient(host, port, timeout=20)
    observer = SolverClient(host, port, timeout=20)

    try:
        session_id = "pending6-session"
        assert control.begin_session(session_id, 2.0)["ok"]
        ack = control.precompute_requests(_requests(session_id))
        assert ack["scheduled"] == 6

        before = observer.status()
        assert before["background"] == 6
        assert before["foreground"] == 0
        assert before["pending_background_keys"] == 6
        assert before["idle"] is False

        result: dict[str, object] = {}
        finished = threading.Event()

        def freeze_from_tcp() -> None:
            try:
                result.update(control.freeze(session_id, timeout=5.0))
            finally:
                finished.set()

        freezer = threading.Thread(target=freeze_from_tcp, daemon=True)
        freezer.start()

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            state = observer.status()
            if state["freeze_pending"]:
                break
            time.sleep(0.005)
        else:
            raise AssertionError("global freeze never entered freeze_pending")

        # Admission is closed immediately, but no exclusive token exists while
        # even one physically running job remains.
        frozen_wait = observer.status()
        assert frozen_wait["admission_open"] is False
        assert frozen_wait["freeze_pending"] is True
        assert frozen_wait["frozen"] is False
        assert frozen_wait["background"] == 6
        assert finished.is_set() is False

        # Queued/new background and foreground work must fail closed now.
        assert engine.fence.enter_background(session_id) is False
        assert engine.fence.enter_foreground(session_id) is False

        # Finish five jobs. Freeze must still be blocked with background == 1.
        for expected_remaining in (5, 4, 3, 2, 1):
            controlled.complete_one()
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                state = observer.status()
                if state["background"] == expected_remaining:
                    break
                time.sleep(0.005)
            assert state["background"] == expected_remaining
            assert state["foreground"] == 0
            assert finished.is_set() is False

        # The sixth completion is the only event allowed to make the solver
        # globally idle and therefore mint the physical-input ownership token.
        controlled.complete_one()
        freezer.join(timeout=2.0)
        assert freezer.is_alive() is False
        assert finished.is_set() is True
        assert result["ok"] is True
        assert result["idle"] is True
        assert result["foreground"] == 0
        assert result["background"] == 0
        assert result["frozen"] is True
        assert result["admission_open"] is False
        assert result["exclusive_token"]

        token = str(result["exclusive_token"])
        released = control.release(session_id, token)
        assert released["ok"] is True
        assert released["admission_open"] is True
        assert released["foreground"] == 0
        assert released["background"] == 0

        # Admission really reopened; do not leave a synthetic count behind.
        assert engine.fence.enter_foreground(session_id) is True
        engine.fence.leave_foreground()
        final = observer.status()
        assert final["foreground"] == 0
        assert final["background"] == 0
        assert final["idle"] is True
    finally:
        control.close()
        observer.close()
        server.shutdown()
        server.server_close()
        engine.close()
