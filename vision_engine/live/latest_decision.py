from __future__ import annotations

"""One-slot latest-state-wins queue for live solver decisions.

The live monitor can observe a newer jewel/board state before an older WSL solve
finishes.  This lane guarantees at most one executing decision plus one pending
LATEST state. Intermediate states are overwritten instead of accumulating in the
ThreadPoolExecutor/socket queue.
"""

from dataclasses import dataclass
import threading
from typing import Callable, Generic, TypeVar

from .solver_admission import BridgeAdmissionGate

ReqT = TypeVar("ReqT")
RespT = TypeVar("RespT")


@dataclass(slots=True)
class _Pending(Generic[ReqT, RespT]):
    generation: int
    request: ReqT
    on_result: Callable[[RespT], None]
    on_error: Callable[[BaseException], None]


class LatestDecisionLane(Generic[ReqT, RespT]):
    def __init__(
        self,
        *,
        admission: BridgeAdmissionGate,
        submit_worker: Callable[[Callable[[], None]], object],
        run_request: Callable[[ReqT], RespT | None],
    ) -> None:
        self._admission = admission
        self._submit_worker = submit_worker
        self._run_request = run_request
        self._lock = threading.RLock()
        self._generation = 0
        self._pending: _Pending[ReqT, RespT] | None = None
        self._worker_active = False
        self._superseded = 0

    def submit(
        self,
        request: ReqT,
        on_result: Callable[[RespT], None],
        on_error: Callable[[BaseException], None],
    ) -> dict[str, int | bool]:
        # Preserve the bridge's immediate contract if input-exclusive is already
        # active. Accepted work that becomes stale later is silently dropped.
        if self._admission.ticket("solve") is None:
            on_error(RuntimeError("solver bridge admission closed"))
            return {"accepted": False, "generation": self._generation, "superseded": False}

        with self._lock:
            had_older = self._worker_active or self._pending is not None
            if had_older:
                self._superseded += 1
            self._generation += 1
            generation = self._generation
            self._pending = _Pending(generation, request, on_result, on_error)
            start_worker = not self._worker_active
            if start_worker:
                self._worker_active = True

        if start_worker:
            try:
                self._submit_worker(self._loop)
            except BaseException as exc:
                with self._lock:
                    self._worker_active = False
                    pending = self._pending
                    self._pending = None
                if pending is not None and pending.generation == generation:
                    on_error(exc)
                return {"accepted": False, "generation": generation, "superseded": had_older}

        return {"accepted": True, "generation": generation, "superseded": had_older}

    def _loop(self) -> None:
        while True:
            with self._lock:
                item = self._pending
                self._pending = None
                if item is None:
                    self._worker_active = False
                    return

            ticket = self._admission.ticket("solve")
            if ticket is None or not self._admission.enter(ticket):
                continue
            response: RespT | None = None
            error: BaseException | None = None
            try:
                response = self._run_request(item.request)
            except BaseException as exc:
                error = exc
            finally:
                self._admission.leave()

            with self._lock:
                latest = item.generation == self._generation
            if not latest or not self._admission.is_current(ticket):
                continue
            if error is not None:
                item.on_error(error)
            elif response is not None:
                item.on_result(response)

    def drop_pending(self) -> None:
        with self._lock:
            self._generation += 1
            self._pending = None

    def snapshot(self) -> dict[str, int | bool]:
        with self._lock:
            return {
                "generation": int(self._generation),
                "worker_active": bool(self._worker_active),
                "pending_latest": self._pending is not None,
                "superseded": int(self._superseded),
            }
