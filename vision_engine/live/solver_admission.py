from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from concurrent.futures import Future


@dataclass(frozen=True, slots=True)
class AdmissionTicket:
    epoch: int
    kind: str


class BridgeAdmissionGate:
    """Windows-side admission gate for solver IPC submitted to the bridge pool."""

    def __init__(self) -> None:
        self._cv = threading.Condition(threading.RLock())
        self._open = True
        self._epoch = 0
        self._active = 0
        self._futures: set[Future] = set()

    def ticket(self, kind: str) -> AdmissionTicket | None:
        with self._cv:
            if not self._open:
                return None
            return AdmissionTicket(self._epoch, str(kind))

    def attach(self, ticket: AdmissionTicket, future: Future) -> None:
        with self._cv:
            self._futures.add(future)
            stale = (not self._open) or ticket.epoch != self._epoch
        if stale:
            future.cancel()
        future.add_done_callback(self._forget)

    def _forget(self, future: Future) -> None:
        with self._cv:
            self._futures.discard(future)
            self._cv.notify_all()

    def enter(self, ticket: AdmissionTicket) -> bool:
        with self._cv:
            if not self._open or ticket.epoch != self._epoch:
                return False
            self._active += 1
            return True

    def leave(self) -> None:
        with self._cv:
            self._active = max(0, self._active - 1)
            self._cv.notify_all()

    def is_current(self, ticket: AdmissionTicket) -> bool:
        with self._cv:
            return self._open and ticket.epoch == self._epoch

    def close(self) -> dict[str, int | bool]:
        with self._cv:
            self._open = False
            self._epoch += 1
            futures = list(self._futures)
            active = self._active
        cancelled = 0
        for future in futures:
            try:
                if future.cancel():
                    cancelled += 1
            except BaseException:
                pass
        return {"open": False, "epoch": self._epoch, "active": active, "cancelled": cancelled}

    def wait_idle(self, timeout: float) -> dict[str, int | bool]:
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._cv:
            while self._active:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cv.wait(remaining)
            return {
                "idle": self._active == 0,
                "active": self._active,
                "open": self._open,
                "epoch": self._epoch,
            }

    def open(self) -> dict[str, int | bool]:
        with self._cv:
            self._open = True
            self._epoch += 1
            self._cv.notify_all()
            return {"open": True, "active": self._active, "epoch": self._epoch}

    def snapshot(self) -> dict[str, int | bool]:
        with self._cv:
            return {
                "open": self._open,
                "active": self._active,
                "queued": sum(1 for f in self._futures if not f.running() and not f.done()),
                "epoch": self._epoch,
            }
