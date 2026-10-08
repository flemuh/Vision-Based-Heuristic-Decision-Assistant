from __future__ import annotations

import threading
import time


class MonitorInputGate:
    """Synchronous safe-point handshake between monitor and physical input.

    A commit owner requests quiescence. The monitor acknowledges only between
    complete vision cycles and remains parked until release. Release does not
    return until the monitor has actually left that safe point, preventing the
    next commit from reusing handshake state from the previous cycle.
    """

    def __init__(self) -> None:
        self._cv = threading.Condition(threading.RLock())
        self._requested = False
        self._quiesced = False
        self._owner_lock = threading.Lock()

    @property
    def requested(self) -> bool:
        with self._cv:
            return self._requested

    @property
    def quiesced(self) -> bool:
        with self._cv:
            return self._quiesced

    def request_quiesce(self, timeout: float = 2.0) -> bool:
        """Request a monitor safe point and wait for its acknowledgement."""
        timeout = max(0.0, float(timeout))
        if not self._owner_lock.acquire(blocking=False):
            return False
        deadline = time.monotonic() + timeout
        with self._cv:
            self._requested = True
            self._cv.notify_all()
            while not self._quiesced:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._requested = False
                    self._cv.notify_all()
                    self._release_owner()
                    return False
                self._cv.wait(remaining)
            return True

    def monitor_safe_point(self, *, running: bool = True) -> None:
        """Called only by the monitor thread between complete vision cycles."""
        with self._cv:
            if not self._requested:
                return
            self._quiesced = True
            self._cv.notify_all()
            try:
                while running and self._requested:
                    self._cv.wait(0.05)
            finally:
                self._quiesced = False
                self._cv.notify_all()

    def wait_or_wake(self, timeout: float) -> bool:
        """Interrupt monitor sleeps when input quiescence is requested."""
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._cv:
            while not self._requested:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cv.wait(remaining)
            return True

    def release(self, timeout: float = 1.0) -> None:
        """Release and wait until the monitor has physically left the safe point."""
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._cv:
            self._requested = False
            self._cv.notify_all()
            while self._quiesced:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("monitor did not leave input safe point")
                self._cv.wait(remaining)
        self._release_owner()

    def force_release(self) -> None:
        """Shutdown-safe, non-blocking release used only for recovery."""
        with self._cv:
            self._requested = False
            self._quiesced = False
            self._cv.notify_all()
        self._release_owner()

    def _release_owner(self) -> None:
        if self._owner_lock.locked():
            try:
                self._owner_lock.release()
            except RuntimeError:
                pass
