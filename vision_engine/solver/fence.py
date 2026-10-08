from __future__ import annotations

from dataclasses import dataclass
import threading
import time
import uuid
from typing import Callable


@dataclass(frozen=True, slots=True)
class FenceToken:
    session_id: str
    token: str


class GlobalSolverFence:
    """Server-side admission and quiescence fence for all solver work.

    Foreground and background work must enter this gate before reaching a
    process pool. A freeze closes both admissions, waits for every admitted
    job to leave, then returns an ownership token. Only that token can reopen
    the server. Session rotation uses the same mechanism so work from an old
    Windows app instance cannot leak into a newly opened session.
    """

    def __init__(self) -> None:
        self._cv = threading.Condition(threading.RLock())
        self._session_id = ""
        self._admission_open = False
        self._freeze_pending = False
        self._owner: FenceToken | None = None
        self._foreground = 0
        self._background = 0
        self._generation = 0

    @property
    def session_id(self) -> str:
        with self._cv:
            return self._session_id

    def snapshot(self) -> dict[str, object]:
        with self._cv:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> dict[str, object]:
        return {
            "session_id": self._session_id,
            "admission_open": self._admission_open,
            "freeze_pending": self._freeze_pending,
            "frozen": self._owner is not None,
            "foreground": self._foreground,
            "background": self._background,
            "idle": self._foreground == 0 and self._background == 0,
            "generation": self._generation,
        }

    def _matches(self, session_id: str) -> bool:
        return bool(session_id) and session_id == self._session_id

    def enter_foreground(self, session_id: str) -> bool:
        with self._cv:
            if not self._admission_open or self._freeze_pending or self._owner is not None:
                return False
            if not self._matches(session_id):
                return False
            self._foreground += 1
            return True

    def leave_foreground(self) -> None:
        with self._cv:
            self._foreground = max(0, self._foreground - 1)
            self._cv.notify_all()

    def enter_background(self, session_id: str) -> bool:
        with self._cv:
            if not self._admission_open or self._freeze_pending or self._owner is not None:
                return False
            if not self._matches(session_id):
                return False
            self._background += 1
            return True

    def leave_background(self) -> None:
        with self._cv:
            self._background = max(0, self._background - 1)
            self._cv.notify_all()

    def _wait_idle_locked(self, deadline: float) -> bool:
        while self._foreground or self._background:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            self._cv.wait(remaining)
        return True

    def begin_session(
        self,
        session_id: str,
        timeout: float,
        cancel_queued: Callable[[], int] | None = None,
    ) -> dict[str, object]:
        session_id = str(session_id or "").strip()
        if not session_id:
            return {"ok": False, "reason": "missing-session-id", **self.snapshot()}
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._cv:
            # Do not steal an active physical-input fence from another GUI.
            # Wait for its explicit release; if the old GUI died, the bounded
            # timeout fails closed and the caller can restart the service.
            while self._freeze_pending or self._owner is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    reason = "exclusive-owner-still-active" if self._owner is not None else "freeze-still-pending"
                    return {"ok": False, "reason": reason, **self._snapshot_locked()}
                self._cv.wait(remaining)
            self._admission_open = False
            self._freeze_pending = True
            self._generation += 1
        cancelled = int(cancel_queued() or 0) if cancel_queued else 0
        with self._cv:
            idle = self._wait_idle_locked(deadline)
            if not idle:
                self._freeze_pending = False
                return {"ok": False, "reason": "old-session-not-idle", "cancelled": cancelled, **self._snapshot_locked()}
            self._session_id = session_id
            self._freeze_pending = False
            self._admission_open = True
            self._cv.notify_all()
            return {"ok": True, "cancelled": cancelled, **self._snapshot_locked()}

    def freeze(
        self,
        session_id: str,
        timeout: float,
        cancel_queued: Callable[[], int] | None = None,
    ) -> dict[str, object]:
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._cv:
            if not self._matches(session_id):
                return {"ok": False, "reason": "stale-session", **self._snapshot_locked()}
            if self._owner is not None or self._freeze_pending:
                return {"ok": False, "reason": "already-frozen", **self._snapshot_locked()}
            self._admission_open = False
            self._freeze_pending = True
            self._generation += 1
        cancelled = int(cancel_queued() or 0) if cancel_queued else 0
        with self._cv:
            idle = self._wait_idle_locked(deadline)
            if not idle:
                self._freeze_pending = False
                self._admission_open = True
                self._cv.notify_all()
                return {"ok": False, "reason": "solver-not-idle", "cancelled": cancelled, **self._snapshot_locked()}
            token = FenceToken(str(session_id), uuid.uuid4().hex)
            self._owner = token
            self._freeze_pending = False
            return {"ok": True, "exclusive_token": token.token, "cancelled": cancelled, **self._snapshot_locked()}

    def release(self, session_id: str, token: str) -> dict[str, object]:
        with self._cv:
            expected = self._owner
            if expected is None:
                return {"ok": False, "reason": "not-frozen", **self._snapshot_locked()}
            if expected.session_id != session_id or expected.token != token:
                return {"ok": False, "reason": "invalid-exclusive-token", **self._snapshot_locked()}
            self._owner = None
            self._admission_open = True
            self._generation += 1
            self._cv.notify_all()
            return {"ok": True, **self._snapshot_locked()}

    def end_session(
        self,
        session_id: str,
        timeout: float = 5.0,
        cancel_queued: Callable[[], int] | None = None,
    ) -> dict[str, object]:
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._cv:
            if not self._matches(session_id):
                return {"ok": True, "reason": "already-detached", **self._snapshot_locked()}
            self._admission_open = False
            self._freeze_pending = True
            # Same-session shutdown intentionally revokes its own exclusive token.
            self._owner = None
            self._generation += 1
            self._cv.notify_all()
        cancelled = int(cancel_queued() or 0) if cancel_queued else 0
        with self._cv:
            idle = self._wait_idle_locked(deadline)
            self._freeze_pending = False
            self._cv.notify_all()
            return {
                "ok": idle,
                "idle": idle,
                "reason": "session-ended" if idle else "session-end-timeout",
                "cancelled": cancelled,
                **self._snapshot_locked(),
            }

