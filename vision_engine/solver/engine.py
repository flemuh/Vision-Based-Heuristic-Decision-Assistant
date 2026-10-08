from __future__ import annotations

import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from threading import RLock

from .cache import SolverLRUCache
from .fence import GlobalSolverFence
from .protocol import SolverRequest, SolverResponse
from .worker import solve_hot_worker


class SolverEngine:
    """Persistent WSL solver with a global foreground/background admission fence."""

    def __init__(self, workers: int | None = None, cache_entries: int = 4096):
        if workers is None:
            workers = max(2, min(6, (os.cpu_count() or 4) - 1))
        self.workers = max(2, int(workers))
        self.cache = SolverLRUCache(cache_entries)
        ctx_name = "forkserver" if os.name != "nt" else "spawn"
        self.mp_context = mp.get_context(ctx_name)
        # Live decisions are serialized by the bridge latest-state-wins lane. A
        # single foreground process keeps one Advisor/transposition cache hot
        # instead of alternating sequential requests across two cold workers.
        fg_workers = 1
        self.foreground = ProcessPoolExecutor(max_workers=fg_workers, mp_context=self.mp_context)
        self.background = ProcessPoolExecutor(max_workers=max(1, self.workers - fg_workers), mp_context=self.mp_context)
        self.fence = GlobalSolverFence()
        self._lock = RLock()
        self._pending: dict[str, object] = {}
        self._background_enabled = True

    def close(self) -> None:
        # Service shutdown must not leave worker processes behind.
        self.foreground.shutdown(wait=True, cancel_futures=True)
        self.background.shutdown(wait=True, cancel_futures=True)

    def _cancel_queued_background(self) -> int:
        cancelled = 0
        with self._lock:
            pending = self._pending
            if hasattr(pending, "values"):
                futures = list(pending.values())
            else:
                futures = list(getattr(self, "_pending_futures", {}).values())
        for future in futures:
            try:
                if future.cancel():
                    cancelled += 1
            except BaseException:
                pass
        return cancelled

    def begin_session(self, session_id: str, timeout: float = 5.0) -> dict[str, object]:
        return self.fence.begin_session(session_id, timeout, self._cancel_queued_background)

    def end_session(self, session_id: str, timeout: float = 5.0) -> dict[str, object]:
        return self.fence.end_session(session_id, timeout, self._cancel_queued_background)

    def freeze(self, session_id: str, timeout: float = 5.0) -> dict[str, object]:
        return self.fence.freeze(session_id, timeout, self._cancel_queued_background)

    def release(self, session_id: str, token: str) -> dict[str, object]:
        return self.fence.release(session_id, token)

    def background_status(self) -> dict[str, object]:
        with self._lock:
            pending = len(self._pending)
            enabled = bool(self._background_enabled)
        return {"enabled": enabled, "pending": pending, "idle": pending == 0}

    def set_background_enabled(self, enabled: bool) -> dict[str, object]:
        with self._lock:
            self._background_enabled = bool(enabled)
        cancelled = self._cancel_queued_background() if not enabled else 0
        return {**self.background_status(), "cancelled": cancelled}

    def wait_background_idle(self, timeout: float) -> dict[str, object]:
        deadline = time.monotonic() + max(0.0, float(timeout))
        while time.monotonic() < deadline:
            status = self.background_status()
            if status["idle"]:
                return status
            time.sleep(0.005)
        return self.background_status()

    def status(self) -> dict[str, object]:
        with self._lock:
            pending = len(self._pending)
        return {**self.fence.snapshot(), "pending_background_keys": pending}

    def _session_for(self, request: SolverRequest) -> str:
        if request.session_id:
            return request.session_id
        if not self.fence.session_id:
            self.fence.begin_session("__direct__", 0.0)
        return "__direct__" if self.fence.session_id == "__direct__" else ""

    def solve(self, request: SolverRequest) -> SolverResponse:
        session_id = self._session_for(request)
        if not self.fence.enter_foreground(session_id):
            raise RuntimeError("solver foreground admission closed or stale session")
        started = time.perf_counter()
        try:
            key = self.cache.key(request)
            cached = self.cache.get(key)
            if cached is not None:
                return SolverResponse.for_request(
                    request, cached, cache_hit=True,
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                    source="persistent-cache",
                )
            recs = self.foreground.submit(solve_hot_worker, request.to_legacy_request()).result()
            self.cache.put(key, recs)
            return SolverResponse.for_request(
                request, recs, cache_hit=False,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                source="persistent-worker",
            )
        finally:
            self.fence.leave_foreground()

    def _background_done(self, key: str, future) -> None:
        try:
            if not future.cancelled():
                recs = future.result()
                self.cache.put(key, recs)
        except BaseException:
            pass
        finally:
            with self._lock:
                self._pending.pop(key, None)
            self.fence.leave_background()

    def precompute(self, requests: list[SolverRequest]) -> dict[str, int]:
        scheduled = cached = duplicate = suppressed = 0
        with self._lock:
            background_enabled = bool(self._background_enabled)
        if not background_enabled:
            return {"scheduled": 0, "cached": 0, "pending": len(self._pending), "suppressed": len(requests)}
        for request in requests:
            key = self.cache.key(request)
            if self.cache.get(key) is not None:
                cached += 1
                continue
            with self._lock:
                if key in self._pending:
                    duplicate += 1
                    continue
            session_id = self._session_for(request)
            if not self.fence.enter_background(session_id):
                suppressed += 1
                continue
            try:
                future = self.background.submit(solve_hot_worker, request.to_legacy_request())
            except BaseException:
                self.fence.leave_background()
                raise
            with self._lock:
                self._pending[key] = future
            future.add_done_callback(lambda fut, k=key: self._background_done(k, fut))
            scheduled += 1
        return {
            "scheduled": scheduled,
            "cached": cached,
            "pending": duplicate,
            "suppressed": suppressed,
        }
