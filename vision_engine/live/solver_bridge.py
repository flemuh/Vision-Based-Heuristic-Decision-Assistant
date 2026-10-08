from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from vision_engine.solver.client import SolverClient
from vision_engine.solver.protocol import SolverRequest, SolverResponse
from vision_engine.solver.wsl import ensure_wsl_solver

from .solver_admission import AdmissionTicket, BridgeAdmissionGate
from .latest_decision import LatestDecisionLane

class LiveSolverBridge:
    """Windows/UI bridge with explicit IPC admission and session ownership."""

    def __init__(
        self,
        root_dir: Path,
        host: str = "127.0.0.1",
        port: int = 57641,
        workers: int = 6,
        wsl_distro: str = "Ubuntu",
        auto_start_wsl: bool = True,
        session_id: str = "",
        session_timeout: float = 5.0,
    ):
        self.root_dir = Path(root_dir)
        self.host = host
        self.port = int(port)
        self.workers = int(workers)
        self.wsl_distro = str(wsl_distro or "Ubuntu").strip() or "Ubuntu"
        self.auto_start_wsl = bool(auto_start_wsl)
        self.session_id = str(session_id or "")
        self.session_timeout = max(0.05, float(session_timeout))
        self.client = SolverClient(host, port, timeout=120.0)
        self.executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="solver-ipc")
        self.admission = BridgeAdmissionGate()
        self.available = False
        self.status = "not-started"
        self.wsl_process = None
        self._lock = threading.RLock()
        self._starting = False
        self._exclusive_token: str | None = None
        self.decision_lane = LatestDecisionLane(
            admission=self.admission, submit_worker=lambda fn: self.executor.submit(fn),
            run_request=self._run_live_decision,
        )

    def set_session_id(self, session_id: str) -> None:
        with self._lock:
            self.session_id = str(session_id or "")

    def start_async(self, callback: Callable[[bool, str], None] | None = None) -> None:
        with self._lock:
            if self._starting:
                return
            self._starting = True

        def work() -> None:
            ok, status = self._start_sync()
            with self._lock:
                self.available = ok
                self.status = status
                self._starting = False
            if callback:
                callback(ok, status)

        self.executor.submit(work)

    def _start_sync(self) -> tuple[bool, str]:
        ok = False
        status = "unavailable"
        if self.auto_start_wsl:
            ok, proc, status = ensure_wsl_solver(
                self.root_dir, self.host, self.port, self.workers, distro=self.wsl_distro,
            )
            self.wsl_process = proc
        else:
            try:
                ok = bool(self.client.ping().get("ok"))
                status = "connected" if ok else "unavailable"
            except Exception as exc:
                status = f"connect-failed:{type(exc).__name__}"
        if ok and self.session_id:
            try:
                self.client.begin_session(self.session_id, timeout=self.session_timeout)
            except Exception as first_exc:
                # A previous GUI may have crashed while owning an exclusive token.
                # Restart the same-version service once rather than inheriting a
                # permanently frozen session on the next Windows launch.
                try:
                    self.client.shutdown_service()
                except Exception:
                    self.client.close()
                ok, proc, restart_status = ensure_wsl_solver(
                    self.root_dir, self.host, self.port, self.workers, distro=self.wsl_distro,
                )
                if not ok:
                    return False, f"session-recovery-failed:{type(first_exc).__name__}:{first_exc}"
                self.wsl_process = proc
                try:
                    self.client.begin_session(self.session_id, timeout=self.session_timeout)
                except Exception as second_exc:
                    return False, f"session-start-failed:{type(second_exc).__name__}:{second_exc}"
                status = f"session-recovered:{restart_status}"
        return ok, status

    def _submit_admitted(self, kind: str, work: Callable[[AdmissionTicket], None]) -> bool:
        ticket = self.admission.ticket(kind)
        if ticket is None:
            return False

        def wrapped() -> None:
            if not self.admission.enter(ticket):
                return
            try:
                work(ticket)
            finally:
                self.admission.leave()

        future = self.executor.submit(wrapped)
        self.admission.attach(ticket, future)
        return True

    def _run_live_decision(self, request: SolverRequest) -> SolverResponse | None:
        try:
            result = self.client.solve(request)
            with self._lock:
                self.available = True
            if result.session_id != self.session_id:
                return None
            return result
        except BaseException as exc:
            with self._lock:
                self.available = False
                self.status = f"solve-failed:{type(exc).__name__}"
            self.client.close()
            raise

    def solve_async(
        self,
        request: SolverRequest,
        on_result: Callable[[SolverResponse], None],
        on_error: Callable[[BaseException], None],
    ) -> dict[str, int | bool]:
        request = self._with_session(request)
        return self.decision_lane.submit(request, on_result, on_error)

    def decision_snapshot(self) -> dict[str, int | bool]:
        lane = getattr(self, "decision_lane", None)
        return lane.snapshot() if lane is not None else {"generation": 0, "worker_active": False, "pending_latest": False, "superseded": 0}

    def _drop_pending_decisions(self) -> None:
        lane = getattr(self, "decision_lane", None)
        if lane is not None: lane.drop_pending()

    def precompute_async(
        self,
        requests: list[SolverRequest],
        on_done: Callable[[dict], None] | None = None,
    ) -> None:
        if not requests:
            return
        requests = [self._with_session(r) for r in requests]

        def work(ticket: AdmissionTicket) -> None:
            try:
                result = self.client.precompute_requests(requests)
                with self._lock:
                    self.available = True
                if on_done and self.admission.is_current(ticket):
                    on_done(result)
            except BaseException:
                with self._lock:
                    self.available = False
                self.client.close()

        self._submit_admitted("precompute", work)

    def _with_session(self, request: SolverRequest) -> SolverRequest:
        if request.session_id == self.session_id:
            return request
        return SolverRequest(
            request_id=request.request_id,
            state=request.state,
            settings=request.settings,
            learned_scores=request.learned_scores,
            purpose=request.purpose,
            session_id=self.session_id,
        )

    def set_background_enabled_async(
        self, enabled: bool, on_done: Callable[[dict], None] | None = None
    ) -> None:
        def work(_ticket: AdmissionTicket) -> None:
            try:
                result = self.client.set_background_enabled(bool(enabled))
                if on_done:
                    on_done(result)
            except BaseException:
                self.client.close()
        self._submit_admitted("background-control", work)

    def prepare_input_exclusive(self, timeout: float = 5.0) -> dict:
        """Close Windows admission, drain IPC, then freeze all server work."""
        if not self.available:
            return {"ok": False, "reason": "solver-unavailable"}
        started = time.monotonic()
        self._drop_pending_decisions()
        closed = self.admission.close()
        remaining = max(0.0, float(timeout) - (time.monotonic() - started))
        ipc = self.admission.wait_idle(remaining)
        if not ipc.get("idle"):
            self.admission.open()
            return {"ok": False, "reason": "windows-ipc-not-idle", "bridge": {**closed, **ipc}}
        remaining = max(0.0, float(timeout) - (time.monotonic() - started))
        try:
            server = self.client.freeze(self.session_id, timeout=remaining)
        except BaseException:
            self.admission.open()
            raise
        if not server.get("ok") or not server.get("idle"):
            self.admission.open()
            return {"ok": False, "reason": server.get("reason", "server-not-idle"), "server": server}
        token = str(server.get("exclusive_token", ""))
        if not token:
            self.admission.open()
            return {"ok": False, "reason": "missing-exclusive-token", "server": server}
        self._exclusive_token = token
        return {
            "ok": True,
            "idle": True,
            "exclusive_token": token,
            "bridge_active": int(ipc.get("active", 0)),
            "bridge_cancelled": int(closed.get("cancelled", 0)),
            "foreground": int(server.get("foreground", -1)),
            "background": int(server.get("background", -1)),
            "server": server,
        }

    def release_input_exclusive(self, token: str | None = None) -> dict:
        actual = str(token or self._exclusive_token or "")
        if not actual:
            return {"ok": False, "reason": "missing-exclusive-token"}
        try:
            result = self.client.release(self.session_id, actual)
        except BaseException:
            # Retry lifecycle control on an independent socket. Admission stays
            # closed unless the server confirms that the exclusive token ended.
            control = SolverClient(self.host, self.port, timeout=max(1.0, self.session_timeout + 1.0))
            try:
                result = control.release(self.session_id, actual)
            finally:
                control.close()
        if not result.get("ok", False):
            self.status = f"exclusive-release-failed:{result.get('reason', 'rejected')}"
            return result
        self._exclusive_token = None
        self.admission.open()
        return result

    def restart_async(self, callback: Callable[[bool, str], None] | None = None) -> None:
        self._drop_pending_decisions()
        with self._lock:
            self.available = False
            self.status = "restarting"
        try:
            self.client.close()
        except Exception:
            pass
        self.start_async(callback)

    def close(self) -> None:
        """Detach this Windows session without depending on worker IPC health."""
        self._drop_pending_decisions()
        self.admission.close()
        self.admission.wait_idle(1.0)

        control = SolverClient(self.host, self.port, timeout=max(1.0, self.session_timeout + 1.0))
        try:
            if self.session_id:
                try:
                    control.end_session(self.session_id, timeout=self.session_timeout)
                except Exception:
                    pass
            self._exclusive_token = None
            if self.wsl_process is not None:
                try:
                    control.shutdown_service()
                except Exception:
                    control.close()
        finally:
            control.close()
            self.client.close()
            self.executor.shutdown(wait=False, cancel_futures=True)
