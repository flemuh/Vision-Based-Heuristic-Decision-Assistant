from __future__ import annotations

import socket
from dataclasses import replace
from threading import RLock
import uuid

from vision_engine.live.state import GameState

from .protocol import (
    SolverRequest,
    SolverResponse,
    SolverSettings,
    request_to_wire,
    response_from_wire,
    settings_to_wire,
    state_to_wire,
)
from .transport import recv_message, send_message


class SolverClient:
    """Persistent TCP client used by the Windows bridge against WSL localhost."""

    def __init__(self, host: str = "127.0.0.1", port: int = 57641, timeout: float = 30.0):
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)
        self._sock: socket.socket | None = None
        self._lock = RLock()
        self._implicit_session_id = f"client_{uuid.uuid4().hex}"
        self._implicit_session_started = False

    def close(self) -> None:
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.close()
                except OSError:
                    pass
            self._sock = None

    def _connect(self) -> socket.socket:
        if self._sock is None:
            self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
            self._sock.settimeout(self.timeout)
        return self._sock

    def _roundtrip(self, payload: dict) -> dict:
        with self._lock:
            for attempt in range(2):
                try:
                    sock = self._connect()
                    send_message(sock, payload)
                    return recv_message(sock)
                except (OSError, ConnectionError):
                    self.close()
                    if attempt:
                        raise
            raise RuntimeError("unreachable")

    @staticmethod
    def _require_ok(msg: dict, default: str) -> dict:
        if not msg.get("ok"):
            raise RuntimeError(msg.get("error") or msg.get("reason") or default)
        return msg

    def _ensure_implicit_session(self) -> str:
        if not self._implicit_session_started:
            self.begin_session(self._implicit_session_id, timeout=5.0)
            self._implicit_session_started = True
        return self._implicit_session_id

    def ping(self) -> dict:
        return self._roundtrip({"command": "ping"})

    def begin_session(self, session_id: str, timeout: float = 5.0) -> dict:
        return self._require_ok(self._roundtrip({
            "command": "begin_session", "session_id": session_id, "timeout": float(timeout),
        }), "solver begin-session error")

    def end_session(self, session_id: str, timeout: float = 5.0) -> dict:
        return self._require_ok(self._roundtrip({
            "command": "end_session", "session_id": session_id, "timeout": float(timeout),
        }), "solver end-session error")

    def status(self) -> dict:
        return self._require_ok(self._roundtrip({"command": "solver_status"}), "solver status error")

    def freeze(self, session_id: str, timeout: float = 5.0) -> dict:
        return self._require_ok(self._roundtrip({
            "command": "freeze", "session_id": session_id, "timeout": float(timeout),
        }), "solver freeze error")

    def release(self, session_id: str, exclusive_token: str) -> dict:
        return self._require_ok(self._roundtrip({
            "command": "release", "session_id": session_id, "exclusive_token": exclusive_token,
        }), "solver release error")

    def background_status(self) -> dict:
        return self._require_ok(self._roundtrip({"command": "background_status"}), "solver background status error")

    def set_background_enabled(self, enabled: bool) -> dict:
        return self._require_ok(self._roundtrip({
            "command": "set_background_enabled", "enabled": bool(enabled),
        }), "solver background control error")

    def wait_background_idle(self, timeout: float = 3.0) -> dict:
        return self._require_ok(self._roundtrip({
            "command": "wait_background_idle", "timeout": max(0.0, float(timeout)),
        }), "solver background barrier error")

    def precompute_state(
        self,
        state: GameState,
        settings: SolverSettings,
        learned_scores: tuple[tuple[int, float], ...] = (),
        *,
        session_id: str = "",
    ) -> dict:
        actual_session = session_id or self._ensure_implicit_session()
        return self._require_ok(self._roundtrip({
            "command": "precompute",
            "state": state_to_wire(state),
            "settings": settings_to_wire(settings),
            "learned_scores": [[p, score] for p, score in learned_scores],
            "session_id": actual_session,
        }), "solver precompute error")

    def precompute_requests(self, requests: list[SolverRequest]) -> dict:
        if requests and any(not r.session_id for r in requests):
            session_id = self._ensure_implicit_session()
            requests = [replace(r, session_id=r.session_id or session_id) for r in requests]
        return self._require_ok(self._roundtrip({
            "command": "precompute_requests",
            "requests": [request_to_wire(r) for r in requests],
        }), "solver precompute error")

    def solve(self, request: SolverRequest) -> SolverResponse:
        if not request.session_id:
            request = replace(request, session_id=self._ensure_implicit_session())
        msg = self._require_ok(self._roundtrip({
            "command": "solve", "request": request_to_wire(request),
        }), "solver service error")
        return response_from_wire(msg["response"])

    def shutdown_service(self) -> bool:
        try:
            msg = self._roundtrip({"command": "shutdown"})
            return bool(msg.get("ok"))
        finally:
            self.close()
