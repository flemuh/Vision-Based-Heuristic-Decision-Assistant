from __future__ import annotations

import argparse
import socketserver
import threading

from vision_engine import __version__
from vision_engine.constants import JEWELS
from vision_engine.versioning import PROTOCOL_VERSION

from .engine import SolverEngine
from .protocol import (
    SolverRequest,
    request_from_wire,
    response_to_wire,
    settings_from_wire,
    state_from_wire,
)
from .transport import recv_message, send_message


class SolverRequestHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        server: SolverTCPServer = self.server  # type: ignore[assignment]
        self.request.settimeout(server.socket_timeout)
        while True:
            try:
                msg = recv_message(self.request)
            except (ConnectionError, OSError):
                return
            if not self._dispatch(server, msg):
                return

    def _dispatch(self, server: "SolverTCPServer", msg: dict) -> bool:
        command = str(msg.get("command", ""))
        try:
            if command == "ping":
                self._send({
                    "ok": True,
                    "protocol_version": PROTOCOL_VERSION,
                    "app_version": __version__,
                    "cache_entries": len(server.engine.cache),
                    "workers": server.engine.workers,
                    **server.engine.status(),
                })
            elif command == "shutdown":
                self._send({"ok": True})
                threading.Thread(target=server.shutdown, daemon=True).start()
                return False
            elif command == "begin_session":
                self._send({"ok": True, **server.engine.begin_session(
                    str(msg.get("session_id", "")), float(msg.get("timeout", 5.0))
                )})
            elif command == "end_session":
                info = server.engine.end_session(
                    str(msg.get("session_id", "")), float(msg.get("timeout", 5.0))
                )
                self._send({"ok": bool(info.get("ok")), **info})
            elif command == "solver_status":
                self._send({"ok": True, **server.engine.status()})
            elif command == "background_status":
                self._send({"ok": True, **server.engine.background_status()})
            elif command == "set_background_enabled":
                self._send({"ok": True, **server.engine.set_background_enabled(bool(msg.get("enabled", True)))})
            elif command == "wait_background_idle":
                self._send({"ok": True, **server.engine.wait_background_idle(float(msg.get("timeout", 0.0)))})
            elif command == "freeze":
                info = server.engine.freeze(str(msg.get("session_id", "")), float(msg.get("timeout", 5.0)))
                self._send({"ok": bool(info.get("ok")), **info})
            elif command == "release":
                info = server.engine.release(str(msg.get("session_id", "")), str(msg.get("exclusive_token", "")))
                self._send({"ok": bool(info.get("ok")), **info})
            elif command == "solve":
                req = request_from_wire(msg["request"])
                self._send({"ok": True, "response": response_to_wire(server.engine.solve(req))})
            elif command == "precompute":
                self._handle_precompute_state(server, msg)
            elif command == "precompute_requests":
                requests = [request_from_wire(item) for item in msg.get("requests", ())]
                self._send({"ok": True, **server.engine.precompute(requests)})
            else:
                self._send({"ok": False, "error": f"unknown command: {command}"})
        except BaseException as exc:
            self._send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return True

    def _handle_precompute_state(self, server: "SolverTCPServer", msg: dict) -> None:
        state = state_from_wire(msg["state"])
        settings = settings_from_wire(msg["settings"])
        learned = tuple((int(p), float(s)) for p, s in msg.get("learned_scores", ()))
        session_id = str(msg.get("session_id", ""))
        requests = [
            SolverRequest(
                request_id=-(i + 1), state=state.with_jewel(jewel), settings=settings,
                learned_scores=learned, purpose="precompute", session_id=session_id,
            )
            for i, jewel in enumerate(JEWELS)
        ]
        self._send({"ok": True, **server.engine.precompute(requests)})

    def _send(self, payload: dict) -> None:
        send_message(self.request, payload)


class SolverTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, engine: SolverEngine, *, socket_timeout: float = 120.0):
        self.engine = engine
        self.socket_timeout = socket_timeout
        super().__init__(address, SolverRequestHandler)


def serve(host: str = "127.0.0.1", port: int = 57641, workers: int | None = None) -> None:
    engine = SolverEngine(workers=workers)
    server = SolverTCPServer((host, int(port)), engine)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
        engine.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Speedlora Jewel Bingo persistent solver service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=57641)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    serve(args.host, args.port, args.workers)


if __name__ == "__main__":
    main()
