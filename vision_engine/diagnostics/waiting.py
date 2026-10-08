from __future__ import annotations

from dataclasses import asdict
import json
import socketserver
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Callable

from .resource_trend import trend_rows
from .ring_buffer import InMemoryDiagnostics
from .win32 import win32_snapshot


class WaitingDiagnostics:
    """In-memory runtime timeline plus a localhost-only diagnostic probe.

    Nothing is written to disk by the Auto Play hot path. A separate diagnostic
    process can request snapshots while the app is stuck and write them later.
    """

    def __init__(self, capacity: int = 4096) -> None:
        self.events = InMemoryDiagnostics(capacity=capacity)
        self._provider: Callable[[], dict] | None = None
        self._server: socketserver.ThreadingTCPServer | None = None
        self._thread: threading.Thread | None = None
        self._port = 0
        self._sink: Callable[[str, dict], None] | None = None

    @property
    def port(self) -> int:
        return self._port

    def set_provider(self, provider: Callable[[], dict]) -> None:
        self._provider = provider

    def set_sink(self, sink: Callable[[str, dict], None] | None) -> None:
        """Mirror ring-buffer events to an optional low-overhead external sink."""
        self._sink = sink

    def record(self, kind: str, **data: object) -> None:
        self.events.record(kind, **data)
        if self._sink is not None:
            try:
                self._sink(str(kind), dict(data))
            except BaseException:
                pass

    def snapshot(self, *, deep: bool = False) -> dict:
        runtime: dict = {}
        if self._provider is not None:
            try:
                runtime = dict(self._provider())
            except BaseException as exc:
                runtime = {"provider_error": f"{type(exc).__name__}: {exc}"}
        events = [asdict(event) for event in self.events.snapshot()[-256:]]
        win32 = [asdict(event) for event in win32_snapshot()[-256:]]
        out = {
            "ok": True,
            "probe_monotonic": time.monotonic(),
            "runtime": runtime,
            "events": events,
            "win32_events": win32,
            "resource_trend": trend_rows(512),
        }
        if deep:
            out["threads"] = _thread_snapshot()
        return out

    def start(self, marker_path: Path, preferred_port: int = 57642) -> int:
        if self._server is not None:
            return self._port
        diag = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                raw = self.rfile.readline(65536)
                try:
                    request = json.loads(raw.decode("utf-8") or "{}")
                except Exception:
                    request = {}
                command = str(request.get("command", "snapshot"))
                if command == "ping":
                    response = {"ok": True, "port": diag.port}
                elif command == "deep_snapshot":
                    response = diag.snapshot(deep=True)
                else:
                    response = diag.snapshot(deep=False)
                self.wfile.write((json.dumps(response, default=str) + "\n").encode("utf-8"))

        class Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True

        last_error: BaseException | None = None
        for port in range(int(preferred_port), int(preferred_port) + 8):
            try:
                server = Server(("127.0.0.1", port), Handler)
                self._server = server
                self._port = int(port)
                break
            except OSError as exc:
                last_error = exc
        if self._server is None:
            raise RuntimeError(f"could not bind waiting diagnostic probe: {last_error}")

        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(
            json.dumps({"host": "127.0.0.1", "port": self._port, "pid": _pid()}),
            encoding="utf-8",
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="waiting-diagnostic-probe",
            daemon=True,
        )
        self._thread.start()
        return self._port

    def stop(self, marker_path: Path | None = None) -> None:
        server = self._server
        self._server = None
        if server is not None:
            try:
                server.shutdown()
            except Exception:
                pass
            try:
                server.server_close()
            except Exception:
                pass
        if marker_path is not None:
            try:
                marker_path.unlink(missing_ok=True)
            except Exception:
                pass


def _pid() -> int:
    import os
    return int(os.getpid())


def _thread_snapshot() -> list[dict]:
    frames = sys._current_frames()
    rows: list[dict] = []
    for thread in threading.enumerate():
        frame = frames.get(thread.ident or -1)
        stack: list[str] = []
        if frame is not None:
            stack = [line.rstrip() for line in traceback.format_stack(frame, limit=14)]
        rows.append({
            "name": thread.name,
            "ident": thread.ident,
            "daemon": thread.daemon,
            "alive": thread.is_alive(),
            "stack": stack,
        })
    return rows
