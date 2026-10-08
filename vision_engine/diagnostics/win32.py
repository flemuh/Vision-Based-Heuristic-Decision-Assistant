from __future__ import annotations

from typing import Callable

from .ring_buffer import InMemoryDiagnostics

WIN32_DIAGNOSTICS = InMemoryDiagnostics(capacity=1024)
_WIN32_SINK: Callable[[str, dict], None] | None = None


def set_win32_sink(sink: Callable[[str, dict], None] | None) -> None:
    """Mirror Win32 events to an optional non-blocking sink.

    The GUI installs the asynchronous RuntimeJournal here. The mouse hot path
    still performs no disk I/O; it only enqueues a small dict.
    """
    global _WIN32_SINK
    _WIN32_SINK = sink


def record_win32(kind: str, **data: object) -> None:
    WIN32_DIAGNOSTICS.record(kind, **data)
    sink = _WIN32_SINK
    if sink is not None:
        try:
            sink(str(kind), dict(data))
        except BaseException:
            pass


def win32_snapshot():
    return WIN32_DIAGNOSTICS.snapshot()
