from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time


@dataclass(frozen=True, slots=True)
class DiagnosticEvent:
    sequence: int
    monotonic: float
    kind: str
    data: dict[str, object]


class InMemoryDiagnostics:
    """Bounded diagnostics buffer. No disk I/O is performed on the input path."""

    def __init__(self, capacity: int = 512):
        self._events: deque[DiagnosticEvent] = deque(maxlen=max(16, int(capacity)))
        self._lock = threading.Lock()
        self._sequence = 0

    def record(self, kind: str, **data: object) -> None:
        with self._lock:
            self._sequence += 1
            self._events.append(DiagnosticEvent(self._sequence, time.monotonic(), str(kind), dict(data)))

    def snapshot(self) -> tuple[DiagnosticEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
