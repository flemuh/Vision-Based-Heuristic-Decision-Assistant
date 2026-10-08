from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from threading import RLock

from .protocol import SolverRequest, request_to_wire


class SolverLRUCache:
    def __init__(self, max_entries: int = 4096):
        self.max_entries = max(16, int(max_entries))
        self._data: OrderedDict[str, tuple] = OrderedDict()
        self._lock = RLock()

    @staticmethod
    def key(request: SolverRequest) -> str:
        wire = request_to_wire(request)
        # Transport/request identity does not affect the mathematical answer.
        wire.pop("request_id", None)
        wire.pop("purpose", None)
        wire["state"].pop("episode_id", None)
        wire["state"].pop("state_version", None)
        wire["state"].pop("strategy_version", None)
        raw = json.dumps(wire, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        return hashlib.blake2b(raw, digest_size=16).hexdigest()

    def get(self, key: str):
        with self._lock:
            value = self._data.get(key)
            if value is not None:
                self._data.move_to_end(key)
            return value

    def put(self, key: str, value) -> None:
        with self._lock:
            self._data[key] = tuple(value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)
