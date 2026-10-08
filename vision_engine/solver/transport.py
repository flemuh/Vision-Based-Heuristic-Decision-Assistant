from __future__ import annotations

import json
import socket
import struct
from typing import Any

_HEADER = struct.Struct("!I")
_MAX_MESSAGE = 8 * 1024 * 1024


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    parts = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("solver connection closed")
        parts.append(chunk)
        remaining -= len(chunk)
    return b"".join(parts)


def send_message(sock: socket.socket, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if len(body) > _MAX_MESSAGE:
        raise ValueError("solver message too large")
    sock.sendall(_HEADER.pack(len(body)) + body)


def recv_message(sock: socket.socket) -> dict[str, Any]:
    (size,) = _HEADER.unpack(_recv_exact(sock, _HEADER.size))
    if size > _MAX_MESSAGE:
        raise ValueError("solver message too large")
    return json.loads(_recv_exact(sock, size).decode("utf-8"))
