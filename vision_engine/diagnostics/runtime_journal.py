from __future__ import annotations

"""Low-overhead continuous JSONL runtime journal.

The live path only enqueues small Python dicts.  A dedicated daemon writer rotates
files by local day/hour so diagnostics can be collected after lag/WAITING/input
failures without synchronous disk writes in the mouse/vision hot path.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import threading
import time
from typing import Any


@dataclass(frozen=True)
class JournalStats:
    queued: int
    dropped: int
    written: int


class RuntimeJournal:
    def __init__(
        self, base_dir: str | Path, session_id: str, *, capacity: int = 8192,
        keep_days: int = 7, max_total_mb: int = 250,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.session_id = str(session_id)
        self.keep_days = max(1, int(keep_days))
        self.max_total_bytes = max(16, int(max_total_mb)) * 1024 * 1024
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=max(256, int(capacity)))
        self._dropped = 0
        self._written = 0
        self._closed = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._writer, name="runtime-json-journal", daemon=True)
        self._thread.start()

    def stats(self) -> JournalStats:
        with self._lock:
            return JournalStats(self._queue.qsize(), self._dropped, self._written)

    def record(self, event: str, *, level: str = "INFO", **data: object) -> None:
        if self._closed:
            return
        now = datetime.now().astimezone()
        row: dict[str, Any] = {
            "ts": now.isoformat(timespec="milliseconds"),
            "utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "monotonic": round(time.monotonic(), 6),
            "event": str(event),
            "level": str(level),
            "session_id": self.session_id,
            "pid": os.getpid(),
            "thread": threading.current_thread().name,
        }
        row.update(data)
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            with self._lock:
                self._dropped += 1

    def close(self, timeout: float = 1.5) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            # Never block the GUI close path. The daemon writer will flush what it can.
            pass
        self._thread.join(timeout=max(0.0, float(timeout)))


    def _apply_retention(self, *, exclude: Path | None = None) -> None:
        """Best-effort cleanup of old JSONL logs on the writer thread only."""
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            files = [p for p in self.base_dir.rglob("runtime_*.jsonl") if p.is_file()]
            if not files:
                return
            now = time.time()
            cutoff = now - (self.keep_days * 86400)
            # Age cap first. Never touch the file currently open by this journal.
            for path in list(files):
                try:
                    if exclude is not None and path.resolve() == exclude.resolve():
                        continue
                    if path.stat().st_mtime < cutoff:
                        path.unlink(missing_ok=True)
                except OSError:
                    pass
            files = [p for p in self.base_dir.rglob("runtime_*.jsonl") if p.is_file()]
            sized: list[tuple[float, int, Path]] = []
            total = 0
            for path in files:
                try:
                    st = path.stat()
                except OSError:
                    continue
                total += int(st.st_size)
                sized.append((float(st.st_mtime), int(st.st_size), path))
            if total > self.max_total_bytes:
                for _mtime, size, path in sorted(sized, key=lambda x: x[0]):
                    if total <= self.max_total_bytes:
                        break
                    try:
                        if exclude is not None and path.resolve() == exclude.resolve():
                            continue
                        path.unlink(missing_ok=True)
                        total -= size
                    except OSError:
                        pass
            # Remove empty day folders left after cleanup.
            for folder in sorted((p for p in self.base_dir.iterdir() if p.is_dir()), reverse=True):
                try:
                    if not any(folder.iterdir()):
                        folder.rmdir()
                except OSError:
                    pass
        except Exception:
            # Diagnostics retention must never affect the application.
            pass

    def _writer(self) -> None:
        handle = None
        current_path: Path | None = None
        current_key: tuple[str, str] | None = None
        try:
            self._apply_retention()
            while True:
                try:
                    row = self._queue.get(timeout=0.5)
                except queue.Empty:
                    if self._closed:
                        break
                    continue
                if row is None:
                    break
                try:
                    stamp = datetime.fromisoformat(str(row["ts"]))
                    day = stamp.strftime("%Y-%m-%d")
                    hour = stamp.strftime("%Y%m%d_%H")
                    key = (day, hour)
                    if key != current_key:
                        if handle is not None:
                            handle.flush(); handle.close()
                        folder = self.base_dir / day
                        folder.mkdir(parents=True, exist_ok=True)
                        current_path = folder / f"runtime_{hour}.jsonl"
                        handle = current_path.open("a", encoding="utf-8", buffering=1)
                        current_key = key
                        self._apply_retention(exclude=current_path)
                    handle.write(json.dumps(row, ensure_ascii=False, default=str, separators=(",", ":")) + "\n")
                    with self._lock:
                        self._written += 1
                except Exception:
                    # Diagnostics must never be able to take down live play.
                    continue
        finally:
            if handle is not None:
                try:
                    handle.flush(); handle.close()
                except Exception:
                    pass
