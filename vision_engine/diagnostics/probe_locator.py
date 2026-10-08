from __future__ import annotations

import json
import os
from pathlib import Path

from vision_engine.data_home import default_user_data_dir


def _valid_marker(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return bool(data.get("port")) and int(data.get("pid", 0)) > 0
    except Exception:
        return False


def locate_live_probe_marker() -> Path | None:
    """Find the live Assistant waiting_probe.json without mutating anything."""
    candidates: list[Path] = [default_user_data_dir() / "waiting_probe.json"]
    local = os.environ.get("LOCALAPPDATA")
    if local:
        speedlora = Path(local) / "Speedlora"
        if speedlora.exists():
            candidates.extend(speedlora.glob("JewelBingo*/waiting_probe.json"))
    seen: set[Path] = set()
    valid: list[Path] = []
    for p in candidates:
        try:
            r = p.resolve()
        except Exception:
            r = p
        if r in seen:
            continue
        seen.add(r)
        if p.exists() and _valid_marker(p):
            valid.append(p)
    if not valid:
        return None
    valid.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return valid[0]
