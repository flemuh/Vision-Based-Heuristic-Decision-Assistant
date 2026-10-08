from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import shutil
import socket
import sqlite3
import tempfile
import zipfile


def _data_dir() -> Path:
    override = os.environ.get("SPEEDLORA_JEWEL_DATA_DIR")
    if override:
        return Path(override).expanduser()
    local = os.environ.get("LOCALAPPDATA")
    if local:
        p1 = Path(local) / "Speedlora" / "JewelBingo-V1826-StateIntegrity"
        if p1.exists():
            return p1
        return Path(local) / "Speedlora" / "JewelBingo"
    return Path.home() / ".speedlora" / "jewel-bingo"


def _probe_snapshot(data_dir: Path) -> dict | None:
    marker = data_dir / "waiting_probe.json"
    if not marker.exists():
        return None
    try:
        info = json.loads(marker.read_text(encoding="utf-8"))
        host = str(info.get("host", "127.0.0.1"))
        port = int(info["port"])
        with socket.create_connection((host, port), timeout=2.0) as sock:
            sock.sendall(b'{"command":"deep_snapshot"}\n')
            fh = sock.makefile("rb")
            raw = fh.readline(4 * 1024 * 1024)
        return json.loads(raw.decode("utf-8"))
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _sqlite_backup(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    source = sqlite3.connect(f"file:{src.as_posix()}?mode=ro", uri=True, timeout=2.0)
    target = sqlite3.connect(dst)
    try:
        source.backup(target)
    finally:
        target.close(); source.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Export recent Speedlora runtime diagnostics")
    ap.add_argument("--hours", type=int, default=8, help="hours of JSONL history to include")
    args = ap.parse_args()

    data_dir = _data_dir()
    downloads = Path.home() / "Downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = downloads / f"speedlora_runtime_diagnostics_{stamp}.zip"

    with tempfile.TemporaryDirectory(prefix="speedlora_diag_") as tmp:
        root = Path(tmp)
        summary = {
            "created_at": datetime.now().astimezone().isoformat(),
            "data_dir": str(data_dir),
            "hours": int(args.hours),
        }
        (root / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        cutoff = datetime.now() - timedelta(hours=max(1, int(args.hours)))
        runtime_root = data_dir / "logs" / "runtime"
        if runtime_root.exists():
            for path in runtime_root.rglob("runtime_*.jsonl"):
                try:
                    if datetime.fromtimestamp(path.stat().st_mtime) < cutoff:
                        continue
                    rel = path.relative_to(data_dir)
                    dest = root / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, dest)
                except OSError:
                    pass

        for rel in (
            Path("logs") / "assistant.log",
            Path("app_config.json"),
            Path("vision_config.json"),
            Path("waiting_probe.json"),
        ):
            src = data_dir / rel
            if src.exists():
                dest = root / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(src, dest)
                except OSError:
                    pass

        snap = _probe_snapshot(data_dir)
        if snap is not None:
            (root / "live_deep_snapshot.json").write_text(
                json.dumps(snap, indent=2, default=str), encoding="utf-8"
            )

        try:
            _sqlite_backup(data_dir / "bingo.sqlite3", root / "bingo_snapshot.sqlite3")
        except Exception as exc:
            (root / "sqlite_backup_error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")

        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in root.rglob("*"):
                if path.is_file():
                    zf.write(path, path.relative_to(root).as_posix())

    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
