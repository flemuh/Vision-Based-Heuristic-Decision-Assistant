"""Launch a persistent solver with the *same Windows Python* as the GUI.

The service runs independently of Tk, retains hot solver caches, and never uses WSL.
Never shut down an unrelated listener on the configured port.
"""
from __future__ import annotations
import os
import subprocess
import sys
import time
from pathlib import Path

from vision_engine import __version__
from .client import SolverClient


def _matches(client: SolverClient, root: Path) -> bool:
    info = client.ping()
    return (bool(info.get("ok"))
            and str(info.get("app_version")) == __version__
            and os.path.normcase(os.path.abspath(str(info.get("project_root", ""))))
            == os.path.normcase(os.path.abspath(str(root))))


def ensure_native_solver(root_dir: Path, host: str, port: int, workers: int, *, timeout: float = 12.0):
    """Returns (ready, owned_child_or_none, explanation). No silent WSL fallback."""
    root = Path(root_dir).resolve()
    if host not in {"127.0.0.1", "localhost"}:
        return False, None, "native backend requires loopback host"
    client = SolverClient(host, port, timeout=0.6)
    try:
        try:
            info = client.ping()
            if _matches(client, root):
                return True, None, "native-existing-matching-service"
            return False, None, ("port-in-use-different-service:" + str(info.get("project_root", "unknown")))
        except (OSError, ConnectionError, TimeoutError):
            pass
        finally:
            client.close()
        log = root / "data" / "native-solver.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        with log.open("ab", buffering=0) as handle:
            proc = subprocess.Popen(
                [sys.executable, "-m", "vision_engine.solver.service", "--host", "127.0.0.1",
                 "--port", str(port), "--workers", str(max(2, workers))],
                cwd=str(root), stdin=subprocess.DEVNULL, stdout=handle,
                stderr=subprocess.STDOUT, creationflags=creationflags,
            )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                return False, None, f"native-exited:{proc.returncode}; see {log}"
            try:
                if _matches(client, root):
                    return True, proc, "native-started"
            except (OSError, ConnectionError, TimeoutError):
                pass
            finally:
                client.close()
            time.sleep(0.08)
        proc.terminate()
        return False, None, f"native-start-timeout; see {log}"
    finally:
        client.close()
