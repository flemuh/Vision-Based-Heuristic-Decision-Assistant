from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from vision_engine import __version__
from .client import SolverClient


def _creationflags() -> int:
    if os.name != "nt":
        return 0
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _clean_distro(distro: str) -> str:
    return (distro or "Ubuntu").strip() or "Ubuntu"


def _wsl_prefix(distro: str) -> list[str]:
    """Command prefix for an explicit distro without relying on WSL default."""
    return ["wsl.exe", "-d", _clean_distro(distro), "--"]


def wsl_project_command(root_dir: Path, distro: str, *command: str) -> list[str]:
    """Run *command* in the project directory inside the chosen WSL distro.

    Modern WSL accepts an absolute Windows directory in ``--cd``. This is much
    more robust than shelling out to ``wslpath``: the launcher already knows its
    own project directory, so no path discovery/translation step is necessary.
    """
    root = str(Path(root_dir).resolve())
    return [
        "wsl.exe", "-d", _clean_distro(distro),
        "--cd", root,
        "--", *command,
    ]


def check_wsl_python(distro: str = "Ubuntu") -> tuple[bool, str]:
    """Return whether python3 is available in the selected solver distro."""
    if os.name != "nt":
        return False, "not-windows"
    try:
        proc = subprocess.run(
            [*_wsl_prefix(distro), "python3", "--version"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=8,
            creationflags=_creationflags(),
        )
    except Exception as exc:
        return False, f"{type(exc).__name__}:{exc}"
    output = (proc.stdout or "").strip()
    return proc.returncode == 0, output or f"exit={proc.returncode}"


def check_wsl_project(root_dir: Path, distro: str = "Ubuntu") -> tuple[bool, str]:
    """Validate that WSL can enter this exact project directory and import it."""
    if os.name != "nt":
        return False, "not-windows"
    try:
        proc = subprocess.run(
            wsl_project_command(
                root_dir, distro, "python3", "-c",
                "import os, vision_engine; print(os.getcwd()); print(vision_engine.__version__)",
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=12,
            creationflags=_creationflags(),
        )
    except Exception as exc:
        return False, f"{type(exc).__name__}:{exc}"
    output = (proc.stdout or "").strip()
    return proc.returncode == 0, output or f"exit={proc.returncode}"


def launch_wsl_solver(
    root_dir: Path,
    port: int = 57641,
    workers: int = 6,
    distro: str = "Ubuntu",
) -> subprocess.Popen | None:
    """Start the persistent solver from this project's own directory in WSL.

    No ``wslpath``, shell ``cd`` or global/default distro is involved. The same
    extracted folder that launched the Windows UI is used as WSL's working
    directory via ``wsl.exe --cd <windows-project-path>``.
    """
    if os.name != "nt":
        return None

    ok, detail = check_wsl_python(distro)
    if not ok:
        raise RuntimeError(f"WSL distro {distro!r} has no usable python3: {detail}")

    ok, detail = check_wsl_project(root_dir, distro=distro)
    if not ok:
        raise RuntimeError(f"WSL cannot open/import this project directory: {detail}")

    log_path = Path(root_dir) / "data" / "wsl-solver.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = wsl_project_command(
        root_dir, distro,
        "python3", "-m", "vision_engine.solver.service",
        "--host", "0.0.0.0",
        "--port", str(int(port)),
        "--workers", str(int(workers)),
    )

    # Redirect Linux service output straight to a file owned by the Windows
    # project. This avoids bash/redirection/path quoting entirely.
    log_handle = log_path.open("ab", buffering=0)
    try:
        proc = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            creationflags=_creationflags(),
        )
    except Exception:
        log_handle.close()
        raise
    log_handle.close()
    return proc


def _running_solver_matches(info: dict) -> bool:
    return bool(info.get("ok")) and str(info.get("app_version", "")) == __version__


def _stop_stale_solver(client: SolverClient) -> None:
    try:
        client.shutdown_service()
    except Exception:
        client.close()
    # Give the old listener a short window to release the fixed localhost port.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        time.sleep(0.10)
        probe = SolverClient(client.host, client.port, timeout=0.20)
        try:
            probe.ping()
        except Exception:
            probe.close()
            return
        finally:
            probe.close()


def ensure_wsl_solver(
    root_dir: Path,
    host: str = "127.0.0.1",
    port: int = 57641,
    workers: int = 6,
    wait_seconds: float = 5.0,
    distro: str = "Ubuntu",
) -> tuple[bool, subprocess.Popen | None, str]:
    """Ensure the current release's persistent solver is running.

    A solver left behind by an older extracted release is treated as stale and
    restarted, preventing a new GUI from silently talking to old solver code.
    """
    client = SolverClient(host, port, timeout=0.5)
    stale_version = ""
    try:
        info = client.ping()
        if _running_solver_matches(info):
            client.close()
            return True, None, f"already-running:{_clean_distro(distro)}:v{__version__}"
        if info.get("ok"):
            stale_version = str(info.get("app_version") or "unknown")
            _stop_stale_solver(client)
        else:
            client.close()
    except Exception:
        client.close()

    if os.name != "nt":
        return False, None, "not-windows"
    try:
        proc = launch_wsl_solver(root_dir, port=port, workers=workers, distro=distro)
    except Exception as exc:
        return False, None, f"launch-failed:{_clean_distro(distro)}:{type(exc).__name__}:{exc}"

    deadline = time.monotonic() + max(0.5, wait_seconds)
    while time.monotonic() < deadline:
        time.sleep(0.10)
        probe = SolverClient(host, port, timeout=0.4)
        try:
            info = probe.ping()
            if _running_solver_matches(info):
                probe.close()
                prefix = f"restarted-stale:{stale_version}->" if stale_version else "started:"
                return True, proc, f"{prefix}{_clean_distro(distro)}:v{__version__}"
        except Exception:
            pass
        finally:
            probe.close()
    return False, proc, f"timeout:{_clean_distro(distro)}"
