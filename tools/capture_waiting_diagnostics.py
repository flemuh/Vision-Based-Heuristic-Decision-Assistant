from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import zipfile

# Executed directly from tools/ by diagnose_waiting.bat; make the project root
# importable without requiring an editable install.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision_engine.data_home import default_user_data_dir
from vision_engine.solver.client import SolverClient


def _probe(marker: Path, deep: bool = False, timeout: float = 1.0) -> dict:
    info = json.loads(marker.read_text(encoding="utf-8"))
    host = str(info.get("host", "127.0.0.1"))
    port = int(info["port"])
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        command = "deep_snapshot" if deep else "snapshot"
        sock.sendall((json.dumps({"command": command}) + "\n").encode("utf-8"))
        data = b""
        while not data.endswith(b"\n"):
            chunk = sock.recv(65536)
            if not chunk:
                break
            data += chunk
    return json.loads(data.decode("utf-8") or "{}")


def _solver_status(runtime: dict) -> dict:
    host = str(runtime.get("solver_host", "127.0.0.1"))
    port = int(runtime.get("solver_port", 57641))
    client = SolverClient(host, port, timeout=1.5)
    try:
        return client.status()
    except BaseException as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        client.close()


def _command_output(args: list[str], timeout: float = 4.0) -> str:
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
        return (proc.stdout or "") + (proc.stderr or "")
    except BaseException as exc:
        return f"{type(exc).__name__}: {exc}\n"


def _wsl_processes(distro: str) -> str:
    if os.name != "nt":
        return "WSL process capture is available only on Windows.\n"
    command = "ps -eo pid,ppid,stat,etime,cmd | grep -E 'python|solver|jewel|speedlora' | grep -v grep || true"
    args = ["wsl.exe"]
    if distro:
        args += ["-d", distro]
    args += ["--", "bash", "-lc", command]
    return _command_output(args, timeout=5.0)


def _windows_processes() -> str:
    if os.name != "nt":
        return "Windows tasklist capture is available only on Windows.\n"
    raw = _command_output(["tasklist", "/FO", "CSV", "/NH"], timeout=5.0)
    keep = []
    for line in raw.splitlines():
        low = line.lower()
        if any(name in low for name in ("python", "wsl", "ubuntu", "docker")):
            keep.append(line)
    return "\n".join(keep) + "\n"


def _stable(values: list[object]) -> bool:
    return bool(values) and all(value == values[0] for value in values)


def _classify(samples: list[dict]) -> tuple[str, list[str]]:
    if not samples:
        return "PROBE_UNREACHABLE", ["No runtime sample was collected from the Assistant."]
    runtime = [sample.get("probe", {}).get("runtime", {}) for sample in samples]
    solver = [sample.get("solver", {}) for sample in samples]
    phases = [str(r.get("commit_phase", "unknown")) for r in runtime]
    masks = [r.get("mask") for r in runtime]
    heartbeats = [r.get("monitor_heartbeat") for r in runtime]
    bridge_open = [r.get("bridge_admission", {}).get("open") for r in runtime]
    bridge_active = [int(r.get("bridge_admission", {}).get("active", 0) or 0) for r in runtime]
    monitor_q = [bool(r.get("monitor_quiesced")) for r in runtime]
    monitor_req = [bool(r.get("monitor_requested")) for r in runtime]
    frozen = [bool(s.get("frozen")) for s in solver if s]
    freeze_pending = [bool(s.get("freeze_pending")) for s in solver if s]
    fg = [int(s.get("foreground", 0) or 0) for s in solver if s]
    bg = [int(s.get("background", 0) or 0) for s in solver if s]
    details: list[str] = []

    if any(bridge_active):
        details.append(f"Windows bridge active IPC remained non-zero: max={max(bridge_active)}")
    if any(fg) or any(bg):
        details.append(f"WSL solver work remained active: foreground max={max(fg or [0])}, background max={max(bg or [0])}")
    if any(freeze_pending):
        details.append("WSL GLOBAL FREEZE remained pending during capture.")
    if any(frozen):
        details.append("WSL solver remained frozen during capture.")
    if any(monitor_q) or any(monitor_req):
        details.append("Monitor safe-point gate remained requested/quiesced during capture.")

    last = runtime[-1]
    phase = str(last.get("commit_phase", "unknown"))
    if phase == "await-confirm" and _stable(masks):
        heartbeat_moves = not _stable(heartbeats)
        solver_clean = not any(frozen) and not any(freeze_pending) and not any(fg) and not any(bg)
        bridge_clean = all(bool(x) for x in bridge_open if x is not None) and not any(bridge_active)
        monitor_clean = not any(monitor_q) and not any(monitor_req)
        if heartbeat_moves and solver_clean and bridge_clean and monitor_clean:
            details.append("Click transaction was released cleanly, monitor is still cycling, but board mask never advanced.")
            details.append("Primary suspect: vision/transition confirmation after the click, not solver concurrency.")
            return "AWAIT_CONFIRM_VISION_STALL", details
        if not heartbeat_moves:
            details.append("Monitor heartbeat did not advance while waiting for board confirmation.")
            return "MONITOR_LOOP_STALLED", details
        if not monitor_clean:
            return "MONITOR_RELEASE_STUCK", details
        if not bridge_clean or not solver_clean:
            return "SOLVER_FENCE_RELEASE_STUCK", details

    if phase in {"preparing", "quiescing", "committing", "scheduled"} and _stable(phases):
        details.append(f"Commit state machine remained in {phase!r} for the complete capture window.")
        return f"COMMIT_{phase.upper().replace('-', '_')}_STUCK", details

    if phase == "idle" and last.get("autoplay_enabled") and last.get("current_jewel") is None:
        details.append("Auto Play is idle and no current jewel is accepted.")
        return "WAITING_FOR_CURRENT_JEWEL", details

    if any(freeze_pending) or any(frozen) or any(fg) or any(bg) or any(bridge_active):
        return "CONCURRENCY_NOT_IDLE", details

    details.append("No single stuck invariant dominated the capture window; inspect timeline and thread stacks.")
    return "INCONCLUSIVE", details


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture a live Auto Play WAITING diagnostic bundle.")
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args()

    data_dir = default_user_data_dir()
    marker = data_dir / "waiting_probe.json"
    if not marker.exists():
        print("ERROR: waiting probe marker was not found.")
        print("Keep the Assistant open and use the Phase 13 diagnostic build.")
        print(f"Expected: {marker}")
        return 2

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = data_dir / "diagnostics" / f"waiting_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    samples: list[dict] = []
    deep: dict = {}

    print(f"Capturing WAITING state for {args.seconds:.1f}s ...")
    deadline = time.monotonic() + max(1.0, args.seconds)
    while time.monotonic() < deadline:
        try:
            probe = _probe(marker)
            solver = _solver_status(probe.get("runtime", {}))
            samples.append({"captured_at": time.time(), "probe": probe, "solver": solver})
            rt = probe.get("runtime", {})
            print(
                f"  phase={rt.get('commit_phase')} mask={rt.get('mask_count')}/14 "
                f"monitor={rt.get('monitor_quiesced')} bridge={rt.get('bridge_admission', {}).get('active')} "
                f"fg={solver.get('foreground')} bg={solver.get('background')} frozen={solver.get('frozen')}"
            )
        except BaseException as exc:
            samples.append({"captured_at": time.time(), "probe_error": f"{type(exc).__name__}: {exc}"})
        time.sleep(max(0.1, args.interval))

    try:
        deep = _probe(marker, deep=True, timeout=2.0)
    except BaseException as exc:
        deep = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    classification, details = _classify(samples)
    (out_dir / "samples.json").write_text(json.dumps(samples, indent=2, default=str), encoding="utf-8")
    (out_dir / "deep_snapshot.json").write_text(json.dumps(deep, indent=2, default=str), encoding="utf-8")
    (out_dir / "windows_processes.txt").write_text(_windows_processes(), encoding="utf-8")
    runtime = deep.get("runtime", {}) if isinstance(deep, dict) else {}
    distro = str(runtime.get("solver_wsl_distro", ""))
    (out_dir / "wsl_processes.txt").write_text(_wsl_processes(distro), encoding="utf-8")

    report = [
        "Speedlora Jewel Bingo - WAITING Diagnostic",
        "=" * 48,
        f"Classification: {classification}",
        f"Samples: {len(samples)}",
        "",
        "Automatic analysis:",
    ]
    report.extend(f"- {line}" for line in details)
    report += [
        "",
        "Key final runtime state:",
        json.dumps(runtime, indent=2, default=str),
        "",
        "Files in this bundle:",
        "- samples.json: timeline of GUI + WSL fence state",
        "- deep_snapshot.json: in-memory events, Win32 events, thread stacks",
        "- windows_processes.txt: relevant Windows processes",
        "- wsl_processes.txt: relevant WSL processes",
    ]
    (out_dir / "REPORT.txt").write_text("\n".join(report) + "\n", encoding="utf-8")

    zip_path = out_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(out_dir))

    print()
    print(f"CLASSIFICATION: {classification}")
    for line in details:
        print(f"  - {line}")
    print()
    print(f"Diagnostic ZIP: {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
