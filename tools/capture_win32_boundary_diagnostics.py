from __future__ import annotations

import csv
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision_engine.data_home import default_user_data_dir
from vision_engine.diagnostics.probe_locator import locate_live_probe_marker
from vision_engine.diagnostics.win32_boundary import classify_boundary_snapshot
from vision_engine.solver.client import SolverClient


def _probe(marker: Path, deep: bool = False, timeout: float = 1.5) -> dict:
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


def _latest_failure(probe: dict) -> dict | None:
    failures = [event for event in probe.get("win32_events", []) if event.get("kind") == "move-step-failure"]
    return failures[-1] if failures else None


def _solver_status(runtime: dict) -> dict:
    client = SolverClient(str(runtime.get("solver_host", "127.0.0.1")), int(runtime.get("solver_port", 57641)), timeout=1.5)
    try:
        return client.status()
    except BaseException as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        client.close()


def _powershell_processes() -> str:
    command = (
        "Get-Process -ErrorAction SilentlyContinue | "
        "Where-Object {$_.ProcessName -match 'python|mu|main|game|wsl|ubuntu'} | "
        "Select-Object Id,ProcessName,Path,StartTime,CPU,MainWindowTitle | Format-List"
    )
    try:
        proc = subprocess.run(["powershell.exe", "-NoProfile", "-Command", command], capture_output=True, text=True, timeout=8, check=False)
        return (proc.stdout or "") + (proc.stderr or "")
    except BaseException as exc:
        return f"{type(exc).__name__}: {exc}\n"


def _trajectory(events: list[dict]) -> tuple[list[dict], dict | None]:
    begins = [e for e in events if e.get("kind") == "move-begin"]
    if not begins:
        return [], None
    begin = begins[-1]
    begin_seq = int(begin.get("sequence", 0))
    related = [e for e in events if int(e.get("sequence", 0)) >= begin_seq]
    path = list(begin.get("data", {}).get("path", []))
    by_step: dict[int, dict] = {}
    for event in related:
        if event.get("kind") not in {"move-step", "move-step-failure"}:
            continue
        data = event.get("data", {})
        by_step[int(data.get("step", 0))] = event
    rows: list[dict] = []
    for i, xy in enumerate(path, start=1):
        event = by_step.get(i, {})
        data = event.get("data", {}) if event else {}
        boundary = data.get("boundary", {}) if isinstance(data.get("boundary"), dict) else {}
        target_window = boundary.get("points", {}).get("target", {}) if boundary else {}
        process = target_window.get("process", {}) if isinstance(target_window, dict) else {}
        rows.append({
            "step": i,
            "x": xy[0],
            "y": xy[1],
            "kind": event.get("kind", "not-reached") if event else "not-reached",
            "set_cursor_ok": data.get("set_cursor_ok"),
            "sendinput_ok": data.get("sendinput_ok"),
            "actual": data.get("actual"),
            "elapsed_ms": data.get("elapsed_ms"),
            "window_pid": target_window.get("pid"),
            "window_title": target_window.get("title"),
            "window_class": target_window.get("class"),
            "process_path": process.get("path"),
            "integrity": process.get("integrity"),
            "boundary_classification": boundary.get("classification"),
        })
    return rows, begin


def _trend_lines(deep: dict) -> list[str]:
    from vision_engine.diagnostics.resource_trend import trend_summary

    rows = deep.get("resource_trend", []) or []
    summary = trend_summary(rows)
    lines = ["", f"Resource trend ({summary.get('samples', 0)} samples, {summary.get('duration_s', 0)}s):"]
    if not summary.get("metrics"):
        lines.append("- not enough samples (need at least 2 Auto Play board transitions)")
        return lines
    for name, info in summary["metrics"].items():
        flag = "  <-- GROWING" if info.get("growing") else ""
        lines.append(
            f"- {name}: {info['first']} -> {info['last']} (delta {info['delta']}, "
            f"{info['per_minute']}/min, nondecreasing {info['nondecreasing_fraction']}){flag}"
        )
    lines.append(f"- leak suspected: {summary['leak_suspected'] or 'none'}")
    return lines


def _extra_evidence(deep: dict, failure: dict, boundary: dict) -> list[str]:
    """Rejection probe, DPI context, GDI/USER pressure and process landscape."""
    lines: list[str] = ["", "Rejection probe (SetCursorPos retried after the failure):"]
    seq = int(failure.get("sequence", 0))
    probes = [
        e for e in deep.get("win32_events", [])
        if e.get("kind") == "move-recovery-probe" and int(e.get("sequence", 0)) >= seq
    ]
    if probes:
        pdata = probes[0].get("data", {})
        lines.append(f"- classification: {pdata.get('classification')} first_recovery_ms={pdata.get('first_recovery_ms')} error_codes={pdata.get('failure_error_codes')}")
        for a in pdata.get("attempts", []):
            lines.append(
                f"  +{a.get('planned_delay_ms')}ms ok={a.get('set_cursor_ok')} err={a.get('last_error')} "
                f"cursor {a.get('cursor_before')} -> {a.get('cursor_after')} reached={a.get('reached_target')}"
            )
    else:
        lines.append("- no probe recorded for this failure (tolerated step or older build)")
    dpi = boundary.get("dpi") if boundary else None
    lines += ["", "DPI context:", f"- {json.dumps(dpi, default=str)}"]
    gui = boundary.get("gui_resources") if boundary else None
    lines += ["", "GDI/USER handles (Assistant process):", f"- {json.dumps(gui, default=str)}"]
    procs = boundary.get("processes") if boundary else None
    if isinstance(procs, dict) and "error" not in procs:
        lines += ["", "Process landscape:", f"- other python PIDs: {procs.get('other_python_pids')}"]
        lines.append(
            f"- MU-like processes: {procs.get('mu_like_count', 0)} "
            f"totals={json.dumps(procs.get('mu_like_totals', {}), default=str)}"
        )
        for label, key in (("top GDI", "top_gdi"), ("top USER", "top_user")):
            top = ", ".join(
                f"{r.get('name')}[{r.get('pid')}] gdi={r.get('gdi')} user={r.get('user')}"
                for r in (procs.get(key) or [])[:5]
            )
            lines.append(f"- {label}: {top}")
    return lines


def _report(deep: dict, solver: dict, failure: dict) -> tuple[str, str, list[str]]:
    data = failure.get("data", {})
    boundary = data.get("boundary", {}) if isinstance(data.get("boundary"), dict) else {}
    classification, details = classify_boundary_snapshot(boundary) if boundary else ("NO_BOUNDARY_SNAPSHOT", ["Failure has no Phase 14 boundary snapshot."])
    points = boundary.get("points", {}) if boundary else {}
    target = points.get("target", {}) if isinstance(points, dict) else {}
    previous = points.get("previous", {}) if isinstance(points, dict) else {}
    foreground = boundary.get("foreground", {}) if boundary else {}
    assistant = boundary.get("assistant", {}) if boundary else {}
    target_process = target.get("process", {}) if isinstance(target, dict) else {}
    lines = [
        "Speedlora Jewel Bingo - Win32 Boundary Forensics",
        "=" * 58,
        f"Classification: {classification}",
        f"Failure step: {data.get('step')}/{data.get('total')}",
        f"Target: {data.get('target')}",
        f"Actual: {data.get('actual')}",
        f"SetCursorPos: {data.get('set_cursor_ok')} last_error={data.get('last_error')}",
        f"SendInput: {data.get('sendinput_ok')} last_error={data.get('sendinput_last_error')}",
        "",
        "Automatic analysis:",
    ]
    lines += [f"- {line}" for line in details]
    lines += [
        "",
        "Boundary evidence:",
        f"- previous root/pid: {previous.get('root_hwnd')}/{previous.get('pid')}",
        f"- failed root/pid: {target.get('root_hwnd')}/{target.get('pid')}",
        f"- failed window: {target.get('title')!r} class={target.get('class')!r}",
        f"- failed process: {target_process.get('path')!r}",
        f"- Assistant integrity: {assistant.get('integrity')}",
        f"- Target integrity: {target_process.get('integrity')}",
        f"- Foreground: {foreground.get('title')!r} pid={foreground.get('pid')} integrity={foreground.get('process', {}).get('integrity') if isinstance(foreground, dict) else None}",
        f"- ClipCursor: {boundary.get('clip_rect')}",
        f"- Process desktop: {boundary.get('process_desktop')}",
        f"- Input desktop: {boundary.get('input_desktop')}",
        f"- Window station: {boundary.get('window_station')}",
        f"- Virtual screen: {boundary.get('virtual_screen')}",
        "",
        "Current solver state after failure:",
        json.dumps(solver, indent=2, default=str),
    ]
    lines += _extra_evidence(deep, failure, boundary)
    lines += _trend_lines(deep)
    if classification == "GUI_RESOURCE_PRESSURE":
        lines += ["", "NEXT ACTION: handle exhaustion suspected; inspect top_gdi/top_user below and restart the heaviest consumers (and check leaked python/WSL processes)."]
    elif classification == "UIPI_INTEGRITY_MISMATCH":
        lines += ["", "NEXT ACTION: run Assistant and MU at the same integrity level; this dump proves the target is higher integrity."]
    elif classification == "CURSOR_CLIPPED":
        lines += ["", "NEXT ACTION: investigate which process calls ClipCursor and whether MU captures the cursor at that boundary."]
    elif classification == "WINDOW_BOUNDARY_CORRELATED":
        lines += ["", "NEXT ACTION: repeat once more. If the first failure lands on the same root-window/process transition, boundary correlation is strong."]
    elif classification == "INPUT_DESKTOP_MISMATCH":
        lines += ["", "NEXT ACTION: investigate desktop/session switching; SetCursorPos requires the input desktop to be current."]
    elif classification == "CURSOR_POSITION_DIVERGED":
        lines += ["", "NEXT ACTION: a Win32 API reported success but the physical cursor remained outside tolerance; compare repeated failures and the target window/process boundary."]
    else:
        lines += ["", "NEXT ACTION: compare this bundle with a second failure; the report preserves exact window/integrity/clip evidence."]
    return "\n".join(lines) + "\n", classification, details


def main() -> int:
    marker = locate_live_probe_marker()
    if marker is None:
        print("ERROR: Phase 14 diagnostic probe is not running.")
        print("Keep the Assistant open and run this script from the ROOT-FIX diagnostic build.")
        return 2

    data_dir = marker.parent
    print(f"Using live diagnostic probe: {marker}")
    initial = _probe(marker)
    initial_failure = _latest_failure(initial)
    now_mono = float(initial.get("probe_monotonic", 0.0) or 0.0)
    use_existing = False
    baseline_seq = 0
    if initial_failure:
        baseline_seq = int(initial_failure.get("sequence", 0))
        age = now_mono - float(initial_failure.get("monotonic", 0.0) or 0.0)
        use_existing = 0.0 <= age <= 60.0

    if use_existing:
        print(f"Recent Win32 failure found (sequence {baseline_seq}); capturing now ...")
    else:
        print("No recent Win32 failure found.")
        print("Watching for up to 90 seconds. Trigger Auto Play now and leave the Assistant open if it fails.")
        deadline = time.monotonic() + 90.0
        while time.monotonic() < deadline:
            probe = _probe(marker)
            latest = _latest_failure(probe)
            if latest and int(latest.get("sequence", 0)) > baseline_seq:
                print(f"Captured new move failure at sequence {latest.get('sequence')}.")
                break
            time.sleep(0.20)
        else:
            print("ERROR: no new move-step-failure was observed in 90 seconds.")
            return 3

    deep = _probe(marker, deep=True, timeout=2.5)
    failure = _latest_failure(deep)
    if not failure:
        print("ERROR: failure disappeared from in-memory ring buffer.")
        return 4
    runtime = deep.get("runtime", {})
    solver = _solver_status(runtime)
    report, classification, _details = _report(deep, solver, failure)
    rows, begin = _trajectory(deep.get("win32_events", []))

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = data_dir / "diagnostics" / f"win32_boundary_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "REPORT.txt").write_text(report, encoding="utf-8")
    (out_dir / "deep_snapshot.json").write_text(json.dumps(deep, indent=2, default=str), encoding="utf-8")
    (out_dir / "solver_status.json").write_text(json.dumps(solver, indent=2, default=str), encoding="utf-8")
    (out_dir / "failure.json").write_text(json.dumps(failure, indent=2, default=str), encoding="utf-8")
    (out_dir / "move_begin.json").write_text(json.dumps(begin, indent=2, default=str), encoding="utf-8")
    trend = deep.get("resource_trend", []) or []
    if trend:
        keys = ["sequence", "monotonic", "label", "gdi", "user", "gdi_peak", "user_peak", "handles", "working_set_mb", "private_mb", "games", "placed"]
        with (out_dir / "resource_trend.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(trend)
    (out_dir / "windows_processes.txt").write_text(_powershell_processes(), encoding="utf-8")
    with (out_dir / "trajectory.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    zip_path = out_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(out_dir))

    print()
    print(f"CLASSIFICATION: {classification}")
    print(f"Diagnostic ZIP: {zip_path}")
    print("Send this win32_boundary_*.zip back for analysis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
