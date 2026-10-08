"""Game-independent Win32 input matrix (V18.0.27 forensic harness).

Examples (Windows) -- orthogonal 2 x 3 matrix:
  run_win32_input_matrix.bat --label A1_mu_closed_mss_off --games 10 --mss-load off
  run_win32_input_matrix.bat --label A2_mu_open_mss_off --games 10 --mss-load off
  run_win32_input_matrix.bat --label B1_mu_closed_mss_percall --games 10 --mss-load per-call
  run_win32_input_matrix.bat --label B2_mu_open_mss_percall --games 10 --mss-load per-call
  run_win32_input_matrix.bat --label C1_mu_closed_mss_persistent --games 10 --mss-load persistent
  run_win32_input_matrix.bat --label C2_mu_open_mss_persistent --games 10 --mss-load persistent

Open MS Paint (maximized, any drawing tool) on top, start the script, and do not
touch the mouse. Clicks go through the same windows_move_and_left_click used by
Auto Play, on the Tk main thread, with monitor-parked-during-click semantics.
The matrix cell "C" is the real app with MU as target (not this script).
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import sys
import threading
import time
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# DPI awareness must be declared before Tk / mss exist (same order as app.py).
from vision_engine.dpi import enable_dpi_awareness, thread_dpi_report  # noqa: E402

enable_dpi_awareness()

import tkinter as tk  # noqa: E402

from vision_engine.diagnostics.input_matrix import MatrixConfig, MatrixRunner  # noqa: E402
from vision_engine.diagnostics.resource_trend import (  # noqa: E402
    RESOURCE_TREND, sample_resources, trend_rows, trend_summary,
)
from vision_engine.diagnostics.win32 import WIN32_DIAGNOSTICS  # noqa: E402
from vision_engine.diagnostics.win32_boundary import classify_boundary_snapshot  # noqa: E402
from vision_engine.diagnostics.win32_resources import collect_process_summary  # noqa: E402
from vision_engine.input.win32 import windows_input_preflight, windows_move_and_left_click  # noqa: E402


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="matrix")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--clicks", type=int, default=14, help="clicks per game (1..25)")
    ap.add_argument("--region", default="", help="left,top,width,height (default: centre 50%% of the screen)")
    ap.add_argument("--delay-min", type=int, default=900)
    ap.add_argument("--delay-max", type=int, default=1600)
    ap.add_argument("--move-min", type=int, default=220)
    ap.add_argument("--move-max", type=int, default=380)
    ap.add_argument("--mss-load", choices=("off", "per-call", "persistent"), default="off",
                    help="background screen capture every --capture-ms, parked during each click like the monitor")
    ap.add_argument("--capture-ms", type=int, default=75)
    ap.add_argument("--start-delay", type=int, default=5, help="seconds to focus the target window")
    ap.add_argument("--seed", type=int, default=None)
    return ap.parse_args()


class CaptureLoad:
    """Background capture thread that mimics bingo-monitor and parks during clicks."""

    def __init__(self, mode: str, region: tuple[int, int, int, int], period_ms: int) -> None:
        self.mode, self.region, self.period = mode, region, max(10, int(period_ms)) / 1000.0
        self.lock = threading.Lock()  # held by the monitor during a capture, by the click during movement
        self.stop = threading.Event()
        self.captures = 0
        self.thread = threading.Thread(target=self._run, name="matrix-capture-load", daemon=True)

    def start(self) -> None:
        if self.mode != "off":
            self.thread.start()

    def close(self) -> None:
        self.stop.set()
        if self.mode != "off":
            self.thread.join(timeout=3.0)

    def _run(self) -> None:
        import mss
        import numpy as np

        left, top, width, height = self.region
        box = {"left": left, "top": top, "width": width, "height": height}
        sct = mss.mss() if self.mode == "persistent" else None
        try:
            while not self.stop.is_set():
                with self.lock:
                    if sct is not None:
                        np.array(sct.grab(box))
                    else:
                        with mss.mss() as one:
                            np.array(one.grab(box))
                    self.captures += 1
                self.stop.wait(self.period)
        finally:
            if sct is not None:
                sct.close()


def _events(buffer) -> list[dict]:
    return [{"sequence": e.sequence, "monotonic": e.monotonic, "kind": e.kind, "data": e.data}
            for e in buffer.snapshot()]


def main() -> int:
    args = _parse_args()
    ok, backend = windows_input_preflight()
    if not ok:
        print(f"ERROR: Windows input preflight failed: {backend}")
        return 2

    root = tk.Tk()
    root.title("Win32 input matrix")
    root.geometry("+10+10")
    root.attributes("-topmost", True)
    status = tk.StringVar(value="starting...")
    tk.Label(root, textvariable=status, justify="left", font=("Segoe UI", 10), padx=10, pady=8).pack()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    if args.region:
        region = tuple(int(v) for v in args.region.split(","))
    else:
        region = (int(sw * 0.25), int(sh * 0.25), int(sw * 0.5), int(sh * 0.5))

    cfg = MatrixConfig(
        games=args.games, clicks_per_game=args.clicks, region=region,  # type: ignore[arg-type]
        delay_ms=(args.delay_min, args.delay_max), move_ms=(args.move_min, args.move_max),
        seed=args.seed, label=args.label,
    )
    load = CaptureLoad(args.mss_load, region, args.capture_ms)  # type: ignore[arg-type]

    def click(point, move_ms):
        with load.lock:  # monitor is parked while the cursor moves, as in Auto Play
            return windows_move_and_left_click(point, duration_ms=move_ms)

    def avoid():
        return (root.winfo_rootx(), root.winfo_rooty(),
                root.winfo_rootx() + root.winfo_width(), root.winfo_rooty() + root.winfo_height())

    runner = MatrixRunner(cfg, click_fn=click, sample_fn=sample_resources, avoid_fn=avoid)
    dpi_start = thread_dpi_report()
    process_start = collect_process_summary()
    sample_resources("matrix-start")
    stop_requested = {"v": False}

    def finish() -> None:
        load.close()
        sample_resources("matrix-end")
        out = _write_outputs(args, cfg, runner, load, dpi_start, process_start)
        print()
        print(out["verdict"])
        print(f"Output: {out['dir']}")
        print(f"ZIP:    {out['zip']}")
        status.set(out["verdict"].splitlines()[0])
        root.after(1500, root.destroy)

    def drive() -> None:
        if stop_requested["v"]:
            runner.stop("closed by user")
        delay = None if runner.done else runner.step()
        done = runner.done
        status.set(f"{cfg.label}: {len(runner.rows)}/{runner.total_clicks} clicks"
                   + ("  FAILED" if runner.failure and not runner.failure.get("stopped_by_user") else ""))
        if done or delay is None:
            finish()
        else:
            root.after(delay, drive)

    def begin(countdown: int) -> None:
        if countdown > 0:
            status.set(f"{cfg.label}: focus the target window... {countdown}s")
            root.after(1000, begin, countdown - 1)
            return
        load.start()
        drive()

    root.protocol("WM_DELETE_WINDOW", lambda: stop_requested.update(v=True))
    root.after(100, begin, max(0, args.start_delay))
    root.mainloop()
    return 0 if (runner.failure is None) else 1


def _write_outputs(args, cfg, runner, load, dpi_start, process_start) -> dict:
    base = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "Speedlora" / "win32_matrix"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = base / f"{stamp}_{cfg.label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = runner.summary()
    rows = trend_rows(4096)
    trend = trend_summary(rows)
    win32_events = _events(WIN32_DIAGNOSTICS)
    failure_step = next((e for e in reversed(win32_events) if e["kind"] == "move-step-failure"), None)
    probe = next((e for e in reversed(win32_events) if e["kind"] == "move-recovery-probe"), None)
    boundary = (failure_step or {}).get("data", {}).get("boundary") or {}
    boundary_class = None
    if boundary:
        boundary_class, _ = classify_boundary_snapshot(boundary)

    process_end = collect_process_summary()
    meta = {
        "summary": summary, "trend": trend, "dpi_at_start": dpi_start,
        "process_summary_start": process_start, "process_summary_end": process_end,
        "capture_load": {"mode": args.mss_load, "captures": load.captures, "period_ms": args.capture_ms},
        "config": asdict(cfg), "argv": sys.argv[1:], "python": sys.version, "platform": platform.platform(),
        "boundary_classification": boundary_class,
        "probe_classification": (probe or {}).get("data", {}).get("classification"),
    }
    (out_dir / "summary.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    (out_dir / "win32_events.json").write_text(json.dumps(win32_events, indent=2, default=str), encoding="utf-8")
    if failure_step:
        (out_dir / "failure.json").write_text(
            json.dumps({"step": failure_step, "probe": probe}, indent=2, default=str), encoding="utf-8")
    if runner.rows:
        with (out_dir / "clicks.csv").open("w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=sorted({k for r in runner.rows for k in r}))
            w.writeheader(); w.writerows(runner.rows)
    if rows:
        keys = sorted({k for r in rows for k in r})
        with (out_dir / "resource_trend.csv").open("w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)

    zip_path = out_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out_dir.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(out_dir))

    leak = trend.get("leak_suspected") or "none"
    if summary["failed"]:
        f = summary["failure"] or {}
        verdict = (f"RESULT: FAIL at game {f.get('game')} click {f.get('click')} "
                   f"({summary['clicks_ok']}/{summary['clicks_total_expected']} ok)\n"
                   f"error: {f.get('error')}\nboundary: {boundary_class} | probe: {meta['probe_classification']} | leak: {leak}")
    elif summary["completed"]:
        verdict = (f"RESULT: PASS {summary['clicks_ok']}/{summary['clicks_total_expected']} clicks\n"
                   f"leak suspected: {leak}")
    else:
        verdict = f"RESULT: STOPPED after {summary['clicks_ok']} clicks\nleak suspected: {leak}"
    return {"dir": out_dir, "zip": zip_path, "verdict": verdict}


if __name__ == "__main__":
    raise SystemExit(main())
