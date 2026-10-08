from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random
import threading
import time

from vision_engine.input_gate import MonitorInputGate
from vision_engine.live.input_exclusive import InputExclusiveCoordinator
from vision_engine.live.solver_bridge import LiveSolverBridge
from vision_engine.solver.client import SolverClient
from vision_engine.solver.service import SolverEngine, SolverTCPServer


@dataclass(slots=True)
class InputExclusiveStressReport:
    cycles: int
    acquisitions: int = 0
    releases: int = 0
    session_rotations: int = 0
    busy_cycles: int = 0
    pending6_cycles: int = 0
    stale_bridge_rejections: int = 0
    forbidden_input_admissions: int = 0
    failures: int = 0
    final_bridge_active: int = -1
    final_bridge_queued: int = -1
    final_foreground: int = -1
    final_background: int = -1
    final_frozen: bool = True
    final_freeze_pending: bool = True
    final_monitor_requested: bool = True
    final_monitor_quiesced: bool = True

    def to_dict(self) -> dict[str, int | bool]:
        return asdict(self)


def _serve_engine() -> tuple[SolverEngine, SolverTCPServer, threading.Thread]:
    engine = SolverEngine(workers=2, cache_entries=32)
    server = SolverTCPServer(("127.0.0.1", 0), engine, socket_timeout=10.0)
    thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.01},
        daemon=True,
        name="stress-solver-server",
    )
    thread.start()
    return engine, server, thread


def _start_bridge(port: int, session_id: str) -> LiveSolverBridge:
    bridge = LiveSolverBridge(
        Path("."),
        host="127.0.0.1",
        port=port,
        auto_start_wsl=False,
        session_id=session_id,
        session_timeout=1.0,
    )
    ok, status = bridge._start_sync()
    bridge.available = ok
    bridge.status = status
    if not ok:
        bridge.close()
        raise RuntimeError(status)
    return bridge


def _monitor_loop(gate: MonitorInputGate, stop: threading.Event) -> None:
    while not stop.is_set():
        gate.wait_or_wake(0.005)
        if gate.requested:
            gate.monitor_safe_point(running=not stop.is_set())


def _release_server_work(
    engine: SolverEngine,
    foreground: int,
    background: int,
    delay: float,
) -> threading.Thread:
    def work() -> None:
        if delay > 0:
            time.sleep(delay)
        # Reverse the order on purpose. The fence must wait for both classes,
        # not accidentally treat one counter as sufficient.
        for _ in range(background):
            engine.fence.leave_background()
        for _ in range(foreground):
            engine.fence.leave_foreground()

    thread = threading.Thread(target=work, daemon=True, name="stress-server-work-release")
    thread.start()
    return thread


def _release_bridge_ipc(bridge: LiveSolverBridge, delay: float) -> threading.Thread:
    def work() -> None:
        if delay > 0:
            time.sleep(delay)
        bridge.admission.leave()

    thread = threading.Thread(target=work, daemon=True, name="stress-bridge-ipc-release")
    thread.start()
    return thread


def run_input_exclusive_stress(
    cycles: int = 1000,
    *,
    rotate_every: int = 100,
    seed: int = 1826,
) -> InputExclusiveStressReport:
    """Exercise the real TCP bridge/fence/monitor sequence repeatedly.

    No physical mouse input is generated. The critical input window is
    represented by an admission check while both solver and monitor fences are
    held. Scheduling jitter is deterministic via ``seed``.
    """
    cycles = max(1, int(cycles))
    rotate_every = max(0, int(rotate_every))
    report = InputExclusiveStressReport(cycles=cycles)
    rng = random.Random(int(seed))
    engine, server, server_thread = _serve_engine()
    host, port = server.server_address
    observer = SolverClient(host, port, timeout=3.0)
    monitor = MonitorInputGate()
    monitor_stop = threading.Event()
    monitor_thread = threading.Thread(
        target=_monitor_loop,
        args=(monitor, monitor_stop),
        daemon=True,
        name="stress-monitor",
    )
    monitor_thread.start()
    bridge: LiveSolverBridge | None = None
    session_index = 0
    session_id = f"stress-{session_index}"

    try:
        bridge = _start_bridge(port, session_id)
        coordinator = InputExclusiveCoordinator(bridge, monitor, remote_solver=True)

        for i in range(cycles):
            if rotate_every and i and i % rotate_every == 0:
                bridge.close()
                session_index += 1
                session_id = f"stress-{session_index}"
                bridge = _start_bridge(port, session_id)
                coordinator = InputExclusiveCoordinator(bridge, monitor, remote_solver=True)
                report.session_rotations += 1

            stale_ticket = bridge.admission.ticket("queued-stale")
            if stale_ticket is None:
                report.failures += 1
                break

            bridge_worker: threading.Thread | None = None
            if i % 4 == 0:
                active_ticket = bridge.admission.ticket("active-ipc")
                if active_ticket is None or not bridge.admission.enter(active_ticket):
                    report.failures += 1
                    break
                bridge_worker = _release_bridge_ipc(bridge, rng.uniform(0.0002, 0.0015))
                report.busy_cycles += 1

            foreground = 1 if i % 3 == 0 else 0
            background = 6 if i % 50 == 0 else (1 if i % 5 == 0 else 0)
            admitted_fg = 0
            admitted_bg = 0
            for _ in range(foreground):
                if engine.fence.enter_foreground(session_id):
                    admitted_fg += 1
            for _ in range(background):
                if engine.fence.enter_background(session_id):
                    admitted_bg += 1
            if admitted_fg != foreground or admitted_bg != background:
                report.failures += 1
                break
            if background == 6:
                report.pending6_cycles += 1
            server_worker: threading.Thread | None = None
            if foreground or background:
                server_worker = _release_server_work(
                    engine, foreground, background, rng.uniform(0.0002, 0.0018)
                )
                report.busy_cycles += 1

            try:
                lease = coordinator.acquire(solver_timeout=1.0, monitor_timeout=1.0)
                report.acquisitions += 1
            except BaseException:
                report.failures += 1
                break

            if bridge.admission.enter(stale_ticket):
                report.failures += 1
                bridge.admission.leave()
            else:
                report.stale_bridge_rejections += 1

            status = observer.status()
            if (
                int(status.get("foreground", -1)) != 0
                or int(status.get("background", -1)) != 0
                or not bool(status.get("frozen"))
                or not monitor.quiesced
            ):
                report.failures += 1

            if bridge.admission.ticket("forbidden-during-input") is not None:
                report.forbidden_input_admissions += 1
                report.failures += 1

            try:
                released = lease.release()
                if not released.get("ok", False):
                    report.failures += 1
                    break
                report.releases += 1
            except BaseException:
                report.failures += 1
                break

            if bridge_worker is not None:
                bridge_worker.join(0.5)
                if bridge_worker.is_alive():
                    report.failures += 1
                    break
            if server_worker is not None:
                server_worker.join(0.5)
                if server_worker.is_alive():
                    report.failures += 1
                    break

            bridge_state = bridge.admission.snapshot()
            status = observer.status()
            if (
                int(bridge_state.get("active", -1)) != 0
                or int(status.get("foreground", -1)) != 0
                or int(status.get("background", -1)) != 0
                or bool(status.get("frozen"))
                or bool(status.get("freeze_pending"))
                or monitor.requested
                or monitor.quiesced
            ):
                report.failures += 1
                break

        if bridge is not None:
            bridge_state = bridge.admission.snapshot()
            report.final_bridge_active = int(bridge_state.get("active", -1))
            report.final_bridge_queued = int(bridge_state.get("queued", -1))
        final = observer.status()
        report.final_foreground = int(final.get("foreground", -1))
        report.final_background = int(final.get("background", -1))
        report.final_frozen = bool(final.get("frozen"))
        report.final_freeze_pending = bool(final.get("freeze_pending"))
        report.final_monitor_requested = monitor.requested
        report.final_monitor_quiesced = monitor.quiesced
    finally:
        monitor_stop.set()
        monitor.force_release()
        monitor_thread.join(timeout=1.0)
        if bridge is not None:
            bridge.close()
        observer.close()
        server.shutdown()
        server.server_close()
        engine.close()
        server_thread.join(timeout=2.0)

    if (
        report.final_bridge_active != 0
        or report.final_bridge_queued != 0
        or report.final_foreground != 0
        or report.final_background != 0
        or report.final_frozen
        or report.final_freeze_pending
        or report.final_monitor_requested
        or report.final_monitor_quiesced
    ):
        report.failures += 1
    return report
