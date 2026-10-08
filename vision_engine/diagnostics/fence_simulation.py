from __future__ import annotations

from dataclasses import dataclass, asdict
import threading
import time

from vision_engine.live.solver_admission import BridgeAdmissionGate
from vision_engine.solver.fence import GlobalSolverFence


@dataclass(slots=True)
class FenceSimulationReport:
    cycles: int
    freezes: int = 0
    releases: int = 0
    stale_bridge_rejections: int = 0
    frozen_solver_rejections: int = 0
    session_rotations: int = 0
    failures: int = 0
    final_bridge_active: int = -1
    final_foreground: int = -1
    final_background: int = -1

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def run_fence_simulation(cycles: int = 1000) -> FenceSimulationReport:
    cycles = max(1, int(cycles))
    report = FenceSimulationReport(cycles=cycles)
    bridge = BridgeAdmissionGate()
    server = GlobalSolverFence()
    session = "sim-0"
    if not server.begin_session(session, 0.2).get("ok"):
        report.failures += 1
        return report

    for i in range(cycles):
        if i and i % 100 == 0:
            session = f"sim-{i // 100}"
            rotated = server.begin_session(session, 0.2)
            if not rotated.get("ok"):
                report.failures += 1
                break
            report.session_rotations += 1

        active = bridge.ticket("solve")
        stale = bridge.ticket("queued-precompute")
        if active is None or stale is None or not bridge.enter(active):
            report.failures += 1
            break
        if not server.enter_foreground(session) or not server.enter_background(session):
            report.failures += 1
            break

        def finish_admitted_work() -> None:
            # Deliberately finish foreground/background in opposite order across
            # cycles to exercise both wait paths without timing assumptions.
            if i & 1:
                server.leave_background(); server.leave_foreground()
            else:
                server.leave_foreground(); server.leave_background()
            bridge.leave()

        worker = threading.Thread(target=finish_admitted_work)
        worker.start()
        bridge.close()
        if bridge.enter(stale):
            report.failures += 1
        else:
            report.stale_bridge_rejections += 1
        ipc = bridge.wait_idle(0.2)
        worker.join(0.5)
        if not ipc.get("idle") or worker.is_alive():
            report.failures += 1
            break

        frozen = server.freeze(session, 0.2)
        if not frozen.get("ok") or not frozen.get("idle"):
            report.failures += 1
            break
        report.freezes += 1
        if server.enter_foreground(session) or server.enter_background(session):
            report.failures += 1
            break
        report.frozen_solver_rejections += 1

        released = server.release(session, str(frozen.get("exclusive_token", "")))
        if not released.get("ok"):
            report.failures += 1
            break
        report.releases += 1
        bridge.open()

    bridge_state = bridge.snapshot()
    server_state = server.snapshot()
    report.final_bridge_active = int(bridge_state.get("active", -1))
    report.final_foreground = int(server_state.get("foreground", -1))
    report.final_background = int(server_state.get("background", -1))
    if report.final_bridge_active or report.final_foreground or report.final_background:
        report.failures += 1
    return report
