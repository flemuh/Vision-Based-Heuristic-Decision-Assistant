from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vision_engine.input_gate import MonitorInputGate

from .solver_bridge import LiveSolverBridge


@dataclass(slots=True)
class InputExclusiveLease:
    coordinator: "InputExclusiveCoordinator"
    solver_token: str | None = None
    monitor_quiesced: bool = False
    info: dict[str, Any] = field(default_factory=dict)
    released: bool = False

    def release(self) -> dict[str, Any]:
        if self.released:
            return {"ok": True, "reason": "already-released"}
        result = self.coordinator.release(self)
        self.released = True
        return result


class InputExclusiveCoordinator:
    """Owns the complete global fence order around one physical input commit."""

    def __init__(
        self,
        solver_bridge: LiveSolverBridge,
        monitor_gate: MonitorInputGate,
        *,
        remote_solver: bool,
    ) -> None:
        self.solver_bridge = solver_bridge
        self.monitor_gate = monitor_gate
        self.remote_solver = bool(remote_solver)

    def acquire(self, *, solver_timeout: float = 5.0, monitor_timeout: float = 2.0) -> InputExclusiveLease:
        lease = InputExclusiveLease(self)
        if self.remote_solver:
            barrier = self.solver_bridge.prepare_input_exclusive(timeout=solver_timeout)
            lease.info.update(barrier)
            if not barrier.get("ok") or not barrier.get("idle"):
                raise RuntimeError(self._barrier_error(barrier))
            lease.solver_token = str(barrier.get("exclusive_token") or "") or None
            if lease.solver_token is None:
                self.solver_bridge.admission.open()
                raise RuntimeError("global solver fence returned no exclusive token")
        else:
            lease.info.update({
                "ok": True, "idle": True, "foreground": 0,
                "background": 0, "bridge_active": 0,
            })

        if not self.monitor_gate.request_quiesce(timeout=monitor_timeout):
            if lease.solver_token is not None:
                self.solver_bridge.release_input_exclusive(lease.solver_token)
                lease.solver_token = None
            raise RuntimeError("monitor did not reach the input safe point")
        lease.monitor_quiesced = True
        return lease

    def release(self, lease: InputExclusiveLease) -> dict[str, Any]:
        # Always attempt both halves. A monitor-release error must never skip
        # the WSL token release, and a failed solver release keeps the token on
        # the lease so callers can retry without reopening admission early.
        errors: list[str] = []
        if lease.monitor_quiesced:
            try:
                self.monitor_gate.release()
                lease.monitor_quiesced = False
            except BaseException as exc:
                errors.append(f"monitor-release:{type(exc).__name__}:{exc}")
                try:
                    self.monitor_gate.force_release()
                    lease.monitor_quiesced = False
                except BaseException as force_exc:
                    errors.append(f"monitor-force-release:{type(force_exc).__name__}:{force_exc}")

        result: dict[str, Any] = {"ok": True}
        if lease.solver_token is not None:
            token = lease.solver_token
            try:
                result = self.solver_bridge.release_input_exclusive(token)
                if not result.get("ok", False):
                    raise RuntimeError(str(result.get("reason", "solver release rejected")))
                lease.solver_token = None
            except BaseException as exc:
                errors.append(f"solver-release:{type(exc).__name__}:{exc}")

        if errors:
            raise RuntimeError("input exclusive release incomplete: " + "; ".join(errors))
        return result

    def force_release(self) -> None:
        self.monitor_gate.force_release()

    @staticmethod
    def _barrier_error(barrier: dict) -> str:
        return (
            f"global solver fence failed: {barrier.get('reason', 'not-idle')} "
            f"(bridge={barrier.get('bridge_active')}, fg={barrier.get('foreground')}, "
            f"bg={barrier.get('background')})"
        )
