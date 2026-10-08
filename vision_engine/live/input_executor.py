from __future__ import annotations

from vision_engine.input.geometry import AutoPlayPoint
from vision_engine.input.win32 import windows_move_and_left_click

from .input_exclusive import InputExclusiveLease


class LiveInputExecutor:
    """Execute physical Windows input only while an exclusive lease is valid."""

    def move_and_left_click(
        self,
        lease: InputExclusiveLease,
        point: AutoPlayPoint,
        *,
        duration_ms: int,
    ) -> str:
        if lease.released:
            raise RuntimeError("input-exclusive lease is already released")
        if not lease.monitor_quiesced:
            raise RuntimeError("monitor is not quiesced for physical input")
        if lease.coordinator.remote_solver and not lease.solver_token:
            raise RuntimeError("global solver fence token is missing")
        return windows_move_and_left_click(point, duration_ms=duration_ms)
