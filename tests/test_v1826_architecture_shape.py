from pathlib import Path

ROOT = Path(__file__).parents[1]

CORE_RUNTIME = [
    "vision_engine/solver/worker.py",
    "vision_engine/solver/fence.py",
    "vision_engine/solver/engine.py",
    "vision_engine/solver/service.py",
    "vision_engine/solver/client.py",
    "vision_engine/solver/protocol.py",
    "vision_engine/live/solver_admission.py",
    "vision_engine/live/solver_bridge.py",
    "vision_engine/diagnostics/ring_buffer.py",
    "vision_engine/diagnostics/win32.py",
    "vision_engine/diagnostics/fence_simulation.py",
]


def test_v1826_gui_uses_named_runtime_components_not_background_only_barrier():
    gui = (ROOT / "vision_engine" / "gui.py").read_text(encoding="utf-8")
    assert "BridgeAdmissionGate" not in gui  # implementation detail stays out of UI
    assert "wait_background_idle" not in gui
    assert "set_background_enabled_async(False)" not in gui

