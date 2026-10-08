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
    "vision_engine/live/input_exclusive.py",
    "vision_engine/input/geometry.py",
    "vision_engine/input/pacing.py",
    "vision_engine/input/motion.py",
    "vision_engine/input/win32.py",
    "vision_engine/diagnostics/ring_buffer.py",
    "vision_engine/diagnostics/win32.py",
    "vision_engine/diagnostics/fence_simulation.py",
    "vision_engine/autoplay.py",
    "vision_engine/input_gate.py",
    "vision_engine/input_coordinator.py",
]


def test_v1826_core_runtime_files_are_focused_and_under_300_lines():
    oversized = {}
    for relative in CORE_RUNTIME:
        lines = (ROOT / relative).read_text(encoding="utf-8").splitlines()
        if len(lines) > 300:
            oversized[relative] = len(lines)
    assert oversized == {}


def test_v1826_gui_uses_named_runtime_components_not_background_only_barrier():
    gui = (ROOT / "vision_engine" / "gui.py").read_text(encoding="utf-8")
    assert "InputExclusiveCoordinator" in gui
    assert "BridgeAdmissionGate" not in gui  # implementation detail stays out of UI
    assert "wait_background_idle" not in gui
    assert "set_background_enabled_async(False)" not in gui
    assert '"trajectory": "v11-linear-smoothstep"' in gui
