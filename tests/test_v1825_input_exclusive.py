from __future__ import annotations

import hashlib
from pathlib import Path
import threading
import time

from vision_engine.config import AppConfig
from vision_engine.input_coordinator import (
    AutoPlayCommitCoordinator,
    InputActionToken,
    InputCommitPhase,
)
from vision_engine.input_gate import MonitorInputGate
from vision_engine.live.solver_bridge import LiveSolverBridge
from vision_engine.solver.service import SolverEngine

ROOT = Path(__file__).parents[1]
V11_AUTOPLAY_SHA256 = "e9b9a4cebf8e9d5efe158b41f996ab92615a17683b08e7df6efa7669e8d44bb2"


def test_v1825_transport_contract_survives_v26_module_split():
    autoplay = (ROOT / "vision_engine" / "autoplay.py").read_text(encoding="utf-8")
    motion = (ROOT / "vision_engine" / "input" / "motion.py").read_text(encoding="utf-8")
    win32 = (ROOT / "vision_engine" / "input" / "win32.py").read_text(encoding="utf-8")
    assert "safe_cell_point" in autoplay
    assert "bezier_motion_points" in motion
    assert "linear_motion_points" in motion
    assert "SetCursorPos" in win32 and "SendInput" in win32


def test_v1825_commit_state_machine_single_owner_and_generation():
    gate = AutoPlayCommitCoordinator()
    token = gate.reserve("ep", 3, "CR", 17)
    assert isinstance(token, InputActionToken)
    assert token.generation == 0
    assert gate.phase is InputCommitPhase.SCHEDULED
    assert gate.reserve("ep", 3, "CR", 18) is None
    assert gate.begin_prepare(token)
    assert gate.begin_quiesce(token)
    assert gate.begin_commit(token)
    assert not gate.begin_commit(token)
    assert gate.mark_clicked(token)
    assert gate.phase is InputCommitPhase.AWAIT_CONFIRM
    assert gate.confirm(token)
    assert gate.phase is InputCommitPhase.IDLE
    assert gate.generation == 1


def test_v1825_stale_generation_cannot_commit():
    gate = AutoPlayCommitCoordinator()
    stale = gate.reserve("ep", 1, "SO", 4)
    assert stale is not None
    gate.invalidate()
    assert not gate.is_current(stale)
    assert not gate.begin_prepare(stale)
    fresh = gate.reserve("ep", 1, "SO", 4)
    assert fresh is not None
    assert fresh.generation == stale.generation + 1


def test_v1825_abort_invalidates_late_callbacks():
    gate = AutoPlayCommitCoordinator()
    token = gate.reserve("ep", 2, "HA", 8)
    assert token is not None and gate.begin_prepare(token)
    assert gate.abort(token)
    assert gate.phase is InputCommitPhase.IDLE
    assert not gate.begin_quiesce(token)


def _bare_engine() -> SolverEngine:
    # Avoid spawning process pools: these tests exercise the synchronization
    # contract itself, not Monte-Carlo worker execution.
    engine = SolverEngine.__new__(SolverEngine)
    engine._pending = set()
    engine._pending_futures = {}
    engine._lock = threading.RLock()
    engine._background_idle = threading.Condition(engine._lock)
    engine._background_enabled = True
    return engine


def test_v1825_background_barrier_waits_for_real_idle_not_elapsed_sleep():
    engine = _bare_engine()
    with engine._lock:
        engine._pending.add("job")

    released = threading.Event()

    def finish_job():
        time.sleep(0.04)
        with engine._background_idle:
            engine._pending.discard("job")
            engine._background_idle.notify_all()
        released.set()

    t = threading.Thread(target=finish_job)
    t.start()
    started = time.perf_counter()
    result = engine.wait_background_idle(0.5)
    elapsed = time.perf_counter() - started
    t.join(1.0)

    assert released.is_set()
    assert result["idle"] is True
    assert result["pending"] == 0
    assert elapsed >= 0.02


def test_v1825_background_barrier_times_out_closed_not_optimistically_open():
    engine = _bare_engine()
    with engine._lock:
        engine._pending.add("stuck")
    result = engine.wait_background_idle(0.02)
    assert result["idle"] is False
    assert result["pending"] == 1


def test_v1825_disabling_background_is_server_side_and_visible():
    engine = _bare_engine()
    result = engine.set_background_enabled(False)
    assert result["enabled"] is False
    assert engine.background_status()["enabled"] is False


def test_v1825_gui_restores_atomic_snapshot_and_global_input_barrier():
    src = (ROOT / "vision_engine" / "gui.py").read_text(encoding="utf-8")
    assert "self._live_state_lock = threading.RLock()" in src
    assert "def _autoplay_live_snapshot" in src
    assert "self.input_exclusive.acquire(solver_timeout=5.0, monitor_timeout=2.0)" in src
    exclusive_src = (ROOT / "vision_engine" / "live" / "input_exclusive.py").read_text(encoding="utf-8")
    assert "prepare_input_exclusive" in exclusive_src
    assert "request_quiesce" in exclusive_src
    assert "with self._live_state_lock:" in src
    assert "self.input_executor.move_and_left_click" in src
    assert "windows_move_and_left_click(point, duration_ms=move_ms)" not in src


def test_v1825_background_suppression_evolved_to_v26_global_fence():
    src = (ROOT / "vision_engine" / "gui.py").read_text(encoding="utf-8")
    # Precompute is restored between clicks; correctness now comes from the
    # global bridge+server fence immediately before physical input.
    assert "def _schedule_precompute(self) -> None:" in src
    assert "def _schedule_speculative_best_child" in src
    assert "self._schedule_speculative_best_child(mask, recs)" in src
    assert "self._ui_call(self._schedule_precompute)" in src
    assert "self.input_exclusive.acquire" in src


def test_v1825_diagnostic_disk_io_defaults_off():
    cfg = AppConfig()
    assert cfg.save_runtime_logs is False
    assert cfg.save_game_images is False


def test_v1825_no_diagnostic_disk_io_required_by_input_path():
    src = (ROOT / "vision_engine" / "gui.py").read_text(encoding="utf-8")
    assert "if self.app_cfg.save_runtime_logs:" in src
    assert "if not self.app_cfg.save_game_images" in src
    commit = src[src.index("    def _execute_autoplay_click"):src.index("    def _autoplay_transition_watchdog")]
    assert ".open(" not in commit
    assert "cv2.imwrite" not in commit
    assert "time.sleep" not in commit


def test_v1825_1000_generation_cycles_never_accept_previous_token():
    gate = AutoPlayCommitCoordinator()
    previous = None
    for i in range(1000):
        token = gate.reserve(f"ep-{i}", i & 0xFFFF, "BL", i % 25)
        assert token is not None
        if previous is not None:
            assert not gate.is_current(previous)
            assert not gate.begin_prepare(previous)
        assert gate.begin_prepare(token)
        assert gate.begin_quiesce(token)
        assert gate.begin_commit(token)
        assert gate.mark_clicked(token)
        assert gate.confirm(token)
        previous = token


def test_v1825_server_suppresses_precompute_while_background_disabled():
    engine = _bare_engine()
    engine.set_background_enabled(False)
    result = engine.precompute([object(), object(), object()])
    assert result["scheduled"] == 0
    assert result["suppressed"] == 3


def test_v1825_bridge_exclusive_barrier_evolved_to_global_freeze():
    from vision_engine.live.solver_admission import BridgeAdmissionGate
    calls = []

    class FakeClient:
        def freeze(self, session_id, timeout):
            calls.append(("freeze", session_id))
            return {
                "ok": True, "idle": True, "foreground": 0, "background": 0,
                "exclusive_token": "tok",
            }

        def release(self, session_id, token):
            calls.append(("release", session_id, token))
            return {"ok": True}

    bridge = LiveSolverBridge.__new__(LiveSolverBridge)
    bridge.available = True
    bridge.client = FakeClient()
    bridge.admission = BridgeAdmissionGate()
    bridge.session_id = "session-test"
    bridge._exclusive_token = None
    result = bridge.prepare_input_exclusive(timeout=1.25)
    assert result["ok"] is True and result["idle"] is True
    assert result["foreground"] == 0 and result["background"] == 0
    assert calls == [("freeze", "session-test")]
    bridge.release_input_exclusive(result["exclusive_token"])
    assert calls[-1] == ("release", "session-test", "tok")


def test_v1825_combined_background_then_monitor_barrier_sequence():
    commit = AutoPlayCommitCoordinator()
    monitor = MonitorInputGate()
    engine = _bare_engine()
    token = commit.reserve("ep", 7, "CH", 12)
    assert token is not None and commit.begin_prepare(token)

    with engine._lock:
        engine._pending.add("bg")

    monitor_parked = threading.Event()

    def monitor_thread():
        while not monitor.requested:
            time.sleep(0.001)
        monitor_parked.set()
        monitor.monitor_safe_point(running=True)

    def background_thread():
        time.sleep(0.03)
        with engine._background_idle:
            engine._pending.discard("bg")
            engine._background_idle.notify_all()

    mt = threading.Thread(target=monitor_thread)
    bt = threading.Thread(target=background_thread)
    mt.start(); bt.start()

    # The intended V25 ordering: solver background must be truly idle first.
    assert engine.wait_background_idle(0.5)["idle"]
    assert commit.begin_quiesce(token)
    assert monitor.request_quiesce(timeout=0.5)
    assert monitor_parked.is_set()
    assert commit.begin_commit(token)
    assert commit.mark_clicked(token)
    monitor.release()
    assert commit.confirm(token)

    mt.join(1.0); bt.join(1.0)
    assert not mt.is_alive() and not bt.is_alive()
