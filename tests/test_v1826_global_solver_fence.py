from __future__ import annotations

import threading
import time

from vision_engine.solver.fence import GlobalSolverFence


def test_v1826_new_session_rotates_and_opens_admission():
    fence = GlobalSolverFence()
    info = fence.begin_session("session-a", 0.2)
    assert info["ok"] is True
    assert info["session_id"] == "session-a"
    assert info["admission_open"] is True
    assert fence.enter_foreground("session-a") is True
    fence.leave_foreground()
    assert fence.enter_foreground("session-old") is False


def test_v1826_freeze_waits_for_foreground_and_background():
    fence = GlobalSolverFence()
    assert fence.begin_session("s", 0.2)["ok"]
    assert fence.enter_foreground("s")
    assert fence.enter_background("s")

    def finish():
        time.sleep(0.03)
        fence.leave_foreground()
        time.sleep(0.02)
        fence.leave_background()

    t = threading.Thread(target=finish)
    t.start()
    started = time.perf_counter()
    frozen = fence.freeze("s", 0.5)
    elapsed = time.perf_counter() - started
    t.join(1)

    assert frozen["ok"] is True
    assert frozen["foreground"] == 0
    assert frozen["background"] == 0
    assert frozen["frozen"] is True
    assert frozen["admission_open"] is False
    assert elapsed >= 0.04
    assert fence.enter_foreground("s") is False
    assert fence.enter_background("s") is False

    released = fence.release("s", frozen["exclusive_token"])
    assert released["ok"] is True
    assert released["admission_open"] is True
    assert fence.enter_foreground("s") is True
    fence.leave_foreground()


def test_v1826_invalid_token_cannot_reopen_solver():
    fence = GlobalSolverFence()
    assert fence.begin_session("s", 0.2)["ok"]
    frozen = fence.freeze("s", 0.2)
    assert frozen["ok"]
    bad = fence.release("s", "wrong")
    assert bad["ok"] is False
    assert fence.snapshot()["admission_open"] is False
    assert fence.release("s", frozen["exclusive_token"])["ok"] is True


def test_v1826_session_rotation_waits_for_old_work():
    fence = GlobalSolverFence()
    assert fence.begin_session("old", 0.2)["ok"]
    assert fence.enter_background("old")

    def finish_old():
        time.sleep(0.03)
        fence.leave_background()

    t = threading.Thread(target=finish_old)
    t.start()
    rotated = fence.begin_session("new", 0.5)
    t.join(1)
    assert rotated["ok"] is True
    assert rotated["session_id"] == "new"
    assert fence.enter_foreground("old") is False
    assert fence.enter_foreground("new") is True
    fence.leave_foreground()

from vision_engine.live.solver_admission import BridgeAdmissionGate


def test_v1826_windows_bridge_admission_closes_and_waits_active_ipc():
    gate = BridgeAdmissionGate()
    ticket = gate.ticket("solve")
    assert ticket is not None
    assert gate.enter(ticket)

    def finish():
        time.sleep(0.03)
        gate.leave()

    t = threading.Thread(target=finish)
    t.start()
    closed = gate.close()
    assert closed["open"] is False
    idle = gate.wait_idle(0.5)
    t.join(1)
    assert idle["idle"] is True
    assert gate.ticket("solve") is None
    gate.open()
    assert gate.ticket("solve") is not None


def test_v1826_ticket_issued_before_close_becomes_stale():
    gate = BridgeAdmissionGate()
    ticket = gate.ticket("solve")
    assert ticket is not None
    gate.close()
    assert gate.enter(ticket) is False
    gate.open()
    assert gate.is_current(ticket) is False


def test_v1826_new_session_cannot_steal_active_exclusive_owner():
    fence = GlobalSolverFence()
    assert fence.begin_session("old", 0.2)["ok"]
    frozen = fence.freeze("old", 0.2)
    assert frozen["ok"]
    started = time.perf_counter()
    rotated = fence.begin_session("new", 0.03)
    assert rotated["ok"] is False
    assert rotated["reason"] == "exclusive-owner-still-active"
    assert time.perf_counter() - started >= 0.02
    assert fence.release("old", frozen["exclusive_token"])["ok"]
    assert fence.begin_session("new", 0.2)["ok"]
