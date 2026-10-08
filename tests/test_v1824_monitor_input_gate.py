import threading
import time

from vision_engine.input_gate import MonitorInputGate


def test_monitor_gate_handshake_parks_and_releases():
    gate = MonitorInputGate()
    entered = threading.Event()
    left = threading.Event()

    def monitor():
        while not gate.requested:
            time.sleep(0.001)
        entered.set()
        gate.monitor_safe_point(running=True)
        left.set()

    t = threading.Thread(target=monitor)
    t.start()
    assert gate.request_quiesce(timeout=1.0)
    assert entered.wait(0.5)
    assert gate.quiesced
    assert not left.is_set()
    gate.release()
    t.join(1.0)
    assert left.is_set()
    assert not gate.quiesced


def test_monitor_gate_prevents_second_commit_owner():
    gate = MonitorInputGate()

    def monitor():
        while not gate.requested:
            time.sleep(0.001)
        gate.monitor_safe_point(running=True)

    t = threading.Thread(target=monitor)
    t.start()
    assert gate.request_quiesce(timeout=1.0)
    assert gate.request_quiesce(timeout=0.01) is False
    gate.release()
    t.join(1.0)
