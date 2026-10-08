import threading
import time
from pathlib import Path

from vision_engine.config import AppConfig
from vision_engine.live.coordinator import LiveCoordinator
from vision_engine.live.solver_bridge import LiveSolverBridge
from vision_engine.live.state import GameState
from vision_engine.solver.client import SolverClient
from vision_engine.solver.protocol import SolverRequest, SolverSettings
from vision_engine.solver.service import SolverEngine, SolverTCPServer


CELLS=("BL","CR","CH","CH","CR","CR","SO","BL","LI","CR","CH","HA",None,"SO","LI","SO","LI","HA","BL","LI","BL","CH","SO","HA","HA")


def state(jewel=None, version=4):
    return GameState("ep", version, 1, "board", CELLS, (1<<4)|(1<<13)|(1<<17), jewel, "score")


def test_precompute_purpose_does_not_break_cache_hit():
    engine = SolverEngine(workers=3, cache_entries=64)
    server = SolverTCPServer(("127.0.0.1",0), engine, socket_timeout=20)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host,port=server.server_address
    c=SolverClient(host,port,timeout=20)
    settings=SolverSettings(rollouts=8)
    try:
        pre=SolverRequest(-1,state("LI"),settings,purpose="next-jewel-precompute")
        ack=c.precompute_requests([pre])
        assert ack["scheduled"] + ack["cached"] + ack["pending"] == 1
        deadline=time.time()+20
        while time.time()<deadline:
            hot=c.solve(SolverRequest(99,state("LI"),settings,purpose="live-decision"))
            if hot.cache_hit:
                assert hot.elapsed_ms < 50
                break
            time.sleep(.05)
        else:
            raise AssertionError("precomputed result never became a live cache hit")
    finally:
        c.close(); server.shutdown(); server.server_close(); engine.close()


def test_coordinator_versions_are_absolute_publish_contract():
    s=state("LI")
    c=LiveCoordinator(s)
    assert c.state.identity == ("ep",4,1)
    c.update_board(accepted_mask=s.accepted_mask | (1<<8))
    assert c.state.state_version == 5
    assert c.state.current_jewel is None
    c.update_strategy("adaptive")
    assert c.state.strategy_version == 2


def test_packaged_v7_config_prefers_wsl_and_fast_live_polling():
    cfg=AppConfig.load(Path(__file__).parents[1]/"data"/"app_config.json")
    assert cfg.solver_backend == "wsl"
    assert cfg.solver_port == 57641
    assert cfg.solver_wsl_distro == "Ubuntu"
    assert cfg.poll_ms <= 100
    assert cfg.transition_poll_ms <= 50


def test_gui_source_uses_final_remote_decision_and_precompute():
    src=(Path(__file__).parents[1]/"vision_engine"/"gui.py").read_text(encoding="utf-8")
    assert "_start_remote_solver" in src
    assert "_schedule_precompute" in src
    assert "coord.accepts(response)" in src
    assert "self._render_recommendations(mask, jewel, recs, refined=True)" in src
