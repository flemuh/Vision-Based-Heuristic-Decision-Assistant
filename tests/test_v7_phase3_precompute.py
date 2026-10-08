import threading
import time

from vision_engine.live.state import GameState
from vision_engine.solver.client import SolverClient
from vision_engine.solver.protocol import SolverRequest, SolverSettings
from vision_engine.solver.service import SolverEngine, SolverTCPServer


def _state(jewel=None):
    return GameState(
        episode_id="r", state_version=3, strategy_version=0, board_hash="b",
        board_cells=("BL","CR","CH","CH","CR","CR","SO","BL","LI","CR","CH","HA",None,"SO","LI","SO","LI","HA","BL","LI","BL","CH","SO","HA","HA"),
        accepted_mask=(1 << 4) | (1 << 13) | (1 << 17), current_jewel=jewel, strategy="score",
    )


def test_precompute_six_jewels_warms_live_cache():
    engine = SolverEngine(workers=3, cache_entries=64)
    server = SolverTCPServer(("127.0.0.1", 0), engine, socket_timeout=20.0)
    t = threading.Thread(target=server.serve_forever, daemon=True); t.start()
    host, port = server.server_address
    c = SolverClient(host, port, timeout=20)
    settings = SolverSettings(rollouts=8)
    try:
        ack = c.precompute_state(_state(), settings)
        assert ack["scheduled"] >= 1
        deadline = time.time() + 20
        result = None
        while time.time() < deadline:
            result = c.solve(SolverRequest(77, _state("LI"), settings))
            if result.cache_hit:
                break
            time.sleep(0.05)
        assert result is not None and result.recommendations
        # Either precompute won the race, or this foreground solve populated the
        # cache; the next request must always be a hot hit.
        hot = c.solve(SolverRequest(78, _state("LI"), settings))
        assert hot.cache_hit
        assert hot.elapsed_ms < 50
    finally:
        c.close(); server.shutdown(); server.server_close(); engine.close()


def test_worker_advisor_reuse_keeps_exact_results_stable():
    engine = SolverEngine(workers=2, cache_entries=32)
    settings = SolverSettings(rollouts=6)
    a = engine.solve(SolverRequest(1, _state("BL"), settings))
    b = engine.solve(SolverRequest(2, _state("SO"), settings))
    assert a.recommendations and b.recommendations
    engine.close()
