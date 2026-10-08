import threading
import time

from vision_engine import __version__
from vision_engine.live.state import GameState
from vision_engine.solver.client import SolverClient
from vision_engine.solver.protocol import SolverRequest, SolverSettings
from vision_engine.solver.service import SolverEngine, SolverTCPServer


def _state():
    return GameState(
        episode_id="round1", state_version=1, strategy_version=0, board_hash="b",
        board_cells=("BL","CR","CH","CH","CR","CR","SO","BL","LI","CR","CH","HA",None,"SO","LI","SO","LI","HA","BL","LI","BL","CH","SO","HA","HA"),
        accepted_mask=(1 << 4) | (1 << 13), current_jewel="LI", strategy="score",
    )


def test_persistent_service_cache_and_protocol():
    engine = SolverEngine(workers=1, cache_entries=32)
    server = SolverTCPServer(("127.0.0.1", 0), engine, socket_timeout=5.0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    c = SolverClient(host, port, timeout=10)
    try:
        ping = c.ping()
        assert ping["ok"]
        assert ping["app_version"] == __version__
        req = SolverRequest(1, _state(), SolverSettings(rollouts=20))
        first = c.solve(req)
        second = c.solve(SolverRequest(2, _state(), SolverSettings(rollouts=20)))
        assert first.recommendations
        assert not first.cache_hit
        assert second.cache_hit
        assert [(r.position, r.utility) for r in first.recommendations] == [(r.position, r.utility) for r in second.recommendations]
        assert second.request_id == 2
    finally:
        c.close(); server.shutdown(); server.server_close(); engine.close()
