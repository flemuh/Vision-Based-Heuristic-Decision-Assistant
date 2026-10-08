from __future__ import annotations

import threading

from vision_engine.live.state import GameState
from vision_engine.solver.client import SolverClient
from vision_engine.solver.protocol import SolverRequest, SolverSettings
from vision_engine.solver.service import SolverEngine, SolverTCPServer


BOARD = (
    "BL","CR","CH","CH","CR",
    "CR","SO","BL","LI","CR",
    "CH","HA",None,"SO","LI",
    "SO","LI","HA","BL","LI",
    "BL","CH","SO","HA","HA",
)


def _state(jewel: str | None = None) -> GameState:
    return GameState(
        episode_id="ep", state_version=3, strategy_version=0, board_hash="board",
        board_cells=BOARD, accepted_mask=(1 << 4) | (1 << 13) | (1 << 17),
        current_jewel=jewel, strategy="score",
    )


def _requests(session_id: str, count: int = 6) -> list[SolverRequest]:
    jewels = ("BL", "SO", "LI", "CR", "HA", "CH")
    settings = SolverSettings(rollouts=2, exact_horizon=2, rollout_exact_horizon=1)
    return [
        SolverRequest(-(i + 1), _state(j), settings, purpose="test-precompute", session_id=session_id)
        for i, j in enumerate(jewels[:count])
    ]


def _server():
    engine = SolverEngine(workers=3, cache_entries=64)
    server = SolverTCPServer(("127.0.0.1", 0), engine, socket_timeout=20.0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return engine, server


def test_v1826_pending_six_are_drained_before_exclusive_token():
    engine, server = _server()
    host, port = server.server_address
    client = SolverClient(host, port, timeout=20)
    try:
        assert client.begin_session("session-a", 2.0)["ok"]
        ack = client.precompute_requests(_requests("session-a"))
        assert ack["scheduled"] + ack["cached"] + ack["pending"] + ack.get("suppressed", 0) == 6

        frozen = client.freeze("session-a", timeout=10.0)
        assert frozen["ok"] is True
        assert frozen["idle"] is True
        assert frozen["foreground"] == 0
        assert frozen["background"] == 0
        assert frozen["admission_open"] is False
        assert frozen["exclusive_token"]

        # Server must fail closed while physical input owns the token.
        req = SolverRequest(99, _state("LI"), SolverSettings(rollouts=1), session_id="session-a")
        try:
            client.solve(req)
        except RuntimeError as exc:
            assert "admission closed" in str(exc)
        else:
            raise AssertionError("foreground solve was admitted while server was frozen")

        released = client.release("session-a", frozen["exclusive_token"])
        assert released["ok"] is True
        solved = client.solve(req)
        assert solved.recommendations
    finally:
        client.close(); server.shutdown(); server.server_close(); engine.close()


def test_v1826_restart_session_drains_old_background_before_new_admission():
    engine, server = _server()
    host, port = server.server_address
    client = SolverClient(host, port, timeout=20)
    try:
        assert client.begin_session("old-session", 2.0)["ok"]
        ack = client.precompute_requests(_requests("old-session"))
        assert ack["scheduled"] >= 1

        rotated = client.begin_session("new-session", 10.0)
        assert rotated["ok"] is True
        assert rotated["session_id"] == "new-session"
        assert rotated["foreground"] == 0
        assert rotated["background"] == 0

        old_req = SolverRequest(1, _state("BL"), SolverSettings(rollouts=1), session_id="old-session")
        try:
            client.solve(old_req)
        except RuntimeError as exc:
            assert "stale session" in str(exc)
        else:
            raise AssertionError("old session remained admissible after restart rotation")

        new_req = SolverRequest(2, _state("BL"), SolverSettings(rollouts=1), session_id="new-session")
        assert client.solve(new_req).recommendations
    finally:
        client.close(); server.shutdown(); server.server_close(); engine.close()
