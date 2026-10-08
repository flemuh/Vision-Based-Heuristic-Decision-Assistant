from vision_engine.board import Board
from vision_engine.live.coordinator import LiveCoordinator
from vision_engine.live.state import GameState
from vision_engine.solver.protocol import SolverRequest, SolverResponse, SolverSettings
from vision_engine.solver_process import solve_request


def _board():
    # Known valid 5x5 fixture with MU in the center.
    cells = (
        "BL","CR","CH","CH","CR",
        "CR","SO","BL","LI","CR",
        "CH","HA",None,"SO","LI",
        "SO","LI","HA","BL","LI",
        "BL","CH","SO","HA","HA",
    )
    return Board(cells)


def _state(mask=0, jewel="LI", strategy="score"):
    b = _board()
    return GameState(
        episode_id="e1", state_version=7, strategy_version=2,
        board_hash="fixture", board_cells=b.cells, accepted_mask=mask,
        current_jewel=jewel, strategy=strategy,
    )


def test_protocol_preserves_legacy_solver_result():
    state = _state(mask=(1 << 4) | (1 << 13), jewel="LI")
    settings = SolverSettings(rollouts=60, exact_horizon=6, rollout_exact_horizon=3)
    req = SolverRequest(11, state, settings)
    via_protocol = solve_request(req.to_legacy_request())
    direct = solve_request(req.to_legacy_request())
    assert [(r.position, r.utility) for r in via_protocol] == [(r.position, r.utility) for r in direct]


def test_coordinator_rejects_stale_board_result():
    state = _state()
    c = LiveCoordinator(state)
    resp = SolverResponse.for_request(SolverRequest(1, state, SolverSettings(rollouts=10)), [])
    assert c.accepts(resp)
    c.update_board(accepted_mask=1 << 0)
    assert not c.accepts(resp)


def test_coordinator_rejects_stale_strategy_result():
    state = _state()
    c = LiveCoordinator(state)
    resp = SolverResponse.for_request(SolverRequest(1, state, SolverSettings(rollouts=10)), [])
    c.update_strategy("adaptive")
    assert not c.accepts(resp)


def test_complete_is_absolute_publish_barrier():
    state = _state()
    c = LiveCoordinator(state)
    c.update_board(accepted_mask=state.accepted_mask, phase="complete")
    now = c.state
    resp = SolverResponse.for_request(SolverRequest(2, now, SolverSettings(rollouts=10)), [])
    assert not c.accepts(resp)
