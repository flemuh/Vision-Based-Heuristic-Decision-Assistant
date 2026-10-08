from vision_engine.solver_process import solve_request
from vision_engine.board import Board


def base_board():
    return Board.from_rows([
        ["CH", "HA", "SO", "CH", "HA"],
        ["SO", "BL", "BL", "CR", "LI"],
        ["BL", "CH", None, "CR", "HA"],
        ["BL", "SO", "CH", "LI", "SO"],
        ["LI", "CR", "CR", "LI", "HA"],
    ])


def _request(mode: str):
    b = base_board()
    return {
        "board_cells": list(b.cells),
        "mask": 0,
        "jewel": "BL",
        "mode": mode,
        "exact_horizon": 1,
        "rollouts": 40,
        "bias_weights": [1.0] * 6,
        "learned_scores": {},
        "p3_min": 0.08,
        "p2l1n_min": 0.12,
        "p1l2n_min": 0.16,
        "p1l1n_min": 0.22,
        "fallback_override_pp": 0.32,
        "p3_near_best_tolerance": 0.025,
        "ml_near_tie_probability": 0.012,
        "ml_near_tie_score": 15.0,
    }


def test_solver_process_payload_supports_all_three_strategies():
    for mode in ("adaptive", "score", "lucky"):
        recs = solve_request(_request(mode))
        assert len(recs) == 4
        assert all(r.position in base_board().positions("BL") for r in recs)
        assert recs[0].active_goal
