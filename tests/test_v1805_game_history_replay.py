from __future__ import annotations

from collections import Counter

from vision_engine.board import Board
from vision_engine.constants import JEWELS
from vision_engine.persistence import BingoDB
from vision_engine.replay import actual_sequence_hindsight, format_replay_report, list_stored_games, stored_replay
from vision_engine.scoring import score_mask


def _board() -> Board:
    return Board((
        "CH", "CR", "HA", "SO", "HA",
        "CR", "BL", "HA", "HA", "SO",
        "BL", "LI", None, "CH", "CH",
        "SO", "LI", "LI", "LI", "CH",
        "CR", "BL", "CR", "SO", "BL",
    ))


def _sequence_and_positions():
    # This draw composition can make 3 Lucky on H+V+\\ (or H+\\+/).
    # Taking the off-route BL at L5C2 as the first move makes 3 Lucky
    # impossible with the same future jewel sequence.
    sequence = ["BL", "CH", "LI", "HA", "SO", "BL", "CH", "LI", "HA", "CR", "BL", "CH", "LI", "SO"]
    positions_by_jewel = {
        "BL": iter((21, 6, 10)),       # first BL is the irreversible hindsight loss
        "CH": iter((0, 13, 14)),
        "LI": iter((11, 17, 18)),
        "HA": iter((2, 7)),
        "CR": iter((22,)),
        "SO": iter((3, 9)),
    }
    positions = [next(positions_by_jewel[j]) for j in sequence]
    return sequence, positions


def test_exact_sequence_hindsight_finds_first_three_lucky_loss():
    board = _board()
    sequence, positions = _sequence_and_positions()
    actions = [
        {"step": i, "jewel": jewel, "action_pos": pos, "recommended_pos": pos}
        for i, (jewel, pos) in enumerate(zip(sequence, positions))
    ]
    selected = sum(1 << p for p in positions)

    h = actual_sequence_hindsight(board, sequence, actions, selected)
    assert h["available"]
    assert h["three_lucky_possible"]
    assert h["best_lucky"] >= 3
    assert h["actual_lucky"] < 3
    loss = h["first_loss_three_lucky"]
    assert loss is not None
    assert loss["step"] == 1
    assert loss["jewel"] == "BL"
    assert loss["actual"] == "L5C2"
    assert {"L2C2", "L3C1", "L5C5"} <= set(loss["alternative_coords"])


def test_history_lists_games_and_specific_replay_includes_saved_candidates(tmp_path):
    db = BingoDB(tmp_path / "bingo.sqlite3")
    board = _board()
    sequence, positions = _sequence_and_positions()
    game_id = "history-test"
    db.ensure_game(game_id, board=board, box_color="green", operation_mode="recommend", risk_profile="auto")

    mask = 0
    for step, (jewel, pos) in enumerate(zip(sequence, positions)):
        legal = board.legal_positions(jewel, mask)
        rows = []
        for rank, candidate in enumerate(legal, 1):
            rows.append({
                "position": candidate,
                "rank": rank,
                "method": "test",
                "active_goal": "3 Lucky",
                "samples": 100,
                "p3": 0.50 if rank == 1 else 0.25,
                "p2l1n": 0.0,
                "p1l2n": 0.0,
                "p1l1n": 0.0,
                "p_gt1000": 0.55 if rank == 1 else 0.30,
                "p_ge999": 0.60,
                "expected_score": 1000.0 - rank,
                "solver_target": 0.0,
                "solver_regret": float(rank - 1),
            })
        db.save_decision_samples(game_id, step, mask, jewel, board, rows)
        recommended = legal[0]
        db.save_action(
            game_id, step, mask, jewel, pos,
            recommended_pos=recommended,
            solver_regret=0.0 if pos == recommended else 1.0,
        )
        mask |= 1 << pos

    counts = Counter(sequence)
    score = score_mask(mask)
    db.finalize_game(
        game_id, board, [counts[j] for j in JEWELS], sequence, mask,
        {"lucky": score.lucky, "normal": score.normal, "jewel_cells": score.jewel_cells, "total": score.total},
        None, None,
    )

    games = list_stored_games(db)
    assert games and games[0]["game_id"] == game_id
    assert games[0]["actions"] == 14

    replay = stored_replay(db, game_id)
    assert replay is not None
    assert replay["actions"] == 14
    assert replay["steps"][0]["candidates"]
    assert replay["hindsight"]["available"]
    report = format_replay_report(replay)
    assert "ACTUAL-SEQUENCE HINDSIGHT" in report
    assert "First irreversible loss of 3 Lucky: move 1 BL -> L5C2" in report
