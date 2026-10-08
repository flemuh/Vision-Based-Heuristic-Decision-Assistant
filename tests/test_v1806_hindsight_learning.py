from __future__ import annotations

from collections import Counter

from vision_engine.board import Board
from vision_engine.constants import JEWELS
from vision_engine.experience import ExperiencePolicy
from vision_engine.persistence import BingoDB
from vision_engine.scoring import score_mask


def _board() -> Board:
    return Board((
        "CH", "CR", "HA", "SO", "HA",
        "CR", "BL", "HA", "HA", "SO",
        "BL", "LI", None, "CH", "CH",
        "SO", "LI", "LI", "LI", "CH",
        "CR", "BL", "CR", "SO", "BL",
    ))


def _record_complete_game(db: BingoDB, game_id: str = "reward-test") -> tuple[Board, list[str], list[int]]:
    board = _board()
    sequence = ["BL", "CH", "LI", "HA", "SO", "BL", "CH", "LI", "HA", "CR", "BL", "CH", "LI", "SO"]
    positions_by_jewel = {
        "BL": iter((21, 6, 10)),  # first BL destroys the hindsight 3-Lucky path
        "CH": iter((0, 13, 14)),
        "LI": iter((11, 17, 18)),
        "HA": iter((2, 7)),
        "CR": iter((22,)),
        "SO": iter((3, 9)),
    }
    positions = [next(positions_by_jewel[j]) for j in sequence]
    db.ensure_game(game_id, board=board, box_color="green", operation_mode="recommend", risk_profile="auto")

    mask = 0
    for step, (jewel, actual) in enumerate(zip(sequence, positions)):
        legal = board.legal_positions(jewel, mask)
        rows = []
        for rank, candidate in enumerate(legal, 1):
            rows.append({
                "position": candidate,
                "rank": rank,
                "method": "test",
                "active_goal": "3 Lucky",
                "samples": 100,
                "p3": 0.5,
                "p2l1n": 0.5,
                "p1l2n": 0.5,
                "p1l1n": 0.5,
                "p_gt1000": 0.5,
                "p_ge999": 0.5,
                "expected_score": 1000.0,
                "solver_target": 5000.0,
                "solver_regret": 0.0,
            })
        db.save_decision_samples(game_id, step, mask, jewel, board, rows)
        db.save_action(game_id, step, mask, jewel, actual, recommended_pos=actual, solver_regret=0.0)
        mask |= 1 << actual

    counts = Counter(sequence)
    score = score_mask(mask)
    db.finalize_game(
        game_id, board, [counts[j] for j in JEWELS], sequence, mask,
        {"lucky": score.lucky, "normal": score.normal, "jewel_cells": score.jewel_cells, "total": score.total},
        None, None,
    )
    return board, sequence, positions


def test_hindsight_reward_model_learns_unchosen_winning_alternatives(tmp_path):
    db_path = tmp_path / "bingo.sqlite3"
    db = BingoDB(db_path)
    board, _sequence, _positions = _record_complete_game(db)

    policy = ExperiencePolicy(db_path, tmp_path / "policy.joblib")
    info = policy.train(
        min_counterfactual=999999,
        min_outcome_games=999999,
        min_hindsight_games=1,
        min_hindsight_samples=1,
        holdout_fraction=0.0,
    )

    assert info.hindsight_games == 1
    assert info.hindsight_samples > 0
    assert policy.hindsight_model is not None

    # The historical first move BL->L5C2 (index 21) killed 3-Lucky for that
    # actual draw. BL alternatives 6/10/24 preserved a 3-Lucky allocation.
    legal = board.legal_positions("BL", 0)
    scores = policy.predict(board, 0, "BL", legal)
    assert max(scores[p] for p in (6, 10, 24)) > scores[21]


def test_retraining_is_a_full_rebuild_not_duplicate_reward(tmp_path):
    db_path = tmp_path / "bingo.sqlite3"
    db = BingoDB(db_path)
    _record_complete_game(db)
    policy = ExperiencePolicy(db_path, tmp_path / "policy.joblib")

    first = policy.train(
        min_counterfactual=999999,
        min_outcome_games=999999,
        min_hindsight_games=1,
        min_hindsight_samples=1,
        holdout_fraction=0.0,
    )
    second = policy.train(
        min_counterfactual=999999,
        min_outcome_games=999999,
        min_hindsight_games=1,
        min_hindsight_samples=1,
        holdout_fraction=0.0,
    )
    assert second.hindsight_samples == first.hindsight_samples
    assert second.hindsight_games == first.hindsight_games == 1
