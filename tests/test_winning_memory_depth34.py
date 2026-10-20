"""Regression: only verified-through-replay >1000 exact histories can influence ties."""
from __future__ import annotations


from vision_engine.board import Board
from vision_engine.config import AppConfig
from vision_engine.experience import ExperiencePolicy
from vision_engine.persistence import BingoDB
from vision_engine.scoring import score_mask
from vision_engine.winning_memory import WinningMemory


BOARD = Board(tuple([
    "CH", "HA", "SO", "CH", "HA",
    "SO", "BL", "BL", "CR", "LI",
    "BL", "CH", None, "CR", "HA",
    "BL", "SO", "CH", "LI", "SO",
    "LI", "CR", "CR", "LI", "HA",
]))
WIN_POS = (0, 2, 6, 7, 10, 11, 13, 14, 17, 18, 21, 22, 23, 24)
assert score_mask(sum(1 << pos for pos in WIN_POS)).total > 1000


def save_game(db, game_id, positions=WIN_POS, force_total=None, tamper_step=None):
    jewels = [BOARD.cells[i] for i in positions]
    db.ensure_game(game_id, board=BOARD)
    mask = 0
    for step, (pos, jewel) in enumerate(zip(positions, jewels)):
        db.save_action(game_id, step, mask, jewel, pos)
        mask |= 1 << pos
    actual = score_mask(mask)
    db.finalize_game(game_id, BOARD, list(BOARD.jewel_counts_in_mask(mask)), jewels,
                     mask, {"total": actual.total if force_total is None else force_total}, None, None)
    if tamper_step is not None:
        with db.connect() as con:
            con.execute("UPDATE actions SET jewel='SO' WHERE game_id=? AND step=?", (game_id, tamper_step))
    return actual


def test_winning_memory_requires_complete_consistent_win_and_exact_prefix(tmp_path):
    db = BingoDB(tmp_path / 'b.sqlite3')
    save_game(db, 'winner')
    save_game(db, 'fake', force_total=1200)
    save_game(db, 'corrupt', tamper_step=0)
    memory = WinningMemory.from_db(db)
    assert memory.winning_game_ids == {'winner'}
    jewel = BOARD.cells[WIN_POS[1]]
    mask = 1 << WIN_POS[0]
    hit = memory.matching_scores(BOARD, mask, jewel, BOARD.legal_positions(jewel, mask),
                                 [BOARD.cells[WIN_POS[0]]], [WIN_POS[0]])
    assert list(hit) == [WIN_POS[1]]
    assert memory.matching_scores(BOARD, mask, jewel, BOARD.legal_positions(jewel, mask),
                                   [BOARD.cells[WIN_POS[0]]], [WIN_POS[2]]) == {}
    assert memory.matching_scores(BOARD, mask, 'CH', BOARD.legal_positions('CH', mask),
                                   [BOARD.cells[WIN_POS[0]]], [WIN_POS[0]]) == {}
    assert memory.matching_scores(BOARD, mask, jewel, BOARD.legal_positions(jewel, mask),
                                   [], []) == {}
    assert memory.matching_scores(BOARD, mask, jewel, BOARD.legal_positions(jewel, mask),
                                   None, None) == {}


def test_no_ml_without_win_or_exact_match_and_stale_model_ignored(tmp_path):
    db_path = tmp_path / 'b.sqlite3'
    db = BingoDB(db_path)
    # One made-up >1000 number on a losing final mask does not qualify.
    positions = tuple(i for i in range(25) if i != 12)[:14]
    save_game(db, 'loser', positions, force_total=1100)
    model = tmp_path / 'model.joblib'
    from joblib import dump
    dump({'counterfactual_model': object(), 'info': {'counterfactual_trusted': True}}, model)
    policy = ExperiencePolicy(db_path, model)
    assert policy.counterfactual_model is None
    assert policy.predict(BOARD, 0, BOARD.cells[WIN_POS[0]], [WIN_POS[0]],
                          observed_jewels=[], observed_positions=[]) == {}
    save_game(db, 'winner')
    policy.reload()
    assert policy.predict(BOARD, 0, BOARD.cells[WIN_POS[0]], [WIN_POS[0]],
                          observed_jewels=[], observed_positions=[])[WIN_POS[0]] > 1
    assert policy.predict(BOARD, 0, BOARD.cells[WIN_POS[0]], [WIN_POS[0]],
                          observed_jewels=None, observed_positions=None) == {}


def test_train_only_wins_never_negative_real_outcome_mae(tmp_path):
    db_path = tmp_path / 'b.sqlite3'
    db = BingoDB(db_path)
    save_game(db, 'winner')
    save_game(db, 'fake', force_total=2000)
    policy = ExperiencePolicy(db_path, tmp_path / 'model.joblib')
    info = policy.train(min_counterfactual=99999)
    assert policy.winning_memory.winning_game_ids == {'winner'}
    assert info.outcome_residual_mae is None
    assert info.outcome_trusted is False
    assert policy.predict(BOARD, 0, BOARD.cells[WIN_POS[0]], [WIN_POS[0]],
                          observed_jewels=[], observed_positions=[])


def test_adaptive_3_to_4_is_bounded_and_configurable():
    cfg = AppConfig()
    mask = sum(1 << i for i in [1, 2, 3, 4])
    assert cfg.live_horizon(mask, 2, cfg.fast_rollouts) == 3
    assert cfg.live_horizon(mask, 2, cfg.live_rollouts) == 4
    assert cfg.live_horizon(mask, 3, cfg.live_rollouts) == 3
    assert cfg.live_horizon(0, 2, cfg.live_rollouts) == 3
    assert cfg.live_horizon(mask | 1 << 5 | 1 << 6 | 1 << 7, 2, cfg.live_rollouts) == 3
    cfg.live_rollout_exact_horizon_max = 3
    assert cfg.live_horizon(mask, 2, cfg.live_rollouts) == 3
