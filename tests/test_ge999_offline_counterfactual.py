"""Regression: 999 factual admission, offline >1000 alternate wins, no lookahead live."""

from vision_engine.board import Board
from vision_engine.experience import ExperiencePolicy
from vision_engine.persistence import BingoDB
from vision_engine.retrospective_learning import cached_audit, audit_stored_games, audit_summary
from vision_engine.scoring import score_mask
from vision_engine.winning_memory import WinningMemory, validated_episodes
from tests.test_winning_memory_depth34 import BOARD
from tests.test_v1806_hindsight_learning import _record_complete_game

NINE_NINE_NINE = (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 16, 17, 20, 22)
assert score_mask(sum(1 << i for i in NINE_NINE_NINE)).total == 999


def save_consistent(db, id, positions):
    mask = 0
    seq = []
    db.ensure_game(id, board=BOARD)
    for step, pos in enumerate(positions):
        jewel = BOARD.cells[pos]
        db.save_action(id, step, mask, jewel, pos)
        seq.append(jewel)
        mask |= 1 << pos
    db.finalize_game(id, BOARD, list(BOARD.jewel_counts_in_mask(mask)),
                     seq, mask, {'total': score_mask(mask).total}, None, None)


def test_999_is_inclusive_and_looks_up_only_identical_observed_prefix(tmp_path):
    db=BingoDB(tmp_path / 'record.sqlite3')
    save_consistent(db, 'exactly999', NINE_NINE_NINE)
    memory=WinningMemory.from_db(db)
    assert memory.winning_game_ids == {'exactly999'}
    first=NINE_NINE_NINE[0]
    assert memory.matching_scores(BOARD, 0, BOARD.cells[first], [first], [], [])[first] >= 1
    assert memory.matching_scores(BOARD, 0, BOARD.cells[first], [first], None, None) == {}
    wrong=Board(tuple(BOARD.cells))
    # Different full board is rejected; moving a tile would be a new board.
    assert memory.matching_scores(BOARD, 1 << first, BOARD.cells[1], [1], [], []) == {}


def test_losing_894_has_real_offline_1176_opportunity_without_live_memory(tmp_path):
    db=BingoDB(tmp_path / 'bingo.sqlite3')
    board, seq, positions = _record_complete_game(db)
    examples=validated_episodes(db)
    assert len(examples)==1 and examples[0].total == 894
    report=cached_audit(db,examples[0]); assert report['valid']
    assert report['source']=='offline_counterfactual_known_future'
    assert report['observed_total']==894 and report['best_total']==1176
    assert report['could_exceed_1000'] and report['counterfactual_gain_upper_bound']==282
    assert set(report['first_better_choice']['better_alternative_positions']) >= {6,10,24}
    assert audit_stored_games(db)['recoverable_gt1000']==1
    assert audit_summary(db)['recoverable_gt1000']==1
    policy=ExperiencePolicy(db.path,tmp_path/'policy.joblib')
    info=policy.train(min_counterfactual=99999,min_hindsight_games=1,min_hindsight_samples=1,holdout_fraction=0)
    assert info.counterfactual_recovery_games==1
    assert info.hindsight_samples>0
    assert policy.predict(board,0,seq[0],board.legal_positions(seq[0],0),observed_jewels=[],observed_positions=[]) == {}
    assert cached_audit(db,examples[0],compute=False)==report


def test_invalid_history_and_verified_disagreement_never_train(tmp_path):
    db=BingoDB(tmp_path/'b.sqlite3')
    save_consistent(db,'valid',NINE_NINE_NINE)
    original=validated_episodes(db)[0]
    cached_audit(db,original)
    assert audit_summary(db)['analyzed']==1
    db.verify_score('valid',{'total':99999,'source':'manual'})
    assert not validated_episodes(db)
    assert audit_summary(db)['analyzed']==0
    assert WinningMemory.from_db(db).winning_game_ids==frozenset()


def test_ml_event_idempotent_and_counts_are_not_claimed_gain(tmp_path):
    db=BingoDB(tmp_path/'b.sqlite3')
    db.ensure_game('x',board=BOARD)
    db.record_ml_decision('x',0,'BL',True,True,10,21)
    db.record_ml_decision('x',0,'BL',True,False,21,21)
    db.record_ml_decision('x',1,'BL',False,False,21,21)
    assert db.ml_decision_summary()=={'decisions':2,'matches':1,'changes':0}


def test_repeated_rebuild_does_not_duplicate_hindsight(tmp_path):
    db=BingoDB(tmp_path/'b.sqlite3')
    _record_complete_game(db)
    model=tmp_path/'p.joblib'
    policy=ExperiencePolicy(db.path,model)
    args=dict(min_counterfactual=999999,min_hindsight_games=1,min_hindsight_samples=1,holdout_fraction=0)
    a=policy.train(**args)
    b=policy.train(**args)
    assert a.hindsight_samples == b.hindsight_samples
    assert a.counterfactual_recovery_games == b.counterfactual_recovery_games == 1
    assert audit_summary(db)['analyzed']==1


def test_wal_migration_backup_keeps_committed_user_games(tmp_path):
    import sqlite3
    original=tmp_path/'old.sqlite3'
    connection=sqlite3.connect(original)
    connection.execute('PRAGMA journal_mode=WAL')
    connection.execute('CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
    connection.execute("INSERT INTO metadata VALUES('schema_version','9')")
    connection.execute('CREATE TABLE user_marker(important TEXT)')
    connection.execute("INSERT INTO user_marker VALUES('saved-wal-record')")
    connection.commit()
    # Active WAL may hold committed records that shutil.copy2 would miss.
    db=BingoDB(original)
    backup=db.migration_backup_path
    assert backup and backup.is_file()
    with sqlite3.connect(backup) as source:
        assert source.execute('SELECT important FROM user_marker').fetchone()[0]=='saved-wal-record'
    with db.connect() as migrated:
        assert migrated.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()[0]=='10'
        assert migrated.execute('SELECT important FROM user_marker').fetchone()[0]=='saved-wal-record'
        assert {'retrospective_audits','ml_decision_events'}.issubset({x[0] for x in migrated.execute('SELECT name FROM sqlite_master WHERE type=\'table\'')})
    connection.close()
