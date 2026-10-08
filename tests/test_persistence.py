import json

from vision_engine.board import Board
from vision_engine.persistence import BingoDB, board_hash


def board():
    return Board.from_rows([
        ["CH", "HA", "SO", "CH", "HA"],
        ["SO", "BL", "BL", "CR", "LI"],
        ["BL", "CH", None, "CR", "HA"],
        ["BL", "SO", "CH", "LI", "SO"],
        ["LI", "CR", "CR", "LI", "HA"],
    ])


def test_sqlite_roundtrip_and_counterfactuals(tmp_path):
    db = BingoDB(tmp_path / "bingo.sqlite3")
    b = board()
    db.ensure_game("g1", board=b)
    db.save_decision_samples(
        "g1", 0, 0, "CH", b,
        [{
            "position": 0, "method": "test", "p3": .2, "p2l1n": .3,
            "p1l2n": .4, "p1l1n": .5, "p_gt1000": .45, "p_ge999": .55,
            "expected_score": 1001.0, "solver_target": 9999.0,
        }],
    )
    db.save_action("g1", 0, 0, "CH", 0)
    rows = db.counterfactual_rows()
    assert len(rows) == 1
    assert rows[0]["chosen"] == 1
    assert rows[0]["board_json"] == json.dumps(list(b.cells), ensure_ascii=False, separators=(",", ":"))
    assert board_hash(b)


def test_v4_columns_events_and_backup(tmp_path):
    db = BingoDB(tmp_path / "bingo.sqlite3")
    with db.connect() as con:
        game_cols = {r[1] for r in con.execute("PRAGMA table_info(games)")}
        action_cols = {r[1] for r in con.execute("PRAGMA table_info(actions)")}
        decision_cols = {r[1] for r in con.execute("PRAGMA table_info(decision_samples)")}
    assert {"operation_mode", "risk_profile", "finished_at"} <= game_cols
    assert {"recommended_pos", "solver_regret", "current_margin", "current_entropy"} <= action_cols
    assert {"active_goal", "stderr_p3", "solver_regret", "rank"} <= decision_cols
    db.log_event(None, "test", "hello")
    backup = db.backup(tmp_path / "backups", keep=2)
    assert backup.exists() and backup.stat().st_size > 0
