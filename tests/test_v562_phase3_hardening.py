import sqlite3

from vision_engine.config import AppConfig
from vision_engine.persistence import BingoDB, SCHEMA_VERSION
from vision_engine.temporal import TemporalStateFilter


def test_within_round_mask_regression_is_ignored_not_accepted_or_desync():
    f = TemporalStateFilter(frames=3, required=2, max_mask_jump=1)
    accepted = 0b1111
    f.reset(accepted)
    assert not f.observe(0, "SO").stable
    s = f.observe(0, "SO")
    assert not s.stable
    assert s.mask == accepted
    assert s.reason == "ignored-mask-regression"
    assert "desync" not in s.reason


def test_mask_can_continue_forward_after_temporary_regression():
    f = TemporalStateFilter(frames=3, required=2, max_mask_jump=1)
    f.reset(0b11)
    f.observe(0, "SO")
    f.observe(0, "SO")
    # Return to the accepted state twice to clear the temporary regression vote.
    f.observe(0b11, "SO")
    f.observe(0b11, "SO")
    f.observe(0b111, "SO")
    s = f.observe(0b111, "SO")
    assert s.stable and s.mask == 0b111


def test_v562_defaults_use_exact_horizon_six_and_reproducible_seed():
    cfg = AppConfig()
    assert cfg.exact_horizon == 6
    assert cfg.rng_seed == 1337


def test_schema_persists_reproducibility_and_route_metadata(tmp_path):
    path = tmp_path / "bingo.sqlite3"
    db = BingoDB(path)
    with sqlite3.connect(path) as con:
        cols = {row[1] for row in con.execute("PRAGMA table_info(decision_samples)")}
        assert {"rng_seed", "scenario_set", "routes_json", "primary_stat_tie", "tie_break_metric"} <= cols
        version = con.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()[0]
        assert int(version) == SCHEMA_VERSION == 9


def test_decision_metadata_round_trips_route_and_scenario_fields(tmp_path):
    from vision_engine.board import Board
    path = tmp_path / "bingo.sqlite3"
    db = BingoDB(path)
    board = Board.from_rows([
        ["CH", "HA", "SO", "CH", "HA"],
        ["SO", "BL", "BL", "CR", "LI"],
        ["BL", "CH", None, "CR", "HA"],
        ["BL", "SO", "CH", "LI", "SO"],
        ["LI", "CR", "CR", "LI", "HA"],
    ])
    db.ensure_game("g1", board=board)
    row = {
        "position": 0, "method": "test", "active_goal": ">1000", "samples": 15,
        "stderr_p3": 0.0, "stderr_gt1000": 0.01, "stderr_score": 1.0, "solver_regret": 0.0,
        "p3": .1, "p2l1n": .2, "p1l2n": .3, "p1l1n": .4,
        "p_gt1000": .5, "p_ge999": .6, "expected_score": 900.0, "solver_target": 1.0,
        "rng_seed": 1337, "scenario_set": "seed=1337|demo",
        "routes": [{"target": "3 Lucky", "probability": .25}],
        "primary_stat_tie": True, "tie_break_metric": "P(>=999)",
    }
    db.save_decision_samples("g1", 0, 0, "CH", board, [row])
    with sqlite3.connect(path) as con:
        got = con.execute(
            "SELECT rng_seed,scenario_set,routes_json,primary_stat_tie,tie_break_metric FROM decision_samples"
        ).fetchone()
    assert got[0] == 1337
    assert got[1] == "seed=1337|demo"
    assert '"3 Lucky"' in got[2]
    assert got[3] == 1
    assert got[4] == "P(>=999)"
