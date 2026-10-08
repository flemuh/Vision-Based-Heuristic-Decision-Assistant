from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from vision_engine import __version__
from vision_engine.persistence import BingoDB, SCHEMA_VERSION
from vision_engine.solver.protocol import PROTOCOL_VERSION, request_from_wire
from vision_engine.versioning import MODEL_VERSION, STRATEGY_PLANNER_VERSION


def test_v18_contract_versions_are_explicit():
    assert __version__.startswith("18.")
    assert PROTOCOL_VERSION == 4
    assert SCHEMA_VERSION >= 8
    assert MODEL_VERSION == 6
    assert STRATEGY_PLANNER_VERSION == 22


def test_solver_protocol_rejects_silent_version_drift():
    with pytest.raises(ValueError, match="incompatible solver protocol"):
        request_from_wire({"protocol_version": 2})


def test_database_is_backed_up_before_forward_migration(tmp_path: Path):
    db = tmp_path / "bingo.sqlite3"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
    con.execute("INSERT INTO metadata(key,value) VALUES('schema_version','7')")
    con.commit(); con.close()

    upgraded = BingoDB(db)
    assert upgraded.migration_backup_path is not None
    assert upgraded.migration_backup_path.exists()
    with upgraded.connect() as con:
        assert int(con.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()[0]) == SCHEMA_VERSION
