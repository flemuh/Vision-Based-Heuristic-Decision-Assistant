from __future__ import annotations

import sqlite3
from pathlib import Path

from vision_engine import __version__
from vision_engine.persistence import BingoDB
from vision_engine.versioning import MODEL_VERSION, PROTOCOL_VERSION, SCHEMA_VERSION, STRATEGY_PLANNER_VERSION

ROOT = Path(__file__).parents[1]


def test_v18_final_contracts_are_exact():
    assert __version__ == "18.0.26-STATE-INTEGRITY-P1"
    assert PROTOCOL_VERSION == 4
    assert SCHEMA_VERSION == 9
    assert MODEL_VERSION == 6
    assert STRATEGY_PLANNER_VERSION == 22


def test_final_ui_is_v18_and_keeps_beginner_facing_tabs():
    source = (ROOT / "vision_engine" / "gui.py").read_text(encoding="utf-8")
    assert "Speedlora Jewel Bingo Assistant V18" in source
    for tab in ('text="Live"', 'text="Analysis"', 'text="Learning"', 'text="System"'):
        assert tab in source
    assert "TARGET CHANGED" in source
    assert "Win >1000 / Adaptive" in source
    assert "V7 persistent solver ready" not in source


def test_current_docs_exist_and_legacy_docs_are_archived():
    for name in ("README.md", "ARCHITECTURE.md", "RESEARCH.md", "CHANGELOG.md"):
        assert (ROOT / name).exists()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Speedlora Jewel Bingo Assistant V18" in readme
    assert "Live" in readme and "Learning" in readme and "System" in readme
    legacy = ROOT / "docs" / "archive" / "legacy"
    assert legacy.exists()
    assert any(legacy.rglob("CHANGELOG_V7_0_5.md"))
    assert not (ROOT / "docs" / "changelog").exists()


def test_schema7_to_v18_final_migration_creates_backup_and_telemetry(tmp_path: Path):
    path = tmp_path / "bingo.sqlite3"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
    con.execute("INSERT INTO metadata(key,value) VALUES('schema_version','7')")
    con.commit(); con.close()

    db = BingoDB(path)
    assert db.migration_backup_path is not None and db.migration_backup_path.exists()
    with db.connect() as con:
        version = int(con.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()[0])
        tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert version == 9
    assert "telemetry" in tables


def test_release_validator_covers_final_stabilization():
    source = (ROOT / "tools" / "validate_release.py").read_text(encoding="utf-8")
    for marker in (
        "V18 compatibility contracts",
        "false +1 rollback",
        "result-screen completion recovery",
        "migration/backup + telemetry + learning/export",
        "automatic locked-route planner",
        "automatic fallback",
        "V18.0.26-STATE-INTEGRITY-P1 FINAL offline validation OK",
    ):
        assert marker in source
