from __future__ import annotations

import json
from pathlib import Path

from vision_engine.experience import PolicyModelInfo
from vision_engine.learning_status import build_learning_snapshot
from vision_engine.persistence import BingoDB


def test_learning_snapshot_uses_clear_50_100_game_stages(tmp_path: Path):
    db = BingoDB(tmp_path / "b.sqlite3")
    info = PolicyModelInfo(counterfactual_samples=100, validation_games=3)
    snap = build_learning_snapshot(db, info, min_validation_games=8, rng_min_games=50)
    assert snap["pattern_stage"] == "COLLECTING"
    assert snap["cf_status"] == "LEARNING"
    assert snap["rng_influence_enabled"] is False


def test_rng_influence_stays_disabled_when_blend_is_zero_even_after_100_games(tmp_path: Path):
    db = BingoDB(tmp_path / "b.sqlite3")
    with db.connect() as con:
        for i in range(100):
            counts = [2,2,2,2,3,3]
            con.execute(
                "INSERT INTO games(id,created_at,finished_at,source,status,counts_json) VALUES(?,?,?,?,?,?)",
                (f"g{i}", "x", "y", "local-auto", "complete", json.dumps(counts)),
            )
    snap = build_learning_snapshot(db, PolicyModelInfo(), rng_min_games=50, rng_blend=0.0)
    assert snap["pattern_stage"] == "ELIGIBLE FOR VALIDATION"
    assert snap["pattern_games"] == 100
    assert snap["rng_influence_enabled"] is False


def test_learning_ui_explains_math_control_and_research_only_rng():
    source = Path(__file__).parents[1].joinpath("vision_engine", "gui.py").read_text(encoding="utf-8")
    assert 'text="Learning"' in source
    assert "The math solver stays in control" in source
    assert "DISABLED (research only)" in source
    assert "validation {snap['cf_validation_games']}/{snap['required_validation_games']}" in source
