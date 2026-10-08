from __future__ import annotations

import json
from pathlib import Path

from vision_engine.persistence import BingoDB, SCHEMA_VERSION
from vision_engine.recovery import reconcile_masks
from vision_engine.replay import stored_replay
from vision_engine.research_export import export_research_dataset


def test_schema_v9_persists_latency_and_percentiles(tmp_path: Path):
    db = BingoDB(tmp_path / "b.sqlite3")
    assert SCHEMA_VERSION == 9
    for ms, hit in ((10,True),(20,True),(30,False),(40,True),(50,False)):
        db.save_telemetry(None, "decision-roundtrip", ms, hit)
    s = db.telemetry_summary("decision-roundtrip")
    assert s["count"] == 5 and s["p50"] == 30.0 and s["p95"] == 50.0
    assert 0 < s["cache_hit_rate"] < 1


def test_recovery_reconciles_forward_and_false_plus_one_but_rejects_crossed_masks():
    assert reconcile_masks(0b0011, 0b0111)[0]
    assert reconcile_masks(0b0111, 0b0011)[0]
    ok, _mask, _reason = reconcile_masks(0b0101, 0b0011)
    assert not ok


def test_stored_replay_and_research_export(tmp_path: Path):
    db = BingoDB(tmp_path / "b.sqlite3")
    with db.connect() as con:
        con.execute("INSERT INTO games(id,created_at,finished_at,source,status,computed_score_json) VALUES(?,?,?,?,?,?)",
                    ("g","x","y","local-auto","complete",json.dumps({"total":1026})))
        con.execute("INSERT INTO actions(game_id,step,mask_before,jewel,action_pos,recommended_pos,solver_regret) VALUES(?,?,?,?,?,?,?)",
                    ("g",0,0,"BL",0,0,0.0))
    r = stored_replay(db)
    assert r and r["followed"] == 1 and r["score"]["total"] == 1026
    out = export_research_dataset(db, tmp_path / "export")
    for name in ("episodes.csv","moves.csv","model_metrics.json","pattern_analysis.json","latency_metrics.json"):
        assert (out / name).exists()


def test_system_ui_has_health_recovery_replay_export_and_restart():
    source = Path(__file__).parents[1].joinpath("vision_engine","gui.py").read_text(encoding="utf-8")
    assert 'text="System"' in source
    for label in ("Recover current round","Game history / Replay","Export research dataset","Restart WSL solver"):
        assert label in source
    assert 'telemetry_summary("decision-roundtrip")' in source
