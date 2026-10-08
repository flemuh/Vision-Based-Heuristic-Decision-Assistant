import json
from pathlib import Path

from vision_engine.config import AppConfig
from vision_engine.persistence import BingoDB


def test_legacy_balanced_profile_migrates_to_auto(tmp_path: Path):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"risk_profile": "balanced", "mode": "adaptive"}), encoding="utf-8")
    cfg = AppConfig.load(p)
    assert cfg.risk_profile == "auto"
    assert cfg.mode == "auto"


def test_three_strategy_modes_are_explicit():
    cfg = AppConfig()
    for name, mode in (("adaptive", "adaptive"), ("score", "score"), ("lucky", "lucky")):
        cfg.set_risk_profile(name)
        assert cfg.risk_profile == name
        assert cfg.mode == mode


def test_action_persists_strategy(tmp_path: Path):
    db = BingoDB(tmp_path / "bingo.sqlite3")
    db.ensure_game("g1")
    db.save_action("g1", 0, 0, "BL", 1, strategy="score")
    with db.connect() as con:
        row = con.execute("SELECT strategy FROM actions WHERE game_id='g1' AND step=0").fetchone()
    assert row["strategy"] == "score"
