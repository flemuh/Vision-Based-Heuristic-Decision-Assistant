from __future__ import annotations

import json
from pathlib import Path

from vision_engine.board import Board
from vision_engine.data_home import prepare_user_data
from vision_engine.persistence import BingoDB, SCHEMA_VERSION
from vision_engine.round_lifecycle import CompletionDetector
from vision_engine.temporal import TemporalStateFilter


def test_false_plus_one_can_rollback_to_previous_confirmed_checkpoint():
    base = (1 << 1) | (1 << 3) | (1 << 7) | (1 << 9) | (1 << 10) | (1 << 12)
    false13 = base | (1 << 20)
    f = TemporalStateFilter(
        frames=3,
        required=2,
        max_mask_jump=1,
        catchup_required_frames=5,
        single_step_confirm_frames=4,
        single_step_rollback_frames=3,
        single_step_rollback_window_frames=10,
    )
    f.reset(base)

    states = [f.observe(false13, "SO", 0.95) for _ in range(4)]
    assert all(not s.stable for s in states[:-1])
    assert states[-1].stable and states[-1].mask == false13

    # Glow disappears and the real checkpoint reappears. The recent +1 is the
    # only kind of committed state allowed to roll back automatically.
    back = [f.observe(base, "SO", 0.95) for _ in range(3)]
    assert back[-1].stable
    assert back[-1].mask == base
    assert back[-1].reason.startswith("rollback-false-plus-one")
    assert back[-1].jewel == "SO"


def test_real_plus_one_commits_and_stays_after_rollback_window_expires():
    base = (1 << 2) | (1 << 5)
    moved = base | (1 << 8)
    f = TemporalStateFilter(
        frames=3, required=2, max_mask_jump=1,
        single_step_confirm_frames=4,
        single_step_rollback_frames=3,
        single_step_rollback_window_frames=6,
    )
    f.reset(base)
    for _ in range(4):
        state = f.observe(moved, "HA", 0.9)
    assert state.stable and state.mask == moved
    # Keep seeing the real committed state until the rollback window is gone.
    for _ in range(7):
        f.observe(moved, "CR", 0.9)
    one_bad = f.observe(base, "CR", 0.9)
    assert f.accepted_mask == moved
    assert one_bad.mask == moved


def test_completion_detector_recovers_recent_raw_14_when_reward_screen_replaces_board():
    d = CompletionDetector(min_accepted=12, disappear_required=2, recent_14_window_frames=10)
    accepted13 = sum(1 << i for i in range(13))
    raw14 = accepted13 | (1 << 14)
    first = d.observe(accepted_mask=accepted13, raw_mask=raw14, jewel_visible=True)
    assert not first.confirmed
    a = d.observe(accepted_mask=accepted13, raw_mask=0, jewel_visible=False)
    b = d.observe(accepted_mask=accepted13, raw_mask=0, jewel_visible=False)
    assert not a.confirmed
    assert b.confirmed
    assert b.reason == "recent-14-then-result-screen"
    assert b.final_mask == raw14


def test_completion_detector_does_not_treat_random_board_disappearance_as_complete():
    d = CompletionDetector(min_accepted=12, disappear_required=2, recent_14_window_frames=10)
    accepted12 = sum(1 << i for i in range(12))
    for _ in range(4):
        e = d.observe(accepted_mask=accepted12, raw_mask=0, jewel_visible=False)
    assert not e.confirmed


def test_schema_v7_checkpoints_and_idempotent_database_merge(tmp_path):
    b = Board.from_rows([
        ["BL", "CR", "CH", "HA", "SO"],
        ["LI", "BL", "CR", "CH", "HA"],
        ["SO", "LI", None, "BL", "CR"],
        ["CH", "HA", "SO", "LI", "BL"],
        ["CR", "CH", "HA", "SO", "LI"],
    ])
    old_path = tmp_path / "old.sqlite3"
    old = BingoDB(old_path)
    old.ensure_game("old-g1", board=b)
    old.save_checkpoint("old-g1", 3, 7, 15, "playing", "SO", "adaptive", "3 Lucky", {"x": 1})
    old.finalize_game("old-g1", b, list(b.jewel_counts_in_mask((1 << 14) - 1)), [], (1 << 14) - 1,
                      {"lucky": 0, "normal": 0, "jewel": 630, "total": 630}, None, None)

    new = BingoDB(tmp_path / "new.sqlite3")
    assert SCHEMA_VERSION == 9
    assert new.merge_database(old_path)
    assert not new.merge_database(old_path)  # migration_sources makes it idempotent
    rows = [r for r in new.game_rows() if r["id"] == "old-g1"]
    assert len(rows) == 1
    assert rows[0]["status"] == "complete"
    cp = new.latest_checkpoint("old-g1")
    assert cp is not None and cp["confirmed_mask"] == 7


def test_persistent_data_home_uses_newest_previous_version_and_survives_new_zip(tmp_path, monkeypatch):
    parent = tmp_path / "Downloads"
    old = parent / "speedlora-jewel-bingo-assistant-v7.0.2"
    cur = parent / "speedlora-jewel-bingo-assistant-v7.0.3"
    (old / "data" / "icon_templates").mkdir(parents=True)
    (cur / "data").mkdir(parents=True)
    (old / "data" / "vision_config.json").write_text('{"user_calibration": true}', encoding="utf-8")
    (old / "data" / "history.json").write_text('[{"episode":"saved"}]', encoding="utf-8")
    (cur / "data" / "vision_config.json").write_text('{"user_calibration": false}', encoding="utf-8")
    (cur / "data" / "app_config.json").write_text('{}', encoding="utf-8")

    data_home = tmp_path / "persistent"
    monkeypatch.setenv("SPEEDLORA_JEWEL_DATA_DIR", str(data_home))
    # P5 disables hidden sibling migration by default. This legacy behavior is
    # still available only through an explicit recovery opt-in.
    monkeypatch.setenv("SPEEDLORA_JEWEL_ALLOW_LEGACY_MIGRATION", "1")
    resolved, _dbs = prepare_user_data(cur)
    assert resolved == data_home
    assert json.loads((data_home / "vision_config.json").read_text())["user_calibration"] is True
    assert json.loads((data_home / "history.json").read_text())[0]["episode"] == "saved"
    # Current release fills missing defaults without overwriting calibration.
    assert (data_home / "app_config.json").exists()
