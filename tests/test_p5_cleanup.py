from __future__ import annotations

import os
from pathlib import Path
import time

from vision_engine.data_home import prepare_user_data
from vision_engine.diagnostics.runtime_journal import RuntimeJournal


def test_default_data_home_does_not_scan_or_copy_sibling_versions(tmp_path: Path, monkeypatch):
    parent = tmp_path / "Downloads"
    old = parent / "speedlora-jewel-bingo-assistant-v18.0.25-old"
    cur = parent / "speedlora-jewel-bingo-assistant-v18.0.26-current"
    (old / "data").mkdir(parents=True)
    (cur / "data").mkdir(parents=True)
    (old / "data" / "vision_config.json").write_text('{"old": true}', encoding="utf-8")
    (cur / "data" / "vision_config.json").write_text('{"current": true}', encoding="utf-8")
    user = tmp_path / "profile"
    monkeypatch.setenv("SPEEDLORA_JEWEL_DATA_DIR", str(user))
    monkeypatch.delenv("SPEEDLORA_JEWEL_ALLOW_LEGACY_MIGRATION", raising=False)
    monkeypatch.delenv("SPEEDLORA_JEWEL_CLEAN_PROFILE", raising=False)

    resolved, dbs = prepare_user_data(cur)
    assert resolved == user
    assert dbs == []
    assert '"current": true' in (user / "vision_config.json").read_text(encoding="utf-8")
    assert '"old": true' not in (user / "vision_config.json").read_text(encoding="utf-8")


def test_runtime_journal_age_retention_runs_off_hot_path(tmp_path: Path):
    old_day = tmp_path / "2026-01-01"
    old_day.mkdir(parents=True)
    old_file = old_day / "runtime_20260101_01.jsonl"
    old_file.write_text('{"event":"old"}\n', encoding="utf-8")
    old = time.time() - 10 * 86400
    os.utime(old_file, (old, old))

    journal = RuntimeJournal(tmp_path, "p5-test", keep_days=1, max_total_mb=16)
    journal.record("hello")
    journal.close(timeout=2.0)
    assert not old_file.exists()
    assert list(tmp_path.rglob("runtime_*.jsonl"))


def test_live_gui_no_longer_calls_result_or_remaining_ocr():
    root = Path(__file__).resolve().parents[1]
    src = (root / "vision_engine" / "gui.py").read_text(encoding="utf-8")
    assert "read_result(" not in src
    assert "read_int_roi(" not in src
    assert "ResultConsensus(" not in src
    assert "def calibrate_result_rois" not in src
    assert "def calibrate_remaining_roi" not in src


def test_runtime_requirements_do_not_install_test_or_ocr_stack():
    root = Path(__file__).resolve().parents[1]
    runtime = (root / "requirements.txt").read_text(encoding="utf-8").lower()
    dev = (root / "requirements-dev.txt").read_text(encoding="utf-8").lower()
    assert "pytesseract" not in runtime
    assert "pytest" not in runtime
    assert "pytest" in dev
    assert "pytesseract" in dev


def test_physical_mouse_call_is_outside_live_state_lock_block():
    root = Path(__file__).resolve().parents[1]
    src = (root / "vision_engine" / "gui.py").read_text(encoding="utf-8")
    fn = src[src.index("    def _execute_autoplay_click"):src.index("    def _autoplay_game_completed", src.index("    def _execute_autoplay_click"))]
    marker = "with self._live_state_lock:"
    move = "input_backend = self.input_executor.move_and_left_click("
    i_lock = fn.index(marker)
    i_move = fn.index(move)
    assert i_lock < i_move
    # The move call is dedented to the same level as the lock statement, so the
    # lock protects state validation but not the physical 220-380 ms movement.
    move_line = next(line for line in fn.splitlines() if move in line)
    lock_line = next(line for line in fn.splitlines() if marker in line)
    assert len(move_line) - len(move_line.lstrip()) == len(lock_line) - len(lock_line.lstrip())
