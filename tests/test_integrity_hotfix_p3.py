from __future__ import annotations

import json
import time
from pathlib import Path

from vision_engine.config import AppConfig
from vision_engine.diagnostics.runtime_journal import RuntimeJournal
from vision_engine.diagnostics.win32 import record_win32, set_win32_sink
from vision_engine.result_verification import computed_gt1000_verification

ROOT = Path(__file__).parents[1]


def test_auto_verify_requires_exact_14_and_gt1000():
    score = {"lucky": 3, "normal": 0, "jewel_cells": 2, "total": 1026}
    out = computed_gt1000_verification(score, selected_count=14, enabled=True, finalization_source="board-complete")
    assert out is not None
    assert out["source"] == "computed-auto-gt1000"
    assert out["total"] == 1026
    assert computed_gt1000_verification(score, selected_count=13, enabled=True, finalization_source="x") is None
    score_999 = dict(score, total=999)
    assert computed_gt1000_verification(score_999, selected_count=14, enabled=True, finalization_source="x") is None


def test_runtime_journal_writes_hourly_jsonl_without_blocking(tmp_path):
    journal = RuntimeJournal(tmp_path, "test-session", capacity=32)
    journal.record("alpha", value=1)
    journal.record("beta", level="WARN", value=2)
    time.sleep(0.08)
    journal.close(timeout=1.0)
    files = list(tmp_path.rglob("runtime_*.jsonl"))
    assert files
    rows = [json.loads(line) for line in files[0].read_text(encoding="utf-8").splitlines()]
    assert [r["event"] for r in rows[:2]] == ["alpha", "beta"]
    assert rows[0]["session_id"] == "test-session"


def test_config_defaults_defer_heavy_live_maintenance_and_enable_json_logs():
    cfg = AppConfig()
    assert cfg.auto_verify_computed_gt1000 is True
    assert cfg.continuous_json_logs is True
    assert cfg.maintenance_during_live is False


def test_repair_dialog_no_longer_exposes_manual_result_ocr_controls():
    src = (ROOT / "vision_engine" / "gui.py").read_text(encoding="utf-8")
    block = src[src.index("def _open_repair_dialog"):src.index("def _build", src.index("def _open_repair_dialog"))]
    assert 'text="Result OCR regions"' not in block
    assert 'text="Verify last result"' not in block
    assert 'text="Remaining-counter ROI"' not in block
    assert 'text="Refresh jewel refs"' in block


def test_post_game_maintenance_is_deferred_while_monitoring():
    src = (ROOT / "vision_engine" / "gui.py").read_text(encoding="utf-8")
    assert 'if self.running and not bool(self.app_cfg.maintenance_during_live):' in src
    assert '"maintenance-queued"' in src
    assert 'self.after(2000, self._flush_deferred_maintenance)' in src


def test_win32_events_can_mirror_to_async_journal_sink():
    seen = []
    set_win32_sink(lambda kind, data: seen.append((kind, data)))
    try:
        record_win32("move-test", step=3, distance_px=42.0)
    finally:
        set_win32_sink(None)
    assert seen == [("move-test", {"step": 3, "distance_px": 42.0})]
