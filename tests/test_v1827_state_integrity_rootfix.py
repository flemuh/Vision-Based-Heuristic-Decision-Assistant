from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from vision_engine.icon_matcher import CurrentJewelMatcher
from vision_engine.input.geometry import AutoPlayPoint
from vision_engine.input.win32 import _cursor_within_tolerance
from vision_engine.round_lifecycle import CompletionDetector
from vision_engine.state_machine import GamePhase, GameStateMachine
from vision_engine.temporal import TemporalStateFilter


def _mask(n: int) -> int:
    out = 0
    for i in range(n):
        out |= 1 << i
    return out


def test_temporal_rejects_physically_impossible_mask_without_poisoning_history():
    f = TemporalStateFilter(frames=3, required=2)
    f.reset(0)
    s = f.observe(_mask(15), None)
    assert not s.stable
    assert s.reason == "invalid-mask-count:15/14"
    assert f.accepted_mask == 0
    # A correct frame after the impossible one must not be treated as regression.
    f.observe(0, None)
    s2 = f.observe(0, None)
    assert s2.stable
    assert s2.mask == 0


def test_temporal_reset_fails_closed_on_impossible_bootstrap():
    f = TemporalStateFilter()
    f.reset(_mask(16))
    assert f.accepted_mask == 0


def test_state_machine_never_calls_15_of_14_complete():
    sm = GameStateMachine()
    state = sm.update(has_board=True, selected_count=15, current_jewel=None)
    assert state.phase == GamePhase.DESYNC


def test_completion_detector_remembers_exactly_14_not_more():
    d = CompletionDetector(min_accepted=12)
    d.observe(accepted_mask=_mask(13), raw_mask=_mask(15), jewel_visible=True)
    assert d.recent_full_mask is None
    d.observe(accepted_mask=_mask(13), raw_mask=_mask(14), jewel_visible=True)
    assert d.recent_full_mask == _mask(14)


def test_cursor_tolerance_accepts_small_real_position_error_but_not_hard_block():
    target = AutoPlayPoint(100, 100)
    assert _cursor_within_tolerance(AutoPlayPoint(103, 104), target, 5)
    assert not _cursor_within_tolerance(AutoPlayPoint(160, 100), target, 5)


def test_cross_label_near_duplicate_is_removed_from_live_matching(tmp_path: Path):
    # Identical visual under two labels is the historical contamination mode.
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    cv2.circle(img, (16, 16), 8, (100, 200, 240), -1)
    cv2.imwrite(str(tmp_path / "BL_0001.png"), img)
    cv2.imwrite(str(tmp_path / "SO_0001.png"), img)
    # Give every other label one distinct reference so only BL/SO conflict.
    for idx, label in enumerate(("LI", "CR", "HA", "CH"), start=1):
        other = np.zeros((32, 32, 3), dtype=np.uint8)
        cv2.rectangle(other, (idx, idx), (10 + idx, 20 + idx), (20 * idx, 30, 200 - 20 * idx), -1)
        cv2.imwrite(str(tmp_path / f"{label}_0001.png"), other)
    m = CurrentJewelMatcher(tmp_path)
    assert m.conflict_count() >= 1
    assert len(m.usable_templates["BL"]) == 0
    assert len(m.usable_templates["SO"]) == 0
    assert not m.ready()


def test_new_cross_label_duplicate_is_not_learned(tmp_path: Path):
    base = np.zeros((32, 32, 3), dtype=np.uint8)
    cv2.line(base, (2, 2), (29, 29), (255, 255, 255), 3)
    cv2.imwrite(str(tmp_path / "BL_0001.png"), base)
    m = CurrentJewelMatcher(tmp_path)
    assert m.add_template_if_novel("SO", base.copy(), min_mse=0.001) is None
    assert not (tmp_path / "SO_0001.png").exists()


def test_clean_profile_skips_sibling_migration(tmp_path: Path, monkeypatch):
    from vision_engine.data_home import prepare_user_data

    parent = tmp_path / "Downloads"
    old = parent / "speedlora-jewel-bingo-assistant-v18.0.25-old"
    cur = parent / "speedlora-jewel-bingo-assistant-v18.0.26-rootfix"
    (old / "data").mkdir(parents=True)
    (cur / "data").mkdir(parents=True)
    (old / "data" / "vision_config.json").write_text('{"old": true}', encoding="utf-8")
    (cur / "data" / "vision_config.json").write_text('{"clean": true}', encoding="utf-8")
    (cur / "data" / "app_config.json").write_text('{}', encoding="utf-8")

    user = tmp_path / "clean-user"
    monkeypatch.setenv("SPEEDLORA_JEWEL_DATA_DIR", str(user))
    monkeypatch.setenv("SPEEDLORA_JEWEL_CLEAN_PROFILE", "1")
    resolved, dbs = prepare_user_data(cur)
    assert resolved == user
    assert dbs == []
    assert '"clean": true' in (user / "vision_config.json").read_text(encoding="utf-8")
