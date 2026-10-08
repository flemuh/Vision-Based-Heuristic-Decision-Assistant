from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from vision_engine.icon_matcher import CurrentJewelMatcher
from vision_engine.vision import JewelClassifier, blue_selected_score


def test_selection_glow_uses_outer_ring_not_blue_jewel_center():
    # Natural cyan/blue jewel in the middle of a dark, *unselected* cell.
    center_blue = np.full((30, 30, 3), 18, dtype=np.uint8)
    cv2.circle(center_blue, (15, 15), 8, (255, 180, 40), -1)  # BGR cyan-blue
    assert blue_selected_score(center_blue) < 0.17

    # Selected cells in MU use a blue frame/background around the jewel.
    selected = np.full((30, 30, 3), 18, dtype=np.uint8)
    cv2.rectangle(selected, (0, 0), (29, 29), (255, 120, 20), 6)
    assert blue_selected_score(selected) > 0.17


def test_repo_board_seed_features_are_loadable(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    clf = JewelClassifier(tmp_path / "board_templates", repo / "assets" / "seed_board_features.npz")
    assert clf.model is not None
    assert clf.seed_examples == 120
    counts = {j: 0 for j in ("BL", "SO", "LI", "CR", "HA", "CH")}
    for y in clf.model._y:  # NumPy-only KNN internal training labels
        counts[str(y)] += 1
    assert set(counts.values()) == {20}


def _synthetic_icon(label_idx: int) -> np.ndarray:
    img = np.full((18, 18, 3), 20, dtype=np.uint8)
    colors = [
        (170, 80, 210), (80, 70, 240), (30, 130, 190),
        (135, 70, 245), (235, 220, 65), (20, 150, 245),
    ]
    cv2.circle(img, (9, 9), 5 + (label_idx % 2), colors[label_idx], -1)
    cv2.line(img, (3 + label_idx, 3), (14, 14 - label_idx), (245, 245, 245), 1)
    return img


def test_current_jewel_matcher_scans_dynamic_band(tmp_path: Path):
    labels = ("BL", "SO", "LI", "CR", "HA", "CH")
    matcher = CurrentJewelMatcher(tmp_path)
    for i, label in enumerate(labels):
        matcher.add_template(label, _synthetic_icon(i))

    panel = np.full((240, 240, 3), 24, dtype=np.uint8)
    board_roi = (45, 90, 130, 130)
    # The matcher's search band above this board spans around x=57..106,
    # y=42..87. Put CH there without relying on a persisted current ROI.
    icon = _synthetic_icon(5)
    panel[52:70, 66:84] = icon
    found = matcher.locate(panel, board_roi)
    assert found is not None
    assert found.label == "CH"
    assert found.score > 0.90
    assert found.margin > 0.03
