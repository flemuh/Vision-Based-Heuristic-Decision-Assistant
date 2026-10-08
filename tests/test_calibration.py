import cv2
import numpy as np

from vision_engine.calibration import (
    detect_reference_jewels,
    validate_board_roi,
    validate_game_panel,
)


def _panel():
    img = np.full((290, 260, 3), 28, np.uint8)
    # Header/borders keep the crop panel-like.
    cv2.rectangle(img, (4, 4), (255, 285), (55, 55, 55), 2)
    bx, by, bw, bh = 35, 90, 130, 130
    cv2.rectangle(img, (bx, by), (bx+bw, by+bh), (90, 90, 90), 1)
    for k in range(1, 5):
        x = bx + round(k*bw/5)
        y = by + round(k*bh/5)
        cv2.line(img, (x, by), (x, by+bh), (105,105,105), 2)
        cv2.line(img, (bx, y), (bx+bw, y), (105,105,105), 2)
    # Six saturated reference icons in fixed vertical order.
    colors = [
        (180, 80, 220),  # purple
        (90, 70, 240),   # pink/red
        (40, 130, 190),  # brown/gold
        (130, 80, 240),  # pink
        (240, 220, 70),  # cyan
        (20, 150, 245),  # orange
    ]
    ys = [114, 136, 158, 180, 202, 224]
    for y, color in zip(ys, colors):
        cv2.circle(img, (205, y), 7, color, -1)
    return img, (bx, by, bw, bh)


def test_board_geometry_rejects_whole_panel_and_accepts_grid():
    panel, roi = _panel()
    assert validate_game_panel(panel).ok
    assert validate_board_roi(panel, roi).ok
    bad = validate_board_roi(panel, (0, 0, panel.shape[1], panel.shape[0]))
    assert not bad.ok


def test_reference_jewel_auto_detection_finds_six():
    panel, roi = _panel()
    found = detect_reference_jewels(panel, roi)
    assert found is not None
    templates, confidence = found
    assert list(templates) == ["BL", "SO", "LI", "CR", "HA", "CH"]
    assert confidence > 0.5
    assert all(p.size > 0 for p in templates.values())


def test_quality_bands_are_not_binary_pass_fail():
    from vision_engine.calibration import quality_band
    assert quality_band(0.90) == "HIGH"
    assert quality_band(0.70) == "MEDIUM"
    assert quality_band(0.57) == "LOW"
    assert quality_band(0.49) == "REJECT"


def test_board_check_exposes_explained_subscores():
    panel, roi = _panel()
    check = validate_board_roi(panel, roi)
    for key in ("score_shape", "score_size", "score_location", "score_cell_size", "score_grid"):
        assert key in check.metrics
        assert 0.0 <= check.metrics[key] <= 1.0


def test_number_roi_has_explained_ocr_metrics():
    from vision_engine.calibration import validate_number_roi
    panel = np.full((200, 240, 3), 25, np.uint8)
    cv2.putText(panel, "1026", (120, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)
    check = validate_number_roi(panel, (115, 65, 60, 32), "total score")
    assert check.ok
    assert "score_contrast" in check.metrics
    assert "score_digit_edges" in check.metrics
