from __future__ import annotations

import cv2
import numpy as np

from vision_engine.autocalibration import detect_board_grid, detect_board_from_mu_anchors, infer_current_jewel_roi


def synthetic_board() -> tuple[np.ndarray, tuple[int, int, int, int]]:
    img = np.full((260, 280, 3), 150, dtype=np.uint8)
    # Dark UI panel.
    cv2.rectangle(img, (28, 18), (244, 238), (32, 28, 25), -1)
    x0, y0, cell, gap = 52, 80, 22, 4
    pitch = cell + gap
    for r in range(5):
        for c in range(5):
            x = x0 + c * pitch
            y = y0 + r * pitch
            cv2.rectangle(img, (x, y), (x + cell, y + cell), (8, 8, 8), -1)
            cv2.rectangle(img, (x, y), (x + cell, y + cell), (70, 65, 58), 1)
    # MU center anchor: saturated blue with bright text-like strokes.
    cx, cy = x0 + 2 * pitch, y0 + 2 * pitch
    cv2.rectangle(img, (cx + 2, cy + 2), (cx + cell - 2, cy + cell - 2), (220, 100, 20), -1)
    cv2.putText(img, "MU", (cx + 2, cy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (240, 240, 240), 1, cv2.LINE_AA)
    expected = (x0 - gap // 2, y0 - gap // 2, 5 * pitch, 5 * pitch)
    return img, expected


def test_detect_board_grid_from_geometry_and_mu():
    img, expected = synthetic_board()
    det = detect_board_grid(img)
    assert det is not None
    x, y, w, h = det.roi
    ex, ey, ew, eh = expected
    assert abs(x - ex) <= 8
    assert abs(y - ey) <= 8
    assert abs(w - ew) <= 12
    assert abs(h - eh) <= 12
    assert det.coverage >= 0.80
    assert det.mu_anchor_score >= 0.20


def test_current_jewel_roi_is_above_board_and_small():
    board = (40, 90, 130, 130)
    x, y, w, h = infer_current_jewel_roi(board, (260, 260, 3))
    assert y < board[1]
    assert 12 <= w <= 30
    assert w == h
    assert x >= 0 and y >= 0


def test_fast_mu_anchor_search_on_large_desktop():
    panel, expected = synthetic_board()
    desktop = np.full((900, 1600, 3), 185, dtype=np.uint8)
    ox, oy = 310, 170
    desktop[oy:oy+panel.shape[0], ox:ox+panel.shape[1]] = panel
    det = detect_board_from_mu_anchors(desktop)
    assert det is not None
    ex, ey, ew, eh = expected
    x, y, w, h = det.roi
    assert abs(x - (ox + ex)) <= 12
    assert abs(y - (oy + ey)) <= 12
    assert abs(w - ew) <= 18
    assert abs(h - eh) <= 18
    assert det.mu_anchor_score >= 0.20
