from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import cv2
import numpy as np

from .constants import JEWELS


@dataclass(frozen=True)
class CalibrationCheck:
    ok: bool
    score: float
    messages: tuple[str, ...]
    metrics: dict[str, float]

    @property
    def band(self) -> str:
        return quality_band(self.score)

    @property
    def automatic(self) -> bool:
        return self.ok and self.score >= 0.80

    @property
    def reviewable(self) -> bool:
        return self.ok and self.score >= 0.50


@dataclass(frozen=True)
class ReferenceDetection:
    templates: dict[str, np.ndarray]
    confidence: float
    metrics: dict[str, float]

    @property
    def band(self) -> str:
        return quality_band(self.confidence)


def quality_band(score: float) -> str:
    """Human-facing confidence bands used consistently across setup.

    HIGH   >= .80: automatic validation is trustworthy.
    MEDIUM >= .65: geometry is plausible; user review is required.
    LOW    >= .50: weak detector confidence; explicit visual confirmation required.
    REJECT < .50: do not persist automatically.
    """
    score = float(score)
    if score >= 0.80:
        return "HIGH"
    if score >= 0.65:
        return "MEDIUM"
    if score >= 0.50:
        return "LOW"
    return "REJECT"


def component_band(score: float) -> str:
    score = float(score)
    if score >= 0.80:
        return "GOOD"
    if score >= 0.60:
        return "OK"
    if score >= 0.40:
        return "WEAK"
    return "BAD"


def _inside(img: np.ndarray, roi: tuple[int, int, int, int]) -> bool:
    x, y, w, h = roi
    ih, iw = img.shape[:2]
    return x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= iw and y + h <= ih


def _crop(img: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = roi
    return img[y:y+h, x:x+w]


def _triangle_score(value: float, ideal: float, half_width: float) -> float:
    return float(np.clip(1.0 - abs(float(value) - ideal) / max(half_width, 1e-6), 0.0, 1.0))


def _range_score(value: float, good_lo: float, good_hi: float, outer_lo: float, outer_hi: float) -> float:
    v = float(value)
    if good_lo <= v <= good_hi:
        return 1.0
    if outer_lo <= v < good_lo:
        return float((v - outer_lo) / max(good_lo - outer_lo, 1e-6))
    if good_hi < v <= outer_hi:
        return float((outer_hi - v) / max(outer_hi - good_hi, 1e-6))
    return 0.0


def validate_game_panel(panel: np.ndarray) -> CalibrationCheck:
    """Validate the outer Jewel Bingo working panel.

    This check intentionally separates hard mistakes (Event Inventory / huge
    desktop crop / tiny crop) from soft confidence. A visually correct crop can
    therefore be reviewed instead of being silently accepted or rejected.
    """
    if panel is None or panel.size == 0:
        return CalibrationCheck(False, 0.0, ("The selected area is empty.",), {})
    h, w = panel.shape[:2]
    aspect = w / max(1.0, float(h))
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    dark_fraction = float((gray < 95).mean())

    aspect_score = _range_score(aspect, 0.74, 1.10, 0.60, 1.35)
    # More dark UI is not a problem; the dangerous case is a crop dominated by
    # bright game-world/desktop pixels.
    darkness_score = float(np.clip((dark_fraction - 0.25) / 0.42, 0.0, 1.0))
    size_score = float(np.clip(min(w / 210.0, h / 230.0), 0.0, 1.0))
    score = float(0.40 * aspect_score + 0.35 * darkness_score + 0.25 * size_score)

    msgs: list[str] = []
    if w < 150 or h < 170:
        msgs.append("The selected panel is too small; include the complete Jewel Bingo panel.")
    if not (0.66 <= aspect <= 1.28):
        if aspect < 0.66:
            msgs.append("The selection is too tall. Stop before Event Inventory; do not include the inventory/boxes below.")
        else:
            msgs.append("The selection is too wide. Select only the Jewel Bingo panel, not surrounding game UI.")
    if dark_fraction < 0.35:
        msgs.append("The selection does not look like the dark Jewel Bingo UI panel.")

    return CalibrationCheck(
        not msgs,
        score,
        tuple(msgs),
        {
            "aspect": aspect,
            "dark_fraction": dark_fraction,
            "width_px": float(w),
            "height_px": float(h),
            "score_shape": aspect_score,
            "score_dark_ui": darkness_score,
            "score_size": size_score,
        },
    )


def _grid_periodicity(board_img: np.ndarray) -> tuple[float, float, float]:
    """Legacy edge-energy diagnostic retained as one component of board score."""
    if board_img is None or board_img.size == 0:
        return 0.0, 0.0, 0.0
    gray = cv2.cvtColor(board_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 28, 95).astype(np.float32) / 255.0
    h, w = edges.shape
    boundary_vals: list[float] = []
    interior_vals: list[float] = []
    for frac in (0.2, 0.4, 0.6, 0.8):
        x = int(round(frac * w))
        y = int(round(frac * h))
        xb = edges[:, max(0, x-1):min(w, x+2)]
        yb = edges[max(0, y-1):min(h, y+2), :]
        boundary_vals.extend([float(xb.mean()) if xb.size else 0.0, float(yb.mean()) if yb.size else 0.0])

        dx = max(2, int(round(0.075 * w)))
        dy = max(2, int(round(0.075 * h)))
        for xx in (x-dx, x+dx):
            z = edges[:, max(0, xx-1):min(w, xx+2)]
            interior_vals.append(float(z.mean()) if z.size else 0.0)
        for yy in (y-dy, y+dy):
            z = edges[max(0, yy-1):min(h, yy+2), :]
            interior_vals.append(float(z.mean()) if z.size else 0.0)
    boundary = float(np.mean(boundary_vals)) if boundary_vals else 0.0
    interior = float(np.mean(interior_vals)) if interior_vals else 0.0
    delta = boundary - interior
    return boundary, interior, delta


def _projection_grid_score(board_img: np.ndarray) -> tuple[float, float, float]:
    """Score roughly five repeated cells using dark-border projection spacing.

    MU's board borders are often dark, while selected jewels add bright blue
    edges. Using low-intensity projections is more stable than assuming every
    cell border is a bright Canny edge exactly at 20/40/60/80 percent.
    """
    if board_img is None or board_img.size == 0:
        return 0.0, 0.0, 0.0
    gray = cv2.cvtColor(board_img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    def axis_score(proj: np.ndarray) -> tuple[float, float]:
        n = len(proj)
        if n < 40:
            return 0.0, 0.0
        # Lower values correspond to the dark separator columns/rows.
        smooth = np.convolve(proj, np.ones(3, dtype=np.float32) / 3.0, mode="same")
        target = n / 5.0
        best_score = 0.0
        best_period = target
        for period in np.linspace(target * 0.82, target * 1.12, 25):
            # Search phase only in the first period. Six lines define five cells.
            for phase in np.linspace(0.0, period, 20, endpoint=False):
                pts = [int(round(phase + k * period)) for k in range(6)]
                pts = [p for p in pts if 0 <= p < n]
                if len(pts) < 5:
                    continue
                vals = np.asarray([smooth[p] for p in pts], dtype=np.float32)
                # Compare candidate border darkness to general projection level.
                med = float(np.median(smooth))
                spread = float(np.percentile(smooth, 80) - np.percentile(smooth, 10)) + 1e-6
                darkness = float(np.clip((med - float(vals.mean())) / spread * 1.8 + 0.45, 0.0, 1.0))
                coverage = min(1.0, len(pts) / 6.0)
                score = darkness * coverage
                if score > best_score:
                    best_score = score
                    best_period = period
        period_error = abs(best_period - target) / max(target, 1e-6)
        regularity = float(np.clip(1.0 - period_error / 0.18, 0.0, 1.0))
        return best_score, regularity

    sx, rx = axis_score(gray.mean(axis=0))
    sy, ry = axis_score(gray.mean(axis=1))
    return float((sx + sy) / 2.0), float((rx + ry) / 2.0), float(min(sx, sy))


def validate_board_roi(panel: np.ndarray, roi: tuple[int, int, int, int]) -> CalibrationCheck:
    if panel is None or panel.size == 0 or not _inside(panel, roi):
        return CalibrationCheck(False, 0.0, ("Board selection is outside the Jewel Bingo panel.",), {})
    x, y, w, h = roi
    ph, pw = panel.shape[:2]
    aspect = w / max(1.0, float(h))
    rw, rh = w / max(1.0, float(pw)), h / max(1.0, float(ph))
    cx, cy = (x + w / 2) / max(1.0, pw), (y + h / 2) / max(1.0, ph)
    cell = min(w, h) / 5.0
    board = _crop(panel, roi)
    boundary, interior, delta = _grid_periodicity(board)
    proj_score, period_regularity, weakest_axis = _projection_grid_score(board)

    shape_score = _range_score(aspect, 0.92, 1.08, 0.80, 1.22)
    size_w = _range_score(rw, 0.40, 0.58, 0.32, 0.72)
    size_h = _range_score(rh, 0.37, 0.56, 0.28, 0.69)
    size_score = (size_w + size_h) / 2.0
    location_x = _range_score(cx, 0.22, 0.48, 0.08, 0.65)
    location_y = _range_score(cy, 0.37, 0.59, 0.20, 0.76)
    location_score = (location_x + location_y) / 2.0
    cell_score = _range_score(cell, 17.0, 40.0, 10.0, 65.0)

    # Edge energy can be depressed by blue selected-cell glow, so combine it
    # with the dark-border periodicity detector instead of trusting it alone.
    legacy_grid = float(np.clip((delta - 0.005) / 0.10, 0.0, 1.0))
    # Dark-border projection is the most stable signal on real MU screenshots;
    # selected blue glow often weakens the legacy bright-edge score.
    grid_score = float(np.clip(0.15 * legacy_grid + 0.70 * proj_score + 0.15 * period_regularity, 0.0, 1.0))
    score = float(
        0.20 * shape_score
        + 0.15 * size_score
        + 0.12 * location_score
        + 0.08 * cell_score
        + 0.45 * grid_score
    )

    msgs: list[str] = []
    if not (0.80 <= aspect <= 1.22):
        msgs.append("The board crop must be almost square. Select only the 5x5 cells.")
    if not (0.32 <= rw <= 0.72 and 0.28 <= rh <= 0.69):
        msgs.append("The board crop is too large/small relative to the Jewel Bingo panel. Do not include title, counters or bottom text.")
    if not (0.08 <= cx <= 0.65 and 0.20 <= cy <= 0.76):
        msgs.append("The board should be in the left/center part of the Jewel Bingo panel; this selection looks displaced.")
    if cell < 10:
        msgs.append("Board cells are too small for reliable vision at this capture size.")
    # Only make periodicity a hard failure when BOTH detectors are very weak.
    if grid_score < 0.23 and (boundary < 0.10 or delta < 0.015) and weakest_axis < 0.28:
        msgs.append("The selected area does not look sufficiently periodic for a 5x5 grid. Tighten the crop around the 25 cells.")

    return CalibrationCheck(
        not msgs,
        score,
        tuple(msgs),
        {
            "aspect": aspect,
            "width_fraction": rw,
            "height_fraction": rh,
            "center_x_fraction": cx,
            "center_y_fraction": cy,
            "cell_px": cell,
            "grid_boundary_energy": boundary,
            "grid_interior_energy": interior,
            "grid_delta": delta,
            "projection_grid_score": proj_score,
            "period_regularity": period_regularity,
            "weakest_grid_axis": weakest_axis,
            "score_shape": shape_score,
            "score_size": size_score,
            "score_location": location_score,
            "score_cell_size": cell_score,
            "score_grid": grid_score,
        },
    )


def validate_current_roi(panel: np.ndarray, roi: tuple[int, int, int, int]) -> CalibrationCheck:
    if panel is None or panel.size == 0 or not _inside(panel, roi):
        return CalibrationCheck(False, 0.0, ("Current-jewel selection is outside the Jewel Bingo panel.",), {})
    x, y, w, h = roi
    ph, pw = panel.shape[:2]
    aspect = w / max(1.0, float(h))
    rw, rh = w / max(1.0, pw), h / max(1.0, ph)
    cy = (y + h / 2) / max(1.0, ph)
    patch = _crop(panel, roi)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    colorful = float(((hsv[..., 1] > 55) & (hsv[..., 2] > 55)).mean())
    contrast = float(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).std())

    shape_score = _range_score(aspect, 0.78, 1.28, 0.50, 1.90)
    size_score = (_range_score(rw, 0.045, 0.13, 0.02, 0.24) + _range_score(rh, 0.045, 0.13, 0.02, 0.24)) / 2.0
    location_score = _range_score(cy, 0.08, 0.25, 0.01, 0.40)
    contrast_score = float(np.clip((contrast - 5.0) / 30.0, 0.0, 1.0))
    color_score = float(np.clip((colorful - 0.01) / 0.22, 0.0, 1.0))
    score = float(0.20 * shape_score + 0.20 * size_score + 0.15 * location_score + 0.25 * contrast_score + 0.20 * color_score)

    msgs: list[str] = []
    if not (0.50 <= aspect <= 1.90):
        msgs.append("Current-jewel ROI should tightly surround one icon and be roughly square.")
    if not (0.02 <= rw <= 0.24 and 0.02 <= rh <= 0.24):
        msgs.append("Current-jewel ROI is too large/small. Select only the jewel icon, not the box icon or remaining number.")
    if cy > 0.40:
        msgs.append("Current jewel normally appears near the top of the panel; this selection is too low.")
    if contrast < 8.0 or colorful < 0.025:
        msgs.append("This area looks blank/low-detail. Start a box so the current jewel is visible, then select only its icon.")

    return CalibrationCheck(
        not msgs,
        score,
        tuple(msgs),
        {
            "aspect": aspect,
            "width_fraction": rw,
            "height_fraction": rh,
            "center_y_fraction": cy,
            "colorful_fraction": colorful,
            "contrast": contrast,
            "score_shape": shape_score,
            "score_size": size_score,
            "score_location": location_score,
            "score_contrast": contrast_score,
            "score_color": color_score,
        },
    )


def validate_number_roi(panel: np.ndarray, roi: tuple[int, int, int, int], role: str = "number") -> CalibrationCheck:
    """Validate optional OCR number areas (remaining/result values)."""
    if panel is None or panel.size == 0 or not _inside(panel, roi):
        return CalibrationCheck(False, 0.0, (f"The {role} selection is outside the panel.",), {})
    x, y, w, h = roi
    ph, pw = panel.shape[:2]
    rw, rh = w / max(1.0, pw), h / max(1.0, ph)
    patch = _crop(panel, roi)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    contrast = float(gray.std())
    edge_fraction = float((cv2.Canny(gray, 35, 110) > 0).mean())
    size_score = (_range_score(rw, 0.025, 0.22, 0.01, 0.35) + _range_score(rh, 0.018, 0.12, 0.008, 0.22)) / 2.0
    contrast_score = float(np.clip((contrast - 3.0) / 28.0, 0.0, 1.0))
    edge_score = _range_score(edge_fraction, 0.05, 0.30, 0.01, 0.55)
    score = float(0.35 * size_score + 0.40 * contrast_score + 0.25 * edge_score)
    msgs: list[str] = []
    if w < 5 or h < 5:
        msgs.append(f"The {role} ROI is too small.")
    if rw > 0.35 or rh > 0.22:
        msgs.append(f"The {role} ROI is too large; select only the digits, not the full label.")
    if contrast < 5.0:
        msgs.append(f"The {role} ROI has too little contrast to look like visible digits.")
    return CalibrationCheck(
        not msgs,
        score,
        tuple(msgs),
        {
            "width_fraction": rw,
            "height_fraction": rh,
            "contrast": contrast,
            "edge_fraction": edge_fraction,
            "score_size": size_score,
            "score_contrast": contrast_score,
            "score_digit_edges": edge_score,
        },
    )


def detect_reference_jewels_detailed(panel: np.ndarray, board_roi: tuple[int, int, int, int]) -> ReferenceDetection | None:
    """Detect six colored x4 reference icons and expose geometry diagnostics."""
    if not _inside(panel, board_roi):
        return None
    x, y, bw, bh = board_roi
    ph, pw = panel.shape[:2]
    sx0 = max(0, int(round(x + bw + 0.08 * pw)))
    sx1 = min(pw, int(round(x + bw + 0.23 * pw)))
    sy0 = max(0, int(round(y + 0.10 * bh)))
    sy1 = min(ph, int(round(y + bh + 0.12 * ph)))
    if sx1 - sx0 < 12 or sy1 - sy0 < 40:
        return None
    search = panel[sy0:sy1, sx0:sx1]
    hsv = cv2.cvtColor(search, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 1] > 72) & (hsv[..., 2] > 65)).astype(np.uint8) * 255
    row = mask.mean(axis=1)
    smooth = np.convolve(row, np.ones(5, dtype=np.float64) / 5.0, mode="same")
    candidates = np.argsort(smooth)[::-1]
    min_sep = max(8, int(round(bh / 8.2)))
    peaks: list[int] = []
    for p in candidates:
        p = int(p)
        if smooth[p] < 18.0:
            break
        if all(abs(p - q) >= min_sep for q in peaks):
            peaks.append(p)
        if len(peaks) >= 6:
            break
    if len(peaks) != 6:
        return None
    peaks = sorted(peaks)
    gaps = np.diff(peaks).astype(np.float64)
    if gaps.mean() <= 0:
        return None
    spacing_cv = float(gaps.std() / gaps.mean())
    if spacing_cv > 0.30:
        return None
    span_ratio = float((peaks[-1] - peaks[0]) / max(1.0, float(bh)))
    if not (0.60 <= span_ratio <= 1.12):
        return None

    cell = max(14, int(round(min(bw, bh) / 5.0)))
    patch_size = max(14, int(round(cell * 0.88)))
    out: dict[str, np.ndarray] = {}
    x_centers: list[float] = []
    peak_strengths: list[float] = []
    color_fractions: list[float] = []
    for label, rel_y in zip(JEWELS, peaks):
        py = sy0 + rel_y
        yy0, yy1 = max(0, py - patch_size // 2), min(ph, py + patch_size // 2 + 1)
        sub = panel[yy0:yy1, sx0:sx1]
        shsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
        smask = (shsv[..., 1] > 72) & (shsv[..., 2] > 65)
        xs = np.where(smask)[1]
        if len(xs) < 3:
            return None
        cx = sx0 + float(np.median(xs))
        x_centers.append(cx)
        peak_strengths.append(float(smooth[rel_y]))
        color_fractions.append(float(smask.mean()))
        half = patch_size // 2
        px0, px1 = max(0, int(round(cx)) - half), min(pw, int(round(cx)) + half + 1)
        py0, py1 = max(0, py - half), min(ph, py + half + 1)
        patch = panel[py0:py1, px0:px1]
        if patch.size == 0:
            return None
        out[label] = patch.copy()
    x_std = float(np.std(x_centers))
    if x_std > max(6.0, 0.05 * pw):
        return None

    spacing_score = float(np.clip(1.0 - spacing_cv / 0.28, 0.0, 1.0))
    alignment_score = float(np.clip(1.0 - x_std / max(5.0, 0.04 * pw), 0.0, 1.0))
    color_signal = float(np.clip(np.mean(peak_strengths) / 80.0, 0.0, 1.0))
    span_score = _range_score(span_ratio, 0.68, 1.02, 0.58, 1.15)
    confidence = float(np.clip(0.32 * spacing_score + 0.24 * alignment_score + 0.26 * color_signal + 0.18 * span_score, 0.0, 1.0))
    return ReferenceDetection(
        templates=out,
        confidence=confidence,
        metrics={
            "spacing_cv": spacing_cv,
            "span_ratio": span_ratio,
            "x_std_px": x_std,
            "mean_peak_strength": float(np.mean(peak_strengths)),
            "mean_color_fraction": float(np.mean(color_fractions)),
            "score_spacing": spacing_score,
            "score_alignment": alignment_score,
            "score_color_signal": color_signal,
            "score_span": span_score,
        },
    )


def detect_reference_jewels(panel: np.ndarray, board_roi: tuple[int, int, int, int]) -> tuple[dict[str, np.ndarray], float] | None:
    """Backwards-compatible wrapper used by older scripts/tests."""
    found = detect_reference_jewels_detailed(panel, board_roi)
    if found is None:
        return None
    return found.templates, found.confidence
