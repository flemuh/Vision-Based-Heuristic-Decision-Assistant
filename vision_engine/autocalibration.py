from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import itertools
import math

import cv2
import numpy as np

from .calibration import CalibrationCheck, validate_board_roi, validate_game_panel


@dataclass(frozen=True)
class BoardDetection:
    roi: tuple[int, int, int, int]
    score: float
    coverage: float
    regularity: float
    square_score: float
    mu_anchor_score: float
    cell_size: float
    candidate_count: int
    stability: float = 1.0

    @property
    def high_confidence(self) -> bool:
        return self.score >= 0.82 and self.mu_anchor_score >= 0.20 and self.coverage >= 0.84


@dataclass(frozen=True)
class PanelDetection:
    roi: tuple[int, int, int, int]
    board_roi_screen: tuple[int, int, int, int]
    board_roi_panel: tuple[int, int, int, int]
    score: float
    board: BoardDetection
    panel_check: CalibrationCheck

    @property
    def high_confidence(self) -> bool:
        return self.score >= 0.78 and self.board.high_confidence and self.panel_check.ok


def _dedupe_rects(rects: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    out: list[tuple[int, int, int, int]] = []
    for r in sorted(rects, key=lambda x: x[2] * x[3], reverse=True):
        x, y, w, h = r
        cx, cy = x + w / 2.0, y + h / 2.0
        if any(abs(cx - (ox + ow / 2.0)) < 2.2 and abs(cy - (oy + oh / 2.0)) < 2.2 for ox, oy, ow, oh in out):
            continue
        out.append(r)
    return out


def _cluster_values(values: Iterable[float], tol: float) -> list[float]:
    clusters: list[list[float]] = []
    for v in sorted(float(x) for x in values):
        if not clusters or abs(v - float(np.mean(clusters[-1]))) > tol:
            clusters.append([v])
        else:
            clusters[-1].append(v)
    return [float(np.mean(c)) for c in clusters]


def _regular_sequences(values: list[float], n: int = 5) -> list[tuple[float, float, tuple[float, ...]]]:
    """Return low-CV equally-spaced n-value sequences.

    We cap the candidate list before combinations to keep full-screen detection
    bounded. Duplicate contour centers have already been collapsed.
    """
    vals = sorted(values)
    if len(vals) > 20:
        # Dense UI can produce many x/y clusters. Keep positions supported by
        # nearby neighbours by regular sampling; the board itself is dense.
        idx = np.linspace(0, len(vals) - 1, 20).round().astype(int)
        vals = [vals[i] for i in sorted(set(idx))]
    out: list[tuple[float, float, tuple[float, ...]]] = []
    for comb in itertools.combinations(vals, n):
        dif = np.diff(comb)
        mean = float(np.mean(dif))
        if mean < 8.0:
            continue
        cv = float(np.std(dif) / max(mean, 1e-6))
        if cv <= 0.16:
            out.append((cv, mean, tuple(float(x) for x in comb)))
    out.sort(key=lambda x: x[0])
    return out[:80]


def _mu_anchor_score(img: np.ndarray, roi: tuple[int, int, int, int]) -> float:
    x, y, w, h = roi
    y0, y1 = y + round(2 * h / 5), y + round(3 * h / 5)
    x0, x1 = x + round(2 * w / 5), x + round(3 * w / 5)
    patch = img[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
    if patch.size == 0:
        return 0.0
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    blue = float(((hh >= 88) & (hh <= 138) & (ss >= 70) & (vv >= 55)).mean())
    contrast = float(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).std())
    blue_score = float(np.clip((blue - 0.08) / 0.45, 0.0, 1.0))
    contrast_score = float(np.clip((contrast - 12.0) / 35.0, 0.0, 1.0))
    return float(0.78 * blue_score + 0.22 * contrast_score)


def _detect_board_grid_core(img: np.ndarray) -> BoardDetection | None:
    """Template-free 5x5 board detector.

    The detector looks for the repeated rectangular cell contours instead of
    asking the user to trace the board by hand. A candidate must form five
    regularly-spaced columns and rows and its center must look like the blue MU
    anchor. This works on both empty and populated/blue-glow boards.
    """
    if img is None or img.size == 0:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ih, iw = gray.shape
    edges = cv2.Canny(gray, 25, 90)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rects: list[tuple[int, int, int, int]] = []
    max_dim = max(18, int(min(iw, ih) * 0.35))
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if not (9 <= w <= max_dim and 9 <= h <= max_dim):
            continue
        ar = w / max(float(h), 1.0)
        if not (0.68 <= ar <= 1.45):
            continue
        if w * h < 90:
            continue
        rects.append((x, y, w, h))
    rects = _dedupe_rects(rects)
    if len(rects) < 14:
        return None

    sizes = np.asarray([(w + h) / 2.0 for _, _, w, h in rects], dtype=np.float64)
    hypotheses = sorted(set(int(round(s)) for s in sizes if s >= 11.0))
    best: BoardDetection | None = None

    for s in hypotheses:
        tol_size = max(3.0, s * 0.20)
        rr = [r for r in rects if abs((r[2] + r[3]) / 2.0 - s) <= tol_size]
        if len(rr) < 14:
            continue
        xs = _cluster_values((x + w / 2.0 for x, y, w, h in rr), max(2.0, s * 0.16))
        ys = _cluster_values((y + h / 2.0 for x, y, w, h in rr), max(2.0, s * 0.16))
        xseqs = _regular_sequences(xs)
        yseqs = _regular_sequences(ys)
        if not xseqs or not yseqs:
            continue
        centers = np.asarray([[x + w / 2.0, y + h / 2.0] for x, y, w, h in rr], dtype=np.float64)

        for cvx, px, xs5 in xseqs[:32]:
            if not (s * 0.76 <= px <= s * 1.48):
                continue
            for cvy, py, ys5 in yseqs[:32]:
                if not (s * 0.76 <= py <= s * 1.48):
                    continue
                pitch_diff = abs(px - py) / max(px, py, 1e-6)
                if pitch_diff > 0.18:
                    continue
                matches = 0
                dists: list[float] = []
                for xx in xs5:
                    for yy in ys5:
                        dist = float(np.min(np.sqrt((centers[:, 0] - xx) ** 2 + (centers[:, 1] - yy) ** 2)))
                        if dist <= s * 0.30:
                            matches += 1
                            dists.append(dist)
                coverage = matches / 25.0
                if coverage < 0.56:
                    continue
                regularity = float(np.clip(1.0 - (cvx + cvy) / 0.28, 0.0, 1.0))
                square = float(np.clip(1.0 - pitch_diff / 0.20, 0.0, 1.0))
                x0 = int(round(xs5[0] - px / 2.0))
                x1 = int(round(xs5[-1] + px / 2.0))
                y0 = int(round(ys5[0] - py / 2.0))
                y1 = int(round(ys5[-1] + py / 2.0))
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(iw, x1), min(ih, y1)
                if x1 - x0 < 40 or y1 - y0 < 40:
                    continue
                roi = (x0, y0, x1 - x0, y1 - y0)
                mu = _mu_anchor_score(img, roi)
                fit = 1.0 - min(1.0, (float(np.mean(dists)) if dists else s) / max(s * 0.30, 1e-6))
                score = float(np.clip(0.46 * coverage + 0.15 * regularity + 0.10 * square + 0.24 * mu + 0.05 * fit, 0.0, 1.0))
                det = BoardDetection(roi, score, coverage, regularity, square, mu, float((px + py) / 2.0), len(rr))
                if best is None or det.score > best.score:
                    best = det
    return best



def detect_board_from_mu_anchors(img: np.ndarray, max_candidates: int = 18) -> BoardDetection | None:
    """Fast bounded board search for large screenshots.

    The MU center is a saturated blue square in both idle and active boards.
    Full-screen exhaustive rectangle clustering can become combinatorial on a
    desktop full of UI, so this detector first finds a small set of plausible
    blue MU anchors and evaluates only 5x5 board boxes around them.

    This is intentionally bounded: at most ``max_candidates`` connected
    components and four cell-size hypotheses per component are evaluated.
    """
    if img is None or img.size == 0:
        return None
    ih, iw = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (((hh >= 88) & (hh <= 138) & (ss >= 70) & (vv >= 55)).astype(np.uint8) * 255)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)

    candidates: list[tuple[float, int, int, int, int, int]] = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if not (8 <= w <= 64 and 8 <= h <= 64 and 45 <= area <= 2200):
            continue
        ar = w / max(float(h), 1.0)
        if not (0.58 <= ar <= 1.72):
            continue
        density = area / max(float(w * h), 1.0)
        if density < 0.22:
            continue
        # Favor compact, saturated square-ish blobs of roughly cell size.
        square = 1.0 - min(1.0, abs(math.log(max(ar, 1e-6))) / 0.55)
        size_pref = float(np.clip(min(w, h) / 18.0, 0.0, 1.0))
        rank = 0.52 * density + 0.30 * square + 0.18 * size_pref
        candidates.append((rank, x, y, w, h, area))

    candidates.sort(reverse=True)
    best: BoardDetection | None = None
    for _rank, x, y, bw, bh, _area in candidates[:max_candidates]:
        cx, cy = x + bw / 2.0, y + bh / 2.0
        base_cell = float(max(bw, bh))
        for factor in (0.92, 1.00, 1.08, 1.16, 1.24, 1.36, 1.48):
            cell = base_cell * factor
            side = int(round(cell * 5.05))
            if side < 55 or side > 360:
                continue
            x0 = int(round(cx - side / 2.0)); y0 = int(round(cy - side / 2.0))
            if x0 < 0 or y0 < 0 or x0 + side > iw or y0 + side > ih:
                continue
            roi = (x0, y0, side, side)
            check = validate_board_roi(img, roi)
            mu = _mu_anchor_score(img, roi)
            grid = float(check.metrics.get("score_grid", 0.0))
            period = float(check.metrics.get("period_regularity", 0.0))
            weakest = float(check.metrics.get("weakest_grid_axis", 0.0))
            # The panel-relative size/location terms in validate_board_roi are
            # intentionally ignored here because ``img`` may be the full screen.
            delta = float(check.metrics.get("grid_delta", 0.0))
            delta_score = float(np.clip((delta + 0.01) / 0.12, 0.0, 1.0))
            # Reward five-cell periodicity and positive boundary-vs-interior
            # separation. This prevents a single blue HUD orb or a 3x3 crop
            # around MU from outscoring the true full 5x5 grid.
            local_score = float(np.clip(
                0.25 * grid + 0.25 * period + 0.10 * weakest + 0.20 * mu + 0.20 * delta_score,
                0.0, 1.0,
            ))
            coverage = float(np.clip(0.50 + 0.30 * grid + 0.20 * period, 0.0, 1.0))
            regularity = float(np.clip(0.55 * period + 0.45 * weakest, 0.0, 1.0))
            det = BoardDetection(
                roi=roi,
                score=local_score,
                coverage=coverage,
                regularity=regularity,
                square_score=1.0,
                mu_anchor_score=mu,
                cell_size=side / 5.0,
                candidate_count=len(candidates),
            )
            if best is None or det.score > best.score:
                best = det
    return best

def detect_board_grid(img: np.ndarray) -> BoardDetection | None:
    """Detect the 5x5 board in either a panel crop or a larger screenshot.

    Large screenshots are searched with overlapping tiles first. This avoids
    unrelated inventory/UI rectangles overwhelming the global x/y clustering.
    """
    if img is None or img.size == 0:
        return None
    ih, iw = img.shape[:2]
    if iw <= 420 and ih <= 420:
        return _detect_board_grid_core(img)

    tile_w = min(iw, 380)
    tile_h = min(ih, 360)
    stride_x = max(160, int(tile_w * 0.62))
    stride_y = max(150, int(tile_h * 0.62))

    def starts(total: int, size: int, stride: int) -> list[int]:
        vals = list(range(0, max(1, total - size + 1), stride))
        last = max(0, total - size)
        if not vals or vals[-1] != last:
            vals.append(last)
        return sorted(set(vals))

    best: BoardDetection | None = None
    for y0 in starts(ih, tile_h, stride_y):
        for x0 in starts(iw, tile_w, stride_x):
            crop = img[y0:y0+tile_h, x0:x0+tile_w]
            d = _detect_board_grid_core(crop)
            if d is None:
                continue
            x, y, w, h = d.roi
            mapped = BoardDetection(
                roi=(x + x0, y + y0, w, h),
                score=d.score, coverage=d.coverage, regularity=d.regularity,
                square_score=d.square_score, mu_anchor_score=d.mu_anchor_score,
                cell_size=d.cell_size, candidate_count=d.candidate_count, stability=d.stability,
            )
            if best is None or mapped.score > best.score:
                best = mapped
    return best


def detect_board_consensus(frames: list[np.ndarray]) -> BoardDetection | None:
    detections = [detect_board_grid(f) for f in frames]
    detections = [d for d in detections if d is not None]
    if not detections:
        return None
    # Use the strongest cluster of near-identical boxes.
    best_cluster: list[BoardDetection] = []
    for seed in detections:
        sx, sy, sw, sh = seed.roi
        cluster = []
        for d in detections:
            x, y, w, h = d.roi
            tol = max(5.0, seed.cell_size * 0.45)
            if abs(x - sx) <= tol and abs(y - sy) <= tol and abs(w - sw) <= tol and abs(h - sh) <= tol:
                cluster.append(d)
        if len(cluster) > len(best_cluster) or (len(cluster) == len(best_cluster) and np.mean([d.score for d in cluster]) > np.mean([d.score for d in best_cluster] or [0])):
            best_cluster = cluster
    if not best_cluster:
        return max(detections, key=lambda d: d.score)
    rois = np.asarray([d.roi for d in best_cluster], dtype=np.float64)
    roi = tuple(int(round(v)) for v in np.median(rois, axis=0))
    positional_std = float(np.mean(np.std(rois, axis=0)))
    cell = float(np.mean([d.cell_size for d in best_cluster]))
    stability = float(np.clip(1.0 - positional_std / max(cell * 0.35, 1.0), 0.0, 1.0))
    representative = max(best_cluster, key=lambda d: d.score)
    score = float(np.clip(np.mean([d.score for d in best_cluster]) * (0.80 + 0.20 * stability), 0.0, 1.0))
    return BoardDetection(
        roi=roi,
        score=score,
        coverage=float(np.mean([d.coverage for d in best_cluster])),
        regularity=float(np.mean([d.regularity for d in best_cluster])),
        square_score=float(np.mean([d.square_score for d in best_cluster])),
        mu_anchor_score=float(np.mean([d.mu_anchor_score for d in best_cluster])),
        cell_size=cell,
        candidate_count=max(d.candidate_count for d in best_cluster),
        stability=stability,
    )


def infer_panel_roi_from_board(screen: np.ndarray, board_roi: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Infer a conservative Jewel Bingo work area from the detected 5x5 board.

    The ratios are intentionally broad and are only a first proposal. The panel
    is subsequently checked by validate_game_panel and can fall back to manual
    selection if confidence is not high.
    """
    x, y, w, h = board_roi
    ih, iw = screen.shape[:2]
    x0 = int(round(x - 0.22 * w))
    # Leave generous headroom for the title + box/current/count row. Generic
    # full-screen MU detection can initially lock onto an inner/smaller grid;
    # the refined board pass below then uses this margin to recover the complete
    # panel instead of clipping the current-jewel row.
    y0 = int(round(y - 0.68 * h))
    x1 = int(round(x + 1.65 * w))
    y1 = int(round(y + 1.16 * h))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(iw, x1), min(ih, y1)
    return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def detect_panel_from_screen(screen: np.ndarray) -> PanelDetection | None:
    """Fast full-screen Jewel Bingo detector.

    Large desktops used to run the exhaustive grid search on many cluttered
    tiles, which could block the UI for minutes. Full-screen detection is now
    anchored on plausible blue MU components and is strictly bounded. Small
    panel crops still use the more exhaustive contour detector.
    """
    if screen is None or screen.size == 0:
        return None
    ih, iw = screen.shape[:2]
    if iw > 520 or ih > 520:
        board = detect_board_from_mu_anchors(screen)
    else:
        board = detect_board_grid(screen)
    if board is None or board.mu_anchor_score < 0.18:
        return None
    proi = infer_panel_roi_from_board(screen, board.roi)
    px, py, pw, ph = proi
    panel = screen[py:py+ph, px:px+pw]
    panel_check = validate_game_panel(panel)
    bx, by, bw, bh = board.roi
    brel = (bx - px, by - py, bw, bh)
    board_check = validate_board_roi(panel, brel)
    score = float(np.clip(0.56 * board.score + 0.24 * panel_check.score + 0.20 * board_check.score, 0.0, 1.0))
    return PanelDetection(proi, board.roi, brel, score, board, panel_check)


def infer_current_jewel_roi(board_roi: tuple[int, int, int, int], panel_shape: tuple[int, int] | tuple[int, int, int]) -> tuple[int, int, int, int]:
    """Infer the top current-jewel icon from fixed geometry around the board.

    Current jewel sits roughly one cell above and just right of the first board
    column. We deliberately return a tight square so the count and box icon stay
    outside. The live recognition test still decides whether a visible jewel is
    actually present.
    """
    bx, by, bw, bh = board_roi
    ph, pw = int(panel_shape[0]), int(panel_shape[1])
    cell = (bw + bh) / 10.0
    side = int(round(max(12.0, cell * 0.80)))
    cx = bx + 1.15 * cell
    cy = by - 1.00 * cell
    x = int(round(cx - side / 2.0))
    y = int(round(cy - side / 2.0))
    x = max(0, min(max(0, pw - side), x))
    y = max(0, min(max(0, ph - side), y))
    return (x, y, min(side, pw - x), min(side, ph - y))


def board_detection_metrics(det: BoardDetection) -> dict[str, float]:
    return {
        "auto_grid_score": det.score,
        "auto_grid_coverage": det.coverage,
        "auto_grid_regularity": det.regularity,
        "auto_grid_square": det.square_score,
        "auto_mu_anchor": det.mu_anchor_score,
        "auto_stability": det.stability,
        "auto_cell_px": det.cell_size,
    }
