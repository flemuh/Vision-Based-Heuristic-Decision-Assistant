from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import cv2
import numpy as np
from .simple_ml import NumpyKNNClassifier

from .board import Board
from .constants import CENTER, JEWELS


@dataclass
class VisionConfig:
    # Pixel ROIs inside selected game panel.
    board_roi: tuple[int, int, int, int] | None = None
    current_roi: tuple[int, int, int, int] | None = None
    remaining_roi: tuple[int, int, int, int] | None = None
    result_lucky_roi: tuple[int, int, int, int] | None = None
    result_normal_roi: tuple[int, int, int, int] | None = None
    result_jewel_roi: tuple[int, int, int, int] | None = None
    result_total_roi: tuple[int, int, int, int] | None = None

    # Calibration validation flags. Older configs load these as False so a bad
    # V4.3 crop/template set cannot silently unlock START in V4.4.
    board_roi_validated: bool = False
    current_roi_validated: bool = False
    templates_validated: bool = False
    game_area_geometry_score: float = 0.0
    board_geometry_score: float = 0.0
    template_geometry_score: float = 0.0
    current_geometry_score: float = 0.0

    selected_blue_threshold: float = 0.17
    selected_ambiguity_band: float = 0.025
    # A true selected cell normally glows around more than one edge. A click on
    # a neighboring cell can spill blue into only one edge and used to create
    # false selected cells.
    selected_side_threshold: float = 0.14
    selected_min_glow_sides: int = 2

    # Confidence is a composite of top probability, margin and entropy. These
    # values are intentionally conservative; uncertain states are surfaced for
    # confirmation instead of being silently learned.
    board_min_confidence: float = 0.20
    board_min_cell_probability: float = 0.08
    board_max_ambiguous_cells: int = 3
    current_min_confidence: float = 0.22
    current_min_margin: float = 0.10
    current_max_entropy: float = 0.82

    @classmethod
    def load(cls, path: str | Path) -> "VisionConfig":
        p = Path(path)
        if not p.exists():
            return cls()
        raw = json.loads(p.read_text(encoding="utf-8"))
        allowed = {f.name for f in fields(cls)}
        d = {k: v for k, v in raw.items() if k in allowed}
        for key in (
            "board_roi", "current_roi", "remaining_roi", "result_lucky_roi",
            "result_normal_roi", "result_jewel_roi", "result_total_roi",
        ):
            if d.get(key) is not None:
                d[key] = tuple(d[key])
        return cls(**d)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


@dataclass(frozen=True)
class VisionPrediction:
    label: str
    confidence: float
    top_probability: float
    margin: float
    entropy: float
    distribution: dict[str, float]

    @property
    def reliable(self) -> bool:
        return self.confidence >= 0.22 and self.margin >= 0.10 and self.entropy <= 0.82


@dataclass(frozen=True)
class BoardReadDetails:
    board: Board
    selected_mask: int
    selected_scores: tuple[float, ...]
    board_confidence: float
    min_assigned_probability: float
    ambiguous_cells: tuple[int, ...]
    distributions: tuple[dict[str, float], ...]
    assigned_probabilities: tuple[float, ...]


def crop_roi(img: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = roi
    return img[y:y+h, x:x+w]


def board_cell_rois(board_img: np.ndarray):
    h, w = board_img.shape[:2]
    for r in range(5):
        for c in range(5):
            x0 = round(c * w / 5)
            x1 = round((c + 1) * w / 5)
            y0 = round(r * h / 5)
            y1 = round((r + 1) * h / 5)
            yield r * 5 + c, board_img[y0:y1, x0:x1]


def feature(patch: np.ndarray) -> np.ndarray:
    """Feature robust to blue selection glow and modest brightness changes.

    V4 adds local edge structure to V3's grayscale + HSV histograms. We still
    keep the feature compact because the user-specific template set is small.
    """
    if patch.size == 0:
        return np.zeros(16 * 16 * 2 + 16 * 3, dtype=np.float32)
    h, w = patch.shape[:2]
    y0, y1 = int(h * 0.18), max(int(h * 0.82), int(h * 0.18) + 1)
    x0, x1 = int(w * 0.18), max(int(w * 0.82), int(w * 0.18) + 1)
    core = patch[y0:y1, x0:x1]
    core = cv2.resize(core, (16, 16), interpolation=cv2.INTER_AREA)
    gray_u8 = cv2.cvtColor(core, cv2.COLOR_BGR2GRAY)
    gray = gray_u8.astype(np.float32) / 255.0
    edges = cv2.Canny(gray_u8, 45, 120).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV)
    hist_parts = []
    for ch, rng in [(0, 180), (1, 256), (2, 256)]:
        hist = cv2.calcHist([hsv], [ch], None, [16], [0, rng]).flatten()
        hist = hist / (hist.sum() + 1e-6)
        hist_parts.append(hist)
    return np.concatenate([gray.flatten(), edges.flatten(), *hist_parts]).astype(np.float32)


def blue_selected_details(patch: np.ndarray) -> tuple[float, tuple[float, float, float, float]]:
    """Return outer-ring blue score plus per-side glow coverage.

    The ring score separates selection glow from naturally blue jewel centers.
    V5.6.4 also measures top/bottom/left/right strips because click glow can spill
    into one neighboring edge. A real selected cell normally lights multiple
    sides of its own frame, while spillover is strongly one-sided.
    """
    if patch is None or patch.size == 0:
        return 0.0, (0.0, 0.0, 0.0, 0.0)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]
    blue = (h >= 90) & (h <= 135) & (s >= 105) & (v >= 85)

    ph, pw = blue.shape[:2]
    ry = max(1, int(round(ph * 0.22)))
    rx = max(1, int(round(pw * 0.22)))
    ring = np.ones_like(blue, dtype=bool)
    y0, y1 = ry, ph - ry
    x0, x1 = rx, pw - rx
    if y1 > y0 and x1 > x0:
        ring[y0:y1, x0:x1] = False
    vals = blue[ring]
    ring_score = float(vals.mean()) if vals.size else float(blue.mean())
    # Exclude corners when measuring each side. A spill on the right edge also
    # colors the top-right/bottom-right corners; counting full-width top/bottom
    # strips would incorrectly make that one-sided spill look like three sides.
    mid_x0, mid_x1 = rx, max(rx + 1, pw - rx)
    mid_y0, mid_y1 = ry, max(ry + 1, ph - ry)
    top = blue[:ry, mid_x0:mid_x1]
    bottom = blue[ph-ry:, mid_x0:mid_x1]
    left = blue[mid_y0:mid_y1, :rx]
    right = blue[mid_y0:mid_y1, pw-rx:]
    sides = tuple(float(part.mean()) if part.size else 0.0 for part in (top, bottom, left, right))
    return ring_score, sides


def blue_selected_score(patch: np.ndarray) -> float:
    """Backwards-compatible ring-only selection score."""
    return blue_selected_details(patch)[0]


def selected_glow_decision(patch: np.ndarray, cfg: VisionConfig) -> tuple[bool, float, int]:
    """Decide selection using both ring strength and multi-edge coherence."""
    score, sides = blue_selected_details(patch)
    side_count = sum(float(x) >= float(cfg.selected_side_threshold) for x in sides)
    selected = score >= float(cfg.selected_blue_threshold) and side_count >= int(cfg.selected_min_glow_sides)
    return bool(selected), float(score), int(side_count)


def distribution_metrics(dist: dict[str, float]) -> tuple[str, float, float, float, float]:
    vals = sorted(((float(p), j) for j, p in dist.items()), reverse=True)
    top_p, label = vals[0]
    second_p = vals[1][0] if len(vals) > 1 else 0.0
    margin = max(0.0, top_p - second_p)
    probs = np.asarray([max(float(dist.get(j, 0.0)), 1e-12) for j in JEWELS], dtype=np.float64)
    probs /= probs.sum()
    entropy = float(-(probs * np.log(probs)).sum() / math.log(len(JEWELS)))
    # Conservative composite: a single sharp KNN vote should not be treated as
    # certain unless the margin is also healthy and entropy is low.
    confidence = float(np.clip(top_p * (0.40 + 0.60 * margin) * (1.0 - 0.45 * entropy), 0.0, 1.0))
    return label, confidence, top_p, margin, entropy


class JewelClassifier:
    def __init__(self, template_dir: str | Path, seed_features_path: str | Path | None = None):
        self.template_dir = Path(template_dir)
        self.template_dir.mkdir(parents=True, exist_ok=True)
        self.seed_features_path = Path(seed_features_path) if seed_features_path else None
        self.seed_examples = 0
        self.model: NumpyKNNClassifier | None = None
        self.reload()

    def reload(self) -> None:
        xs, ys = [], []
        self.seed_examples = 0
        if self.seed_features_path is not None and self.seed_features_path.exists():
            try:
                data = np.load(self.seed_features_path, allow_pickle=False)
                feats = np.asarray(data["features"], dtype=np.float32)
                labels = [str(x) for x in data["labels"].tolist()]
                if feats.ndim == 2 and len(feats) == len(labels):
                    for x, label in zip(feats, labels):
                        if label in JEWELS:
                            xs.append(x)
                            ys.append(label)
                            self.seed_examples += 1
            except Exception:
                # Seed descriptors are an optional bootstrap. User-confirmed
                # templates still keep the classifier functional if the asset
                # is missing/corrupt.
                pass
        for label in JEWELS:
            for p in sorted(self.template_dir.glob(f"{label}_*.png")):
                img = cv2.imread(str(p))
                if img is None:
                    continue
                xs.append(feature(img))
                ys.append(label)
        if xs:
            # A small bank of board-domain seed descriptors avoids the V5.1
            # failure mode where one x4-list icon per class produced one-hot
            # KNN votes and near-zero constrained board likelihood.
            k = min(5, max(1, len(xs) // 6), len(xs))
            self.model = NumpyKNNClassifier(n_neighbors=k, weights="distance")
            self.model.fit(np.stack(xs), ys)
        else:
            self.model = None

    def template_counts(self) -> dict[str, int]:
        return {j: len(list(self.template_dir.glob(f"{j}_*.png"))) for j in JEWELS}

    def _next_path(self, label: str) -> Path:
        nums = []
        for p in self.template_dir.glob(f"{label}_*.png"):
            try:
                nums.append(int(p.stem.split("_")[-1]))
            except Exception:
                pass
        n = max(nums, default=0) + 1
        return self.template_dir / f"{label}_{n:04d}.png"

    def add_template(self, label: str, patch: np.ndarray) -> Path:
        if label not in JEWELS:
            raise ValueError(label)
        path = self._next_path(label)
        cv2.imwrite(str(path), patch)
        self.reload()
        return path

    def add_template_if_novel(
        self,
        label: str,
        patch: np.ndarray,
        min_distance: float = 0.28,
        max_per_label: int = 160,
    ) -> Path | None:
        """Self-supervised visual learning from confirmed placements.

        The destination cell is ground truth because the game only accepts a
        jewel on a matching cell. Only diverse examples are retained.
        """
        if label not in JEWELS:
            raise ValueError(label)
        existing = sorted(self.template_dir.glob(f"{label}_*.png"))
        x = feature(patch)
        best = math.inf
        for p in existing:
            img = cv2.imread(str(p))
            if img is None:
                continue
            d = float(np.linalg.norm(x - feature(img)))
            best = min(best, d)
        if best < min_distance:
            return None
        path = self.add_template(label, patch)
        existing = sorted(self.template_dir.glob(f"{label}_*.png"))
        if len(existing) > max_per_label:
            # Keep a small immutable seed and the newest confirmed diversity.
            removable = existing[6: max(6, len(existing) - max_per_label + 6)]
            for old in removable:
                try:
                    old.unlink()
                except OSError:
                    pass
            self.reload()
        return path

    def predict_distribution(self, patch: np.ndarray) -> dict[str, float]:
        if self.model is None:
            raise RuntimeError("No jewel templates. Run calibration first.")
        x = feature(patch).reshape(1, -1)
        probs = self.model.predict_proba(x)[0]
        classes = [str(c) for c in self.model.classes_]
        out = {j: 1e-8 for j in JEWELS}
        for c, p in zip(classes, probs):
            if c in out:
                out[c] = max(float(p), 1e-8)
        z = sum(out.values())
        return {k: v / z for k, v in out.items()}

    def predict_detail(self, patch: np.ndarray) -> VisionPrediction:
        dist = self.predict_distribution(patch)
        label, confidence, top, margin, entropy = distribution_metrics(dist)
        return VisionPrediction(label, confidence, top, margin, entropy, dist)

    def predict(self, patch: np.ndarray) -> tuple[str, float]:
        p = self.predict_detail(patch)
        return p.label, p.confidence


def _constrained_board_assignment_full(
    distributions: list[dict[str, float]],
) -> tuple[list[str], float, list[float]]:
    """Maximum-likelihood 24-cell assignment with exactly four of each jewel.

    Pure-Python/NumPy dynamic programming replacement for SciPy's Hungarian
    assignment. The state tracks counts of the first five jewel types; the
    sixth count is implied by the number of processed cells. There are at most
    5^5 states per layer, so this is fast for the fixed 24-cell board.
    """
    if len(distributions) != 24:
        raise ValueError("Need 24 non-center cell distributions")
    eps = 1e-10
    # state -> best log-likelihood at current layer
    current: dict[tuple[int, int, int, int, int], float] = {(0, 0, 0, 0, 0): 0.0}
    layers: list[dict[tuple[int, int, int, int, int], tuple[tuple[int, int, int, int, int], int]]] = []

    for t, probs in enumerate(distributions):
        nxt: dict[tuple[int, int, int, int, int], float] = {}
        back: dict[tuple[int, int, int, int, int], tuple[tuple[int, int, int, int, int], int]] = {}
        for state, score in current.items():
            sixth = t - sum(state)
            counts = (*state, sixth)
            for label_idx, jewel in enumerate(JEWELS):
                if counts[label_idx] >= 4:
                    continue
                if label_idx < 5:
                    ns = list(state); ns[label_idx] += 1; new_state = tuple(ns)
                else:
                    new_state = state
                val = score + math.log(max(float(probs.get(jewel, eps)), eps))
                if val > nxt.get(new_state, -math.inf):
                    nxt[new_state] = val
                    back[new_state] = (state, label_idx)
        current = nxt
        layers.append(back)

    final_state = (4, 4, 4, 4, 4)
    if final_state not in current:
        raise RuntimeError("Could not find a valid 4x6 jewel assignment")
    labels_idx = [0] * 24
    state = final_state
    for t in range(23, -1, -1):
        prev_state, label_idx = layers[t][state]
        labels_idx[t] = label_idx
        state = prev_state
    assigned = [JEWELS[i] for i in labels_idx]
    assigned_probs = [float(distributions[i].get(assigned[i], 0.0)) for i in range(24)]
    gm = math.exp(sum(math.log(max(x, eps)) for x in assigned_probs) / 24.0)
    return assigned, float(gm), assigned_probs


def _constrained_board_assignment(distributions: list[dict[str, float]]) -> tuple[list[str], float]:
    # Backwards-compatible helper used by tests/research scripts.
    assigned, gm, _ = _constrained_board_assignment_full(distributions)
    return assigned, gm


def read_board_detailed(panel: np.ndarray, cfg: VisionConfig, clf: JewelClassifier) -> BoardReadDetails:
    if cfg.board_roi is None:
        raise RuntimeError("board_roi is not calibrated")
    bimg = crop_roi(panel, cfg.board_roi)
    selected_mask = 0
    scores = [0.0] * 25
    noncenter_indices: list[int] = []
    distributions: list[dict[str, float]] = []
    selected_ambiguous: list[int] = []

    for i, patch in board_cell_rois(bimg):
        if i == CENTER:
            continue
        noncenter_indices.append(i)
        distributions.append(clf.predict_distribution(patch))
        is_selected, sc, side_count = selected_glow_decision(patch, cfg)
        scores[i] = sc
        if abs(sc - cfg.selected_blue_threshold) <= cfg.selected_ambiguity_band or (
            sc >= cfg.selected_blue_threshold and side_count < cfg.selected_min_glow_sides
        ):
            selected_ambiguous.append(i)
        if is_selected:
            selected_mask |= 1 << i

    assigned, board_conf, assigned_probs = _constrained_board_assignment_full(distributions)
    cells: list[str | None] = [None] * 25
    ambiguous = list(selected_ambiguous)
    for idx, label, prob in zip(noncenter_indices, assigned, assigned_probs):
        cells[idx] = label
        if prob < cfg.board_min_cell_probability:
            ambiguous.append(idx)
    board = Board(tuple(cells))
    probs25 = [0.0] * 25
    probs25[CENTER] = 1.0
    for idx, prob in zip(noncenter_indices, assigned_probs):
        probs25[idx] = float(prob)
    return BoardReadDetails(
        board=board,
        selected_mask=selected_mask,
        selected_scores=tuple(scores),
        board_confidence=board_conf,
        min_assigned_probability=min(assigned_probs) if assigned_probs else 0.0,
        ambiguous_cells=tuple(sorted(set(ambiguous))),
        distributions=tuple(distributions),
        assigned_probabilities=tuple(probs25),
    )


def read_board(panel: np.ndarray, cfg: VisionConfig, clf: JewelClassifier) -> tuple[Board, int, list[float], float]:
    d = read_board_detailed(panel, cfg, clf)
    return d.board, d.selected_mask, list(d.selected_scores), d.board_confidence


def read_current_detailed(panel: np.ndarray, cfg: VisionConfig, clf: JewelClassifier) -> tuple[VisionPrediction, np.ndarray]:
    if cfg.current_roi is None:
        raise RuntimeError("current_roi is not calibrated")
    patch = crop_roi(panel, cfg.current_roi)
    return clf.predict_detail(patch), patch


def read_current(panel: np.ndarray, cfg: VisionConfig, clf: JewelClassifier) -> tuple[str, float, np.ndarray]:
    pred, patch = read_current_detailed(panel, cfg, clf)
    return pred.label, pred.confidence, patch


def read_selected_mask(panel: np.ndarray, cfg: VisionConfig) -> tuple[int, list[float]]:
    if cfg.board_roi is None:
        raise RuntimeError("board_roi is not calibrated")
    bimg = crop_roi(panel, cfg.board_roi)
    selected_mask = 0
    scores = [0.0] * 25
    for i, patch in board_cell_rois(bimg):
        if i == CENTER:
            continue
        is_selected, sc, _side_count = selected_glow_decision(patch, cfg)
        scores[i] = sc
        if is_selected:
            selected_mask |= 1 << i
    return selected_mask, scores


def board_patch(panel: np.ndarray, cfg: VisionConfig, index: int) -> np.ndarray:
    if cfg.board_roi is None:
        raise RuntimeError("board_roi is not calibrated")
    bimg = crop_roi(panel, cfg.board_roi)
    for i, patch in board_cell_rois(bimg):
        if i == index:
            return patch.copy()
    raise IndexError(index)
