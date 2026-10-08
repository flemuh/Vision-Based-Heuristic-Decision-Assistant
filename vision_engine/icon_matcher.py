from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import math
import shutil

import cv2
import numpy as np

from .constants import JEWELS


@dataclass(frozen=True)
class IconMatch:
    label: str
    score: float
    margin: float
    roi: tuple[int, int, int, int]
    patch: np.ndarray
    second_label: str | None = None

    @property
    def reliable(self) -> bool:
        return self.score >= 0.76 and self.margin >= 0.035


class CurrentJewelMatcher:
    """Locate/classify the current jewel using user-specific x4 reference icons.

    The V5.1 implementation inferred one static ROI and then ran the same KNN
    used for board cells. In practice the top-row icon shifts a few pixels with
    window scaling, and list icons/board cells live in different visual domains.
    This matcher instead scans only the small band above the board and performs
    multi-scale normalized template matching against references learned from the
    game's own colored x4 list. No process/memory access is used.
    """

    def __init__(self, template_dir: str | Path):
        self.template_dir = Path(template_dir)
        self.template_dir.mkdir(parents=True, exist_ok=True)
        self.templates: dict[str, list[np.ndarray]] = {j: [] for j in JEWELS}
        self.usable_templates: dict[str, list[np.ndarray]] = {j: [] for j in JEWELS}
        self.cross_label_conflicts: list[tuple[str, str, float]] = []
        self.reload()

    def reload(self) -> None:
        self.templates = {j: [] for j in JEWELS}
        for label in JEWELS:
            for p in sorted(self.template_dir.glob(f"{label}_*.png")):
                img = cv2.imread(str(p))
                if img is not None and img.size:
                    self.templates[label].append(img)
        self._rebuild_usable_templates()

    def _rebuild_usable_templates(self, conflict_mse: float = 0.008) -> None:
        """Ignore near-identical references that exist under different labels.

        Adaptive learning historically checked novelty only inside one label, so
        the same visual patch could be persisted under two jewel names. That
        makes the best and second-best template scores nearly identical forever.
        We leave files untouched for forensics, but exclude both conflicting
        references from live matching.
        """
        self.usable_templates = {j: list(self.templates.get(j, [])) for j in JEWELS}
        bad: dict[str, set[int]] = {j: set() for j in JEWELS}
        conflicts: list[tuple[str, str, float]] = []
        thumbs = {j: [self._normalized_thumb(x) for x in self.templates.get(j, [])] for j in JEWELS}
        for a_i, a in enumerate(JEWELS):
            for b in JEWELS[a_i + 1:]:
                for ia, ta in enumerate(thumbs[a]):
                    for ib, tb in enumerate(thumbs[b]):
                        mse = float(np.mean((ta - tb) ** 2))
                        if mse <= float(conflict_mse):
                            bad[a].add(ia)
                            bad[b].add(ib)
                            conflicts.append((a, b, mse))
        self.cross_label_conflicts = conflicts
        for label in JEWELS:
            self.usable_templates[label] = [
                img for idx, img in enumerate(self.templates.get(label, [])) if idx not in bad[label]
            ]

    def conflict_count(self) -> int:
        return len(self.cross_label_conflicts)

    def template_counts(self) -> dict[str, int]:
        return {j: len(self.templates.get(j, [])) for j in JEWELS}

    def ready(self) -> bool:
        return all(len(self.usable_templates.get(j, [])) >= 1 for j in JEWELS)

    def _next_path(self, label: str) -> Path:
        nums = []
        for p in self.template_dir.glob(f"{label}_*.png"):
            try:
                nums.append(int(p.stem.split("_")[-1]))
            except Exception:
                pass
        return self.template_dir / f"{label}_{max(nums, default=0)+1:04d}.png"

    @staticmethod
    def _normalized_thumb(img: np.ndarray, size: int = 28) -> np.ndarray:
        if img is None or img.size == 0:
            return np.zeros((size, size), dtype=np.float32)
        g = cv2.cvtColor(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        g = g.astype(np.float32)
        g -= float(g.mean())
        sd = float(g.std())
        if sd > 1e-6:
            g /= sd
        return g

    def add_template(self, label: str, patch: np.ndarray) -> Path:
        if label not in JEWELS:
            raise ValueError(label)
        p = self._next_path(label)
        cv2.imwrite(str(p), patch)
        self.reload()
        return p

    def add_template_if_novel(
        self, label: str, patch: np.ndarray, min_mse: float = 0.07, max_per_label: int = 80,
        cross_label_mse: float = 0.012,
    ) -> Path | None:
        if label not in JEWELS:
            raise ValueError(label)
        cand = self._normalized_thumb(patch)
        for old in self.templates.get(label, []):
            ref = self._normalized_thumb(old)
            mse = float(np.mean((cand - ref) ** 2))
            if mse < min_mse:
                return None
        # Never learn a patch that is visually almost identical to a different
        # jewel label. This is the contamination mode that creates 1.000/0.999
        # matches and permanent low-margin WAITING states.
        for other in JEWELS:
            if other == label:
                continue
            for old in self.templates.get(other, []):
                ref = self._normalized_thumb(old)
                if float(np.mean((cand - ref) ** 2)) <= float(cross_label_mse):
                    return None
        p = self.add_template(label, patch)
        files = sorted(self.template_dir.glob(f"{label}_*.png"))
        if len(files) > max_per_label:
            for old in files[:-max_per_label]:
                try:
                    old.unlink()
                except OSError:
                    pass
            self.reload()
        return p

    def replace_batch(self, batch: dict[str, np.ndarray]) -> None:
        """Replace current-jewel references with one fresh trusted x4 batch.

        AUTO SETUP is the explicit recovery boundary. Older versions kept every
        adaptive reference forever, so a mislabeled reference survived upgrades
        and future AUTO SETUP runs. Archive the old bank and rebuild from the six
        simultaneously observed, trusted list icons.
        """
        missing = [j for j in JEWELS if j not in batch or batch[j] is None or batch[j].size == 0]
        if missing:
            raise ValueError(f"incomplete current-jewel reference batch: {missing}")
        existing = [p for p in self.template_dir.glob("*.png") if p.is_file()]
        if existing:
            archive = self.template_dir / ("_archive_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
            archive.mkdir(parents=True, exist_ok=True)
            for old in existing:
                try:
                    shutil.move(str(old), str(archive / old.name))
                except OSError:
                    pass
        for label in JEWELS:
            cv2.imwrite(str(self.template_dir / f"{label}_0001.png"), batch[label])
        self.reload()

    def locate(self, panel: np.ndarray, board_roi: tuple[int, int, int, int]) -> IconMatch | None:
        if not self.ready() or panel is None or panel.size == 0:
            return None
        bx, by, bw, bh = board_roi
        cell = (bw + bh) / 10.0
        ph, pw = panel.shape[:2]
        # Current jewel is between the box icon and the remaining-count text.
        # Restricting the search band prevents the x4 list/board from producing
        # accidental high matches.
        x0 = max(0, int(round(bx + 0.45 * cell)))
        x1 = min(pw, int(round(bx + 2.35 * cell)))
        y0 = max(0, int(round(by - 1.85 * cell)))
        y1 = min(ph, int(round(by - 0.10 * cell)))
        if x1 - x0 < 12 or y1 - y0 < 12:
            return None
        band = panel[y0:y1, x0:x1]

        best_per_label: dict[str, tuple[float, tuple[int, int], tuple[int, int]]] = {}
        scales = (0.72, 0.78, 0.84, 0.90, 0.96, 1.02, 1.08, 1.14, 1.20, 1.26)
        for label in JEWELS:
            best = (-1.0, (0, 0), (0, 0))
            for templ in self.usable_templates.get(label, []):
                for scale in scales:
                    tw = max(8, int(round(templ.shape[1] * scale)))
                    th = max(8, int(round(templ.shape[0] * scale)))
                    if tw > band.shape[1] or th > band.shape[0]:
                        continue
                    inter = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
                    t = cv2.resize(templ, (tw, th), interpolation=inter)
                    res = cv2.matchTemplate(band, t, cv2.TM_CCOEFF_NORMED)
                    _, score, _, loc = cv2.minMaxLoc(res)
                    if float(score) > best[0]:
                        best = (float(score), loc, (tw, th))
            best_per_label[label] = best

        ordered = sorted(((v[0], label, v[1], v[2]) for label, v in best_per_label.items()), reverse=True)
        if not ordered:
            return None
        score, label, loc, size = ordered[0]
        second_score = ordered[1][0] if len(ordered) > 1 else -1.0
        second_label = ordered[1][1] if len(ordered) > 1 else None
        margin = max(0.0, float(score - second_score))
        # Idle/no-jewel states on the supplied UI stay below ~0.69; visible
        # jewels are ~0.86-0.95. Keep a safety gap and let temporal consensus
        # handle animation frames near the boundary.
        if score < 0.72:
            return None
        rx, ry = x0 + loc[0], y0 + loc[1]
        rw, rh = size
        patch = panel[ry:ry+rh, rx:rx+rw].copy()
        return IconMatch(label, float(score), margin, (rx, ry, rw, rh), patch, second_label)
