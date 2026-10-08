from __future__ import annotations

import re
from collections import Counter, deque
from dataclasses import dataclass, asdict

import cv2
import numpy as np

from .vision import VisionConfig, crop_roi


@dataclass(frozen=True)
class VerifiedResult:
    lucky_score: int | None = None
    normal_score: int | None = None
    jewel_score: int | None = None
    total_score: int | None = None
    source: str = "ocr"

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def complete(self) -> bool:
        return all(x is not None for x in (self.lucky_score, self.normal_score, self.jewel_score, self.total_score))

    @property
    def plausible(self) -> bool:
        if not self.complete:
            return False
        l, n, j, t = self.lucky_score, self.normal_score, self.jewel_score, self.total_score
        assert l is not None and n is not None and j is not None and t is not None
        if min(l, n, j, t) < 0 or t > 5000:
            return False
        return l + n + j == t

    @property
    def model_consistent(self) -> bool:
        if not self.complete:
            return False
        l, n, j = self.lucky_score, self.normal_score, self.jewel_score
        assert l is not None and n is not None and j is not None
        return l % 312 == 0 and n % 240 == 0 and j % 45 == 0


class ResultConsensus:
    """Accept OCR only after the same plausible result appears repeatedly."""

    def __init__(self, required: int = 2, frames: int = 4):
        self.required = max(1, int(required))
        self.values = deque(maxlen=max(self.required, int(frames)))

    def reset(self) -> None:
        self.values.clear()

    def observe(self, result: VerifiedResult | None) -> VerifiedResult | None:
        if result is None or not result.plausible:
            return None
        key = (result.lucky_score, result.normal_score, result.jewel_score, result.total_score)
        self.values.append(key)
        k, n = Counter(self.values).most_common(1)[0]
        if n < self.required:
            return None
        return VerifiedResult(k[0], k[1], k[2], k[3], "ocr-consensus")


def _ocr_int(patch: np.ndarray) -> int | None:
    try:
        import pytesseract
    except Exception:
        return None
    if patch.size == 0:
        return None
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    try:
        text = pytesseract.image_to_string(bw, config="--psm 7 -c tessedit_char_whitelist=0123456789")
    except Exception:
        return None
    nums = re.findall(r"\d+", text)
    if not nums:
        return None
    try:
        return int(max(nums, key=len))
    except ValueError:
        return None


def read_int_roi(panel: np.ndarray, roi: tuple[int, int, int, int] | None) -> int | None:
    if roi is None:
        return None
    return _ocr_int(crop_roi(panel, roi))


def read_result(panel: np.ndarray, cfg: VisionConfig) -> VerifiedResult | None:
    rois = (cfg.result_lucky_roi, cfg.result_normal_roi, cfg.result_jewel_roi, cfg.result_total_roi)
    if not all(rois):
        return None
    vals = [_ocr_int(crop_roi(panel, roi)) for roi in rois if roi is not None]
    if len(vals) != 4 or all(v is None for v in vals):
        return None
    return VerifiedResult(vals[0], vals[1], vals[2], vals[3], "ocr")
