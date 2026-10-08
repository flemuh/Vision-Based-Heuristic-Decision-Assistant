from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mss
import numpy as np


@dataclass
class ScreenRegion:
    left: int
    top: int
    width: int
    height: int

    def as_dict(self) -> dict[str, int]:
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}


def screenshot_all() -> tuple[np.ndarray, dict]:
    with mss.mss() as sct:
        mon = sct.monitors[0]
        raw = np.array(sct.grab(mon))
    return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR), mon


def capture_region(region: ScreenRegion) -> np.ndarray:
    with mss.mss() as sct:
        raw = np.array(sct.grab(region.as_dict()))
    return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)


def save_region_anchor(panel: np.ndarray, path: str | Path) -> Path:
    """Save a stable title/header anchor beginning at panel top-left."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = panel.shape[:2]
    ah = max(28, min(h, int(h * 0.18)))
    anchor = panel[:ah, :w].copy()
    cv2.imwrite(str(path), anchor)
    return path


def relocate_region_by_anchor(
    anchor_path: str | Path,
    previous: ScreenRegion,
    threshold: float = 0.80,
    scales: tuple[float, ...] = (0.90, 0.95, 1.0, 1.05, 1.10),
) -> tuple[ScreenRegion | None, float]:
    """Find a moved or modestly resized game panel from a saved header anchor.

    This is ordinary screen template matching only. It does not inspect the game
    process or interact with it. Multi-scale matching tolerates common Windows
    DPI/window-size changes between sessions.
    """
    anchor = cv2.imread(str(anchor_path))
    if anchor is None:
        return None, 0.0
    screen, mon = screenshot_all()
    scr_g = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    best_val = -1.0
    best_loc = None
    best_scale = 1.0
    for scale in scales:
        if scale <= 0:
            continue
        anc = cv2.resize(anchor, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        ah, aw = anc.shape[:2]
        sh, sw = scr_g.shape[:2]
        if ah > sh or aw > sw or ah < 8 or aw < 8:
            continue
        anc_g = cv2.cvtColor(anc, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(scr_g, anc_g, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if float(max_val) > best_val:
            best_val = float(max_val); best_loc = max_loc; best_scale = float(scale)
    if best_loc is None or best_val < threshold:
        return None, max(0.0, best_val)
    x, y = best_loc
    region = ScreenRegion(
        mon["left"] + x, mon["top"] + y,
        max(1, round(previous.width * best_scale)),
        max(1, round(previous.height * best_scale)),
    )
    return region, best_val



def relocate_region_by_anchor_in_image(
    anchor_path: str | Path,
    previous: ScreenRegion,
    screen: np.ndarray,
    mon: dict,
    threshold: float = 0.62,
    scales: tuple[float, ...] = (0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15),
) -> tuple[ScreenRegion | None, float]:
    """Relocate a saved panel inside an already captured desktop image.

    AUTO SETUP uses this before falling back to generic MU/grid search. It makes
    moving the Jewel Bingo window a normal automatic case rather than forcing a
    new manual panel selection.
    """
    anchor = cv2.imread(str(anchor_path))
    if anchor is None or screen is None or screen.size == 0:
        return None, 0.0
    scr_g = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    best_val = -1.0
    best_loc = None
    best_scale = 1.0
    for scale in scales:
        anc = cv2.resize(anchor, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        ah, aw = anc.shape[:2]
        if ah < 8 or aw < 8 or ah > scr_g.shape[0] or aw > scr_g.shape[1]:
            continue
        res = cv2.matchTemplate(scr_g, cv2.cvtColor(anc, cv2.COLOR_BGR2GRAY), cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(res)
        if float(mx) > best_val:
            best_val = float(mx); best_loc = loc; best_scale = float(scale)
    if best_loc is None or best_val < threshold:
        return None, max(0.0, best_val)
    x, y = best_loc
    return ScreenRegion(
        int(mon["left"] + x), int(mon["top"] + y),
        max(1, round(previous.width * best_scale)),
        max(1, round(previous.height * best_scale)),
    ), best_val
