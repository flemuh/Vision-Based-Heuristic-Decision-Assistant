"""Game-independent physical-input matrix runner (V18.0.27 forensic harness).

Replays the Auto Play click pattern (random cells, bounded move/delay times,
14 clicks per "game") through the SAME ``windows_move_and_left_click`` the app
uses, against any on-screen target (MS Paint, an empty desktop, ...). It has no
vision, solver or WSL dependency, so it isolates "my Win32 input path" from
"the game". All Win32 effects are injected, so the logic is testable off Windows.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
import time
from types import SimpleNamespace
from typing import Any, Callable

from vision_engine.input.geometry import AutoPlayPoint, safe_cell_point


@dataclass(slots=True)
class MatrixConfig:
    games: int = 10
    clicks_per_game: int = 14
    region: tuple[int, int, int, int] = (0, 0, 1000, 800)  # left, top, width, height
    delay_ms: tuple[int, int] = (900, 1600)
    move_ms: tuple[int, int] = (220, 380)
    jitter_fraction: float = 0.18
    seed: int | None = None
    label: str = "matrix"


class MatrixRunner:
    def __init__(
        self,
        cfg: MatrixConfig,
        *,
        click_fn: Callable[[AutoPlayPoint, int], str],
        sample_fn: Callable[..., Any],
        avoid_fn: Callable[[], tuple[int, int, int, int] | None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 1 <= int(cfg.clicks_per_game) <= 25:
            raise ValueError("clicks_per_game must be between 1 and 25")
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self._click = click_fn
        self._sample = sample_fn
        self._avoid = avoid_fn
        self._clock = clock
        self._started = clock()
        self._game = 0
        self._click_in_game = 0
        self._positions = self._new_game_positions()
        self.rows: list[dict[str, Any]] = []
        self.failure: dict[str, Any] | None = None
        self.done = False
        self.avoided = 0

    @property
    def total_clicks(self) -> int:
        return int(self.cfg.games) * int(self.cfg.clicks_per_game)

    def _new_game_positions(self) -> list[int]:
        return self.rng.sample(range(25), int(self.cfg.clicks_per_game))

    def _pick_point(self, position: int) -> AutoPlayPoint:
        left, top, width, height = self.cfg.region
        region = SimpleNamespace(left=int(left), top=int(top))
        roi = (0, 0, int(width), int(height))
        point = safe_cell_point(region, roi, position, jitter_fraction=self.cfg.jitter_fraction, rng=self.rng)
        for _ in range(30):
            rect = self._avoid() if self._avoid else None
            if not rect or not (rect[0] <= point.x < rect[2] and rect[1] <= point.y < rect[3]):
                return point
            self.avoided += 1
            position = self.rng.randrange(25)
            point = safe_cell_point(region, roi, position, jitter_fraction=self.cfg.jitter_fraction, rng=self.rng)
        return point

    def step(self) -> int | None:
        """Execute one click. Returns the delay (ms) before the next one, or None when finished."""
        if self.done:
            return None
        position = self._positions[self._click_in_game]
        point = self._pick_point(position)
        move_ms = self.rng.randint(*self.cfg.move_ms)
        row: dict[str, Any] = {
            "t": round(self._clock() - self._started, 3), "game": self._game + 1,
            "click": self._click_in_game + 1, "position": int(position),
            "x": point.x, "y": point.y, "move_ms": move_ms,
        }
        try:
            backend = self._click(point, move_ms)
        except Exception as exc:
            row.update(ok=False, error=f"{type(exc).__name__}: {exc}")
            self.rows.append(row)
            self.failure = dict(row)
            self.done = True
            self._sample("failure", game=self._game + 1, click=self._click_in_game + 1)
            return None
        row.update(ok=True, backend=str(backend))
        self.rows.append(row)
        self._sample("click", game=self._game + 1, click=self._click_in_game + 1)

        self._click_in_game += 1
        if self._click_in_game >= int(self.cfg.clicks_per_game):
            self._sample("game-end", game=self._game + 1)
            self._game += 1
            self._click_in_game = 0
            if self._game >= int(self.cfg.games):
                self.done = True
                return None
            self._positions = self._new_game_positions()
        return self.rng.randint(*self.cfg.delay_ms)

    def stop(self, reason: str = "stopped") -> None:
        if not self.done:
            self.done = True
            self.failure = self.failure or {"ok": False, "error": f"stopped: {reason}", "stopped_by_user": True}

    def summary(self) -> dict[str, Any]:
        ok_rows = [r for r in self.rows if r.get("ok")]
        failed = bool(self.failure) and not (self.failure or {}).get("stopped_by_user")
        return {
            "label": self.cfg.label,
            "games_requested": int(self.cfg.games),
            "clicks_per_game": int(self.cfg.clicks_per_game),
            "clicks_ok": len(ok_rows),
            "clicks_total_expected": self.total_clicks,
            "completed": bool(self.done and self.failure is None),
            "failed": failed,
            "failure": self.failure,
            "avoided_overlap_draws": self.avoided,
            "elapsed_s": round(self._clock() - self._started, 2),
        }
