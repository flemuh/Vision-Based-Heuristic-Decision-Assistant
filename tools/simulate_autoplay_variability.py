from __future__ import annotations

from collections import Counter
import random

from _bootstrap import ROOT  # adds repository root to sys.path
from types import SimpleNamespace

from vision_engine.autoplay import bounded_random_ms, safe_cell_point


def main(samples: int = 10000) -> None:
    rng = random.Random(1808)
    region = SimpleNamespace(left=0, top=0)
    roi = (0, 0, 135, 135)  # representative ~27 px cells
    records = []
    for _ in range(samples):
        point = safe_cell_point(region, roi, 12, jitter_fraction=0.18, rng=rng)
        base_delay = bounded_random_ms(550, 950, rng=rng)
        extra_settle = bounded_random_ms(50, 500, rng=rng)
        total_delay = base_delay + extra_settle
        move = bounded_random_ms(180, 320, rng=rng)
        records.append((point.x, point.y, total_delay, move))
    counts = Counter(records)
    repeated_samples = sum(v - 1 for v in counts.values() if v > 1)
    endpoints = Counter((x, y) for x, y, _d, _m in records)
    print(f"samples={samples}")
    print(f"unique_full_tuples={len(counts)}")
    print(f"repeated_full_tuples={repeated_samples}")
    print(f"unique_pixel_endpoints={len(endpoints)}")
    print("NOTE: this measures input variability only; it does not estimate human-likeness or anti-cheat detectability.")


if __name__ == "__main__":
    main()
