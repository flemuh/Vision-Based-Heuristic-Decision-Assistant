import random
from types import SimpleNamespace

from vision_engine.autoplay import (
    AutoPlayPoint,
    bounded_random_ms,
    motion_points,
    safe_cell_point,
)


def test_safe_cell_point_stays_in_central_area():
    region = SimpleNamespace(left=100, top=200)
    roi = (50, 60, 500, 500)  # each cell 100x100
    rng = random.Random(1234)
    for _ in range(200):
        p = safe_cell_point(region, roi, 0, jitter_fraction=0.18, rng=rng)
        # Cell 0 spans x=150..250, y=260..360. Safe jitter is +/-18px around center.
        assert 182 <= p.x <= 218
        assert 292 <= p.y <= 328


def test_bounded_random_delay_respects_bounds_and_varies():
    rng = random.Random(99)
    values = [bounded_random_ms(550, 950, rng=rng) for _ in range(100)]
    assert min(values) >= 550
    assert max(values) <= 950
    assert len(set(values)) > 20


def test_motion_path_finishes_exactly_at_target():
    start = AutoPlayPoint(10, 20)
    target = AutoPlayPoint(310, 220)
    pts = motion_points(start, target, 20)
    assert pts[-1] == target
    assert len(pts) >= 10
    assert all(10 <= p.x <= 310 for p in pts)
    assert all(20 <= p.y <= 220 for p in pts)
