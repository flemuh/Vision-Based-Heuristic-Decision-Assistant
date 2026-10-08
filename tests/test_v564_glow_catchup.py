import numpy as np

from vision_engine.temporal import TemporalStateFilter
from vision_engine.vision import VisionConfig, blue_selected_details, selected_glow_decision


def test_false_multicell_glow_never_promotes_accepted_state():
    f = TemporalStateFilter(frames=3, required=2, max_mask_jump=1, catchup_required_frames=5)
    base = (1 << 1) | (1 << 3) | (1 << 7) | (1 << 9) | (1 << 10) | (1 << 12)
    false_extra = base | (1 << 19) | (1 << 23) | (1 << 24)
    f.reset(base)

    a = f.observe(false_extra, "CR", 0.9)
    b = f.observe(false_extra, "CR", 0.9)
    assert not a.stable
    assert not b.stable
    assert "catch-up-quarantine" in b.reason
    assert f.accepted_mask == base

    # Glow disappears. The accepted checkpoint must still be the real 6/14 mask.
    f.observe(base, "CR", 0.9)
    settled = f.observe(base, "CR", 0.9)
    assert settled.mask == base
    assert f.accepted_mask == base


def test_real_multicell_catchup_requires_extended_raw_stability():
    f = TemporalStateFilter(frames=3, required=2, max_mask_jump=1, catchup_required_frames=5)
    f.reset(0)
    mask = (1 << 1) | (1 << 3) | (1 << 7) | (1 << 9)
    states = [f.observe(mask, "HA", 0.9) for _ in range(5)]
    assert all(not x.stable for x in states[:-1])
    assert states[-1].stable
    assert states[-1].mask == mask
    assert states[-1].reason == "catch-up:4"


def test_ordinary_plus_one_still_uses_fast_consensus():
    f = TemporalStateFilter(frames=3, required=2, max_mask_jump=1, catchup_required_frames=5)
    base = (1 << 1) | (1 << 3)
    f.reset(base)
    moved = base | (1 << 7)
    assert not f.observe(moved, "SO", 0.9).stable
    accepted = f.observe(moved, "SO", 0.9)
    assert accepted.stable
    assert accepted.mask == moved
    assert accepted.reason == "stable"


def _blue_bgr():
    # HSV-ish vivid blue in OpenCV BGR space.
    return np.array([255, 60, 20], dtype=np.uint8)


def test_one_sided_neighbor_glow_is_rejected_even_with_high_ring_score():
    cfg = VisionConfig(selected_blue_threshold=0.17, selected_side_threshold=0.14, selected_min_glow_sides=2)
    patch = np.zeros((40, 40, 3), dtype=np.uint8)
    patch[:, -10:] = _blue_bgr()  # spill from neighbor on only the right edge
    score, sides = blue_selected_details(patch)
    selected, _, side_count = selected_glow_decision(patch, cfg)
    assert score >= cfg.selected_blue_threshold
    assert side_count < cfg.selected_min_glow_sides
    assert not selected


def test_multi_edge_selection_glow_is_accepted():
    cfg = VisionConfig(selected_blue_threshold=0.17, selected_side_threshold=0.14, selected_min_glow_sides=2)
    patch = np.zeros((40, 40, 3), dtype=np.uint8)
    blue = _blue_bgr()
    patch[:10, :] = blue
    patch[-10:, :] = blue
    patch[:, :10] = blue
    patch[:, -10:] = blue
    selected, score, side_count = selected_glow_decision(patch, cfg)
    assert score >= cfg.selected_blue_threshold
    assert side_count >= 2
    assert selected
