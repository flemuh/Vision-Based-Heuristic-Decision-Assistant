from vision_engine.temporal import TemporalStateFilter


def test_monotonic_multi_cell_jump_recovers_only_after_quarantine():
    f = TemporalStateFilter(frames=3, required=2, max_mask_jump=1, catchup_required_frames=5)
    f.reset(0)
    mask = (1 << 1) | (1 << 3) | (1 << 7) | (1 << 9)
    states = [f.observe(mask, "HA", 0.9) for _ in range(5)]
    assert all(not state.stable for state in states[:-1])
    assert states[-1].stable
    assert states[-1].mask == mask
    assert states[-1].reason == "catch-up:4"
    assert states[-1].jewel is None


def test_transition_clears_old_jewel_votes():
    f = TemporalStateFilter(frames=3, required=2, max_mask_jump=1)
    f.reset(0)
    f.observe(0, "LI", 0.9)
    old = f.observe(0, "LI", 0.9)
    assert old.jewel == "LI"
    moved = 1 << 2
    f.observe(moved, "LI", 0.9)
    accepted = f.observe(moved, "LI", 0.9)
    assert accepted.stable
    assert accepted.mask == moved
    assert accepted.jewel is None


def test_mask_regression_is_still_rejected():
    f = TemporalStateFilter(frames=2, required=1, max_mask_jump=1)
    f.reset((1 << 1) | (1 << 2))
    state = f.observe(1 << 1, None, 0.0)
    assert not state.stable
    assert "regression" in state.reason
