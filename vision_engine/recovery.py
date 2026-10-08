from __future__ import annotations


def reconcile_masks(checkpoint_mask: int | None, screen_mask: int) -> tuple[bool, int, str]:
    """Conservative explicit-recovery reconciliation.

    A stable screen may advance from the checkpoint (missed placements) or roll
    back a checkpoint that contained transient glow. Non-subset disagreements
    are ambiguous and require normal monitoring/manual recovery instead.
    """
    screen = int(screen_mask)
    if checkpoint_mask is None:
        return True, screen, "no checkpoint; use stable screen"
    cp = int(checkpoint_mask)
    if cp == screen:
        return True, screen, "checkpoint matches stable screen"
    if (cp & screen) == cp:
        return True, screen, "stable screen advanced beyond checkpoint"
    if (cp & screen) == screen:
        return True, screen, "stable screen rolled back transient checkpoint"
    return False, cp, "checkpoint and screen disagree on different cells"
