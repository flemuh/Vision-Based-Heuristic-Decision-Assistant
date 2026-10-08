"""Transient-vs-permanent probe for a rejected SetCursorPos (V18.0.27 diagnostics).

After both SetCursorPos and SendInput fail, retry SetCursorPos at fixed delays
and record the outcome of each attempt. The Win32 calls are injected so the
logic is unit-testable off Windows. The probe only moves the cursor to the
waypoint the move loop was already trying to reach; it never clicks.
"""
from __future__ import annotations

from typing import Any, Callable

PROBE_DELAYS_MS: tuple[int, ...] = (10, 50, 200, 1000)

TRANSIENT = "TRANSIENT_REJECTION"
PERSISTENT = "PERSISTENT_REJECTION"
PARTIAL = "INTERMITTENT_REJECTION"


def run_rejection_probe(
    *,
    target: tuple[int, int],
    set_cursor: Callable[[int, int], bool],
    get_cursor: Callable[[], tuple[int, int] | None],
    get_last_error: Callable[[], int],
    sleep: Callable[[float], None],
    clock: Callable[[], float],
    delays_ms: tuple[int, ...] = PROBE_DELAYS_MS,
    tolerance_px: int = 8,
) -> dict[str, Any]:
    """Return per-attempt rows plus a classification.

    Delays are measured from the moment the probe starts (10 ms, 50 ms, 200 ms,
    1000 ms after the failure), not cumulatively between attempts.
    """
    t0 = clock()
    attempts: list[dict[str, Any]] = []
    tx, ty = int(target[0]), int(target[1])
    for delay_ms in delays_ms:
        remaining = (t0 + delay_ms / 1000.0) - clock()
        if remaining > 0:
            sleep(remaining)
        before = get_cursor()
        ok = bool(set_cursor(tx, ty))
        err = int(get_last_error())
        after = get_cursor()
        reached = (
            after is not None
            and abs(after[0] - tx) <= tolerance_px
            and abs(after[1] - ty) <= tolerance_px
        )
        attempts.append({
            "planned_delay_ms": int(delay_ms),
            "actual_delay_ms": round((clock() - t0) * 1000.0, 3),
            "set_cursor_ok": ok,
            "last_error": err,
            "cursor_before": None if before is None else [int(before[0]), int(before[1])],
            "cursor_after": None if after is None else [int(after[0]), int(after[1])],
            "reached_target": bool(reached),
        })

    reached = [bool(a["reached_target"]) for a in attempts]
    # The physical cursor is authoritative. Win32 has historically returned FALSE
    # even when GetCursorPos showed that the requested waypoint was reached.
    if not attempts or not any(reached):
        classification = PERSISTENT
    elif reached[-1]:
        classification = TRANSIENT
    else:
        classification = PARTIAL
    first_ok = next((a["planned_delay_ms"] for a in attempts if a["reached_target"]), None)
    errors = sorted({a["last_error"] for a in attempts if not a["reached_target"]})
    return {
        "classification": classification,
        "first_recovery_ms": first_ok,
        "failure_error_codes": errors,
        "attempts": attempts,
        "target": [tx, ty],
    }
