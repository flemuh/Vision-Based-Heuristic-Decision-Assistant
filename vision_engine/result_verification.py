from __future__ import annotations

from typing import Any


def computed_gt1000_verification(
    score: dict[str, Any],
    *,
    selected_count: int,
    enabled: bool,
    finalization_source: str,
) -> dict[str, Any] | None:
    """Promote a trusted complete board score to an internal verified result.

    This is intentionally strict: exactly 14 accepted cells and a total strictly
    greater than 1000. It is not OCR; the source field makes that distinction
    explicit in SQLite/exports.
    """
    if not enabled or int(selected_count) != 14:
        return None
    try:
        total = int(score.get("total") or 0)
    except Exception:
        return None
    if total <= 1000:
        return None
    out = dict(score)
    out.update({
        "source": "computed-auto-gt1000",
        "verification_basis": "trusted-14-cell-final-mask",
        "finalization_source": str(finalization_source),
    })
    return out
