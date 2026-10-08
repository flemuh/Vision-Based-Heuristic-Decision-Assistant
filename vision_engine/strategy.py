from __future__ import annotations

from dataclasses import dataclass

from .constants import LUCKY_LINES, NORMAL_LINES


def viability_label(probability: float, low_threshold: float, high_threshold: float) -> str:
    """Translate a probability into a compact strategy-viability label."""
    p = float(probability)
    if p <= 1e-12:
        return "IMPOSSIBLE"
    if p < float(low_threshold):
        return "LOW"
    if p < float(high_threshold):
        return "MEDIUM"
    return "HIGH"


@dataclass(frozen=True)
class LineImpact:
    kind: str
    name: str
    before: int
    after: int
    size: int

    @property
    def completed(self) -> bool:
        return self.after >= self.size

    @property
    def missing_after(self) -> int:
        return max(0, self.size - self.after)


def move_line_impacts(mask: int, position: int) -> tuple[LineImpact, ...]:
    """Return every Lucky/Normal line directly advanced by a candidate cell."""
    out: list[LineImpact] = []
    for name, cells in LUCKY_LINES.items():
        if position in cells:
            before = sum(1 for i in cells if (mask >> i) & 1)
            after = before if (mask >> position) & 1 else before + 1
            out.append(LineImpact("Lucky", name, before, after, len(cells)))
    for name, cells in NORMAL_LINES.items():
        if position in cells:
            before = sum(1 for i in cells if (mask >> i) & 1)
            after = before if (mask >> position) & 1 else before + 1
            out.append(LineImpact("Normal", name, before, after, len(cells)))
    return tuple(out)


def compact_move_line_summary(mask: int, position: int) -> str:
    """Human-readable, compact explanation used by the live ranking UI."""
    impacts = move_line_impacts(mask, position)
    lucky = [x for x in impacts if x.kind == "Lucky"]
    normal = [x for x in impacts if x.kind == "Normal"]

    def one(x: LineImpact) -> str:
        if x.completed:
            return f"{x.name} {x.before}/{x.size}->{x.after}/{x.size} COMPLETE"
        return f"{x.name} {x.before}/{x.size}->{x.after}/{x.size} ({x.missing_after} missing)"

    lucky_text = ", ".join(one(x) for x in lucky) if lucky else "none"
    normal_text = ", ".join(one(x) for x in normal) if normal else "none"
    return f"Lucky: {lucky_text} | Normal: {normal_text}"
