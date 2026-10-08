from __future__ import annotations

from collections import Counter
from typing import Iterable

from .constants import JEWELS


def learned_bias_weights(
    history: list[dict],
    alpha: float = 40.0,
    blend: float = 0.0,
    min_games: int = 50,
    local_only: bool = True,
) -> tuple[float, ...]:
    """Dirichlet-smoothed empirical jewel weights with a safety gate.

    The chat-recovered games are useful for research but are selected examples,
    so they are deliberately excluded from RNG adaptation by default. Only a
    sufficiently large consecutive local sample can change draw probabilities.
    """
    eligible = []
    for row in history:
        counts = row.get("counts")
        if not counts or len(counts) != 6:
            continue
        if local_only and not str(row.get("source", "")).startswith("local-"):
            continue
        eligible.append(row)
    if len(eligible) < int(min_games) or blend <= 0:
        return (1.0,) * 6

    totals = [0.0] * 6
    for row in eligible:
        for i, v in enumerate(row["counts"]):
            totals[i] += float(v)
    mean = sum(totals) / 6.0
    if mean <= 0:
        return (1.0,) * 6
    # Prior is centered at no bias. alpha is measured in pseudo-games.
    empirical = [((totals[i] + alpha * mean) / (len(eligible) + alpha)) / mean for i in range(6)]
    return tuple((1.0 - blend) + blend * x for x in empirical)


def pattern_signature(counts: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted((int(x) for x in counts), reverse=True))


def history_summary(history: list[dict]) -> dict:
    valid = [row for row in history if row.get("counts") and len(row["counts"]) == 6]
    totals = [sum(row["counts"][i] for row in valid) for i in range(6)] if valid else [0] * 6
    sigs = Counter(pattern_signature(row["counts"]) for row in valid)
    local = [r for r in valid if str(r.get("source", "")).startswith("local-")]
    return {
        "games": len(valid),
        "local_games": len(local),
        "totals": dict(zip(JEWELS, totals)),
        "patterns": {str(k): v for k, v in sigs.most_common()},
    }
