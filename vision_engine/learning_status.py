from __future__ import annotations

import json
from typing import Any

from .constants import JEWELS
from .experience import PolicyModelInfo
from .persistence import BingoDB


def _local_complete_vectors(db: BingoDB) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []
    for row in db.game_rows():
        if not str(row.get("source", "")).startswith("local-") or not row.get("finished_at"):
            continue
        raw = row.get("counts_json")
        if not raw:
            continue
        try:
            counts = tuple(int(x) for x in json.loads(raw))
        except Exception:
            continue
        if len(counts) == 6 and sum(counts) == 14:
            out.append(counts)
    return out


def build_learning_snapshot(
    db: BingoDB,
    info: PolicyModelInfo,
    *,
    min_validation_games: int = 8,
    rng_min_games: int = 50,
    rng_blend: float = 0.0,
) -> dict[str, Any]:
    games = db.game_rows()
    completed = [g for g in games if g.get("finished_at")]
    verified = [g for g in completed if g.get("verified_score_json")]
    vectors = _local_complete_vectors(db)
    n = len(vectors)
    if n < rng_min_games:
        stage = "COLLECTING"
    elif n < 100:
        stage = "ANALYZING"
    else:
        stage = "ELIGIBLE FOR VALIDATION"

    totals = [0] * 6
    for counts in vectors:
        for i, value in enumerate(counts):
            totals[i] += int(value)
    denom = sum(totals)
    expected = 1.0 / 6.0
    jewel_rates = []
    for i, jewel in enumerate(JEWELS):
        observed = totals[i] / denom if denom else 0.0
        jewel_rates.append({
            "jewel": jewel,
            "observed": observed,
            "expected": expected,
            "delta": observed - expected,
        })

    return {
        "completed_games": len(completed),
        "verified_games": len(verified),
        "cf_samples": int(info.counterfactual_samples),
        "outcome_samples": int(info.outcome_samples),
        "cf_validation_games": int(info.validation_games),
        "outcome_validation_games": int(info.outcome_validation_games),
        "required_validation_games": int(min_validation_games),
        "cf_status": "TRUSTED" if info.counterfactual_trusted else ("LEARNING" if info.counterfactual_samples else "WAITING"),
        "outcome_status": "TRUSTED" if info.outcome_trusted else ("LEARNING" if info.outcome_samples else "WAITING"),
        "hindsight_status": "TRUSTED" if info.hindsight_trusted else ("LEARNING" if info.hindsight_samples else "WAITING"),
        "hindsight_samples": int(info.hindsight_samples),
        "hindsight_games": int(info.hindsight_games),
        "hindsight_validation_games": int(info.hindsight_validation_games),
        "hindsight_mae_p3": info.hindsight_mae_p3,
        "hindsight_mae_gt1000": info.hindsight_mae_gt1000,
        "hindsight_mae_score": info.hindsight_mae_score,
        "mae_p3": info.mae_p3,
        "mae_gt1000": info.mae_gt1000,
        "mae_score": info.mae_score,
        "outcome_residual_mae": info.outcome_residual_mae,
        "outcome_baseline_mae": info.outcome_baseline_mae,
        "outcome_improvement": info.outcome_improvement,
        "pattern_games": n,
        "pattern_stage": stage,
        "repeated_layouts": len(db.repeated_layouts(2)),
        "jewel_rates": jewel_rates,
        # Even after enough games, adaptation remains opt-in. V18 does not silently
        # turn observed frequency noise into solver probability changes.
        "rng_influence_enabled": bool(rng_blend > 0 and n >= max(100, rng_min_games)),
    }
