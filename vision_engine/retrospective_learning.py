"""Offline, exact-sequence counterfactual analysis, never a live oracle.

The 14 jewels observed *after* a game ends can be used to check which other
legal placements could have produced a higher score. They must never be fed
back to the live solver as known future draws. An alternative is a POSSIBLE
hindsight result, not an observed victory or proof that the solver was wrong.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone

from .board import Board
from .constants import JEWELS
from .persistence import BingoDB
from .replay import _terminal_masks_for_counts
from .scoring import score_mask
from .winning_memory import WinningExample, validated_episodes

AUDIT_VERSION = 2


def compute_audit(episode: WinningExample) -> dict:
    board = Board(episode.board)
    counts = Counter(episode.sequence)
    terminal_masks = _terminal_masks_for_counts(episode.board, tuple(counts[j] for j in JEWELS))
    if not terminal_masks:
        return {"valid": False, "reason": "no legal terminal allocation"}
    # Score every complete assignment once, NOT every possible action prefix.
    terminal = [(int(m), score_mask(m)) for m in terminal_masks]
    best_mask, best_score = max(terminal, key=lambda pair: (pair[1].total, pair[1].lucky, pair[1].normal))
    actual_mask = sum(1 << p for p in episode.actions)
    if score_mask(actual_mask).total != episode.total:
        return {"valid": False, "reason": "recorded score mismatch"}

    rows = []
    prefix = 0
    first_missed = None
    first_lost_gt1000 = None
    for step, (jewel, actual_position) in enumerate(zip(episode.sequence, episode.actions)):
        candidates = []
        for candidate in board.legal_positions(jewel, prefix):
            after_mask = prefix | 1 << candidate
            reachable = [score for mask, score in terminal if (mask & after_mask) == after_mask]
            if not reachable:
                continue
            best = max(reachable, key=lambda s: (s.total, s.lucky, s.normal))
            candidates.append({
                "position": candidate,
                "best_score": best.total,
                "possible_gt1000": int(best.total > 1000),
                "possible_ge999": int(best.total >= 999),
                "possible_3lucky": int(any(sc.lucky >= 3 for sc in reachable)),
                "possible_2l1n": int(any(sc.lucky >= 2 and sc.normal >= 1 for sc in reachable)),
                "possible_1l2n": int(any(sc.lucky >= 1 and sc.normal >= 2 for sc in reachable)),
                "possible_1l1n": int(any(sc.lucky >= 1 and sc.normal >= 1 for sc in reachable)),
                "best_lucky": max(sc.lucky for sc in reachable),
                "best_normal": max(sc.normal for sc in reachable),
            })
        actual = next((c for c in candidates if c["position"] == actual_position), None)
        best_alternatives = sorted(
            (c for c in candidates if c["position"] != actual_position and c["best_score"] > (actual or {}).get("best_score", -1)),
            key=lambda c: (-c["best_score"], c["position"]),
        )
        if first_missed is None and best_alternatives:
            first_missed = {
                "step": step + 1, "jewel": jewel, "actual_pos": actual_position,
                "actual_best_remaining": actual["best_score"] if actual else None,
                "better_alternative_positions": [c["position"] for c in best_alternatives],
                "best_alternative_score": best_alternatives[0]["best_score"],
            }
        possible_other_wins = [c for c in candidates if c["position"] != actual_position and c["possible_gt1000"]]
        if first_lost_gt1000 is None and actual is not None and not actual["possible_gt1000"] and possible_other_wins:
            first_lost_gt1000 = {
                "step": step + 1, "jewel": jewel, "actual_pos": actual_position,
                "winning_alternative_positions": [c["position"] for c in possible_other_wins],
                "best_winning_alternative_score": max(c["best_score"] for c in possible_other_wins),
            }
        rows.append({"step": step, "mask_before": prefix, "jewel": jewel,
                     "actual_pos": actual_position, "candidates": candidates})
        prefix |= 1 << actual_position

    return {
        "valid": True, "audit_version": AUDIT_VERSION, "game_id": episode.game_id,
        "source": "offline_counterfactual_known_future", "observed_total": episode.total,
        "best_total": best_score.total, "best_lucky": best_score.lucky,
        "best_normal": best_score.normal, "best_mask": best_mask,
        "could_exceed_1000": best_score.total > 1000,
        "counterfactual_gain_upper_bound": max(0, best_score.total - episode.total),
        "terminal_allocations": len(terminal), "first_better_choice": first_missed,
        "first_lost_gt1000": first_lost_gt1000,
        "steps": rows,
    }


def cached_audit(db: BingoDB, episode: WinningExample, *, compute: bool = True) -> dict | None:
    """Idempotent per-game audited labels, stored separately from factual scores."""
    with db.connect() as con:
        row = con.execute(
            "SELECT audit_json FROM retrospective_audits WHERE game_id=? AND audit_version=?",
            (episode.game_id, AUDIT_VERSION),
        ).fetchone()
    if row:
        try:
            return json.loads(row["audit_json"])
        except (ValueError, TypeError):
            pass
    if not compute:
        return None
    data = compute_audit(episode)
    with db.connect() as con:
        con.execute(
            """INSERT INTO retrospective_audits(game_id,audit_version,computed_at,audit_json)
            VALUES(?,?,?,?) ON CONFLICT(game_id) DO UPDATE SET
            audit_version=excluded.audit_version,computed_at=excluded.computed_at,
            audit_json=excluded.audit_json""",
            (episode.game_id, AUDIT_VERSION, datetime.now(timezone.utc).isoformat(), json.dumps(data, separators=(",", ":"))),
        )
    return data


def audit_stored_games(db: BingoDB) -> dict[str, int]:
    """Heavy offline maintenance only: all replay-valid games, at most one audit/version."""
    episodes = validated_episodes(db)
    stats = {"validated": len(episodes), "analyzed": 0, "potential_gt1000": 0,
             "recoverable_gt1000": 0, "observed_ge999": 0}
    # Invalidate only derived caches for no-longer-valid episodes. Raw games,
    # results, actions and images are never deleted.
    valid_ids = {episode.game_id for episode in episodes}
    with db.connect() as con:
        cached_ids = {row[0] for row in con.execute("SELECT game_id FROM retrospective_audits")}
        for stale_id in cached_ids - valid_ids:
            con.execute("DELETE FROM retrospective_audits WHERE game_id=?", (stale_id,))
    for episode in episodes:
        data = cached_audit(db, episode)
        if not data or not data.get("valid"):
            continue
        stats["analyzed"] += 1
        stats["observed_ge999"] += int(episode.total >= 999)
        stats["potential_gt1000"] += int(data["could_exceed_1000"])
        stats["recoverable_gt1000"] += int(episode.total <= 1000 and data["could_exceed_1000"])
    return stats


def audit_summary(db: BingoDB) -> dict[str, int]:
    """Cheap UI aggregate: does not enumerate allocations and does not retrain."""
    totals = {"analyzed": 0, "potential_gt1000": 0, "recoverable_gt1000": 0,
              "observed_ge999": 0}
    with db.connect() as con:
        for row in con.execute("SELECT audit_json FROM retrospective_audits WHERE audit_version=?", (AUDIT_VERSION,)):
            try:
                data = json.loads(row[0])
                if not data.get("valid"):
                    continue
                totals["analyzed"] += 1
                totals["potential_gt1000"] += int(data["could_exceed_1000"])
                totals["recoverable_gt1000"] += int(data["observed_total"] <= 1000 and data["could_exceed_1000"])
                totals["observed_ge999"] += int(data["observed_total"] >= 999)
            except (ValueError, TypeError, KeyError):
                pass
    return totals
