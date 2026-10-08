from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from functools import lru_cache
from itertools import combinations, product
from typing import Any, Iterable

from .board import Board
from .constants import JEWELS, MAX_DRAWS
from .persistence import BingoDB
from .scoring import score_mask


def _load_json(raw: str | None, fallback=None):
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except Exception:
        return fallback


def _score_from_game(game) -> dict | None:
    raw = game["verified_score_json"] or game["computed_score_json"]
    value = _load_json(raw)
    return value if isinstance(value, dict) else None


def _local_time_text(raw: str | None) -> str:
    if not raw:
        return "?"
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.astimezone()
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return raw[:19].replace("T", " ")


def list_stored_games(db: BingoDB, limit: int = 100) -> list[dict[str, Any]]:
    """Return newest recorded games that have at least one accepted action.

    This is intentionally read-only and uses the same persistent SQLite database
    as live play. It powers the Game History UI without changing solver state.
    """
    with db.connect() as con:
        rows = con.execute(
            """
            SELECT g.*,
                   COUNT(a.id) AS action_count,
                   SUM(CASE WHEN a.recommended_pos IS NOT NULL THEN 1 ELSE 0 END) AS comparable_count,
                   SUM(CASE WHEN a.recommended_pos IS NOT NULL AND a.recommended_pos=a.action_pos THEN 1 ELSE 0 END) AS followed_count
            FROM games g
            JOIN actions a ON a.game_id=g.id
            GROUP BY g.id
            ORDER BY g.created_at DESC
            LIMIT ?
            """,
            (max(1, int(limit)),),
        ).fetchall()

    out: list[dict[str, Any]] = []
    for row in rows:
        score = _score_from_game(row) or {}
        out.append({
            "game_id": row["id"],
            "created_at": row["created_at"],
            "created_local": _local_time_text(row["created_at"]),
            "status": row["status"] or ("complete" if row["finished_at"] else "playing"),
            "box_color": row["box_color"],
            "actions": int(row["action_count"] or 0),
            "comparable": int(row["comparable_count"] or 0),
            "followed": int(row["followed_count"] or 0),
            "score": score,
            "total": score.get("total"),
            "lucky": score.get("lucky"),
            "normal": score.get("normal"),
        })
    return out


@lru_cache(maxsize=128)
def _terminal_masks_for_counts(
    cells: tuple[str | None, ...], counts: tuple[int, ...]
) -> tuple[int, ...]:
    board = Board(cells)
    choices: list[tuple[int, ...]] = []
    for jewel, count in zip(JEWELS, counts):
        positions = board.positions(jewel)
        if count < 0 or count > len(positions):
            return ()
        jewel_masks = []
        for combo in combinations(positions, count):
            mask = 0
            for pos in combo:
                mask |= 1 << pos
            jewel_masks.append(mask)
        choices.append(tuple(jewel_masks))

    masks: list[int] = []
    for parts in product(*choices):
        mask = 0
        for part in parts:
            mask |= part
        masks.append(mask)
    return tuple(masks)


def _coords(positions: Iterable[int]) -> list[str]:
    out = []
    for pos in positions:
        r, c = Board.rc(int(pos))
        out.append(f"L{r}C{c}")
    return out


def _first_irreversible_loss(
    board: Board,
    actions: list[dict[str, Any]],
    winning_masks: tuple[int, ...],
) -> dict[str, Any] | None:
    """Find first accepted move that makes a hindsight target unreachable.

    `winning_masks` is built using the *actual complete jewel sequence*, so this
    is retrospective diagnosis only. It must not be described as information the
    live planner could have known at the time of the move.
    """
    if not winning_masks:
        return None
    prefix = 0
    for idx, action in enumerate(actions):
        try:
            pos = int(action["action_pos"])
            jewel = str(action["jewel"])
        except Exception:
            continue
        before = any((mask & prefix) == prefix for mask in winning_masks)
        after_mask = prefix | (1 << pos)
        after = any((mask & after_mask) == after_mask for mask in winning_masks)
        if before and not after:
            alternatives = []
            for candidate in board.legal_positions(jewel, prefix):
                if candidate == pos:
                    continue
                candidate_prefix = prefix | (1 << candidate)
                if any((mask & candidate_prefix) == candidate_prefix for mask in winning_masks):
                    alternatives.append(candidate)
            r, c = Board.rc(pos)
            recommended = action.get("recommended_pos")
            rec_coord = None
            if recommended is not None:
                rr, rc = Board.rc(int(recommended))
                rec_coord = f"L{rr}C{rc}"
            return {
                "step": idx + 1,
                "jewel": jewel,
                "actual_pos": pos,
                "actual": f"L{r}C{c}",
                "recommended_pos": recommended,
                "recommended": rec_coord,
                "alternatives": alternatives,
                "alternative_coords": _coords(alternatives),
            }
        prefix = after_mask
    return None


def actual_sequence_hindsight(
    board: Board,
    sequence: list[str],
    actions: list[dict[str, Any]],
    selected_mask: int | None = None,
) -> dict[str, Any]:
    """Exact retrospective result for the jewel sequence that actually arrived.

    Each jewel exists four times on the board, so a fixed 14-draw composition has
    at most 6^6 = 46,656 terminal allocations. Exhaustive evaluation is therefore
    cheap enough for an on-demand historical replay and avoids Monte Carlo here.
    """
    seq = [str(x) for x in sequence if str(x) in JEWELS]
    if len(seq) != MAX_DRAWS:
        return {
            "available": False,
            "reason": f"needs a complete {MAX_DRAWS}-jewel sequence; recorded {len(seq)}",
        }
    counts_counter = Counter(seq)
    counts = tuple(int(counts_counter.get(j, 0)) for j in JEWELS)
    if any(c > 4 for c in counts):
        return {"available": False, "reason": "recorded sequence contains more than four copies of a jewel"}

    terminal_masks = _terminal_masks_for_counts(board.cells, counts)
    if not terminal_masks:
        return {"available": False, "reason": "no legal terminal allocation matches the recorded sequence"}

    scored = [(mask, score_mask(mask)) for mask in terminal_masks]
    best_mask, best_score = max(scored, key=lambda item: (item[1].total, item[1].lucky, item[1].normal))
    three_lucky = tuple(mask for mask, score in scored if score.lucky >= 3)
    over_1000 = tuple(mask for mask, score in scored if score.total > 1000)

    action_mask = 0
    for action in actions:
        try:
            action_mask |= 1 << int(action["action_pos"])
        except Exception:
            pass
    actual_mask = int(selected_mask) if selected_mask is not None and int(selected_mask).bit_count() == MAX_DRAWS else action_mask
    actual_score = score_mask(actual_mask) if actual_mask.bit_count() == MAX_DRAWS else None

    return {
        "available": True,
        "sequence": seq,
        "counts": dict(zip(JEWELS, counts)),
        "terminal_allocations": len(terminal_masks),
        "best_total": best_score.total,
        "best_lucky": best_score.lucky,
        "best_normal": best_score.normal,
        "best_mask": best_mask,
        "best_cells": _coords(i for i in range(25) if (best_mask >> i) & 1),
        "three_lucky_possible": bool(three_lucky),
        "three_lucky_allocations": len(three_lucky),
        "over_1000_possible": bool(over_1000),
        "over_1000_allocations": len(over_1000),
        "first_loss_three_lucky": _first_irreversible_loss(board, actions, three_lucky),
        "first_loss_over_1000": _first_irreversible_loss(board, actions, over_1000),
        "actual_total": None if actual_score is None else actual_score.total,
        "actual_lucky": None if actual_score is None else actual_score.lucky,
        "actual_normal": None if actual_score is None else actual_score.normal,
        "actual_is_best_score": actual_score is not None and actual_score.total == best_score.total,
    }


def stored_replay(db: BingoDB, game_id: str | None = None) -> dict[str, Any] | None:
    with db.connect() as con:
        if game_id:
            game = con.execute("SELECT * FROM games WHERE id=?", (game_id,)).fetchone()
        else:
            game = con.execute(
                "SELECT * FROM games WHERE id IN (SELECT game_id FROM actions) ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        if game is None:
            return None
        actions = [dict(r) for r in con.execute(
            "SELECT * FROM actions WHERE game_id=? ORDER BY step", (game["id"],)
        ).fetchall()]
        decisions = [dict(r) for r in con.execute(
            "SELECT * FROM decision_samples WHERE game_id=? ORDER BY step,rank,candidate_pos", (game["id"],)
        ).fetchall()]

    by_step: dict[int, list[dict[str, Any]]] = {}
    for decision in decisions:
        by_step.setdefault(int(decision["step"]), []).append(decision)

    followed = 0
    comparable = 0
    total_regret = 0.0
    steps = []
    for action in actions:
        actual = int(action["action_pos"])
        recommended = action.get("recommended_pos")
        if recommended is not None:
            comparable += 1
            followed += int(actual == int(recommended))
        total_regret += float(action.get("solver_regret") or 0.0)
        ar, ac = Board.rc(actual)
        rec = None
        if recommended is not None:
            rr, rc = Board.rc(int(recommended))
            rec = f"L{rr}C{rc}"
        candidates = []
        for d in by_step.get(int(action["step"]), [])[:5]:
            rr, rc = Board.rc(int(d["candidate_pos"]))
            candidates.append({
                "position": int(d["candidate_pos"]),
                "coord": f"L{rr}C{rc}",
                "chosen": int(d["candidate_pos"]) == actual,
                "rank": d.get("rank"),
                "active_goal": d.get("active_goal"),
                "p3": float(d.get("p3") or 0.0),
                "p_gt1000": float(d.get("p_gt1000") or 0.0),
                "p_ge999": float(d.get("p_ge999") or 0.0),
                "expected_score": float(d.get("expected_score") or 0.0),
                "solver_regret": float(d.get("solver_regret") or 0.0),
            })
        steps.append({
            "step": int(action["step"]) + 1,
            "jewel": action["jewel"],
            "actual_pos": actual,
            "actual": f"L{ar}C{ac}",
            "recommended_pos": recommended,
            "recommended": rec,
            "followed": recommended is not None and actual == int(recommended),
            "regret": float(action.get("solver_regret") or 0.0),
            "phase": action.get("phase"),
            "strategy": action.get("strategy"),
            "candidates": candidates,
        })

    score = _score_from_game(game)
    board = None
    hindsight = {"available": False, "reason": "board not recorded"}
    board_cells = _load_json(game["board_json"])
    if isinstance(board_cells, list) and len(board_cells) == 25:
        try:
            board = Board(tuple(board_cells))
        except Exception:
            board = None
    sequence = _load_json(game["sequence_json"], [])
    if not isinstance(sequence, list) or len(sequence) < len(actions):
        sequence = [str(a["jewel"]) for a in actions]
    if board is not None:
        hindsight = actual_sequence_hindsight(
            board,
            sequence,
            actions,
            selected_mask=game["selected_mask"],
        )

    return {
        "game_id": game["id"],
        "created_at": game["created_at"],
        "created_local": _local_time_text(game["created_at"]),
        "status": game["status"],
        "box_color": game["box_color"],
        "board": board,
        "steps": steps,
        "actions": len(actions),
        "comparable": comparable,
        "followed": followed,
        "total_regret": total_regret,
        "score": score,
        "hindsight": hindsight,
    }


def format_replay_report(replay: dict[str, Any]) -> str:
    score = replay.get("score") or {}
    total = score.get("total", "?")
    lucky = score.get("lucky", "?")
    normal = score.get("normal", "?")
    lines = [
        f"Game: {replay['game_id']}",
        f"Started: {replay.get('created_local') or '?'} | Box: {replay.get('box_color') or 'not set'}",
        f"Recorded actions: {replay['actions']}",
        f"Recommendation followed: {replay['followed']}/{replay['comparable']}",
        f"Final score: {total} | Lucky {lucky} | Normal {normal}",
        f"Cumulative stored solver regret: {replay['total_regret']:.1f}",
    ]

    hindsight = replay.get("hindsight") or {}
    lines.extend(["", "ACTUAL-SEQUENCE HINDSIGHT (retrospective; not information available live)"])
    if not hindsight.get("available"):
        lines.append(f"Unavailable: {hindsight.get('reason', 'incomplete historical data')}")
    else:
        lines.append(
            f"Best possible with the jewels that actually arrived: {hindsight['best_total']} pts | "
            f"Lucky {hindsight['best_lucky']} | Normal {hindsight['best_normal']}"
        )
        lines.append(
            f"3 Lucky possible: {'YES' if hindsight['three_lucky_possible'] else 'NO'} "
            f"({hindsight['three_lucky_allocations']} legal final allocations) | "
            f">1000 possible: {'YES' if hindsight['over_1000_possible'] else 'NO'} "
            f"({hindsight['over_1000_allocations']} allocations)"
        )
        lines.append(
            "ML reward: INCLUDED automatically from stored candidate states. "
            "Reopening this replay does not duplicate its weight."
        )
        if hindsight.get("actual_total") is not None:
            lines.append(
                f"Actual accepted final mask: {hindsight['actual_total']} pts | "
                f"Lucky {hindsight['actual_lucky']} | Normal {hindsight['actual_normal']} | "
                f"best-score allocation: {'YES' if hindsight['actual_is_best_score'] else 'NO'}"
            )
        for label, key in (("3 Lucky", "first_loss_three_lucky"), (">1000", "first_loss_over_1000")):
            loss = hindsight.get(key)
            if not (hindsight.get("three_lucky_possible") if key.endswith("three_lucky") else hindsight.get("over_1000_possible")):
                lines.append(f"First irreversible loss of {label}: n/a — target was impossible for this draw sequence.")
            elif loss is None:
                lines.append(f"First irreversible loss of {label}: none in the recorded actions.")
            else:
                alts = ", ".join(loss.get("alternative_coords") or []) or "none"
                planner = f" | live suggestion {loss['recommended']}" if loss.get("recommended") else ""
                lines.append(
                    f"First irreversible loss of {label}: move {loss['step']} {loss['jewel']} -> {loss['actual']}"
                    f"{planner} | hindsight-preserving alternatives: {alts}"
                )

    lines.extend(["", "MOVES (saved live decision vs accepted placement)"])
    for row in replay.get("steps", [])[:MAX_DRAWS]:
        mark = "FOLLOWED" if row["followed"] else "DIFF"
        lines.append(
            f"{row['step']:02d}. {row['jewel']} actual {row['actual']} | "
            f"suggested {row['recommended'] or '?'} | {mark} | regret {row['regret']:.1f}"
        )
        for candidate in row.get("candidates", [])[:3]:
            chosen = "*" if candidate["chosen"] else " "
            lines.append(
                f"    {chosen}#{candidate.get('rank') or '?'} {candidate['coord']} | "
                f"target {candidate.get('active_goal') or '?'} | P3 {candidate['p3']*100:.1f}% | "
                f">1000 {candidate['p_gt1000']*100:.1f}% | E {candidate['expected_score']:.1f}"
            )
    return "\n".join(lines)
