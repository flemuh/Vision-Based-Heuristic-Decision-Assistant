from __future__ import annotations

from _bootstrap import ROOT  # adds repository root to sys.path

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from vision_engine.constants import JEWELS
from vision_engine.learning import pattern_signature
from vision_engine.persistence import BingoDB
from vision_engine.probability import all_count_states

ROOT = Path(__file__).resolve().parents[1]
DB = BingoDB(ROOT / "data" / "bingo.sqlite3")


def main() -> None:
    games = DB.game_rows()
    local = [g for g in games if str(g.get("source", "")).startswith("local-")]
    completed = [g for g in local if g.get("computed_score_json")]
    verified = [g for g in completed if g.get("verified_score_json")]
    print(f"Games in DB: {len(games)} | local: {len(local)} | completed: {len(completed)} | verified: {len(verified)}")

    print("\nRepeated exact layouts:")
    reps = DB.repeated_layouts(2)
    if not reps:
        print("  none yet")
    for row in reps:
        print(f"  {row['board_hash']}: {row['games']} games")

    colors = Counter(g.get("box_color") for g in local if g.get("box_color"))
    if colors:
        print("\nBox colors:")
        for color, n in colors.most_common():
            print(f"  {color}: {n}")

    mismatches = []
    for g in verified:
        try:
            c = json.loads(g["computed_score_json"])
            v = json.loads(g["verified_score_json"])
            if v.get("total") is not None and c.get("total") != v.get("total"):
                mismatches.append((g["id"], c.get("total"), v.get("total")))
        except Exception:
            pass
    print(f"\nComputed-vs-verified total mismatches: {len(mismatches)}")
    for row in mismatches[:20]:
        print(" ", row)

    decisions = DB.counterfactual_rows()
    print(f"\nCounterfactual decision samples: {len(decisions)}")
    by_game = Counter(r["game_id"] for r in decisions)
    if by_game:
        print(f"Games with solver samples: {len(by_game)}")

    with DB.connect() as con:
        actions = [dict(r) for r in con.execute("SELECT * FROM actions ORDER BY game_id,step").fetchall()]
        events = [dict(r) for r in con.execute("SELECT * FROM state_events ORDER BY created_at").fetchall()]
    regrets = [float(a["solver_regret"]) for a in actions if a.get("solver_regret") is not None]
    recommended = [a for a in actions if a.get("recommended_pos") is not None]
    followed = sum(int(a["recommended_pos"]) == int(a["action_pos"]) for a in recommended)
    print(f"Actions: {len(actions)} | top recommendation followed: {followed}/{len(recommended)}" if recommended else f"Actions: {len(actions)}")
    if regrets:
        print(f"Mean solver regret: {np.mean(regrets):.2f} | median {np.median(regrets):.2f}")
    ec = Counter(e["kind"] for e in events)
    print(f"Vision corrections: {ec.get('vision-correction',0)} | desync events: {ec.get('desync',0)} | remaining mismatches: {ec.get('remaining-mismatch',0)}")

    # Session -> board hashes tests persistence/template hypotheses.
    session_boards = defaultdict(list)
    for g in local:
        if g.get("session_id") and g.get("board_hash"):
            session_boards[g["session_id"]].append(g["board_hash"])
    print("\nSession layout persistence:")
    shown = 0
    for session, hashes in session_boards.items():
        if len(hashes) >= 2:
            counts = Counter(hashes)
            print(f"  {session}: {len(hashes)} games, {len(counts)} unique layouts, most common {counts.most_common(1)[0]}")
            shown += 1
    if not shown:
        print("  need >=2 local games in the same app session")

    # RNG research uses consecutive LOCAL games only. Chat-selected examples are
    # intentionally excluded because they can create selection bias.
    count_rows = []
    for g in completed:
        try:
            c = json.loads(g["counts_json"]) if g.get("counts_json") else None
            if c and len(c) == 6 and sum(c) == 14:
                count_rows.append(tuple(map(int, c)))
        except Exception:
            pass
    print("\nLocal RNG sample:")
    if not count_rows:
        print("  no complete local count vectors yet")
        return
    totals = np.asarray(count_rows).sum(axis=0)
    print("  totals:", dict(zip(JEWELS, map(int, totals))))
    print("  signatures:", Counter(pattern_signature(x) for x in count_rows).most_common())
    # Exact theoretical probability of each count signature under 14 of 24.
    sig_prob = defaultdict(float)
    for counts, w in all_count_states():
        sig_prob[pattern_signature(counts)] += w
    observed = Counter(pattern_signature(x) for x in count_rows)
    for sig, n in observed.most_common(8):
        print(f"  {sig}: observed {n}/{len(count_rows)}; theoretical per-game {100*sig_prob[sig]:.2f}%")
    if len(count_rows) < 50:
        print("  NOTE: <50 consecutive local games: do not enable empirical RNG bias yet.")


if __name__ == "__main__":
    main()
