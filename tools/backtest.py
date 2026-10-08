from __future__ import annotations

from _bootstrap import ROOT  # adds repository root to sys.path

import argparse
import json
from pathlib import Path

from vision_engine.advisor import Advisor
from vision_engine.board import Board
from vision_engine.config import AppConfig, RISK_PROFILES
from vision_engine.persistence import BingoDB
from vision_engine.scoring import score_mask

ROOT = Path(__file__).resolve().parents[1]


def make_advisor(board: Board, profile: str, rollouts: int, exact_horizon: int) -> Advisor:
    p = RISK_PROFILES.get(profile, RISK_PROFILES["adaptive"])
    return Advisor(
        board,
        mode={"adaptive": "adaptive", "score": "score", "lucky": "lucky"}.get(profile, "adaptive"),
        rollouts=rollouts,
        exact_horizon=exact_horizon,
        p3_min=p["p3_min"],
        p2l1n_min=p["p2l1n_min"],
        p1l2n_min=p["p1l2n_min"],
        p1l1n_min=p["p1l1n_min"],
        fallback_override_pp=p["fallback_override_pp"],
    )


def simulate(board: Board, sequence: list[str], profile: str, rollouts: int, exact_horizon: int):
    mask = 0
    advisor = make_advisor(board, profile, rollouts, exact_horizon)
    trace = []
    for step, jewel in enumerate(sequence[:14]):
        recs = advisor.recommend(mask, jewel)
        if not recs:
            break
        best = recs[0]
        trace.append((step, jewel, best.position, best.active_goal, best.utility.p3, best.utility.p_gt1000))
        mask |= 1 << best.position
    return mask, score_mask(mask), trace


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest V5.6 policy on locally recorded jewel sequences.")
    ap.add_argument("--profile", choices=list(RISK_PROFILES), default="adaptive")
    ap.add_argument("--rollouts", type=int, default=600)
    ap.add_argument("--exact-horizon", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    db = BingoDB(ROOT / "data" / "bingo.sqlite3")
    rows = [g for g in db.game_rows() if g.get("board_json") and g.get("sequence_json")]
    if args.limit > 0:
        rows = rows[-args.limit:]
    if not rows:
        print("No recorded board+sequence games yet.")
        return

    results = []
    for g in rows:
        try:
            board = Board(tuple(json.loads(g["board_json"])))
            seq = list(json.loads(g["sequence_json"]))
            if len(seq) < 14:
                continue
            mask, score, _ = simulate(board, seq, args.profile, args.rollouts, args.exact_horizon)
            actual = None
            raw = g.get("verified_score_json") or g.get("computed_score_json")
            if raw:
                actual = json.loads(raw).get("total")
            results.append((g["id"], score.total, score.lucky, score.normal, actual))
            print(f"{g['id']}: policy={score.total} ({score.lucky}L/{score.normal}N) actual={actual}")
        except Exception as e:
            print(f"{g.get('id')}: skipped: {e}")

    if not results:
        print("No complete usable games.")
        return
    n = len(results)
    gt1000 = sum(x[1] > 1000 for x in results)
    l3 = sum(x[2] >= 3 for x in results)
    avg = sum(x[1] for x in results) / n
    print("\nSummary")
    print(f"  games: {n}")
    print(f"  profile: {args.profile}")
    print(f"  >1000: {gt1000}/{n} ({100*gt1000/n:.1f}%)")
    print(f"  >=3 Lucky: {l3}/{n} ({100*l3/n:.1f}%)")
    print(f"  average score: {avg:.1f}")


if __name__ == "__main__":
    main()
