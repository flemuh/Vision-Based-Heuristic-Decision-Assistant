from __future__ import annotations

from _bootstrap import ROOT  # adds repository root to sys.path

import argparse
import json
from pathlib import Path

from vision_engine.board import Board
from vision_engine.history import HistoryStore
from vision_engine.hindsight import hindsight_metrics
from vision_engine.probability import p_at_least_two_lucky, p_three_lucky, triple_requirements

ROOT = Path(__file__).resolve().parents[1]


def load_boards():
    raw = json.loads((ROOT / "data" / "boards.json").read_text(encoding="utf-8"))
    return {name: Board.from_rows(rows) for name, rows in raw.items()}


def covers_three_lucky(board: Board, counts: list[int]) -> bool:
    for _, req in triple_requirements(board):
        if all(counts[i] >= req[i] for i in range(6)):
            return True
    return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hindsight", action="store_true", help="Also run exact ~1.96M-mask full-score evaluation per board (slow).")
    args = ap.parse_args()
    hist = HistoryStore(ROOT / "data" / "history.json").load()
    for name, board in load_boards().items():
        coverage = sum(1 for row in hist if row.get("counts") and covers_three_lucky(board, row["counts"]))
        print(f"\n{name}")
        print(board.pretty())
        print(f"P(>=3 Lucky): {p_three_lucky(board)*100:.9f}%")
        print(f"P(>=2 Lucky): {p_at_least_two_lucky(board)*100:.9f}%")
        print(f"Observed 3-Lucky-feasible profiles: {coverage}/{len(hist)}")
        if args.hindsight:
            hm = hindsight_metrics(board)
            print(f"Hindsight upper bound P(score>1000): {hm['p_gt1000']*100:.9f}%")
            print(f"Hindsight upper bound P(score>=999): {hm['p_ge999']*100:.9f}%")
            print(f"Hindsight expected best score: {hm['expected_best_score']:.3f}")
