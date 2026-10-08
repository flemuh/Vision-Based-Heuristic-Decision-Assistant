from __future__ import annotations

from _bootstrap import ROOT  # adds repository root to sys.path

import argparse
import json
from pathlib import Path

from vision_engine.board import Board
from vision_engine.probability import p_at_least_two_lucky, p_three_lucky, triple_requirements


def load_board(path: Path) -> Board:
    d = json.loads(path.read_text(encoding="utf-8"))
    return Board.from_rows(d["rows"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("board_json", type=Path)
    args = ap.parse_args()
    board = load_board(args.board_json)
    print(board.pretty())
    print(f"P(>=3 Lucky): {p_three_lucky(board)*100:.9f}%")
    print(f"P(>=2 Lucky): {p_at_least_two_lucky(board)*100:.9f}%")
    print("3-line requirements:")
    for names, req in triple_requirements(board):
        print(names, req)
