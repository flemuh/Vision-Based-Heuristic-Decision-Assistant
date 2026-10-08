from __future__ import annotations

from _bootstrap import ROOT  # adds repository root to sys.path

import argparse
import random

from vision_engine.board import Board
from vision_engine.constants import JEWELS
from vision_engine.probability import p_three_lucky


def random_board(rng: random.Random) -> Board:
    pool = [j for j in JEWELS for _ in range(4)]
    rng.shuffle(pool)
    cells = []
    k = 0
    for i in range(25):
        if i == 12:
            cells.append(None)
        else:
            cells.append(pool[k]); k += 1
    return Board(tuple(cells))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Random-search full boards for high P(>=3 Lucky).")
    ap.add_argument("--iterations", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260927)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    best_p = -1.0
    best = None
    for n in range(1, args.iterations + 1):
        b = random_board(rng)
        p = p_three_lucky(b)
        if p > best_p:
            best_p, best = p, b
            print(f"[{n}] best P(>=3 Lucky)={p*100:.9f}%")
            print(b.pretty())
    print("\nFinal best:")
    print(f"{best_p*100:.9f}%")
    print(best.pretty())
