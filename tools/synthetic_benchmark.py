from __future__ import annotations

from _bootstrap import ROOT  # adds repository root to sys.path

import argparse
import random

from vision_engine.advisor import Advisor
from vision_engine.board import Board
from vision_engine.constants import CENTER, JEWELS
from vision_engine.scoring import score_mask


def random_board(rng: random.Random) -> Board:
    pool = [j for j in JEWELS for _ in range(4)]
    rng.shuffle(pool)
    cells = []
    it = iter(pool)
    for i in range(25):
        cells.append(None if i == CENTER else next(it))
    return Board(tuple(cells))


def random_sequence(rng: random.Random) -> list[str]:
    pool = [j for j in JEWELS for _ in range(4)]
    rng.shuffle(pool)
    return pool[:14]


def main() -> None:
    ap = argparse.ArgumentParser(description="Synthetic regression benchmark for the online policy.")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--rollouts", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261002)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    totals=[]; l3=gt=0
    for n in range(args.games):
        board=random_board(rng); seq=random_sequence(rng); mask=0
        adv=Advisor(board, rollouts=args.rollouts, exact_horizon=6)
        for jewel in seq:
            rec=adv.recommend(mask,jewel)[0]
            mask |= 1 << rec.position
        s=score_mask(mask); totals.append(s.total); l3 += s.lucky>=3; gt += s.total>1000
        print(f"{n+1:03d}: {s.total} ({s.lucky}L/{s.normal}N)")
    print(f"\nmean={sum(totals)/len(totals):.1f} >1000={gt}/{len(totals)} 3L={l3}/{len(totals)}")


if __name__ == "__main__":
    main()
