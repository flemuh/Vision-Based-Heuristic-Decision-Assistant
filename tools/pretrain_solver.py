from __future__ import annotations

from _bootstrap import ROOT  # adds repository root to sys.path

import argparse
import random
from pathlib import Path

import joblib
import numpy as np
from vision_engine.simple_ml import NumpyRidgeRegressor

from vision_engine.advisor import Advisor
from vision_engine.board import Board
from vision_engine.constants import CENTER, JEWELS
from vision_engine.experience import ExperiencePolicy, action_features

ROOT = Path(__file__).resolve().parents[1]


def random_board(rng: random.Random) -> Board:
    pool = [j for j in JEWELS for _ in range(4)]
    rng.shuffle(pool)
    cells=[]; it=iter(pool)
    for i in range(25):
        cells.append(None if i == CENTER else next(it))
    return Board(tuple(cells))


def main() -> None:
    ap=argparse.ArgumentParser(description="Distill exact solver action values into a generic local ML bootstrap model.")
    ap.add_argument("--states", type=int, default=120, help="Exact synthetic states. 500+ gives a stronger model but takes longer.")
    ap.add_argument("--seed", type=int, default=20261002)
    args=ap.parse_args()
    rng=random.Random(args.seed)
    xs=[]; ys=[]
    cols=ExperiencePolicy.UTILITY_COLUMNS
    for n in range(args.states):
        b=random_board(rng)
        # Exact solver is tractable near the end and produces noise-free labels.
        placed=rng.randint(7, 11)
        candidates=[i for i in range(25) if i != CENTER]
        rng.shuffle(candidates)
        mask=sum(1 << i for i in candidates[:placed])
        legal_jewels=[j for j in JEWELS if b.legal_positions(j,mask)]
        if not legal_jewels:
            continue
        jewel=rng.choice(legal_jewels)
        adv=Advisor(b, exact_horizon=14, rollouts=1)
        for rec in adv.recommend(mask,jewel):
            xs.append(action_features(b,mask,jewel,rec.position))
            u=rec.utility
            ys.append([u.p3,u.p2l1n,u.p1l2n,u.p1l1n,u.p_gt1000,u.p_ge999,u.expected_score])
        if (n+1) % 10 == 0:
            print(f"states {n+1}/{args.states} | action samples {len(xs)}")
    if len(xs) < 30:
        raise SystemExit("Not enough samples generated")
    model=NumpyRidgeRegressor(alpha=4.0)
    model.fit(np.stack(xs),np.asarray(ys,dtype=np.float64))
    out=ROOT/"data"/"bootstrap_solver_model.joblib"
    joblib.dump({"model":model,"states":args.states,"samples":len(xs),"utility_columns":cols},out, compress=3)
    print(f"Saved {out} ({len(xs)} solver-labelled action samples)")


if __name__ == "__main__":
    main()
