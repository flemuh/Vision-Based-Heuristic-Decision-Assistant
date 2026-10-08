from __future__ import annotations

from _bootstrap import ROOT  # adds repository root to sys.path

import argparse
from pathlib import Path

import cv2

from vision_engine.advisor import Advisor
from vision_engine.config import AppConfig
from vision_engine.vision import JewelClassifier, VisionConfig, read_board_detailed, read_current_detailed
from vision_engine.board import Board

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze a saved game-panel screenshot without the MU client running.")
    ap.add_argument("image")
    args = ap.parse_args()
    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit("Could not read image")
    vcfg = VisionConfig.load(ROOT / "data" / "vision_config.json")
    acfg = AppConfig.load(ROOT / "data" / "app_config.json")
    clf = JewelClassifier(ROOT / "data" / "templates")
    d = read_board_detailed(img, vcfg, clf)
    pred, _ = read_current_detailed(img, vcfg, clf)
    print(d.board.pretty(d.selected_mask))
    print(f"board confidence={d.board_confidence:.3f} min-cell={d.min_assigned_probability:.3f} ambiguous={d.ambiguous_cells}")
    print(f"current={pred.label} conf={pred.confidence:.3f} margin={pred.margin:.3f} entropy={pred.entropy:.3f}")
    adv = Advisor(
        d.board, mode=acfg.mode, exact_horizon=acfg.exact_horizon, rollouts=acfg.rollouts,
        p3_min=acfg.p3_min, p2l1n_min=acfg.p2l1n_min, p1l2n_min=acfg.p1l2n_min,
        p1l1n_min=acfg.p1l1n_min, fallback_override_pp=acfg.fallback_override_pp,
        p3_near_best_tolerance=acfg.p3_near_best_tolerance,
    )
    print("\nRecommendations:")
    for i,r in enumerate(adv.recommend(d.selected_mask, pred.label),1):
        rr,cc=Board.rc(r.position); u=r.utility
        print(f"{i}. L{rr}C{cc} target={r.active_goal} P3={100*u.p3:.2f}% >1000={100*u.p_gt1000:.2f}% E={u.expected_score:.1f}")


if __name__ == "__main__":
    main()
