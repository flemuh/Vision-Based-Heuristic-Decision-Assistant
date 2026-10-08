from __future__ import annotations

from _bootstrap import ROOT  # adds repository root to sys.path

import argparse
import json
import shutil
from pathlib import Path

from vision_engine.config import AppConfig
from vision_engine.persistence import BingoDB
from vision_engine.vision import VisionConfig

ROOT = Path(__file__).resolve().parents[1]


def copy_tree_files(src: Path, dst: Path, pattern: str = "*", tag: str | None = None) -> int:
    if not src.exists():
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in src.glob(pattern):
        if not p.is_file():
            continue
        if tag:
            # Preserve the jewel prefix expected by the classifier (e.g. BL_*),
            # while avoiding collisions with templates already collected in V4.
            parts = p.stem.split("_", 1)
            prefix = parts[0]
            rest = parts[1] if len(parts) > 1 else p.stem
            target = dst / f"{prefix}_{tag}_{rest}{p.suffix}"
            i = 2
            while target.exists():
                target = dst / f"{prefix}_{tag}{i}_{rest}{p.suffix}"
                i += 1
        else:
            target = dst / p.name
            if target.exists():
                continue
        shutil.copy2(p, target)
        n += 1
    return n


def db_game_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        import sqlite3
        with sqlite3.connect(path) as con:
            row = con.execute("SELECT COUNT(*) FROM games").fetchone()
            return int(row[0]) if row else 0
    except Exception:
        return 0


def main() -> None:
    ap=argparse.ArgumentParser(description="Copy calibration/local data from a V3 folder into V4.")
    ap.add_argument("v3_folder")
    args=ap.parse_args()
    old=Path(args.v3_folder).resolve()
    if not (old/"data").exists():
        raise SystemExit("V3 data folder not found")
    newdata=ROOT/"data"

    # Preserve V4 defaults but carry over fields understood by the new config.
    old_app=AppConfig.load(old/"data"/"app_config.json")
    new_app=AppConfig.load(newdata/"app_config.json")
    for name in ("screen_region","exact_horizon","rollouts","empirical_bias_blend","rng_min_games","poll_ms","auto_relocate","relocate_threshold"):
        setattr(new_app,name,getattr(old_app,name))
    new_app.save(newdata/"app_config.json")

    old_v=VisionConfig.load(old/"data"/"vision_config.json")
    new_v=VisionConfig.load(newdata/"vision_config.json")
    for name in ("board_roi","current_roi","remaining_roi","result_lucky_roi","result_normal_roi","result_jewel_roi","result_total_roi","selected_blue_threshold"):
        setattr(new_v,name,getattr(old_v,name))
    new_v.save(newdata/"vision_config.json")

    nt=copy_tree_files(old/"data"/"templates",newdata/"templates","*.png", tag="v3")
    copy_tree_files(old/"data"/"game_images",newdata/"game_images","*")
    copy_tree_files(old/"data"/"decision_images",newdata/"decision_images","*")
    if (old/"data"/"panel_anchor.png").exists():
        shutil.copy2(old/"data"/"panel_anchor.png",newdata/"panel_anchor.png")
    old_db = old/"data"/"bingo.sqlite3"
    new_db = newdata/"bingo.sqlite3"
    if old_db.exists():
        # If V4 was opened once before migration it may have created an empty DB.
        # Replace only an empty target; never destroy already-collected V4 games.
        if not new_db.exists() or db_game_count(new_db) == 0:
            if new_db.exists():
                new_db.unlink()
            shutil.copy2(old_db, new_db)
        else:
            print("V4 already contains games; keeping its DB. Merge the V3 DB manually/export datasets if needed.")
    # Opening creates/migrates the schema to V4.
    BingoDB(new_db)
    print(f"Migration complete. Added {nt} template images. V3 policy model was not copied because V4 uses different features/targets.")


if __name__ == "__main__":
    main()
