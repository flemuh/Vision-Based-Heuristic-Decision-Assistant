from __future__ import annotations

from _bootstrap import ROOT  # adds repository root to sys.path

import argparse
import json
from pathlib import Path

from vision_engine.board import Board
from vision_engine.persistence import BingoDB
from vision_engine.data_home import default_user_data_dir

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay a recorded game and inspect actual vs recommended moves.")
    ap.add_argument("game_id", nargs="?", help="Game id. Omit for latest game with actions.")
    args = ap.parse_args()
    db = BingoDB(default_user_data_dir() / "bingo.sqlite3")
    with db.connect() as con:
        if args.game_id:
            game = con.execute("SELECT * FROM games WHERE id=?", (args.game_id,)).fetchone()
        else:
            game = con.execute("SELECT * FROM games WHERE id IN (SELECT game_id FROM actions) ORDER BY created_at DESC LIMIT 1").fetchone()
        if game is None:
            print("Game not found.")
            return
        actions = con.execute("SELECT * FROM actions WHERE game_id=? ORDER BY step", (game["id"],)).fetchall()
        decisions = con.execute("SELECT * FROM decision_samples WHERE game_id=? ORDER BY step,rank,candidate_pos", (game["id"],)).fetchall()

    print(f"Game: {game['id']} | session={game['session_id']} | box={game['box_color']} | mode={game['operation_mode']} | risk={game['risk_profile']}")
    if game["board_json"]:
        print(Board(tuple(json.loads(game["board_json"]))).pretty())
    by_step = {}
    for d in decisions:
        by_step.setdefault(int(d["step"]), []).append(d)
    total_regret = 0.0
    for a in actions:
        step = int(a["step"])
        pos = int(a["action_pos"])
        rr, cc = Board.rc(pos)
        recpos = a["recommended_pos"]
        rec_text = "?"
        if recpos is not None:
            r2,c2 = Board.rc(int(recpos)); rec_text=f"L{r2}C{c2}"
        regret = float(a["solver_regret"] or 0.0)
        total_regret += regret
        print(f"\n{step+1:02d}. {a['jewel']} -> actual L{rr}C{cc} | recommended {rec_text} | regret {regret:.1f}")
        for d in by_step.get(step, [])[:4]:
            r,c = Board.rc(int(d["candidate_pos"]))
            mark = "*" if int(d["candidate_pos"]) == pos else " "
            print(
                f"  {mark} L{r}C{c}: P3={100*d['p3']:.2f}% >1000={100*d['p_gt1000']:.2f}% "
                f"E={d['expected_score']:.1f} target={d['active_goal']} regret={d['solver_regret']:.1f}"
            )
    print(f"\nCumulative solver regret: {total_regret:.1f}")


if __name__ == "__main__":
    main()
