from __future__ import annotations

from _bootstrap import ROOT  # adds repository root to sys.path

import html
import json
from collections import Counter
from pathlib import Path

from vision_engine.persistence import BingoDB
from vision_engine.data_home import default_user_data_dir

ROOT = Path(__file__).resolve().parents[1]
OUT = default_user_data_dir() / "exports" / "dashboard.html"


def score_of(g):
    raw = g.get("verified_score_json") or g.get("computed_score_json")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def main() -> None:
    db = BingoDB(default_user_data_dir() / "bingo.sqlite3")
    games = [g for g in db.game_rows() if str(g.get("source", "")).startswith("local-")]
    completed = [g for g in games if score_of(g)]
    scores = [score_of(g) for g in completed]
    gt = sum((s or {}).get("total", 0) > 1000 for s in scores)
    l3 = sum((s or {}).get("lucky", 0) >= 3 for s in scores)
    verified = sum(bool(g.get("verified_score_json")) for g in completed)
    reps = db.repeated_layouts(2)
    with db.connect() as con:
        actions = [dict(r) for r in con.execute("SELECT * FROM actions ORDER BY game_id,step").fetchall()]
        events = [dict(r) for r in con.execute("SELECT * FROM state_events ORDER BY created_at").fetchall()]
    regrets = [float(a.get("solver_regret") or 0) for a in actions if a.get("solver_regret") is not None]
    adherence = [a for a in actions if a.get("recommended_pos") is not None]
    followed = sum(int(a["action_pos"]) == int(a["recommended_pos"]) for a in adherence)
    event_counts = Counter(e["kind"] for e in events)

    rows = []
    for g in completed[-30:][::-1]:
        s = score_of(g) or {}
        rows.append(
            "<tr>" + "".join(
                f"<td>{html.escape(str(v if v is not None else ''))}</td>" for v in (
                    g["id"], g.get("session_id"), g.get("box_color"), g.get("operation_mode"),
                    g.get("risk_profile"), s.get("lucky"), s.get("normal"), s.get("total"), g.get("board_hash")
                )
            ) + "</tr>"
        )

    def pct(a,b): return f"{100*a/b:.1f}%" if b else "—"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(f"""<!doctype html><meta charset='utf-8'><title>Jewel Bingo V18 Dashboard</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;max-width:1200px;margin:32px auto;padding:0 18px;color:#202124}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}}.card{{border:1px solid #ddd;border-radius:10px;padding:14px}}.big{{font-size:28px;font-weight:700}}table{{border-collapse:collapse;width:100%;font-size:13px;margin-top:16px}}th,td{{border-bottom:1px solid #ddd;padding:7px;text-align:left}}code{{background:#f5f5f5;padding:2px 5px}}</style>
<h1>Speedlora Jewel Bingo Assistant V7</h1>
<div class='cards'>
<div class='card'><div class='big'>{len(completed)}</div>completed local games</div>
<div class='card'><div class='big'>{pct(gt,len(completed))}</div>&gt;1000</div>
<div class='card'><div class='big'>{pct(l3,len(completed))}</div>3+ Lucky</div>
<div class='card'><div class='big'>{verified}</div>verified results</div>
<div class='card'><div class='big'>{pct(followed,len(adherence))}</div>followed top recommendation</div>
<div class='card'><div class='big'>{(sum(regrets)/len(regrets) if regrets else 0):.1f}</div>mean solver regret</div>
</div>
<h2>Data quality</h2>
<p>Vision corrections: <b>{event_counts.get('vision-correction',0)}</b> · desync events: <b>{event_counts.get('desync',0)}</b> · repeated layouts: <b>{len(reps)}</b></p>
<h2>Recent games</h2>
<table><thead><tr><th>Game</th><th>Session</th><th>Box</th><th>Mode</th><th>Risk</th><th>Lucky</th><th>Normal</th><th>Total</th><th>Board hash</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p>Generated locally from the persistent Jewel Bingo data directory.</p>""", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
