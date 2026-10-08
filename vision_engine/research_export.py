from __future__ import annotations

import csv
import json
from pathlib import Path

from .experience import PolicyModelInfo
from .learning_status import build_learning_snapshot
from .persistence import BingoDB


def _write_table(out: Path, name: str, rows: list[dict]) -> None:
    (out / f"{name}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    # Always create the CSV companion, even when the dataset is empty. This
    # gives research tooling a stable export contract instead of making file
    # presence depend on whether a user has already completed a game.
    keys = sorted({k for row in rows for k in row}) if rows else []
    with (out / f"{name}.csv").open("w", newline="", encoding="utf-8") as fh:
        if keys:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)


def export_research_dataset(db: BingoDB, out: str | Path, info: PolicyModelInfo | None = None) -> Path:
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    games = db.game_rows(); decisions = db.counterfactual_rows()
    with db.connect() as con:
        actions = [dict(r) for r in con.execute("SELECT * FROM actions ORDER BY game_id,step")]
        events = [dict(r) for r in con.execute("SELECT * FROM state_events ORDER BY created_at")]
        telemetry = [dict(r) for r in con.execute("SELECT * FROM telemetry ORDER BY created_at")]
    for name, rows in (("episodes",games),("moves",actions),("candidates",decisions),("events",events),("latency_metrics",telemetry)):
        _write_table(out, name, rows)
    snapshot = build_learning_snapshot(db, info or PolicyModelInfo())
    (out / "model_metrics.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "pattern_analysis.json").write_text(json.dumps({
        "pattern_games": snapshot["pattern_games"], "stage": snapshot["pattern_stage"],
        "jewel_rates": snapshot["jewel_rates"], "repeated_layouts": snapshot["repeated_layouts"],
        "rng_influence_enabled": snapshot["rng_influence_enabled"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
