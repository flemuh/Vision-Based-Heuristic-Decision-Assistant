from __future__ import annotations

from _bootstrap import ROOT  # adds repository root to sys.path

import sqlite3
import tempfile
from pathlib import Path

from vision_engine import __version__
from vision_engine.advisor import Advisor
from vision_engine.board import Board
from vision_engine.experience import PolicyModelInfo
from vision_engine.learning_status import build_learning_snapshot
from vision_engine.persistence import BingoDB
from vision_engine.research_export import export_research_dataset
from vision_engine.round_lifecycle import CompletionDetector
from vision_engine.solver.protocol import PROTOCOL_VERSION
from vision_engine.temporal import TemporalStateFilter
from vision_engine.versioning import MODEL_VERSION, SCHEMA_VERSION, STRATEGY_PLANNER_VERSION


def validate_versions() -> None:
    assert __version__ == "18.0.26-STATE-INTEGRITY-P1"
    assert PROTOCOL_VERSION == 4
    assert SCHEMA_VERSION == 9
    assert MODEL_VERSION == 6
    assert STRATEGY_PLANNER_VERSION == 22
    print("V18 compatibility contracts: OK")


def validate_temporal_recovery() -> None:
    base = sum(1 << i for i in (1, 3, 7, 9, 10, 12))
    false_plus = base | (1 << 20)
    filt = TemporalStateFilter(
        frames=3,
        required=2,
        max_mask_jump=1,
        single_step_confirm_frames=4,
        single_step_rollback_frames=3,
        single_step_rollback_window_frames=10,
    )
    filt.reset(base)
    for _ in range(4):
        state = filt.observe(false_plus, "SO", 0.95)
    assert state.stable and state.mask == false_plus
    for _ in range(3):
        state = filt.observe(base, "SO", 0.95)
    assert state.stable and state.mask == base and state.reason.startswith("rollback-false-plus-one")
    print("false +1 rollback: OK")

    detector = CompletionDetector(min_accepted=12, disappear_required=2, recent_14_window_frames=10)
    mask13 = sum(1 << i for i in range(13))
    mask14 = mask13 | (1 << 14)
    assert not detector.observe(accepted_mask=mask13, raw_mask=mask14, jewel_visible=True).confirmed
    assert not detector.observe(accepted_mask=mask13, raw_mask=0, jewel_visible=False).confirmed
    done = detector.observe(accepted_mask=mask13, raw_mask=0, jewel_visible=False)
    assert done.confirmed and done.final_mask == mask14
    print("13/14 -> result-screen completion recovery: OK")


def validate_persistence_learning_and_export() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        old = root / "bingo.sqlite3"
        con = sqlite3.connect(old)
        con.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
        con.execute("INSERT INTO metadata(key,value) VALUES('schema_version','7')")
        con.commit()
        con.close()

        db = BingoDB(old)
        assert db.migration_backup_path is not None and db.migration_backup_path.exists()
        with db.connect() as con:
            schema = int(con.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()[0])
            tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert schema == SCHEMA_VERSION
        assert {"state_checkpoints", "migration_sources", "telemetry"} <= tables

        for ms, hit in ((12.0, True), (20.0, True), (45.0, False)):
            db.save_telemetry(None, "decision-roundtrip", ms, hit)
        telemetry = db.telemetry_summary("decision-roundtrip")
        assert telemetry["count"] == 3 and telemetry["p50"] == 20.0

        learning = build_learning_snapshot(
            db,
            PolicyModelInfo(),
            min_validation_games=8,
            rng_min_games=50,
            rng_blend=0.0,
        )
        assert learning["required_validation_games"] == 8
        assert learning["pattern_stage"] == "COLLECTING"
        assert not learning["rng_influence_enabled"]

        exported = export_research_dataset(db, root / "export")
        for name in (
            "episodes.csv",
            "moves.csv",
            "candidates.csv",
            "events.csv",
            "latency_metrics.csv",
            "model_metrics.json",
            "pattern_analysis.json",
        ):
            assert (exported / name).exists()
    print("schema 9 migration/backup + telemetry + learning/export: OK")


def validate_strategy() -> None:
    board = Board.from_rows([
        ["CH", "HA", "SO", "CH", "HA"],
        ["SO", "BL", "BL", "CR", "LI"],
        ["BL", "CH", None, "CR", "HA"],
        ["BL", "SO", "CH", "LI", "SO"],
        ["LI", "CR", "CR", "LI", "HA"],
    ])
    locked = Advisor(
        board,
        mode="auto",
        rollouts=60,
        plan_target="3 Lucky",
        plan_lucky_lines=("H", "V", "\\"),
        route_switch_ratio=100.0,
    ).recommend((1 << 6) | (1 << 13) | (1 << 22), "CH")
    assert locked[0].plan_target == "3 Lucky"
    assert locked[0].plan_on_route and locked[0].position != 3
    print("automatic locked-route planner: OK")

    impossible = Advisor(
        board,
        mode="auto",
        rollouts=60,
        plan_target="3 Lucky",
        plan_lucky_lines=("H", "V", "/"),
    ).recommend((1 << 1) | (1 << 3) | (1 << 5), "CH")
    assert impossible[0].plan_target != "3 Lucky"
    print(f"automatic fallback: OK -> {impossible[0].plan_target}")

    user_like = Board.from_rows([
        ["CR", "HA", "LI", "CR", "CR"],
        ["BL", "CH", "HA", "LI", "CH"],
        ["BL", "CR", None, "SO", "CH"],
        ["CH", "LI", "SO", "SO", "HA"],
        ["LI", "SO", "BL", "BL", "HA"],
    ])
    selected = sum(1 << p for p in (0, 1, 2, 3, 4, 6, 7, 8, 16, 20))
    switched = Advisor(
        user_like, mode="auto", rollouts=30,
        plan_target="3 Lucky", plan_lucky_lines=("H", "V", "\\"),
        route_switch_ratio=100.0,
    ).recommend(selected, "BL")
    assert switched[0].plan_target == "3 Lucky" and switched[0].route_switched
    assert switched[0].plan_probability > 0 and switched[0].target_probability > 0
    print("current-jewel emergency same-target route switch: OK")

    fallback = Advisor(
        user_like, mode="auto", rollouts=30,
        plan_target="3 Lucky", plan_lucky_lines=("H", "V", "\\"),
        route_switch_ratio=100.0,
    ).recommend(selected, "CR")
    assert fallback[0].plan_target == "2 Lucky + 1 Normal"
    print("current-jewel impossible target fallback: OK -> 2 Lucky + 1 Normal")

    harmony = Board.from_rows([
        ["SO", "BL", "BL", "BL", "CH"],
        ["CR", "LI", "CR", "SO", "CR"],
        ["CR", "HA", None, "LI", "BL"],
        ["CH", "HA", "CH", "SO", "LI"],
        ["LI", "HA", "HA", "CH", "SO"],
    ])
    hmask = sum(1 << p for p in (0, 2, 5, 7, 10, 13, 14))
    hrec = Advisor(
        harmony, mode="auto", rollouts=120,
        plan_target="3 Lucky", plan_lucky_lines=("H", "V", "\\"),
    ).recommend(hmask, "HA")
    assert harmony.rc(hrec[0].position) == (3, 2)
    assert hrec[0].lucky_clears_gained == 1
    assert len(hrec[0].plan_lucky_lines) == 3
    print("canonical route + immediate Lucky clear priority: OK -> HA L3C2")


def main() -> None:
    validate_versions()
    validate_temporal_recovery()
    validate_persistence_learning_and_export()
    validate_strategy()
    print("V18.0.26-STATE-INTEGRITY-P1 FINAL offline validation OK")


if __name__ == "__main__":
    main()
