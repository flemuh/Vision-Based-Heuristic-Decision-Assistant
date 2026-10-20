from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_release_validator_covers_final_stabilization():
    source = (ROOT / "tools" / "validate_release.py").read_text(encoding="utf-8")
    for marker in (
        "V18 compatibility contracts",
        "false +1 rollback",
        "result-screen completion recovery",
        "migration/backup + telemetry + learning/export",
        "automatic locked-route planner",
        "automatic fallback",
        "V18.0.26-STATE-INTEGRITY-P1 FINAL offline validation OK",
    ):
        assert marker in source

