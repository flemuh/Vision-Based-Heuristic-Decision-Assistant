from __future__ import annotations

from pathlib import Path

from vision_engine.advisor import Advisor
from vision_engine.board import Board
from vision_engine.config import AppConfig


def base_board() -> Board:
    return Board.from_rows([
        ["CH", "HA", "SO", "CH", "HA"],
        ["SO", "BL", "BL", "CR", "LI"],
        ["BL", "CH", None, "CR", "HA"],
        ["BL", "SO", "CH", "LI", "SO"],
        ["LI", "CR", "CR", "LI", "HA"],
    ])


def late_impossible_three_board() -> Board:
    # User's 11/14 screenshot where the V7.0.4 legacy 3-Lucky strategy kept
    # TARGET 3 Lucky even though exact route feasibility was 0.0%.
    return Board.from_rows([
        ["SO", "CH", "LI", "CR", "CH"],
        ["HA", "LI", "HA", "CH", "SO"],
        ["CR", "BL", None, "LI", "BL"],
        ["HA", "LI", "SO", "HA", "CH"],
        ["CR", "BL", "SO", "BL", "CR"],
    ])


def mask(*positions: int) -> int:
    return sum(1 << p for p in positions)


def test_live_config_defaults_to_single_automatic_mode():
    cfg = AppConfig()
    assert cfg.risk_profile == "auto"
    assert cfg.mode == "auto"


def test_auto_planner_keeps_locked_three_lucky_route_and_rejects_fourth_line():
    b = base_board()
    # Lock H+V+\\. Chaos has three on-route cells (L1C1/L3C2/L4C3) and one
    # unrelated cell L1C4. With the route healthy, the unrelated cell must not
    # outrank an on-route placement merely because a local score proxy likes it.
    adv = Advisor(
        b, mode="auto", rollouts=120,
        plan_target="3 Lucky", plan_lucky_lines=("H", "V", "\\"),
        auto_p3_floor=0.04, auto_fallback_ratio=3.0, route_switch_ratio=100.0,
    )
    recs = adv.recommend(mask(6, 13, 22), "CH")
    assert recs[0].plan_target == "3 Lucky"
    assert recs[0].plan_lucky_lines == ("H", "V", "\\")
    assert recs[0].plan_on_route is True
    assert recs[0].position != 3  # L1C4 is the off-plan fourth-line cell.


def test_auto_planner_falls_back_when_three_lucky_is_impossible():
    b = base_board()
    # Three already-selected cells outside every Lucky line consume more than
    # the global two-placement slack budget, so no 3-Lucky trio can be completed
    # in the remaining 11 draws.
    selected = mask(1, 3, 5)
    adv = Advisor(
        b, mode="auto", rollouts=120,
        plan_target="3 Lucky", plan_lucky_lines=("H", "V", "/"),
    )
    recs = adv.recommend(selected, "CH")
    assert recs
    assert recs[0].plan_target != "3 Lucky"
    assert recs[0].plan_target in {"2 Lucky + 1 Normal", "1 Lucky + 2 Normal", "1 Lucky + 1 Normal", "expected score"}


def test_auto_target_ladder_never_intentionally_targets_four_line_bonus():
    b = base_board()
    adv = Advisor(b, mode="auto", rollouts=60)
    recs = adv.recommend(0, "CH")
    assert recs[0].plan_target in {
        "3 Lucky", "2 Lucky + 1 Normal", "1 Lucky + 2 Normal",
        "1 Lucky + 1 Normal", "expected score",
    }
    assert recs[0].plan_target not in {"3 Lucky + 1 Normal", "2 Lucky + 2 Normal", "1 Lucky + 3 Normal"}


def test_live_ui_exposes_one_automatic_strategy_only():
    source = Path(__file__).parents[1].joinpath("vision_engine", "gui.py").read_text(encoding="utf-8")
    assert 'self.strategy_var = tk.StringVar(value="Win >1000 / Adaptive")' in source
    assert 'values=("Adaptive", "Score >1000", "3 Lucky")' not in source


from vision_engine.live.state import GameState
from vision_engine.live.coordinator import LiveCoordinator
from vision_engine.solver.protocol import state_to_wire, state_from_wire


def test_locked_plan_survives_wsl_protocol_round_trip_and_versions():
    b = base_board()
    state = GameState(
        episode_id="e", state_version=2, strategy_version=3,
        board_hash="h", board_cells=tuple(b.cells), accepted_mask=0,
        current_jewel="CH", strategy="auto", phase="playing",
        plan_target="3 Lucky", plan_lucky_lines=("H", "V", "\\"),
    )
    got = state_from_wire(state_to_wire(state))
    assert got.plan_target == "3 Lucky"
    assert got.plan_lucky_lines == ("H", "V", "\\")
    coord = LiveCoordinator(got)
    v = coord.state.strategy_version
    coord.update_plan(target="2 Lucky + 1 Normal", lucky_lines=("H", "V"), normal_lines=("R2",))
    assert coord.state.strategy_version == v + 1
    assert coord.state.plan_target == "2 Lucky + 1 Normal"


from vision_engine.solver.protocol import SolverRequest, SolverSettings
from vision_engine.solver_process import solve_request


def test_solver_request_carries_locked_plan_into_worker():
    b = base_board()
    state = GameState(
        episode_id="e2", state_version=1, strategy_version=1,
        board_hash="h", board_cells=tuple(b.cells), accepted_mask=mask(6, 13, 22),
        current_jewel="CH", strategy="auto", phase="playing",
        plan_target="3 Lucky", plan_lucky_lines=("H", "V", "\\"),
    )
    req = SolverRequest(1, state, SolverSettings(rollouts=60, route_switch_ratio=100.0))
    recs = solve_request(req.to_legacy_request())
    assert recs[0].plan_target == "3 Lucky"
    assert recs[0].plan_lucky_lines == ("H", "V", "\\")
    assert recs[0].plan_on_route is True
