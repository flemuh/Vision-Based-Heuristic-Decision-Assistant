from __future__ import annotations

from vision_engine.advisor import Advisor
from vision_engine.board import Board


def m(*positions: int) -> int:
    return sum(1 << p for p in positions)


def late_game_board() -> Board:
    return Board.from_rows([
        ["CR", "BL", "CR", "HA", "HA"],
        ["CR", "CH", "SO", "CH", "CH"],
        ["HA", "LI", None, "CH", "SO"],
        ["BL", "HA", "LI", "LI", "SO"],
        ["BL", "BL", "LI", "CR", "SO"],
    ])


def test_endgame_route_hysteresis_relaxes_without_changing_early_game_default():
    b = late_game_board()
    a = Advisor(b, mode="auto", route_switch_ratio=1.35)
    assert a._effective_route_switch_ratio(m(*range(9))) == 1.35
    assert a._effective_route_switch_ratio(m(*range(10))) == 1.20
    assert a._effective_route_switch_ratio(m(*range(12))) == 1.10
    assert a._effective_route_switch_ratio(m(*range(13))) == 1.00


