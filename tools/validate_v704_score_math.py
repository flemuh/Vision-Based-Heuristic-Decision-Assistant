from __future__ import annotations

from vision_engine.advisor import Advisor
from vision_engine.board import Board


def mask(*positions: int) -> int:
    return sum(1 << p for p in positions)


def main() -> None:
    bless = Board.from_rows([
        ["SO", "CH", "CR", "BL", "LI"],
        ["BL", "HA", "CR", "LI", "BL"],
        ["HA", "CH", None, "LI", "LI"],
        ["CH", "HA", "HA", "SO", "CR"],
        ["SO", "BL", "CR", "CH", "SO"],
    ])
    r = Advisor(bless, mode="score", rollouts=120, exact_horizon=6).recommend(mask(2, 11, 13), "BL")
    assert bless.rc(r[0].position) == (1, 4)
    assert r[0].tie_break_metric == ">1000 structural route potential"
    print("Bless 3/14 structural tie-break: L1C4 OK")

    life6 = Board.from_rows([
        ["BL", "CH", "LI", "CR", "HA"],
        ["LI", "HA", "LI", "SO", "SO"],
        ["HA", "CR", None, "LI", "CR"],
        ["CH", "SO", "SO", "BL", "CH"],
        ["BL", "CH", "CR", "HA", "BL"],
    ])
    r = Advisor(life6, mode="score", rollouts=120, exact_horizon=6).recommend(mask(4, 6, 8, 10, 13, 14), "LI")
    assert life6.rc(r[0].position) in {(1, 3), (2, 3)}
    print(f"Life 6/14 statistical tie guardrail: L{life6.rc(r[0].position)[0]}C{life6.rc(r[0].position)[1]} OK")

    life4 = Board.from_rows([
        ["BL", "SO", "LI", "LI", "HA"],
        ["CR", "SO", "HA", "CH", "LI"],
        ["CH", "SO", None, "HA", "CR"],
        ["LI", "BL", "BL", "CH", "CR"],
        ["CR", "SO", "CH", "BL", "HA"],
    ])
    live = Advisor(life4, mode="score", rollouts=120, exact_horizon=6).recommend(mask(0, 6, 18, 24), "LI")
    assert life4.rc(live[0].position) == (1, 4)
    assert live[0].confidence == "LOW_SAMPLE"
    exact = Advisor(life4, mode="score", rollouts=120, exact_horizon=9).recommend(mask(0, 6, 18, 24), "LI")
    assert life4.rc(exact[0].position) == (1, 4)
    by_rc = {life4.rc(x.position): x for x in exact}
    print(
        "Life 4/14 exact confirmation: L1C4 "
        f"P>1000={by_rc[(1,4)].utility.p_gt1000*100:.3f}% vs "
        f"L1C3={by_rc[(1,3)].utility.p_gt1000*100:.3f}% OK"
    )


if __name__ == "__main__":
    main()
