"""Life finishing a certain Lucky should not trigger Monte Carlo rollouts."""
from vision_engine.advisor import Advisor
from vision_engine.scoring import score_mask
from vision_engine.strategy import move_line_impacts
from tests.test_winning_memory_depth34 import BOARD, WIN_POS


def test_life_last_exact_jewel_completes_lucky_and_wins_immediately():
    # A replay-consistent winner: Life at 18 closes diagonal Lucky \\.
    life_position = 18
    selected = sum(1 << pos for pos in WIN_POS if pos != life_position)
    assert BOARD.cells[life_position] == "LI"
    assert any(line.kind == 'Lucky' and line.completed
               for line in move_line_impacts(selected, life_position))
    assert score_mask(selected).total < 1000
    assert score_mask(selected | (1 << life_position)).total > 1000

    advisor = Advisor(BOARD, mode='auto', rollouts=100)
    # A sampled scenario routine must never run on a guaranteed immediate win.
    def fail_mc(*args, **kwargs):
        raise AssertionError('Monte Carlo must be skipped for guaranteed Life Lucky')
    advisor._mc_evaluate = fail_mc
    recs = advisor.recommend(selected, 'LI')
    assert recs[0].position == life_position
    assert recs[0].method == 'immediate-lucky-fastpath'
    assert recs[0].lucky_clears_gained >= 1


def test_uncertain_lucky_can_fall_back_to_full_solver():
    # No guarantee is claimed for a candidate that cannot close Lucky.
    advisor = Advisor(BOARD, mode='auto', rollouts=5)
    mask = (1 << 0) | (1 << 2)
    positions = BOARD.legal_positions('LI', mask)
    assert advisor._immediate_lucky_fastpath(mask, 'LI', positions) is None
