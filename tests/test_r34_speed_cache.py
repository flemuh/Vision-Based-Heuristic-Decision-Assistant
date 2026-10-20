"""R3.4 pure-equivalence and FAST -> REFINE transposition tests."""
from __future__ import annotations

import random

from vision_engine.advisor import Advisor
from vision_engine.board import Board, _board_lookup
from vision_engine.constants import JEWELS
from vision_engine.scoring import LUCKY_MASKS, NORMAL_MASKS, score_mask
from vision_engine.solver import worker


CELLS = (
    'CH','HA','SO','CH','HA',
    'SO','BL','BL','CR','LI',
    'BL','CH',None,'CR','HA',
    'BL','SO','CH','LI','SO',
    'LI','CR','CR','LI','HA',
)


def request(mask=0, rollouts=20, board=CELLS):
    return {
        'board_cells': list(board), 'mask': mask, 'jewel': 'BL',
        'mode': 'auto', 'exact_horizon': 6, 'rollout_exact_horizon': 3,
        'rollouts': rollouts, 'rng_seed': 1337, 'bias_weights': [1.] * 6,
        'learned_scores': {},
    }


def test_cached_score_matches_reference_for_random_masks():
    rng = random.Random(9317)
    for _ in range(1600):
        mask = rng.randrange(1 << 25)
        lucky = tuple(k for k, v in LUCKY_MASKS.items() if mask & v == v)
        normal = tuple(k for k, v in NORMAL_MASKS.items() if mask & v == v)
        clear = 0
        for k in lucky:
            clear |= LUCKY_MASKS[k]
        for k in normal:
            clear |= NORMAL_MASKS[k]
        score = score_mask(mask)
        assert (score.lucky, score.normal, score.jewel_cells) == (
            len(lucky), len(normal), (mask & ~clear).bit_count())
        assert score.lucky_names == lucky and score.normal_names == normal
        assert score is score_mask(mask)


def test_cached_board_jewel_counts_and_positions_are_exact():
    board = Board(CELLS)
    rng = random.Random(124)
    for jewel in JEWELS:
        assert board.positions(jewel) == tuple(i for i, j in enumerate(CELLS) if j == jewel)
    for _ in range(1000):
        mask = rng.randrange(1 << 25)
        expected = tuple(sum(bool(mask & (1 << i)) for i, j in enumerate(CELLS) if j == jewel)
                         for jewel in JEWELS)
        assert board.jewel_counts_in_mask(mask) == expected
        for jewel in JEWELS:
            assert board.legal_positions(jewel, mask) == tuple(
                i for i, j in enumerate(CELLS) if j == jewel and not mask & (1 << i)
            )
    assert _board_lookup(CELLS) is _board_lookup(CELLS)


def test_fast_then_refine_is_same_as_fresh_full_request():
    mask = sum(1 << p for p in (0, 6, 5, 9, 16, 13))
    worker._WORKER_ADVISORS.clear()
    try:
        worker.solve_hot_worker(request(mask, rollouts=20))
        refined = worker.solve_hot_worker(request(mask, rollouts=40))
        assert len(worker._WORKER_ADVISORS) == 1
        cached_advisor = next(iter(worker._WORKER_ADVISORS.values()))
        assert cached_advisor.rollouts == 40
        assert len(cached_advisor._root_sample_cache) > 0
        fresh = Advisor(Board(CELLS), mode='auto', exact_horizon=6,
                        rollout_exact_horizon=3, rollouts=40).recommend(mask, 'BL')
        assert [r.position for r in refined] == [r.position for r in fresh]
        for a, b in zip(refined, fresh):
            assert a.utility.as_tuple() == b.utility.as_tuple()
            assert a.samples == b.samples
            assert (a.stderr_p3, a.stderr_gt1000, a.stderr_score) == (
                b.stderr_p3, b.stderr_gt1000, b.stderr_score)
    finally:
        worker._WORKER_ADVISORS.clear()


def test_policy_affecting_inputs_cannot_share_advisor():
    a = request(rollouts=20)
    b = request(rollouts=40)
    assert worker.advisor_signature(a) == worker.advisor_signature(b)
    for k, v in [('rng_seed', 222), ('rollout_exact_horizon', 4),
                 ('mode', 'score'), ('board_cells', list(reversed(a['board_cells'])))]:
        different = {**b, k: v}
        assert worker.advisor_signature(a) != worker.advisor_signature(different)
