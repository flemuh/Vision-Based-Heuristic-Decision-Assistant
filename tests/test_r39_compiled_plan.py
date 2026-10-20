"""Parity tests: the optional compiled route/features kernel must never change decisions."""
from __future__ import annotations
import random
import math

import pytest

from vision_engine.board import Board
from vision_engine.advisor import Advisor
from vision_engine.constants import JEWELS
from vision_engine.solver.r39_plan_kernel import board_bitmasks, compiled_features, kernel

BOARD = ('CH','HA','SO','CH','HA', 'SO','BL','BL','CR','LI',
         'BL','CH',None,'CR','HA', 'BL','SO','CH','LI','SO',
         'LI','CR','CR','LI','HA')


@pytest.fixture(scope='module')
def compiled():
    fn = kernel()
    board = Board(BOARD)
    compiled_features(0, board_bitmasks(board), fn)  # cold compile outside measurements
    return fn


def test_plan_feature_parity_320_random_legal_masks(compiled):
    rng = random.Random(7731)
    for board_idx in range(4):
        shuffled = [j for j in JEWELS for _ in range(4)]
        rng.shuffle(shuffled)
        shuffled.insert(12, None)
        board = Board(tuple(shuffled))
        advisor = Advisor(board, compiled_plan_features=False)
        bitmasks = board_bitmasks(board)
        for _ in range(80):
            positions = rng.sample([i for i in range(25) if i != 12], rng.randrange(0,15))
            mask = sum(1 << i for i in positions)
            expected = advisor._plan_features(mask)
            observed = compiled_features(mask, bitmasks, compiled)
            assert expected.keys() == observed.keys()
            for key in expected:
                assert math.isclose(expected[key], observed[key], abs_tol=2e-10, rel_tol=1e-12), (board_idx, mask, key, expected[key], observed[key])


@pytest.mark.parametrize('mode', ['auto', 'adaptive', 'score', 'lucky'])
def test_complete_advisor_recommendation_parity(mode):
    board = Board(BOARD)
    mask = sum(1 << i for i in (0,6,5,9,16,13))
    settings = dict(mode=mode, rollouts=16, rng_seed=31415,
                    rollout_exact_horizon=3, exact_horizon=6)
    baseline = Advisor(board, compiled_plan_features=False, **settings).recommend(mask, 'BL')
    optimized = Advisor(board, compiled_plan_features=True, **settings).recommend(mask, 'BL')
    assert [r.position for r in baseline] == [r.position for r in optimized]
    for expected, actual in zip(baseline, optimized):
        for a, b in zip(expected.utility.as_tuple(), actual.utility.as_tuple()):
            assert math.isclose(a, b, abs_tol=2e-10, rel_tol=1e-12)
        assert expected.active_goal == actual.active_goal
        assert expected.reward_guarantee == actual.reward_guarantee


def test_reward_guaranteed_fastpath_unchanged():
    board = Board(BOARD)
    # Endgame paths often return before the numerical policy kernel is invoked.
    mask = sum(1 << i for i in (0,1,2,3,4,5,6,7,8,9,10,11))
    a = Advisor(board, mode='auto', compiled_plan_features=False)
    b = Advisor(board, mode='auto', compiled_plan_features=True)
    left=a.recommend(mask,'CR');right=b.recommend(mask,'CR')
    assert [r.position for r in left]==[r.position for r in right]
    assert [r.reward_guarantee for r in left]==[r.reward_guarantee for r in right]
