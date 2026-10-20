"""R3.3: real 900+ box reward wins over conditional 999 in same reward tier."""

import pytest

from vision_engine.advisor import Advisor, RewardForecast, Utility
from vision_engine.board import Board
from vision_engine.scoring import score_mask
from vision_engine.solver.protocol import recommendation_from_wire, recommendation_to_wire


BOARD = Board.from_rows([
    ['BL', 'SO', 'CH', 'SO', 'LI'],
    ['CH', 'LI', 'HA', 'BL', 'HA'],
    ['BL', 'CR', None, 'CR', 'CR'],
    ['CH', 'BL', 'LI', 'CR', 'SO'],
    ['CH', 'LI', 'HA', 'SO', 'HA'],
])
# Reconstructed from the game screenshot: 2 Lucky complete, 12/14 accepted.
MASK = sum(1 << p for p in (2, 4, 6, 7, 8, 13, 16, 17, 18, 20, 22, 23))


def test_reproduced_soul_turn_guarantees_900_instead_of_gambling_for_999():
    assert score_mask(MASK).total == 804
    a = Advisor(BOARD, mode='auto', rollouts=20)
    recs = a.recommend(MASK, 'SO')
    assert [r.position for r in recs] == [3, 19, 1]
    safe, gamble = recs[:2]
    assert safe.method == 'reward-tier-exact'
    assert safe.normal_clears_gained == 1
    assert safe.reward_guarantee == '900+ guaranteed'
    assert safe.reward_min_final == 954
    assert safe.reward_p_ge900 == pytest.approx(1)
    assert safe.utility.expected_score == pytest.approx(954)
    assert gamble.reward_guarantee == 'not guaranteed'
    assert gamble.reward_min_final == 894
    assert gamble.reward_p_ge900 == pytest.approx(4 / 11)
    assert gamble.utility.p_ge999 == pytest.approx(2 / 11)
    assert gamble.utility.expected_score == pytest.approx(924)
    # The win probabilities are not fictitious MC samples.
    assert not a._last_mc_samples
    assert safe.samples == 0


def test_reward_metrics_survive_worker_json_roundtrip():
    safe = Advisor(BOARD, mode='auto').recommend(MASK, 'SO')[0]
    restored = recommendation_from_wire(recommendation_to_wire(safe))
    assert restored.reward_guarantee == '900+ guaranteed'
    assert restored.reward_min_final == 954
    assert restored.reward_p_ge900 == pytest.approx(1.0)


def test_reward_prioritizes_guaranteed_1000_over_guaranteed_900():
    u900 = Utility(0, 0, 0, 0, 0.0, 0.0, 954.)
    u1000 = Utility(1, 1, 1, 1, 1.0, 1.0, 1026.)
    key = Advisor._reward_key
    assert key(RewardForecast(u1000, 1.0, 1026)) > key(RewardForecast(u900, 1.0, 954))


def test_same_reward_tier_prefers_guaranteed_box_to_risky_higher_points():
    sure = RewardForecast(Utility(0, 0, 0, 0, 0., 0., 954.), 1., 954)
    risky = RewardForecast(Utility(0, 0, 0, 0, 0., .18, 980.), .36, 894)
    assert Advisor._reward_key(sure) > Advisor._reward_key(risky)


def test_still_no_tier_override_in_early_game():
    mask = sum(1 << p for p in (2, 4, 6))
    a = Advisor(BOARD, mode='auto', rollouts=4)
    assert a._reward_tier_fastpath(mask, 'SO', BOARD.legal_positions('SO', mask)) is None


def test_three_lucky_route_remains_primary_while_feasible(monkeypatch):
    a = Advisor(BOARD, mode='auto')
    monkeypatch.setattr(a, '_best_three_lucky_route_probability', lambda mask: 0.25)
    assert a._reward_tier_fastpath(MASK, 'SO', BOARD.legal_positions('SO', MASK)) is None
