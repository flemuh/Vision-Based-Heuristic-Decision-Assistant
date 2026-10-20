from vision_engine.advisor import Advisor
from vision_engine.board import Board

BOARD = ('CH','HA','SO','CH','HA','SO','BL','BL','CR','LI',
         'BL','CH',None,'CR','HA','BL','SO','CH','LI','SO',
         'LI','CR','CR','LI','HA')
MASK = sum(1 << i for i in (0, 6, 5, 9, 16, 13))
CURRENT_JEWEL = 'BL'


def make_advisor(*, optimized: bool, rollouts: int = 60):
    return Advisor(
        Board(BOARD),
        mode='auto',
        exact_horizon=6,
        rollout_exact_horizon=3,
        rollouts=rollouts,
        rng_seed=1337,
        compiled_plan_features=True,
        policy_key_cache_enabled=optimized,
    )


def test_policy_cache_keeps_same_recommendation():
    base = make_advisor(optimized=False, rollouts=60).recommend(MASK, CURRENT_JEWEL)
    opt = make_advisor(optimized=True, rollouts=60).recommend(MASK, CURRENT_JEWEL)
    assert [r.position for r in base] == [r.position for r in opt]
    for a, b in zip(base, opt):
        assert a.utility.as_tuple() == b.utility.as_tuple()


def test_policy_cache_uses_context_sensitive_keys():
    adv = make_advisor(optimized=True, rollouts=16)
    adv._active_plan_target = '3 Lucky'
    adv._active_plan_lucky_lines = ('H', 'V', '\\')
    key1 = adv._policy_key_for_mask(MASK | (1 << 2))
    size1 = len(adv._policy_key_cache)
    adv._active_plan_target = '>=999'
    adv._active_plan_lucky_lines = ('H', 'V')
    adv._active_plan_normal_lines = ('R1',)
    key2 = adv._policy_key_for_mask(MASK | (1 << 2))
    size2 = len(adv._policy_key_cache)
    assert size2 > size1
    assert key1 != key2
