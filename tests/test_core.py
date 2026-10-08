from collections import Counter

from vision_engine.advisor import Advisor, Utility, utility_learning_target
from vision_engine.board import Board
from vision_engine.scoring import score_mask
from vision_engine.temporal import TemporalStateFilter
from vision_engine.vision import _constrained_board_assignment


def base_board():
    return Board.from_rows([
        ["CH", "HA", "SO", "CH", "HA"],
        ["SO", "BL", "BL", "CR", "LI"],
        ["BL", "CH", None, "CR", "HA"],
        ["BL", "SO", "CH", "LI", "SO"],
        ["LI", "CR", "CR", "LI", "HA"],
    ])


def test_board_has_four_each():
    b = base_board()
    assert all(len(b.positions(j)) == 4 for j in ("BL", "SO", "LI", "CR", "HA", "CH"))


def test_three_lucky_1026_shape():
    lucky_cells = {10, 11, 13, 14, 2, 7, 17, 22, 4, 8, 16, 20}
    extra = {0, 5}
    mask = sum(1 << i for i in lucky_cells | extra)
    s = score_mask(mask)
    assert s.lucky >= 3
    assert s.total >= 1026


def test_risk_gate_ignores_tiny_three_lucky_chance():
    adv = Advisor(base_board(), p3_min=0.08)
    tiny_three = Utility(0.004, 0.0, 0.0, 0.0, 0.08, 0.10, 800)
    strong_fallback = Utility(0.0, 0.0, 0.0, 0.0, 0.78, 0.85, 1120)
    assert adv._key(strong_fallback) > adv._key(tiny_three)


def test_risk_gate_keeps_real_three_lucky_focus():
    adv = Advisor(base_board(), p3_min=0.08)
    three = Utility(0.12, 0.12, 0.1, 0.1, 0.25, 0.3, 980)
    fallback = Utility(0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 1200)
    assert adv._key(three) > adv._key(fallback)


def test_temporal_filter_requires_consensus_and_recovers_missed_transitions():
    f = TemporalStateFilter(frames=3, required=2, max_mask_jump=1)
    f.reset(0)
    assert not f.observe(1, "BL").stable
    s = f.observe(1, "BL")
    # Accept the placement, but clear BL votes because BL was the jewel that was
    # just placed; the next jewel must earn a fresh consensus.
    assert s.stable and s.mask == 1 and s.jewel is None
    # A repeated monotonic multi-cell jump is quarantined longer than an
    # ordinary +1 transition so click-glow spill cannot become permanent state.
    states = [f.observe(0b111, "SO") for _ in range(5)]
    assert all(not state.stable for state in states[:-1])
    s = states[-1]
    assert s.stable
    assert s.mask == 0b111
    assert s.reason == "catch-up:2"


def test_global_board_assignment_enforces_four_each():
    # Deliberately make every fifth cell slightly ambiguous; capacity constraint
    # must still return exactly four of every jewel.
    jewels = ("BL", "SO", "LI", "CR", "HA", "CH")
    dists = []
    for j in jewels:
        for k in range(4):
            d = {x: 0.01 for x in jewels}
            d[j] = 0.90 if k else 0.55
            d[jewels[(jewels.index(j) + 1) % 6]] = 0.40 if k == 0 else 0.02
            z = sum(d.values())
            dists.append({x: v / z for x, v in d.items()})
    assigned, conf = _constrained_board_assignment(dists)
    assert Counter(assigned) == Counter({j: 4 for j in jewels})
    assert conf > 0


def test_solver_learning_target_prefers_better_utility():
    a = Utility(0.2, 0.4, 0.5, 0.6, 0.5, 0.6, 1030)
    b = Utility(0.1, 0.4, 0.5, 0.6, 0.5, 0.6, 1030)
    assert utility_learning_target(a) > utility_learning_target(b)

from vision_engine.state_machine import GamePhase, GameStateMachine
from vision_engine.temporal import BoardConsensusFilter
from vision_engine.vision import distribution_metrics
from vision_engine.result_reader import ResultConsensus, VerifiedResult


def test_distribution_metrics_reward_clear_margin():
    clear = {j: 0.01 for j in ("BL", "SO", "LI", "CR", "HA", "CH")}
    clear["LI"] = 0.95
    z = sum(clear.values()); clear = {k:v/z for k,v in clear.items()}
    label, conf, top, margin, entropy = distribution_metrics(clear)
    assert label == "LI"
    assert top > 0.9 and margin > 0.85 and entropy < 0.3 and conf > 0.5


def test_distribution_metrics_flag_ambiguous_prediction():
    dist = {j: 1/6 for j in ("BL", "SO", "LI", "CR", "HA", "CH")}
    _, conf, _, margin, entropy = distribution_metrics(dist)
    assert margin < 1e-9
    assert entropy > 0.99
    assert conf < 0.2


def test_board_consensus_requires_repeated_hash():
    f = BoardConsensusFilter(frames=3, required=2)
    assert not f.observe("a")[0]
    assert not f.observe("b")[0]
    ok, n = f.observe("a")
    assert ok and n == 2


def test_game_phase_machine():
    sm = GameStateMachine()
    assert sm.update(has_board=False, selected_count=0, current_jewel=None).phase == GamePhase.NO_BOARD
    assert sm.update(has_board=True, selected_count=5, current_jewel="BL").phase == GamePhase.PLAYING
    assert sm.update(has_board=True, selected_count=13, current_jewel="LI").phase == GamePhase.ONE_LEFT
    assert sm.update(has_board=True, selected_count=14, current_jewel=None).phase == GamePhase.COMPLETE


def test_result_consensus_rejects_implausible_then_accepts_repeat():
    c = ResultConsensus(required=2)
    bad = VerifiedResult(936, 0, 91, 1027, "ocr")
    assert c.observe(bad) is None
    good = VerifiedResult(936, 0, 90, 1026, "ocr")
    assert c.observe(good) is None
    out = c.observe(good)
    assert out is not None and out.total_score == 1026


def test_monte_carlo_recommendation_is_reproducible():
    b = base_board()
    a1 = Advisor(b, exact_horizon=1, rollouts=300, rng_seed=42)
    a2 = Advisor(b, exact_horizon=1, rollouts=300, rng_seed=42)
    r1 = a1.recommend(0, "CH")
    r2 = a2.recommend(0, "CH")
    assert [(r.position, r.utility) for r in r1] == [(r.position, r.utility) for r in r2]

def test_verified_result_separates_plausibility_from_model_consistency():
    # A real game result that sums correctly should be recordable even if it
    # disproves our currently inferred 312/240/45 scoring model.
    observed = VerifiedResult(937, 0, 90, 1027, "manual")
    assert observed.plausible
    assert not observed.model_consistent


def test_numpy_knn_classifier_exact_match():
    import numpy as np
    from vision_engine.simple_ml import NumpyKNNClassifier
    x = np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    clf = NumpyKNNClassifier(n_neighbors=1).fit(x, ["A", "B", "C"])
    p = clf.predict_proba(np.asarray([[1.0, 1.0]], dtype=np.float32))[0]
    assert clf.classes_[int(np.argmax(p))] == "B"


def test_constrained_assignment_uses_four_of_each():
    from collections import Counter
    from vision_engine.constants import JEWELS
    from vision_engine.vision import _constrained_board_assignment_full
    labels = [j for j in JEWELS for _ in range(4)]
    dists = []
    for lab in labels:
        d = {j: 0.01 for j in JEWELS}
        d[lab] = 0.95
        z = sum(d.values())
        dists.append({k: v / z for k, v in d.items()})
    assigned, confidence, probs = _constrained_board_assignment_full(dists)
    assert Counter(assigned) == Counter({j: 4 for j in JEWELS})
    assert assigned == labels
    assert confidence > 0.8
    assert len(probs) == 24
