from __future__ import annotations

from dataclasses import replace

from vision_engine.config import AppConfig
from vision_engine.solver.engine import SolverEngine
from vision_engine.solver import worker


def _request(scores=None):
    return {
        "board_cells": [
            "CH","HA","SO","CH","HA",
            "SO","BL","BL","CR","LI",
            "BL","CH",None,"CR","HA",
            "BL","SO","CH","LI","SO",
            "LI","CR","CR","LI","HA",
        ],
        "mask": 0,
        "jewel": "BL",
        "mode": "auto",
        "exact_horizon": 6,
        "rollout_exact_horizon": 2,
        "rollouts": 20,
        "rng_seed": 1337,
        "bias_weights": [1.0] * 6,
        "learned_scores": scores or {},
        "p3_min": 0.08,
        "p2l1n_min": 0.12,
        "p1l2n_min": 0.16,
        "p1l1n_min": 0.22,
        "fallback_override_pp": 0.32,
        "adaptive_score_min": 0.08,
        "p3_near_best_tolerance": 0.025,
        "ml_near_tie_probability": 0.012,
        "ml_near_tie_score": 15.0,
        "auto_p3_floor": 0.04,
        "auto_fallback_ratio": 3.0,
        "route_switch_ratio": 1.35,
        "plan_target": "",
        "plan_lucky_lines": (),
        "plan_normal_lines": (),
    }


def test_live_horizon_is_bounded_without_changing_offline_default():
    cfg = AppConfig()
    assert cfg.rollout_exact_horizon == 3
    assert cfg.live_rollout_exact_horizon == 2
    assert cfg.exact_horizon == 6


def test_advisor_identity_ignores_request_local_learned_scores():
    a = _request({6: 0.1})
    b = _request({6: 0.9, 7: -0.2})
    assert worker.advisor_signature(a) == worker.advisor_signature(b)


def test_reused_advisor_receives_fresh_learned_overlay(monkeypatch):
    worker._WORKER_ADVISORS.clear()
    seen = []

    class FakeAdvisor:
        learned_action_scores = None
        def recommend(self, mask, jewel):
            positions = (6, 7)
            values = self.learned_action_scores(mask, jewel, positions) if self.learned_action_scores else {6: 0.0, 7: 0.0}
            seen.append(values)
            return [values]

    built = []
    def fake_build(_request):
        adv = FakeAdvisor()
        # Mirror build_advisor's initial request-local callback.
        worker._apply_request_overlay(adv, _request)
        built.append(adv)
        return adv

    monkeypatch.setattr(worker, "build_advisor", fake_build)
    worker.solve_hot_worker(_request({6: 0.1}))
    worker.solve_hot_worker(_request({6: 0.9, 7: -0.2}))
    assert len(built) == 1
    assert seen == [{6: 0.1, 7: 0.0}, {6: 0.9, 7: -0.2}]
    worker._WORKER_ADVISORS.clear()


def test_solver_engine_uses_one_hot_foreground_worker():
    engine = SolverEngine(workers=8, cache_entries=32)
    try:
        assert engine.foreground._max_workers == 1
        assert engine.background._max_workers == 7
    finally:
        engine.close()
