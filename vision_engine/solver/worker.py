from __future__ import annotations

import hashlib
import json

from vision_engine.solver_process import build_advisor

_WORKER_ADVISORS: dict[str, object] = {}


def advisor_signature(request: dict) -> str:
    # learned_scores only affect the root near-tie ordering. They do not affect
    # exact/Monte-Carlo state values, so including them here rebuilt the Advisor
    # and discarded its transposition caches on almost every live move.
    # FAST (60) -> REFINE (120) must share the same Advisor and exact-state
    # caches. Rollout count changes only the MC sample budget, not the policy.
    stable = {k: v for k, v in request.items() if k not in {"mask", "jewel", "learned_scores", "rollouts"}}
    raw = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=list).encode()
    return hashlib.blake2b(raw, digest_size=16).hexdigest()


def _apply_request_overlay(advisor, legacy_request: dict) -> None:
    """Refresh request-local root ranking data without discarding math caches."""
    if "rollouts" in legacy_request:
        advisor.rollouts = int(legacy_request["rollouts"])
    learned_scores = {int(k): float(v) for k, v in (legacy_request.get("learned_scores") or {}).items()}
    if learned_scores:
        def learned(_mask: int, _jewel: str, positions: tuple[int, ...]) -> dict[int, float]:
            return {p: learned_scores.get(p, 0.0) for p in positions}
        advisor.learned_action_scores = learned
    else:
        advisor.learned_action_scores = None


def solve_hot_worker(legacy_request: dict):
    key = advisor_signature(legacy_request)
    advisor = _WORKER_ADVISORS.get(key)
    if advisor is None:
        advisor = build_advisor(legacy_request)
        if len(_WORKER_ADVISORS) >= 12:
            _WORKER_ADVISORS.pop(next(iter(_WORKER_ADVISORS)))
        _WORKER_ADVISORS[key] = advisor
    _apply_request_overlay(advisor, legacy_request)
    return advisor.recommend(int(legacy_request["mask"]), str(legacy_request["jewel"]))


def warm_worker() -> bool:
    """Warm optional Numba kernel before the first live turn (inside worker)."""
    import os
    if os.environ.get("SPEEDLORA_COMPILED_PLAN", "0") == "1":
        from vision_engine.solver.r39_plan_kernel import (
            board_bitmasks, compiled_features, kernel,
        )
        from vision_engine.board import Board
        from vision_engine.constants import JEWELS
        # Valid synthetic layout only; no user game data or ML consulted.
        cells = tuple((None if i == 12 else JEWELS[(i if i < 12 else i - 1) // 4]) for i in range(25))
        compiled_features(0, board_bitmasks(Board(cells)), kernel())
    return True
