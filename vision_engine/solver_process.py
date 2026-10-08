from __future__ import annotations

import os
import pickle
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from .advisor import Advisor, Recommendation
from .board import Board


def build_advisor(request: dict[str, Any]) -> Advisor:
    """Build an Advisor from a plain solver request.

    Kept separate from :func:`solve_request` so V7 persistent workers can retain
    Advisor/transposition caches across multiple moves without changing the
    mathematical implementation.
    """
    board = Board(tuple(request["board_cells"]))
    learned_scores = {int(k): float(v) for k, v in (request.get("learned_scores") or {}).items()}

    def learned(_mask: int, _jewel: str, positions: tuple[int, ...]) -> dict[int, float]:
        return {p: learned_scores.get(p, 0.0) for p in positions}

    return Advisor(
        board,
        mode=str(request.get("mode", "adaptive")),
        exact_horizon=int(request.get("exact_horizon", 6)),
        rollout_exact_horizon=int(request.get("rollout_exact_horizon", 3)),
        rollouts=int(request.get("rollouts", 60)),
        bias_weights=tuple(float(x) for x in request.get("bias_weights", (1.0,) * 6)),
        rng_seed=int(request.get("rng_seed", 1337)),
        learned_action_scores=learned if learned_scores else None,
        p3_min=float(request.get("p3_min", 0.08)),
        p2l1n_min=float(request.get("p2l1n_min", 0.12)),
        p1l2n_min=float(request.get("p1l2n_min", 0.16)),
        p1l1n_min=float(request.get("p1l1n_min", 0.22)),
        fallback_override_pp=float(request.get("fallback_override_pp", 0.32)),
        adaptive_score_min=float(request.get("adaptive_score_min", 0.08)),
        p3_near_best_tolerance=float(request.get("p3_near_best_tolerance", 0.025)),
        ml_near_tie_probability=float(request.get("ml_near_tie_probability", 0.012)),
        ml_near_tie_score=float(request.get("ml_near_tie_score", 15.0)),
        plan_target=str(request.get("plan_target", "")),
        plan_lucky_lines=tuple(request.get("plan_lucky_lines", ())),
        plan_normal_lines=tuple(request.get("plan_normal_lines", ())),
        auto_p3_floor=float(request.get("auto_p3_floor", 0.04)),
        auto_fallback_ratio=float(request.get("auto_fallback_ratio", 3.0)),
        route_switch_ratio=float(request.get("route_switch_ratio", 1.35)),
    )


def solve_request(request: dict[str, Any]) -> list[Recommendation]:
    """Pure/pickle-safe solver entry point used by a child/persistent worker."""
    adv = build_advisor(request)
    return adv.recommend(int(request["mask"]), str(request["jewel"]))


def solver_entry(request: dict[str, Any], output_queue) -> None:
    """multiprocessing target. Return a tiny tagged payload to the parent."""
    try:
        output_queue.put(("ok", solve_request(request)))
    except BaseException as exc:  # child must report errors instead of hanging polling
        output_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def start_solver_process(ctx, request: dict[str, Any], stage: str, placed_count: int):
    """Start a disposable solver child and return it only after start succeeds.

    Windows can fail during ``Process.start`` before a child exists (for example
    when inherited console handles are invalid). Queue/process resources are
    cleaned here so the GUI never has to commit a half-started solver state.
    """

    q = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=solver_entry,
        args=(request, q),
        name=f"jewel-solver-{stage}-{placed_count}",
        daemon=True,
    )
    try:
        proc.start()
    except Exception:
        try:
            q.cancel_join_thread()
            q.close()
        except Exception:
            pass
        try:
            if proc.is_alive():
                proc.terminate()
        except Exception:
            pass
        raise
    return proc, q


def start_solver_subprocess(
    request: dict[str, Any], root_dir: Path, stage: str, placed_count: int
):
    """Fallback solver process that avoids ``multiprocessing`` pipe spawning.

    On some Windows console configurations CPython's multiprocessing spawn can
    fail before the child starts with an invalid console-buffer handle. A plain
    ``subprocess`` with DEVNULL stdio avoids those inherited console handles
    while preserving process isolation from Tk/GIL.
    """

    tmp_dir = Path(root_dir) / "data" / "solver_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:12]
    req_path = tmp_dir / f"{stage}_{placed_count}_{token}.request.pkl"
    out_path = tmp_dir / f"{stage}_{placed_count}_{token}.result.pkl"
    with req_path.open("wb") as fh:
        pickle.dump(request, fh, protocol=pickle.HIGHEST_PROTOCOL)

    flags = 0
    if os.name == "nt":
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    cmd = [sys.executable, "-m", "vision_engine.solver_subprocess", str(req_path), str(out_path)]
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(root_dir),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
        )
    except Exception:
        req_path.unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)
        raise
    return proc, req_path, out_path


def read_solver_subprocess_result(output_path: Path):
    with Path(output_path).open("rb") as fh:
        return pickle.load(fh)
