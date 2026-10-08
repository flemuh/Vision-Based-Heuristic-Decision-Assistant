from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from vision_engine.advisor import Recommendation, Utility
from vision_engine.live.state import GameState
from vision_engine.route_math import LuckySlackEstimate, RouteEstimate

from vision_engine.versioning import PROTOCOL_VERSION


@dataclass(frozen=True, slots=True)
class SolverSettings:
    exact_horizon: int = 6
    rollout_exact_horizon: int = 3
    rollouts: int = 1600
    rng_seed: int = 1337
    bias_weights: tuple[float, ...] = (1.0,) * 6
    p3_min: float = 0.08
    p2l1n_min: float = 0.12
    p1l2n_min: float = 0.16
    p1l1n_min: float = 0.22
    fallback_override_pp: float = 0.32
    adaptive_score_min: float = 0.08
    p3_near_best_tolerance: float = 0.025
    ml_near_tie_probability: float = 0.012
    ml_near_tie_score: float = 15.0
    auto_p3_floor: float = 0.04
    auto_fallback_ratio: float = 3.0
    route_switch_ratio: float = 1.35


@dataclass(frozen=True, slots=True)
class SolverRequest:
    request_id: int
    state: GameState
    settings: SolverSettings
    learned_scores: tuple[tuple[int, float], ...] = ()
    purpose: str = "decision"
    session_id: str = ""


    @classmethod
    def from_legacy(
        cls, request_id: int, state: GameState, legacy: dict[str, Any], *, purpose: str = "decision"
    ) -> "SolverRequest":
        settings = SolverSettings(
            exact_horizon=int(legacy.get("exact_horizon", 6)),
            rollout_exact_horizon=int(legacy.get("rollout_exact_horizon", 3)),
            rollouts=int(legacy.get("rollouts", 1600)),
            rng_seed=int(legacy.get("rng_seed", 1337)),
            bias_weights=tuple(float(x) for x in legacy.get("bias_weights", (1.0,) * 6)),
            p3_min=float(legacy.get("p3_min", 0.08)),
            p2l1n_min=float(legacy.get("p2l1n_min", 0.12)),
            p1l2n_min=float(legacy.get("p1l2n_min", 0.16)),
            p1l1n_min=float(legacy.get("p1l1n_min", 0.22)),
            fallback_override_pp=float(legacy.get("fallback_override_pp", 0.32)),
            adaptive_score_min=float(legacy.get("adaptive_score_min", 0.08)),
            p3_near_best_tolerance=float(legacy.get("p3_near_best_tolerance", 0.025)),
            ml_near_tie_probability=float(legacy.get("ml_near_tie_probability", 0.012)),
            ml_near_tie_score=float(legacy.get("ml_near_tie_score", 15.0)),
            auto_p3_floor=float(legacy.get("auto_p3_floor", 0.04)),
            auto_fallback_ratio=float(legacy.get("auto_fallback_ratio", 3.0)),
            route_switch_ratio=float(legacy.get("route_switch_ratio", 1.35)),
        )
        learned = tuple(sorted((int(k), float(v)) for k, v in (legacy.get("learned_scores") or {}).items()))
        return cls(request_id=request_id, state=state, settings=settings, learned_scores=learned, purpose=purpose)

    def to_legacy_request(self) -> dict[str, Any]:
        if not self.state.current_jewel:
            raise ValueError("current_jewel is required for a solver decision")
        s = self.settings
        return {
            "board_cells": self.state.board_cells,
            "mask": self.state.accepted_mask,
            "jewel": self.state.current_jewel,
            "mode": self.state.strategy,
            "exact_horizon": s.exact_horizon,
            "rollout_exact_horizon": s.rollout_exact_horizon,
            "rollouts": s.rollouts,
            "bias_weights": s.bias_weights,
            "rng_seed": s.rng_seed,
            "learned_scores": dict(self.learned_scores),
            "p3_min": s.p3_min,
            "p2l1n_min": s.p2l1n_min,
            "p1l2n_min": s.p1l2n_min,
            "p1l1n_min": s.p1l1n_min,
            "fallback_override_pp": s.fallback_override_pp,
            "adaptive_score_min": s.adaptive_score_min,
            "p3_near_best_tolerance": s.p3_near_best_tolerance,
            "ml_near_tie_probability": s.ml_near_tie_probability,
            "ml_near_tie_score": s.ml_near_tie_score,
            "auto_p3_floor": s.auto_p3_floor,
            "auto_fallback_ratio": s.auto_fallback_ratio,
            "route_switch_ratio": s.route_switch_ratio,
            "plan_target": self.state.plan_target,
            "plan_lucky_lines": self.state.plan_lucky_lines,
            "plan_normal_lines": self.state.plan_normal_lines,
        }


@dataclass(frozen=True, slots=True)
class SolverResponse:
    request_id: int
    episode_id: str
    state_version: int
    strategy_version: int
    recommendations: tuple[Recommendation, ...]
    cache_hit: bool = False
    elapsed_ms: float = 0.0
    source: str = "local"
    session_id: str = ""
    protocol_version: int = PROTOCOL_VERSION

    @property
    def identity(self) -> tuple[str, int, int]:
        return self.episode_id, self.state_version, self.strategy_version

    @classmethod
    def for_request(
        cls,
        request: SolverRequest,
        recommendations: list[Recommendation] | tuple[Recommendation, ...],
        *,
        cache_hit: bool = False,
        elapsed_ms: float = 0.0,
        source: str = "local",
    ) -> "SolverResponse":
        st = request.state
        return cls(
            request_id=request.request_id,
            episode_id=st.episode_id,
            state_version=st.state_version,
            strategy_version=st.strategy_version,
            recommendations=tuple(recommendations),
            cache_hit=cache_hit,
            elapsed_ms=float(elapsed_ms),
            source=source,
            session_id=request.session_id,
        )


def recommendation_to_wire(rec: Recommendation) -> dict[str, Any]:
    data = asdict(rec)
    # dataclasses.asdict recursively converts RouteEstimate/LuckySlackEstimate.
    return data


def recommendation_from_wire(data: dict[str, Any]) -> Recommendation:
    raw = dict(data)
    raw["utility"] = Utility(**raw["utility"])
    raw["routes"] = tuple(RouteEstimate(**r) for r in raw.get("routes", ()))
    slack = raw.get("lucky_slack")
    raw["lucky_slack"] = LuckySlackEstimate(**slack) if slack else None
    raw["tied_with"] = tuple(raw.get("tied_with", ()))
    raw["plan_lucky_lines"] = tuple(raw.get("plan_lucky_lines", ()))
    raw["plan_normal_lines"] = tuple(raw.get("plan_normal_lines", ()))
    return Recommendation(**raw)


def settings_to_wire(settings: SolverSettings) -> dict[str, Any]:
    data = asdict(settings)
    data["bias_weights"] = list(settings.bias_weights)
    return data


def settings_from_wire(data: dict[str, Any]) -> SolverSettings:
    raw = dict(data)
    raw["bias_weights"] = tuple(raw.get("bias_weights", (1.0,) * 6))
    return SolverSettings(**raw)


def state_to_wire(state: GameState) -> dict[str, Any]:
    return {
        "episode_id": state.episode_id,
        "state_version": state.state_version,
        "strategy_version": state.strategy_version,
        "board_hash": state.board_hash,
        "board_cells": list(state.board_cells),
        "accepted_mask": state.accepted_mask,
        "current_jewel": state.current_jewel,
        "strategy": state.strategy,
        "phase": state.phase,
        "plan_target": state.plan_target,
        "plan_lucky_lines": list(state.plan_lucky_lines),
        "plan_normal_lines": list(state.plan_normal_lines),
    }


def state_from_wire(data: dict[str, Any]) -> GameState:
    raw = dict(data)
    raw["board_cells"] = tuple(raw["board_cells"])
    raw["plan_lucky_lines"] = tuple(raw.get("plan_lucky_lines", ()))
    raw["plan_normal_lines"] = tuple(raw.get("plan_normal_lines", ()))
    raw.setdefault("plan_target", "")
    return GameState(**raw)


def request_to_wire(request: SolverRequest) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request.request_id,
        "state": state_to_wire(request.state),
        "settings": settings_to_wire(request.settings),
        "learned_scores": [[p, s] for p, s in request.learned_scores],
        "purpose": request.purpose,
        "session_id": request.session_id,
    }


def request_from_wire(data: dict[str, Any]) -> SolverRequest:
    if int(data.get("protocol_version", -1)) != PROTOCOL_VERSION:
        raise ValueError("incompatible solver protocol version")
    return SolverRequest(
        request_id=int(data["request_id"]),
        state=state_from_wire(data["state"]),
        settings=settings_from_wire(data["settings"]),
        learned_scores=tuple((int(p), float(s)) for p, s in data.get("learned_scores", ())),
        purpose=str(data.get("purpose", "decision")),
        session_id=str(data.get("session_id", "")),
    )


def response_to_wire(response: SolverResponse) -> dict[str, Any]:
    return {
        "protocol_version": response.protocol_version,
        "request_id": response.request_id,
        "episode_id": response.episode_id,
        "state_version": response.state_version,
        "strategy_version": response.strategy_version,
        "recommendations": [recommendation_to_wire(r) for r in response.recommendations],
        "cache_hit": response.cache_hit,
        "elapsed_ms": response.elapsed_ms,
        "source": response.source,
        "session_id": response.session_id,
    }


def response_from_wire(data: dict[str, Any]) -> SolverResponse:
    if int(data.get("protocol_version", -1)) != PROTOCOL_VERSION:
        raise ValueError("incompatible solver response protocol version")
    return SolverResponse(
        request_id=int(data["request_id"]),
        episode_id=str(data["episode_id"]),
        state_version=int(data["state_version"]),
        strategy_version=int(data["strategy_version"]),
        recommendations=tuple(recommendation_from_wire(r) for r in data.get("recommendations", ())),
        cache_hit=bool(data.get("cache_hit", False)),
        elapsed_ms=float(data.get("elapsed_ms", 0.0)),
        source=str(data.get("source", "remote")),
        session_id=str(data.get("session_id", "")),
        protocol_version=int(data.get("protocol_version", PROTOCOL_VERSION)),
    )
