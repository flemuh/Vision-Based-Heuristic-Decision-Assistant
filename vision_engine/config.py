from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


RISK_PROFILES = {
    # V7.0.5 normal live-play strategy. The planner starts with a locked
    # 3-Lucky route and automatically falls back when that route becomes weak
    # or impossible. Old strategy names remain accepted only for migrations and
    # offline/replay compatibility; the live UI exposes this one mode.
    "auto": {
        "p3_min": 0.08,
        "p2l1n_min": 0.12,
        "p1l2n_min": 0.16,
        "p1l1n_min": 0.22,
        "fallback_override_pp": 0.32,
        "adaptive_score_min": 0.08,
    },
    # Legacy/internal modes retained so historical replays stay readable.
    "adaptive": {
        "p3_min": 0.08, "p2l1n_min": 0.12, "p1l2n_min": 0.16,
        "p1l1n_min": 0.22, "fallback_override_pp": 0.32, "adaptive_score_min": 0.08,
    },
    "score": {
        "p3_min": 1.01, "p2l1n_min": 1.01, "p1l2n_min": 1.01,
        "p1l1n_min": 1.01, "fallback_override_pp": 0.0,
    },
    "lucky": {
        "p3_min": 0.0, "p2l1n_min": 0.10, "p1l2n_min": 0.14,
        "p1l1n_min": 0.18, "fallback_override_pp": 1.01,
    },
}


LEGACY_STRATEGY_ALIASES = {
    "aggressive": "lucky",
    "balanced": "adaptive",
    "conservative": "score",
    "custom": "auto",
}



@dataclass
class AppConfig:
    screen_region: dict | None = None

    # recommend = show advice, observe = collect unbiased games without showing
    # the recommendation, offline = intended for screenshot/replay scripts.
    operation_mode: str = "recommend"
    # risk_profile is retained as the persistence field name for backwards
    # compatibility. In V5.6 it stores one of: adaptive, score, lucky.
    risk_profile: str = "auto"
    mode: str = "auto"

    # V7 live solver backend. Windows keeps capture/UI; the default WSL backend
    # keeps a persistent Linux solver hot across the whole play session.
    solver_backend: str = "wsl"  # wsl, remote, or local
    solver_host: str = "127.0.0.1"
    solver_port: int = 57641
    solver_workers: int = 8
    # Never rely on the global/default WSL distribution. Docker Desktop often
    # makes docker-desktop the default distro, which has no bash/python3.
    # The live solver uses this explicit Linux distro instead.
    solver_wsl_distro: str = "Ubuntu"
    solver_autostart_wsl: bool = True

    # Solver is staged: a small first pass is shown quickly, then a deeper pass
    # refines it only if the screen state has not changed.
    exact_horizon: int = 6
    # Offline/research quality setting. Live GUI requests use the separate
    # bounded horizon below so Monte-Carlo rollouts do not recursively enter
    # a 3-move exact solve on every sample.
    rollout_exact_horizon: int = 3
    live_rollout_exact_horizon: int = 2
    fast_rollouts: int = 60
    # V7 live path uses a bounded publishable Monte-Carlo budget. The existing
    # 1600-rollout setting remains available to offline/research tools; exact
    # mid/endgame is unchanged. Speculative WSL precompute makes most live
    # decisions cache hits before the next jewel appears.
    live_rollouts: int = 120
    # Root-fix stability default: speculative six-jewel background work is OFF
    # in live play. It can be re-enabled explicitly for research/benchmarking.
    live_speculative_precompute: bool = False
    rollouts: int = 1600
    # Deterministic Monte-Carlo scenario seed. Saved with each decision so a
    # recommendation can be reproduced exactly offline.
    rng_seed: int = 1337
    empirical_bias_blend: float = 0.0
    rng_min_games: int = 50
    poll_ms: int = 400
    # During a placement/current-jewel transition poll much faster so 2-of-3
    # consensus can settle without waiting ~0.8s just for two normal frames.
    transition_poll_ms: int = 120

    # Temporal/state validation. Five-frame voting is more tolerant of the blue
    # glow animation than V3's 3/2 default.
    temporal_frames: int = 3
    temporal_required: int = 2
    max_mask_jump: int = 1
    # Multi-cell catch-up is intentionally slower than an ordinary +1 move.
    # A repeated transient selection glow must not be promoted to permanent state.
    catchup_required_frames: int = 5
    # Ordinary +1 moves are provisional for a short time so a transient click
    # glow cannot become an irreversible accepted state.
    single_step_confirm_frames: int = 4
    single_step_rollback_frames: int = 3
    single_step_rollback_window_frames: int = 10
    board_consensus_frames: int = 3
    board_consensus_required: int = 2

    # Adaptive risk gates. These remain editable for backwards compatibility;
    # the selected risk_profile overrides them unless risk_profile="custom".
    p3_min: float = 0.08
    p2l1n_min: float = 0.12
    p1l2n_min: float = 0.16
    p1l1n_min: float = 0.22
    fallback_override_pp: float = 0.32
    # Below this best P(>1000), Adaptive enters mixed-line salvage instead of
    # chasing a technically non-zero but practically weak score path.
    adaptive_score_min: float = 0.08
    p3_near_best_tolerance: float = 0.025
    # Automatic planner: switch away from 3 Lucky when the locked route is
    # both weak and clearly dominated by the next fallback. Route hysteresis
    # prevents changing concrete line sets for tiny probability differences.
    auto_p3_floor: float = 0.04
    auto_fallback_ratio: float = 3.0
    route_switch_ratio: float = 1.35

    # Local ML is never allowed to override a clearly better mathematical move.
    ml_enabled: bool = True
    ml_near_tie_probability: float = 0.012
    ml_near_tie_score: float = 15.0
    ml_min_validation_games: int = 8
    ml_holdout_fraction: float = 0.25

    # UI / capture. Overlay is OFF by default. Diagnostic disk I/O is also
    # OFF by default so Auto Play correctness never depends on file-write timing.
    overlay_enabled: bool = False
    save_runtime_logs: bool = False
    save_game_images: bool = False

    # V18.0.10 Auto Play TEST. Explicit, visible UI automation only. Timing is
    # intentionally conservative so capture/animation/vision can settle before
    # every action. Bounded variability is for UI robustness, not stealth or
    # anti-cheat evasion. A pacing clock aims for ~51.4 s/round on a 700-game
    # run (~10 h total) but never shortens these safety floors to catch up.
    autoplay_click_delay_min_ms: int = 900
    autoplay_click_delay_max_ms: int = 1600
    autoplay_extra_settle_min_ms: int = 200
    autoplay_extra_settle_max_ms: int = 700
    autoplay_move_duration_min_ms: int = 220
    autoplay_move_duration_max_ms: int = 380
    autoplay_click_jitter_fraction: float = 0.18
    autoplay_transition_timeout_ms: int = 6500
    autoplay_target_game_seconds: float = 51.4
    autoplay_expected_transition_ms: int = 550
    autoplay_max_pacing_wait_ms: int = 4200
    autoplay_default_games: int = 1
    auto_relocate: bool = True
    relocate_threshold: float = 0.80
    save_debug_frames: bool = False
    ask_on_low_confidence: bool = False

    # Result verification. Live OCR was removed from the hot path in P5.
    # Keep the legacy flag only so old app_config.json files continue to load;
    # runtime verification is the trusted 14/14 computed path (>1000).
    result_ocr_enabled: bool = False
    auto_verify_computed_gt1000: bool = True

    # Continuous diagnostics are written asynchronously as hourly JSONL files.
    # Retention is enforced by the writer thread, never by the mouse/vision path.
    continuous_json_logs: bool = True
    runtime_heartbeat_ms: int = 2000
    runtime_resource_sample_ms: int = 5000
    runtime_log_keep_days: int = 7
    runtime_log_max_mb: int = 250

    # Heavy ML/backup maintenance is deliberately deferred while live monitoring
    # is running. Re-training the entire accumulated dataset after every game was
    # able to overlap vision/input and become increasingly expensive as the local
    # database grew.
    maintenance_during_live: bool = False

    # Data durability.
    backup_every_games: int = 10
    keep_backups: int = 8

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        p = Path(path)
        if not p.exists():
            return cls()
        raw = json.loads(p.read_text(encoding="utf-8"))
        allowed = {f.name for f in fields(cls)}
        cfg = cls(**{k: v for k, v in raw.items() if k in allowed})
        # V18.0.10: persistent user data deliberately keeps app_config.json
        # across ZIP upgrades. Migrate only the exact V18.0.9 Auto Play timing
        # defaults so user-customized timings are never overwritten.
        legacy_autoplay_1809 = (
            raw.get("autoplay_click_delay_min_ms") == 550
            and raw.get("autoplay_click_delay_max_ms") == 950
            and raw.get("autoplay_extra_settle_min_ms") == 50
            and raw.get("autoplay_extra_settle_max_ms") == 500
            and raw.get("autoplay_move_duration_min_ms") == 180
            and raw.get("autoplay_move_duration_max_ms") == 320
            and raw.get("autoplay_transition_timeout_ms") == 4500
        )
        if legacy_autoplay_1809:
            cfg.autoplay_click_delay_min_ms = 900
            cfg.autoplay_click_delay_max_ms = 1600
            cfg.autoplay_extra_settle_min_ms = 200
            cfg.autoplay_extra_settle_max_ms = 700
            cfg.autoplay_move_duration_min_ms = 220
            cfg.autoplay_move_duration_max_ms = 380
            cfg.autoplay_transition_timeout_ms = 6500
        # V7.0.5 removes manual live strategy selection. Existing installations
        # migrate transparently to the automatic Win >1000 planner.
        if cfg.risk_profile in {"adaptive", "score", "lucky", "aggressive", "balanced", "conservative", "custom"}:
            cfg.risk_profile = "auto"
        cfg.apply_risk_profile()
        return cfg

    def apply_risk_profile(self) -> None:
        self.risk_profile = LEGACY_STRATEGY_ALIASES.get(self.risk_profile, self.risk_profile)
        if self.risk_profile not in RISK_PROFILES:
            self.risk_profile = "auto"
        profile = RISK_PROFILES[self.risk_profile]
        for k, v in profile.items():
            setattr(self, k, v)
        self.mode = {"auto": "auto", "adaptive": "adaptive", "score": "score", "lucky": "lucky"}[self.risk_profile]

    def set_risk_profile(self, name: str) -> None:
        name = LEGACY_STRATEGY_ALIASES.get(name, name)
        if name not in RISK_PROFILES:
            raise ValueError(f"Unknown strategy: {name}")
        self.risk_profile = name
        self.apply_risk_profile()

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
