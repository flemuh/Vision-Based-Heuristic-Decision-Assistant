from __future__ import annotations

import json
import multiprocessing as mp
import queue
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

import cv2

from .advisor import Advisor, Recommendation, utility_learning_target
from .autocalibration import (
    detect_board_consensus,
    detect_board_grid,
    detect_panel_from_screen,
    infer_current_jewel_roi,
    infer_panel_roi_from_board,
)
from .board import Board
from .capture import (
    ScreenRegion,
    capture_region,
    relocate_region_by_anchor,
    relocate_region_by_anchor_in_image,
    save_region_anchor,
    screenshot_all,
)
from .config import AppConfig
from .calibration import (
    detect_reference_jewels_detailed,
    quality_band,
    validate_board_roi,
    validate_game_panel,
    validate_number_roi,
)
from .constants import CENTER, JEWEL_NAMES, JEWELS
from .experience import ExperiencePolicy
from .history import HistoryStore
from .icon_matcher import CurrentJewelMatcher, IconMatch
from .learning import learned_bias_weights
from .learning_status import build_learning_snapshot
from .overlay import MoveOverlay
from .persistence import BingoDB, board_hash
from .recovery import reconcile_masks
from .replay import format_replay_report, list_stored_games, stored_replay
from .research_export import export_research_dataset
from .data_home import prepare_user_data
from .diagnostics.resource_trend import sample_resources
from .diagnostics.runtime_journal import RuntimeJournal
from .diagnostics.win32_resources import collect_gui_resources, collect_process_vitals
from .diagnostics.waiting import WaitingDiagnostics
from .diagnostics.win32 import set_win32_sink
from .result_reader import VerifiedResult
from .result_verification import computed_gt1000_verification
from .round_lifecycle import CompletionDetector, NewRoundDetector
from .roi_selector import (
    confirm_board_alignment_dialog,
    confirm_geometry_dialog,
    confirm_reference_templates_dialog,
    show_vision_validation_dialog,
    select_roi_dialog,
)
from .scoring import score_mask
from .solver_process import (
    read_solver_subprocess_result,
    solve_request,
    start_solver_process,
    start_solver_subprocess,
)
from .temporal import BoardConsensusFilter, TemporalStateFilter
from .strategy import compact_move_line_summary, move_line_impacts, viability_label
from .route_math import ROUTE_NOMINAL_SCORE, line_feasibilities, route_by_name, route_probability
from .state_machine import GamePhase, GameStateMachine
from .ui_semantics import move_visual_status, visual_status_style
from .live.state import GameState as LiveGameState
from .live.coordinator import LiveCoordinator
from .live.solver_bridge import LiveSolverBridge
from .solver.protocol import SolverRequest
from .versioning import APP_VERSION
from .vision import (
    JewelClassifier,
    VisionConfig,
    board_cell_rois,
    board_patch,
    crop_roi,
    read_board,
    read_board_detailed,
    read_current,
    read_current_detailed,
    read_selected_mask,
)


class BingoApp(tk.Tk):
    def __init__(self, root_dir: Path):
        super().__init__()
        self.root_dir = root_dir
        # Mutable runtime data is intentionally outside the extracted ZIP folder.
        # Upgrading versions must not reset calibration, learned policy or saved games.
        self.data_dir, self._legacy_db_candidates = prepare_user_data(root_dir)
        self._main_thread_id = threading.get_ident()
        self._ui_queue: queue.Queue = queue.Queue()
        self.title(f"Speedlora Jewel Bingo Assistant V18 — {APP_VERSION}")
        self.geometry("980x790")
        self.attributes("-topmost", True)

        self.app_cfg_path = self.data_dir / "app_config.json"
        self.vision_cfg_path = self.data_dir / "vision_config.json"
        self.app_cfg = AppConfig.load(self.app_cfg_path)
        self.vision_cfg = VisionConfig.load(self.vision_cfg_path)

        # Separate visual domains. Board cells have frames/selection glow while
        # the top current-jewel icon matches the colored x4 reference sprites.
        # V5.1 used one classifier for both domains, which caused high geometry
        # confidence but near-zero board classification confidence.
        self.clf = JewelClassifier(
            self.data_dir / "board_templates",
            root_dir / "assets" / "seed_board_features.npz",
        )
        icon_dir = self.data_dir / "icon_templates"
        icon_dir.mkdir(parents=True, exist_ok=True)
        # Best-effort migration from V5/V5.1 where x4/current examples lived in
        # data/templates. They remain only current-icon references in V5.2.
        legacy_icon_dir = self.data_dir / "templates"
        if not any(icon_dir.glob("*.png")) and legacy_icon_dir.exists():
            for old in legacy_icon_dir.glob("*.png"):
                try:
                    shutil.copy2(old, icon_dir / old.name)
                except Exception:
                    pass
        self.icon_matcher = CurrentJewelMatcher(icon_dir)
        if self.icon_matcher.ready() and not self.vision_cfg.templates_validated:
            self.vision_cfg.templates_validated = True
            self.vision_cfg.template_geometry_score = max(0.72, self.vision_cfg.template_geometry_score)
            self.vision_cfg.save(self.vision_cfg_path)
        self.history = HistoryStore(self.data_dir / "history.json")
        self.db = BingoDB(self.data_dir / "bingo.sqlite3")
        # Preserve/migrate V2 data. INSERT OR IGNORE keeps this safe across restarts.
        try:
            self.db.import_legacy_json(self.data_dir / "history.json", self.data_dir / "episodes.json")
        except Exception:
            pass
        # Merge databases from previous extracted versions once. This makes ZIP
        # upgrades cumulative instead of starting a new research history each time.
        for _old_db in self._legacy_db_candidates:
            try:
                self.db.merge_database(_old_db)
            except Exception:
                pass
        self.experience = ExperiencePolicy(
            self.data_dir / "bingo.sqlite3",
            self.data_dir / "policy_model_v4.joblib",
            self.data_dir / "episodes.json",
        )

        # One immutable session id ties together solver ownership and all hourly
        # JSON diagnostics from this Windows launch.
        self.session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S_%f")

        self.image_dir = self.data_dir / "game_images"
        self.backup_dir = self.data_dir / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.decision_image_dir = self.data_dir / "decision_images"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.decision_image_dir.mkdir(parents=True, exist_ok=True)
        self.anchor_path = self.data_dir / "panel_anchor.png"
        self.log_dir = self.data_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.auto_setup_log_path = self.log_dir / "assistant.log"
        self.runtime_journal = RuntimeJournal(
            self.log_dir / "runtime", self.session_id,
            keep_days=self.app_cfg.runtime_log_keep_days,
            max_total_mb=self.app_cfg.runtime_log_max_mb,
        ) if self.app_cfg.continuous_json_logs else None
        set_win32_sink(
            None if self.runtime_journal is None
            else lambda kind, data: self.runtime_journal.record(f"win32-{kind}", **data)
        )
        self._runtime_tick_expected = time.monotonic() + max(0.5, self.app_cfg.runtime_heartbeat_ms / 1000.0)
        self._runtime_last_resource_sample = 0.0
        self._runtime_last_wait_signature: tuple | None = None
        self.auto_setup_running = False
        self.auto_setup_cancel = threading.Event()
        self.auto_setup_thread: threading.Thread | None = None

        # V5.6 solves in a spawned PROCESS, not a Python thread. Monte Carlo and
        # Expectimax are CPU-bound, so a thread still competed with Tk for the GIL.
        # A disposable process also lets latest-state-wins cancel stale work.
        self._solver_lock = threading.Lock()  # retained for synchronous offline helpers
        self._solver_ctx = mp.get_context("spawn")
        self.solver_process = None
        self.solver_queue = None
        self.solver_poll_job = None
        self.solver_started_at = 0.0
        self.solver_stage: str | None = None
        self.solver_subprocess = None
        self.solver_subprocess_paths: tuple[Path, Path] | None = None
        self.solver_fallback_thread: threading.Thread | None = None
        self.solver_fallback_key: tuple | None = None
        self.solver_fallback_stage: str | None = None
        self.recommend_request_id = 0
        # V18.0.26: each Windows launch owns the session created above. The WSL
        # service drains stale work before admitting this session.
        self.recommend_active_key: tuple | None = None
        self.remote_solver_active_key: tuple | None = None
        self.live_coordinator: LiveCoordinator | None = None
        self.solver_bridge = LiveSolverBridge(
            root_dir,
            host=self.app_cfg.solver_host,
            port=self.app_cfg.solver_port,
            workers=self.app_cfg.solver_workers,
            wsl_distro=self.app_cfg.solver_wsl_distro,
            auto_start_wsl=(self.app_cfg.solver_backend == "wsl" and self.app_cfg.solver_autostart_wsl),
            session_id=self.session_id,
            backend=self.app_cfg.solver_backend,
        )

        self.region: ScreenRegion | None = None
        if self.app_cfg.screen_region:
            try:
                self.region = ScreenRegion(**self.app_cfg.screen_region)
            except Exception:
                self.region = None

        # V18.0.25: the monitor publishes several fields as one logical frame.
        # Protect snapshots/commits from observing mixed old/new frame state.
        self._live_state_lock = threading.RLock()

        self.board: Board | None = None
        self.board_confidence = 0.0
        self.selected_mask = 0
        self.current_jewel: str | None = None
        self.current_confidence = 0.0
        self.current_margin = 0.0
        self.current_entropy = 1.0
        # A successful vision test is required once per launch before START is
        # unlocked. ROI/template calibration alone is not enough.
        self.vision_test_passed = False
        self.running = False
        self.worker: threading.Thread | None = None
        self.last_panel = None
        self.last_current_patch = None
        self.last_current_confidence = 0.0
        self.game_sequence: list[str] = []
        self.episode_actions: list[dict] = []
        self.one_left_saved = False
        self.one_left_path: str | None = None
        self.episode_finalized = False
        self.completion_pending = False
        self._last_checkpoint_signature = None
        self.episode_id = self._new_episode_id()
        self.box_color: str | None = None
        self.awaiting_result_until = 0.0
        self.last_verified_result: dict | None = None
        self.new_round_detector = NewRoundDetector(required=2, max_selected=1)
        self.completion_detector = CompletionDetector(min_accepted=12, disappear_required=2, recent_14_window_frames=10)
        self.maintenance_thread: threading.Thread | None = None
        self._maintenance_reload_pending = False
        self._maintenance_pending_episode_id: str | None = None
        self._maintenance_pending_count = 0
        self._last_transition_started_at: float | None = None
        self._last_solver_roundtrip_ms: float | None = None
        self._last_solver_compute_ms: float | None = None
        self.last_recs: list[Recommendation] = []
        # Persist the latest visual ranking across ordinary grid refreshes.
        # V5.6.1 could calculate FAST, then repaint the grid back to neutral before
        # Windows had a chance to draw the recommendation colors.
        self._ranked_grid_mask: int | None = None
        self._ranked_grid_recs: list[Recommendation] = []
        # Amber preview is advisory only; final colored ranking is authoritative.
        self._preview_grid_mask: int | None = None
        self._preview_grid_pos: int | None = None
        self._last_monitor_grid_signature: tuple | None = None
        # V18 live UX remembers the displayed plan so target changes can be
        # explained explicitly instead of silently changing a label.
        self._ui_last_plan_target: str | None = None
        self.cached_recommend_key: tuple | None = None
        self.last_decision_key: tuple | None = None
        self.last_decision_image: str | None = None
        self.last_state_warning: str | None = None
        self.temporal = TemporalStateFilter(
            self.app_cfg.temporal_frames,
            self.app_cfg.temporal_required,
            self.app_cfg.max_mask_jump,
            catchup_required_frames=self.app_cfg.catchup_required_frames,
            single_step_confirm_frames=self.app_cfg.single_step_confirm_frames,
            single_step_rollback_frames=self.app_cfg.single_step_rollback_frames,
            single_step_rollback_window_frames=self.app_cfg.single_step_rollback_window_frames,
        )
        self.temporal_initialized = False
        self.board_consensus = BoardConsensusFilter(
            self.app_cfg.board_consensus_frames, self.app_cfg.board_consensus_required
        )
        self._live_board_failures = 0
        self._live_board_last_log = 0.0
        self._last_icon_log_key: tuple | None = None
        self._last_icon_log_at = 0.0
        self._raw_selected_count = 0
        self._transition_fast_until = 0.0
        self.state_machine = GameStateMachine()
        self.phase = GamePhase.NO_BOARD
        self.overlay = MoveOverlay(self)

        # Public build: recommendation-only. The assistant never sends mouse or
        # keyboard input to the game window; the player makes every move.

        # V18.0.26 diagnostic probe: all hot-path events stay in memory.
        # diagnose_waiting.bat connects over localhost and writes the bundle.
        self.waiting_diagnostics = WaitingDiagnostics(capacity=4096)
        if self.runtime_journal is not None:
            self.waiting_diagnostics.set_sink(
                lambda kind, data: self.runtime_journal.record(kind, **data)
            )
            self.runtime_journal.record(
                "app-start", app_version=APP_VERSION, data_dir=str(self.data_dir),
                solver_backend=str(self.app_cfg.solver_backend),
            )
        self._diag_monitor_heartbeat = 0.0
        self._diag_last_transition = 0.0
        self._diag_last_confirm = 0.0
        self._diag_last_stable_reason = ""
        self._waiting_probe_marker = self.data_dir / "waiting_probe.json"
        self.waiting_diagnostics.set_provider(self._waiting_diagnostic_snapshot)
        try:
            probe_port = self.waiting_diagnostics.start(self._waiting_probe_marker)
            self.waiting_diagnostics.record("probe-started", port=probe_port, session_id=self.session_id)
        except Exception as exc:
            self.waiting_diagnostics.record("probe-start-failed", error=f"{type(exc).__name__}: {exc}")

        self._build()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._startup_status()
        # Do not start a full-history ML rebuild during application startup.
        # Training cost grows with the local DB and used to overlap START/live
        # monitoring when the user began immediately. The Learning tab keeps the
        # explicit rebuild button; normal post-game maintenance is deferred until
        # monitoring is stopped.
        if self.db.completed_game_count() > 0 and self.experience.info.hindsight_samples == 0:
            self._runtime_record(
                "learning-rebuild-pending", completed_games=self.db.completed_game_count(),
                reason="startup-live-safety",
            )
        if self.app_cfg.solver_backend != "local":
            self.solver_bridge.start_async(
                lambda ok, status: self._ui_call(self._solver_service_status, ok, status)
            )
        self.after(50, self._drain_ui_queue)
        self.after(max(500, int(self.app_cfg.runtime_heartbeat_ms)), self._runtime_tick)

    @staticmethod
    def _new_episode_id() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    def _templates_ready(self) -> bool:
        return self.icon_matcher.ready()

    def _calibration_missing(self) -> list[str]:
        missing: list[str] = []
        if self.region is None:
            missing.append("game area")
        if self.vision_cfg.board_roi is None:
            missing.append("5x5 board area")
        elif not self.vision_cfg.board_roi_validated:
            missing.append("validated 5x5 board area")
        if not self._templates_ready():
            missing.append("6 jewel templates")
        elif not self.vision_cfg.templates_validated:
            missing.append("validated jewel templates")
        # Current-jewel location is dynamic in V5.2. It is detected from the
        # x4 reference icons when a box is open, so no manual/static ROI is a
        # setup requirement anymore.
        return missing

    def _calibration_ready(self) -> bool:
        return not self._calibration_missing()

    def _invalidate_vision_test(self):
        self.vision_test_passed = False
        self.board = None
        self.board_confidence = 0.0
        self.current_jewel = None
        self.current_confidence = 0.0
        self.current_margin = 0.0
        self.current_entropy = 1.0
        self.board_meta.set("Board: not validated this session") if hasattr(self, "board_meta") else None
        self.current_label.set("Current jewel: waiting") if hasattr(self, "current_label") else None
        self.vision_label.set("Vision: waiting for test") if hasattr(self, "vision_label") else None
        self.turn_label.set("Placed: 0/14 | Waiting for stable screen state") if hasattr(self, "turn_label") else None

    def _ui_call(self, func, *args, **kwargs):
        if threading.get_ident() == self._main_thread_id:
            try:
                func(*args, **kwargs)
            except tk.TclError:
                pass
        else:
            self._ui_queue.put((func, args, kwargs))

    def _drain_ui_queue(self):
        try:
            while True:
                func, args, kwargs = self._ui_queue.get_nowait()
                try:
                    func(*args, **kwargs)
                except tk.TclError:
                    pass
                except Exception:
                    pass
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(50, self._drain_ui_queue)

    def _set_status(self, text: str):
        self._ui_call(self.status.set, text)

    def _setup_feedback(self, text: str, kind: str = "info"):
        """Prominent first-run feedback kept inside the setup box.

        Users should never have to infer whether a setup step was accepted.
        """
        if not hasattr(self, "setup_feedback_var"):
            return
        prefix = {"ok": "✅ ", "warn": "⚠ ", "error": "❌ ", "info": "ℹ "}.get(kind, "")
        color = {"ok": "#167a2d", "warn": "#8a6500", "error": "#b00020", "info": "#24527a"}.get(kind, "#24527a")
        self.setup_feedback_var.set(prefix + text)
        try:
            self.setup_feedback_label.configure(foreground=color)
        except Exception:
            pass

    def _ask_continue_setup(self, completed_step: int, message: str, next_callable=None):
        self._setup_feedback(message, "ok")
        if next_callable is None:
            return
        go = messagebox.askyesno(
            f"Step {completed_step} complete",
            message + "\n\nContinue to the next setup step now?",
            parent=self,
        )
        if go:
            self.after(120, next_callable)

    def _refresh_setup_state(self):
        if not hasattr(self, "setup_var"):
            return
        counts = self.icon_matcher.template_counts()
        ntempl = sum(1 for j in JEWELS if counts.get(j, 0) >= 1)
        total_examples = sum(counts.values())

        def scored(mark_ok: bool, label: str, score: float) -> str:
            if not mark_ok:
                return f"❌ {label}"
            band = quality_band(score) if score > 0 else "REVIEWED"
            icon = "✅" if band == "HIGH" else "🟠" if band in ("MEDIUM", "LOW") else "✅"
            suffix = f" {score:.2f}/{band}" if score > 0 else ""
            return f"{icon} {label}{suffix}"

        game_text = scored(self.region is not None, "Game area", self.vision_cfg.game_area_geometry_score)
        board_text = scored(
            self.vision_cfg.board_roi is not None and self.vision_cfg.board_roi_validated,
            "Board", self.vision_cfg.board_geometry_score,
        )
        templ_text = scored(
            self._templates_ready() and self.vision_cfg.templates_validated,
            f"Jewels {ntempl}/6 ({total_examples} images)", self.vision_cfg.template_geometry_score,
        )
        current_text = "🔄 Current jewel AUTO" if self._templates_ready() else "○ Current jewel waiting for references"
        vision_text = "✅ Live vision" if self.vision_test_passed else "○ Live check runs on START"
        self.setup_var.set(
            "Setup: " + "   ".join([game_text, board_text, templ_text, current_text, vision_text])
        )
        if hasattr(self, "start_btn") and not self.running:
            can_start = self._calibration_ready() and not self.auto_setup_running
            self.start_btn.config(state="normal" if can_start else "disabled", text="START")

    def _startup_status(self):
        self._refresh_setup_state()
        missing = self._calibration_missing()
        if not missing:
            self.status.set(
                "Calibration loaded. Open/start a box and press START. V5 runs the live vision test automatically before monitoring. "
                f"ML: {self.experience.info.counterfactual_samples} counterfactual + "
                f"{self.experience.info.outcome_samples} real-outcome samples."
            )
            self._setup_feedback("Saved automatic calibration loaded. Open/start a box and press START; the current jewel is located automatically.", "ok")
        else:
            self.status.set("First use: keep Jewel Bingo visible and click AUTO SETUP. Manual calibration is only a fallback.")
            self._setup_feedback("Recommended: keep Jewel Bingo open (idle/x4 list visible) and click AUTO SETUP.", "info")

    def _open_repair_dialog(self):
        """Compact recovery surface for rare manual intervention.

        V5.6.3 removes the always-present Troubleshooting panel from the normal
        workflow. The underlying recovery operations remain available here
        because AUTO SETUP can still legitimately need them on unusual layouts.
        """
        win = tk.Toplevel(self)
        win.title("Repair / recovery")
        win.transient(self)
        win.attributes("-topmost", True)
        win.resizable(False, False)

        outer = ttk.Frame(win, padding=12)
        outer.pack(fill="both", expand=True)
        ttk.Label(
            outer,
            text=(
                "Use these controls only if automatic setup or live tracking cannot recover. "
                "Normal play should use AUTO SETUP + START."
            ),
            wraplength=560,
        ).pack(anchor="w", pady=(0, 10))

        setup = ttk.LabelFrame(outer, text="Setup repair", padding=8)
        setup.pack(fill="x")
        row = ttk.Frame(setup); row.pack(fill="x")
        ttk.Button(row, text="Re-detect moved panel", command=lambda: self.find_moved_area()).pack(side="left", padx=2)
        ttk.Button(row, text="Refresh jewel refs", command=self.calibrate_templates).pack(side="left", padx=2)
        ttk.Button(row, text="Test vision", command=self.read_once).pack(side="left", padx=2)

        recovery = ttk.LabelFrame(outer, text="Advanced recovery", padding=8)
        recovery.pack(fill="x", pady=(10, 0))
        row2 = ttk.Frame(recovery); row2.pack(fill="x")
        ttk.Button(row2, text="Manual panel", command=self.select_area).pack(side="left", padx=2)
        ttk.Button(row2, text="Manual board", command=self.calibrate_board_roi).pack(side="left", padx=2)
        ttk.Button(row2, text="Correct current jewel", command=self.correct_current).pack(side="left", padx=2)
        ttk.Button(row2, text="Reset tracking", command=self.reset_state).pack(side="left", padx=2)
        ttk.Button(row2, text="Help", command=self.show_setup_help).pack(side="left", padx=2)

        ttk.Button(outer, text="Close", command=win.destroy).pack(anchor="e", pady=(10, 0))

    def _build(self):
        """Compact game and ML tabs; technical logs remain available on disk."""
        self.compact_game_ui = True
        # Respect the persisted operation mode; never silently enable the solver.
        if self.app_cfg.operation_mode not in {"observe", "recommend"}:
            self.app_cfg.operation_mode = "recommend"
        self.minsize(660, 530)
        self.geometry("810x660")
        # R3.1: Keep the two operating modes OUTSIDE the Game/ML notebook.
        # They must remain visible in BOTH tabs, even on compact displays.
        self.operation_var = tk.StringVar(value=self.app_cfg.operation_mode)
        self.mode_status_var = tk.StringVar(value=(
            "OBSERVABLE ativo: captura e histórico, sem solver"
            if self.app_cfg.operation_mode == "observe"
            else "PLAY ativo: solver 3/4 e recomendações"
        ))
        self.mode_toolbar = ttk.LabelFrame(self, text="MODO DE OPERAÇÃO — sempre visível", padding=(10, 5, 10, 6))
        self.mode_toolbar.pack(fill="x", padx=9, pady=(7, 3))
        mode_actions = ttk.Frame(self.mode_toolbar)
        mode_actions.pack(fill="x")
        self.observable_mode_button = ttk.Radiobutton(
            mode_actions, text="OBSERVABLE (sem cálculo)", variable=self.operation_var,
            value="observe", command=self._change_operation_mode,
        )
        self.observable_mode_button.pack(side="left", padx=(0, 22))
        self.play_mode_button = ttk.Radiobutton(
            mode_actions, text="PLAY / JOGAR (solver 3/4)", variable=self.operation_var,
            value="recommend", command=self._change_operation_mode,
        )
        self.play_mode_button.pack(side="left")
        ttk.Label(self.mode_toolbar, textvariable=self.mode_status_var,
                  wraplength=770).pack(anchor="w", pady=(4, 0))
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)
        game = ttk.Frame(self.notebook)
        self.notebook.add(game, text="Jogo")
        self.learning_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.learning_tab, text="Aprendizado (ML)")
        self.notebook.bind("<<NotebookTabChanged>>", self._on_main_tab_changed)

        setup = ttk.Frame(game, padding=(10, 8, 10, 4))
        self.setup_frame = setup
        setup.pack(fill="x")
        self.auto_setup_btn = ttk.Button(setup, text="AUTO SETUP", command=self.auto_setup)
        self.auto_setup_btn.pack(side="left", ipadx=8, ipady=4)
        ttk.Button(setup, text="Repair", command=self._open_repair_dialog).pack(side="right", padx=(6, 0))
        self.setup_var = tk.StringVar(value="Setup: checking...")
        ttk.Label(setup, textvariable=self.setup_var, wraplength=510).pack(side="left", fill="x", expand=True, padx=12)
        self.setup_feedback_var = tk.StringVar(value="Keep Jewel Bingo visible; use AUTO SETUP first.")
        # Feedback is still available for errors but no technical log is shown.
        self.setup_feedback_label = ttk.Label(game, textvariable=self.setup_feedback_var,
                                             wraplength=740, foreground="#24527a")
        self.setup_feedback_label.pack(fill="x", padx=12)
        self.setup_progress_row = ttk.Frame(game, padding=(10, 0, 10, 2))
        self.setup_progress_row.pack(fill="x")
        self.auto_progress_var = tk.DoubleVar(value=0.0)
        self.auto_progress = ttk.Progressbar(self.setup_progress_row, maximum=100.0,
                                              variable=self.auto_progress_var)
        self.auto_progress.pack(side="left", fill="x", expand=True)
        self.auto_progress_text = tk.StringVar(value="Ready")
        ttk.Label(self.setup_progress_row, textvariable=self.auto_progress_text,
                  width=27).pack(side="left", padx=(7, 0))

        ctl = ttk.Frame(game, padding=(10, 7, 10, 5))
        self.controls_frame = ctl
        ctl.pack(fill="x")
        self.start_btn = ttk.Button(ctl, text="START", command=self.toggle)
        self.start_btn.pack(side="left", ipadx=12, ipady=4)
        # R3.1: The operating-mode controls live in the global toolbar above
        # the notebook, so they cannot disappear behind Game/ML tab switching.
        self.strategy_var = tk.StringVar(value="Win >1000 / Adaptive")
        self.target_var = tk.StringVar(value="Target: waiting")
        self.phase_var = tk.StringVar(value="Phase: no-board")
        ttk.Label(ctl, text="Box:").pack(side="left", padx=(12, 4))
        self.box_color_var = tk.StringVar(value="")
        for name, value in (("Green", "green"), ("Blue", "blue"), ("Red", "red")):
            ttk.Radiobutton(ctl, text=name, variable=self.box_color_var, value=value,
                            command=self._on_box_color_change).pack(side="left", padx=1)
        ttk.Button(ctl, text="Clear", command=self._clear_box_color).pack(side="left", padx=3)

        main = ttk.Frame(game, padding=(10, 5, 10, 6))
        main.pack(fill="both", expand=True)
        left = ttk.Frame(main)
        left.pack(side="left", anchor="n")
        right = ttk.Frame(main, padding=(14, 0, 0, 0))
        right.pack(side="left", fill="both", expand=True)

        self.cell_vars = []
        grid = ttk.Frame(left)
        grid.pack(anchor="n")
        for r in range(5):
            cells = []
            for c in range(5):
                i = r * 5 + c
                v = tk.StringVar(value="MU" if i == CENTER else "??")
                b = tk.Button(grid, textvariable=v, width=6, height=2,
                              command=lambda idx=i: self.toggle_cell(idx))
                b.grid(row=r, column=c, padx=2, pady=2)
                cells.append((v, b))
            self.cell_vars.append(cells)
        self.board_meta = tk.StringVar(value="Board: not read")
        ttk.Label(left, textvariable=self.board_meta, wraplength=340).pack(
            anchor="w", pady=(8, 0))

        self.current_label = tk.StringVar(value="Current jewel: waiting")
        ttk.Label(right, textvariable=self.current_label,
                  font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.turn_label = tk.StringVar(value="Placed: 0/14")
        ttk.Label(right, textvariable=self.turn_label).pack(anchor="w", pady=(4, 0))
        self.vision_label = tk.StringVar(value="Vision: waiting")
        ttk.Label(right, textvariable=self.vision_label).pack(anchor="w")
        self.target_change_var = tk.StringVar(value="")
        self.target_change_label = ttk.Label(right, textvariable=self.target_change_var,
                                             wraplength=390, foreground="#8a5a00")
        self.target_change_label.pack(anchor="w", pady=(6, 0))

        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=(8, 6))
        self.live_move_var = tk.StringVar(value="WAITING")
        ttk.Label(right, textvariable=self.live_move_var, font=("Segoe UI", 19, "bold"),
                  foreground="#176b2c").pack(anchor="w")
        self.live_target_var = tk.StringVar(value="Target: waiting")
        ttk.Label(right, textvariable=self.live_target_var, wraplength=400,
                  font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(6, 0))
        self.live_route_var = tk.StringVar(value="Route: waiting")
        ttk.Label(right, textvariable=self.live_route_var, wraplength=400).pack(anchor="w")
        self.live_health_var = tk.StringVar(value="Plan: --")
        self.live_slack_var = tk.StringVar(value="Slack: --")
        self.live_confidence_var = tk.StringVar(value="Confidence: --")
        self.live_reason_var = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.live_health_var, wraplength=400).pack(anchor="w", pady=(4, 0))
        ttk.Label(right, textvariable=self.live_slack_var).pack(anchor="w")
        ttk.Label(right, textvariable=self.live_confidence_var).pack(anchor="w")
        ttk.Label(right, textvariable=self.live_reason_var, wraplength=400).pack(
            anchor="w", pady=(4, 0))
        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=7)
        self.live_gt1000_var = tk.StringVar(value="Win >1000: --")
        self.live_target_chance_var = tk.StringVar(value="Goal possible: --")
        self.live_route_chance_var = tk.StringVar(value="Route possible: --")
        self.live_p3_var = tk.StringVar(value="3 Lucky: --")
        self.live_expected_var = tk.StringVar(value="Expected pts: --")
        for value in (self.live_gt1000_var, self.live_target_chance_var,
                      self.live_route_chance_var, self.live_p3_var):
            ttk.Label(right, textvariable=value).pack(anchor="w")

        self.status = tk.StringVar(value="Starting...")
        ttk.Label(game, textvariable=self.status, wraplength=765,
                  padding=(12, 4, 12, 10)).pack(fill="x")

        self._build_learning_tab()

    def _on_main_tab_changed(self, _event=None) -> None:
        if self.notebook.select() == str(self.learning_tab):
            self._refresh_learning_panel()

    def _build_learning_tab(self) -> None:
        """Simple user-facing ML dashboard; research diagnostics are opt-in."""
        # A fixed-height notebook must remain usable on 660px and smaller screens.
        # The entire learning page scrolls, so technical details can never be clipped.
        self.learning_scroll = tk.Canvas(self.learning_tab, highlightthickness=0,
                                         borderwidth=0)
        learning_scrollbar = ttk.Scrollbar(self.learning_tab, orient="vertical",
                                           command=self.learning_scroll.yview)
        self.learning_scroll.configure(yscrollcommand=learning_scrollbar.set)
        learning_scrollbar.pack(side="right", fill="y")
        self.learning_scroll.pack(side="left", fill="both", expand=True)
        content = ttk.Frame(self.learning_scroll, padding=(2, 0, 8, 12))
        self.learning_scroll_window = self.learning_scroll.create_window(
            (0, 0), window=content, anchor="nw",
        )
        content.bind("<Configure>", lambda _event: self.learning_scroll.configure(
            scrollregion=self.learning_scroll.bbox("all")))
        self.learning_scroll.bind("<Configure>", lambda event: self.learning_scroll.itemconfigure(
            self.learning_scroll_window, width=event.width))
        # Wheel over the learning panel itself; scrollbar stays available everywhere.
        self.learning_scroll.bind("<MouseWheel>",
                                  lambda event: self.learning_scroll.yview_scroll(
                                      -int(event.delta / 120), "units"))
        self.learning_scroll.bind("<Button-4>",
                                  lambda _event: self.learning_scroll.yview_scroll(-1, "units"))
        self.learning_scroll.bind("<Button-5>",
                                  lambda _event: self.learning_scroll.yview_scroll(1, "units"))
        ttk.Label(content, text="Aprendizado das partidas",
                  font=("Segoe UI", 14, "bold")).pack(anchor="w")
        ttk.Label(
            content,
            text=("O solver matemático continua responsável pelas jogadas. "
                  "O histórico só pode desempatar quando o tabuleiro e a sequência anterior "
                  "forem idênticos aos de uma partida válida com 999 pontos ou mais."),
            wraplength=745,
        ).pack(anchor="w", pady=(3, 8))

        # Keep existing variable names for maintenance jobs and integrations.
        self.learning_games_var = tk.StringVar(value="Carregando histórico...")
        self.learning_wins_var = tk.StringVar(value="Carregando resultados...")
        self.learning_audit_var = tk.StringVar(value="Carregando análises...")
        self.learning_influence_var = tk.StringVar(value="Carregando decisões...")
        self.learning_last_audit_var = tk.StringVar(value="Nenhuma partida analisada ainda.")
        self.learning_status_var = tk.StringVar(value="Verificando se o ML influenciou decisões...")
        self.learning_games_count_var = tk.StringVar(value="—")
        self.learning_wins_count_var = tk.StringVar(value="—")
        self.learning_opps_count_var = tk.StringVar(value="—")
        self.learning_changes_count_var = tk.StringVar(value="—")

        overview = ttk.Frame(content)
        overview.pack(fill="x", pady=(0, 6))
        for col in range(2):
            overview.columnconfigure(col, weight=1, uniform="learning_summary")

        def summary_card(row: int, col: int, title: str, value: tk.StringVar,
                         subtitle: tk.StringVar) -> None:
            frame = ttk.LabelFrame(overview, text=title, padding=(9, 5, 9, 8))
            frame.grid(row=row, column=col, sticky="nsew", padx=3, pady=3)
            ttk.Label(frame, textvariable=value,
                      font=("Segoe UI", 17, "bold")).pack(anchor="w")
            ttk.Label(frame, textvariable=subtitle,
                      wraplength=315).pack(anchor="w", pady=(0, 1))

        summary_card(0, 0, "Partidas registradas", self.learning_games_count_var,
                     self.learning_games_var)
        summary_card(0, 1, "Resultados 999+", self.learning_wins_count_var,
                     self.learning_wins_var)
        summary_card(1, 0, "Chances >1000 descobertas", self.learning_opps_count_var,
                     self.learning_audit_var)
        summary_card(1, 1, "Jogadas alteradas pelo ML", self.learning_changes_count_var,
                     self.learning_influence_var)

        situation = ttk.LabelFrame(content, text="O ML está ajudando?", padding=9)
        situation.pack(fill="x", pady=(3, 6))
        ttk.Label(situation, textvariable=self.learning_status_var,
                  wraplength=720).pack(anchor="w")

        latest_frame = ttk.LabelFrame(content, text="Última partida analisada", padding=9)
        latest_frame.pack(fill="x", pady=(0, 8))
        ttk.Label(latest_frame, textvariable=self.learning_last_audit_var,
                  wraplength=720).pack(anchor="w")

        self.learning_details_visible = False
        self.learning_details_button = ttk.Button(
            content, text="Mostrar detalhes técnicos ▼",
            command=lambda: self._toggle_learning_details(),
        )
        self.learning_details_button.pack(anchor="w", pady=(0, 4))
        self.learning_details_frame = ttk.Frame(content)
        # Intentionally NOT packed by default: no overwhelming technical page.

        self.learning_cf_var = tk.StringVar(value="Pesquisa de alternativas: --")
        self.learning_outcome_var = tk.StringVar(value="Modelo de resultados: --")
        self.learning_hindsight_var = tk.StringVar(value="Análise retrospectiva: --")
        self.learning_pattern_var = tk.StringVar(value="Pesquisa de padrões: --")
        self.learning_rng_var = tk.StringVar(value="Probabilidades de pedras: não ajustadas pelo ML")

        technical = ttk.LabelFrame(self.learning_details_frame,
                                   text="Treinamento e qualidade das previsões", padding=9)
        technical.pack(fill="x", pady=(2, 6))
        ttk.Label(
            technical,
            text=("MAE é o erro médio de previsão: quanto menor, melhor. "
                  "Esses modelos são de pesquisa e NÃO demonstram ganho de pontuação no jogo."),
            wraplength=690,
        ).pack(anchor="w", pady=(0, 5))
        for variable in (self.learning_cf_var, self.learning_outcome_var,
                         self.learning_hindsight_var, self.learning_pattern_var, self.learning_rng_var):
            ttk.Label(technical, textvariable=variable,
                      wraplength=700).pack(anchor="w", pady=1)
        self.learning_metrics_var = tk.StringVar(value="Aguardando métricas...")
        ttk.Label(technical, textvariable=self.learning_metrics_var,
                  wraplength=700).pack(anchor="w", pady=(6, 4))
        ttk.Button(technical, text="Reprocessar aprendizado do histórico",
                   command=self._rebuild_learning_history_async).pack(anchor="w", pady=(5, 0))
        ttk.Label(technical, text="Reprocessar não apaga partidas. Pare o monitoramento antes de usar.",
                  wraplength=690).pack(anchor="w", pady=(2, 0))

        patterns = ttk.LabelFrame(self.learning_details_frame,
                                  text="Frequência de pedras (pesquisa, sem alterar sorteios)", padding=8)
        patterns.pack(fill="both", expand=True, pady=(1, 0))
        self.learning_pattern_tree = ttk.Treeview(
            patterns, columns=("jewel", "observed", "expected", "delta"),
            show="headings", height=6,
        )
        for col, label, width in (("jewel", "Pedra", 95), ("observed", "Observado", 105),
                                  ("expected", "Esperado", 105), ("delta", "Diferença", 110)):
            self.learning_pattern_tree.heading(col, text=label)
            self.learning_pattern_tree.column(col, width=width, anchor="center", stretch=True)
        scroll = ttk.Scrollbar(patterns, orient="vertical", command=self.learning_pattern_tree.yview)
        self.learning_pattern_tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.learning_pattern_tree.pack(side="left", fill="both", expand=True)

    def _toggle_learning_details(self) -> None:
        """Expand/collapse optional ML diagnostics without rebuilding widgets."""
        if self.learning_details_visible:
            self.learning_details_frame.pack_forget()
            self.learning_details_visible = False
            self.learning_details_button.config(text="Mostrar detalhes técnicos ▼")
        else:
            self.learning_details_frame.pack(fill="both", expand=True, pady=(2, 5))
            self.learning_details_visible = True
            self.learning_details_button.config(text="Ocultar detalhes técnicos ▲")

    def _show_compact_decision(self, mask: int, jewel: str,
                               recs: list[Recommendation], refined: bool) -> None:
        """Update only the visible game information (never build analysis reports)."""
        self.target_var.set(f"Target: {recs[0].active_goal if recs else 'waiting'}")
        self._update_live_summary(mask, jewel, recs)
        if not recs:
            return
        top = recs[0]
        if not refined:
            preview_row, preview_col = Board.rc(top.position)
            self.live_move_var.set(f"PREVIEW: {jewel} → L{preview_row}C{preview_col}")
            self.live_confidence_var.set("Confidence: PRELIMINARY")
            self.live_reason_var.set("Preview may change. Wait for FINAL before moving.")
        # Fast, idempotent telemetry: one final decision per game/step. Never
        # perform retrospective search or retrain inside the monitor/solver path.
        if refined and self.app_cfg.operation_mode == "recommend":
            try:
                self.db.record_ml_decision(
                    self.episode_id, mask.bit_count(), jewel, top.ml_matched,
                    top.ml_changed, top.ml_baseline_pos, top.position,
                )
            except Exception:
                pass
        rr, cc = Board.rc(top.position)
        if not refined:
            self.status.set(f"PREVIEW: {jewel} -> L{rr}C{cc}; refining final decision...")
        elif top.tie_kind:
            ties = "/".join(f"L{Board.rc(p)[0]}C{Board.rc(p)[1]}" for p in top.tied_with)
            kind = "EXACT TIE" if top.tie_kind == "exact" else "CLOSE CALL"
            self.status.set(f"{kind}: {jewel} L{rr}C{cc} / {ties} | target {top.active_goal}")
        else:
            stage = ("900+ SECURED" if top.method == "reward-tier-exact" and top.reward_guarantee == "900+ guaranteed"
                     else "REWARD SECURED" if top.method == "reward-tier-exact" and top.reward_guarantee == ">1000 guaranteed"
                     else "REWARD FIRST" if top.method == "reward-tier-exact"
                     else "LUCKY IMMEDIATE" if top.method == "immediate-lucky-fastpath"
                     else ("BEST" if refined else "PREVIEW — refining"))
            self.status.set(f"{stage}: {jewel} -> L{rr}C{cc} | target {top.active_goal}")
        if refined and self.app_cfg.overlay_enabled and self.region and self.vision_cfg.board_roi:
            self.overlay.show_cell(self.region, self.vision_cfg.board_roi, rr, cc,
                                   f"{jewel} -> {rr},{cc}")
        else:
            self.overlay.hide()

    def _refresh_learning_panel(self) -> None:
        """Update readable summaries from the same existing database snapshot."""
        if not hasattr(self, "learning_games_var"):
            return
        try:
            snap = build_learning_snapshot(
                self.db, self.experience.info,
                min_validation_games=self.app_cfg.ml_min_validation_games,
                rng_min_games=self.app_cfg.rng_min_games,
                rng_blend=self.app_cfg.empirical_bias_blend,
            )
            complete = snap['completed_games']
            verified = snap['verified_games']
            self.learning_games_count_var.set(str(complete))
            self.learning_games_var.set(
                f"{verified} com placar verificado; os demais têm resultado registrado"
            )
            memory = getattr(self.experience, "winning_memory", None)
            wins = len(memory.examples) if memory is not None else 0
            states = len(memory._by_state) if memory is not None else 0
            self.learning_wins_count_var.set(str(wins))
            self.learning_wins_var.set(
                f"Histórico elegível; {states} estados de jogada indexados"
            )
            audit = snap['retrospective']
            influence = snap['live_influence']
            self.learning_opps_count_var.set(str(audit['recoverable_gt1000']))
            self.learning_audit_var.set(
                f"Entre {audit['analyzed']} partidas reconstruídas após o jogo; "
                "não são vitórias reais"
            )
            self.learning_changes_count_var.set(str(influence['changes']))
            self.learning_influence_var.set(
                f"{influence['matches']} correspondências exatas em "
                f"{influence['decisions']} decisões observadas"
            )
            if not influence['decisions']:
                status = "Sem decisões acompanhadas ainda. Não é possível medir o efeito do ML."
            elif influence['changes'] == 0:
                status = (
                    f"Até agora, o ML não alterou nenhuma das {influence['decisions']} "
                    "decisões registradas. Não existe ganho de pontuação comprovado."
                )
            else:
                status = (
                    f"O ML alterou {influence['changes']} de {influence['decisions']} "
                    "decisões registradas, apenas em desempates. Ainda NÃO foi medido "
                    "se isso melhorou a pontuação em relação ao solver puro."
                )
            self.learning_status_var.set(status)

            self.learning_cf_var.set(
                f"Counterfactual (pesquisa): {snap['cf_status']} | "
                f"{snap['cf_samples']} amostras | "
                f"validação {snap['cf_validation_games']}/{snap['required_validation_games']} partidas"
            )
            self.learning_outcome_var.set(
                f"Resultado real (sem influência): {snap['outcome_status']} | "
                f"{snap['outcome_samples']} amostras | "
                f"validação {snap['outcome_validation_games']}/{snap['required_validation_games']} partidas"
            )
            self.learning_hindsight_var.set(
                f"Hindsight (pesquisa): {snap['hindsight_status']} | "
                f"{snap['hindsight_samples']} exemplos de {snap['hindsight_games']} partidas | "
                f"validação {snap['hindsight_validation_games']}/{snap['required_validation_games']} partidas"
            )
            self.learning_pattern_var.set(
                f"Padrões: {snap['pattern_stage']} | "
                f"{snap['pattern_games']} partidas locais | "
                f"{snap['repeated_layouts']} layouts repetidos"
            )
            self.learning_rng_var.set(
                "Influência nas probabilidades de sorteio: " +
                ("HABILITADA" if snap['rng_influence_enabled'] else "DESLIGADA (apenas pesquisa)")
            )
            # Never carry the previous game's outcome into a fresh/empty DB.
            last_text = "Nenhuma análise retrospectiva disponível ainda."
            with self.db.connect() as con:
                latest = con.execute("""SELECT ra.audit_json FROM retrospective_audits ra
                    JOIN games g ON g.id=ra.game_id
                    ORDER BY g.finished_at DESC LIMIT 1""").fetchone()
            if latest:
                value = json.loads(latest[0])
                if value.get("valid"):
                    observed = int(value['observed_total'])
                    best = int(value['best_total'])
                    lost = value.get("first_lost_gt1000")
                    first = value.get("first_better_choice")
                    choice = lost or first
                    if choice and best > observed:
                        positions = (choice.get('winning_alternative_positions') or
                                     choice.get('better_alternative_positions') or [])[:3]
                        where = ', '.join('L%dC%d' % Board.rc(p) for p in positions)
                        hint = (f"Na jogada {choice['step']} ({choice['jewel']}), "
                                f"avaliar {where}.") if where else "Houve outra rota possível."
                    else:
                        hint = "Não foi encontrada rota superior para as mesmas pedras."
                    last_text = (
                        f"Fez {observed} pontos. Melhor resultado calculado após a partida: "
                        f"{best} pontos. {hint} O cálculo retrospectivo conhece as 14 "
                        "pedras e não representa uma vitória real."
                    )
                else:
                    last_text = "Última partida sem reconstrução consistente; não entrou na pesquisa."
            self.learning_last_audit_var.set(last_text)

            def fmt_pct(value):
                return "—" if value is None else f"{value*100:.1f}%"
            def fmt_num(value):
                return "—" if value is None else f"{value:.1f}"
            improvement = snap['outcome_improvement']
            imp = "—" if improvement is None else f"{improvement*100:.1f}% contra a referência"
            self.learning_metrics_var.set(
                f"Counterfactual MAE: 3 Lucky {fmt_pct(snap['mae_p3'])} | "
                f">1000 {fmt_pct(snap['mae_gt1000'])} | "
                f"pontuação {fmt_num(snap['mae_score'])}\n"
                f"Hindsight MAE: 3 Lucky {fmt_pct(snap['hindsight_mae_p3'])} | "
                f">1000 {fmt_pct(snap['hindsight_mae_gt1000'])} | "
                f"pontuação {fmt_num(snap['hindsight_mae_score'])}\n"
                f"Resultado real MAE: {fmt_num(snap['outcome_residual_mae'])} | {imp}"
            )
            for iid in self.learning_pattern_tree.get_children():
                self.learning_pattern_tree.delete(iid)
            for row in snap['jewel_rates']:
                self.learning_pattern_tree.insert("", "end", values=(
                    row['jewel'], f"{row['observed']*100:.1f}%",
                    f"{row['expected']*100:.1f}%", f"{row['delta']*100:+.1f} pp",
                ))
        except Exception as exc:
            self.learning_status_var.set(f"Não foi possível carregar o aprendizado: {exc}")
            self.learning_games_var.set("Verifique o banco de dados e tente novamente.")

    def _refresh_system_logs(self) -> None:
        if not hasattr(self, "system_log_text"):
            return
        try:
            if self.auto_setup_log_path.exists():
                lines = self.auto_setup_log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-120:]
                content = "\n".join(lines) if lines else "Log file is empty."
            else:
                content = "No technical log has been created yet."
        except Exception as exc:
            content = f"Unable to read log: {exc}"
        self.system_log_text.configure(state="normal")
        self.system_log_text.delete("1.0", "end")
        self.system_log_text.insert("end", content)
        self.system_log_text.see("end")
        self.system_log_text.configure(state="disabled")

    def _refresh_system_panel(self) -> None:
        if not hasattr(self, "system_solver_var"):
            return
        bridge = self.solver_bridge
        self.system_solver_var.set(f"Solver: {'ONLINE' if bridge.available else 'OFFLINE'} | {bridge.status}")
        try:
            with self.db.connect() as con:
                ok = con.execute("PRAGMA quick_check").fetchone()[0]
            self.system_db_var.set(f"Database: {'HEALTHY' if ok == 'ok' else ok}")
        except Exception as exc:
            self.system_db_var.set(f"Database: ERROR {exc}")
        self.system_vision_var.set(
            f"Vision: {'READY' if self.vision_test_passed else 'WAITING'} | board {self.board_confidence:.2f}"
        )
        t = self.db.telemetry_summary("decision-roundtrip")
        def ms(v): return "--" if v is None else f"{v:.0f} ms"
        self.system_latency_var.set(
            f"Decision latency: P50 {ms(t['p50'])} | P95 {ms(t['p95'])} | P99 {ms(t['p99'])} | n={t['count']}"
        )
        hit = t['cache_hit_rate']
        self.system_cache_var.set("Cache hit rate: --" if hit is None else f"Cache hit rate: {hit*100:.1f}%")
        backups = sorted(self.backup_dir.glob("bingo_*.sqlite3")) if self.backup_dir.exists() else []
        self.system_backup_var.set(f"Backups: {len(backups)} saved" + (f" | latest {backups[-1].name}" if backups else ""))
        self._refresh_system_logs()

    def _restart_solver_service(self) -> None:
        if hasattr(self, "system_solver_var"):
            self.system_solver_var.set("Solver: RESTARTING...")
        self.solver_bridge.restart_async(lambda ok, status: self._ui_call(self._solver_service_status, ok, status))

    def _show_latest_replay(self) -> None:
        """Backward-compatible entry point; the history window defaults to newest."""
        self._show_game_history()

    def _show_game_history(self) -> None:
        games = list_stored_games(self.db, limit=100)
        if not games:
            messagebox.showinfo("Game history", "No recorded game with actions yet.", parent=self)
            return

        win = tk.Toplevel(self)
        win.title("Game History / Replay")
        win.geometry("980x690")
        win.transient(self)
        try:
            win.attributes("-topmost", True)
        except tk.TclError:
            pass

        header = ttk.Frame(win, padding=8)
        header.pack(fill="x")
        ttk.Label(header, text="Game History / Replay", font=("Segoe UI", 14, "bold")).pack(anchor="w")
        ttk.Label(
            header,
            text=(
                "Select any recorded game. The lower report compares saved live decisions with the accepted moves "
                "and runs exact hindsight using the jewel sequence that actually arrived."
            ),
            wraplength=930,
        ).pack(anchor="w", pady=(2, 2))
        ttk.Label(
            header,
            text="Important: hindsight is retrospective. Completed games are automatically converted into candidate-level rewards for ML, but future jewels are never passed to the live solver. Reopening a replay does not duplicate rewards.",
            wraplength=930,
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w", pady=(0, 4))
        ttk.Label(header, text=f"Database: {self.db.path}", wraplength=930).pack(anchor="w")

        table_frame = ttk.Frame(win, padding=(8, 0, 8, 4))
        table_frame.pack(fill="x")
        columns = ("started", "score", "result", "moves", "followed", "box", "game_id")
        tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=8, selectmode="browse")
        specs = (
            ("started", "Started", 145, "w"),
            ("score", "Score", 70, "center"),
            ("result", "Lucky / Normal", 110, "center"),
            ("moves", "Moves", 60, "center"),
            ("followed", "Followed", 75, "center"),
            ("box", "Box", 65, "center"),
            ("game_id", "Game ID", 235, "w"),
        )
        for col, label, width, anchor in specs:
            tree.heading(col, text=label)
            tree.column(col, width=width, anchor=anchor, stretch=(col == "game_id"))
        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        tree.pack(side="left", fill="x", expand=True)

        game_ids: dict[str, str] = {}
        for idx, game in enumerate(games):
            result = "? / ?" if game.get("lucky") is None else f"{game.get('lucky', '?')} / {game.get('normal', '?')}"
            followed = f"{game['followed']}/{game['comparable']}"
            iid = f"g{idx}"
            game_ids[iid] = str(game["game_id"])
            tree.insert(
                "", "end", iid=iid,
                values=(
                    game.get("created_local") or "?",
                    game.get("total") if game.get("total") is not None else "?",
                    result, game.get("actions", 0), followed, game.get("box_color") or "-", game["game_id"],
                ),
            )

        report_frame = ttk.LabelFrame(win, text="Replay analysis", padding=6)
        report_frame.pack(fill="both", expand=True, padx=8, pady=(0, 6))
        report_text = tk.Text(report_frame, wrap="none", font=("Consolas", 9), state="disabled")
        report_y = ttk.Scrollbar(report_frame, orient="vertical", command=report_text.yview)
        report_x = ttk.Scrollbar(report_frame, orient="horizontal", command=report_text.xview)
        report_text.configure(yscrollcommand=report_y.set, xscrollcommand=report_x.set)
        report_y.pack(side="right", fill="y")
        report_x.pack(side="bottom", fill="x")
        report_text.pack(side="left", fill="both", expand=True)

        current_report = {"text": ""}

        def show_selected(_event=None):
            sel = tree.selection()
            if not sel:
                return
            game_id = game_ids.get(sel[0])
            if not game_id:
                return
            try:
                replay = stored_replay(self.db, game_id)
                text = "Game not found." if replay is None else format_replay_report(replay)
            except Exception as exc:
                text = f"Unable to replay {game_id}: {exc}"
            current_report["text"] = text
            report_text.configure(state="normal")
            report_text.delete("1.0", "end")
            report_text.insert("end", text)
            report_text.configure(state="disabled")
            report_text.yview_moveto(0.0)

        def copy_report():
            text = current_report.get("text", "")
            if not text:
                return
            self.clipboard_clear()
            self.clipboard_append(text)
            self.update_idletasks()

        tree.bind("<<TreeviewSelect>>", show_selected)

        buttons = ttk.Frame(win, padding=(8, 0, 8, 8))
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Copy selected report", command=copy_report).pack(side="left")
        ttk.Button(buttons, text="Close", command=win.destroy).pack(side="right")

        first = tree.get_children()
        if first:
            tree.selection_set(first[0])
            tree.focus(first[0])
            tree.see(first[0])
            show_selected()

    def _export_research_dataset(self) -> None:
        try:
            out = export_research_dataset(self.db, self.data_dir / "exports", self.experience.info)
            messagebox.showinfo("Research export", f"Dataset exported to:\n{out}", parent=self)
        except Exception as exc:
            messagebox.showerror("Research export", str(exc), parent=self)

    def _recover_current_round(self) -> None:
        """Explicit one-click reconciliation using repeated screen observations."""
        if self.region is None or self.vision_cfg.board_roi is None:
            messagebox.showwarning("Recovery", "Board setup is not ready.", parent=self); return
        was_running = self.running
        if was_running:
            self.running = False
            self.start_btn.config(text="START", state="disabled")
            if self.worker and self.worker.is_alive():
                self.worker.join(timeout=2.0)
        try:
            observations = []
            for _ in range(3):
                panel = self._panel(); detail = read_board_detailed(panel, self.vision_cfg, self.clf)
                raw_mask, _ = read_selected_mask(panel, self.vision_cfg)
                observations.append((board_hash(detail.board), detail.board, int(raw_mask), float(detail.board_confidence)))
                time.sleep(0.08)
            from collections import Counter
            pair, count = Counter((x[0], x[2]) for x in observations).most_common(1)[0]
            if count < 2:
                raise RuntimeError("Screen did not stabilize across recovery frames.")
            bh, screen_mask = pair
            chosen = next(x for x in observations if x[0] == bh and x[2] == screen_mask)
            if self.board is not None and board_hash(self.board) != bh:
                raise RuntimeError("Visible board is different from the tracked round. Wait for automatic new-round detection.")
            checkpoint = self.db.latest_checkpoint(self.episode_id)
            cp_mask = int(checkpoint['confirmed_mask']) if checkpoint is not None else None
            ok, recovered_mask, reason = reconcile_masks(cp_mask, screen_mask)
            if not ok:
                raise RuntimeError(reason)
            self.board = chosen[1]; self.board_confidence = chosen[3]; self.selected_mask = recovered_mask
            self.temporal.reset(recovered_mask); self.temporal_initialized = True
            self.current_jewel = None
            if self.live_coordinator is not None:
                self.live_coordinator.update_board(accepted_mask=recovered_mask, phase=self.phase.value)
            self._checkpoint_state("manual-recovery", payload={"reason": reason, "screen_mask": screen_mask})
            self._invalidate_recommendation("manual recovery", f"Recovered to {recovered_mask.bit_count()}/14 confirmed placements.")
            self._refresh_grid(); self._update_phase()
            self.turn_label.set(f"Placed: {recovered_mask.bit_count()}/14 | recovered from stable screen")
            self.status.set(f"Recovery complete: {reason}.")
        except Exception as exc:
            messagebox.showerror("Recovery", str(exc), parent=self)
        finally:
            if was_running:
                self.running = True
                self.start_btn.config(text="STOP", state="normal")
                self.worker = threading.Thread(target=self._loop, name="bingo-monitor", daemon=True)
                self.worker.start()
            else:
                self.start_btn.config(state="normal")
            self._refresh_system_panel()

    def show_setup_help(self):
        messagebox.showinfo(
            "Setup",
            """Recommended V5 flow:

1. Open Jewel Bingo. If possible, keep it idle so the six colored x4 references are visible.
2. Click AUTO SETUP. The app tries to find the Jewel Bingo panel, detect the 5x5 grid from its 25 cell rectangles + MU center, learn the six jewel references, and infer the current-jewel location.
3. Open/start a box.
4. Press START. If the live vision test has not run yet, START runs it automatically first.

Normally you should NOT draw the board/current-jewel rectangles manually.

If automatic detection cannot reach a safe confidence level, open Repair / recovery. Saved calibration and learned images persist between launches.

Scores above 1000 are verified automatically after a trusted 14/14 completion. Lower scores remain saved as completed; legacy OCR stays internal/optional and is no longer part of the normal Repair workflow.""",
            parent=self,
        )

    def _show_observe_state(self) -> None:
        """Read-only observation UI: do not ask the advisor or compute predictions."""
        self.last_recs = []
        self.cached_recommend_key = None
        self._ranked_grid_mask = None
        self._ranked_grid_recs = []
        self._preview_grid_mask = None
        self._preview_grid_pos = None
        self.overlay.hide()
        self._refresh_grid()
        self.target_var.set("Target: disabled (Observable)")
        self.target_change_var.set("")
        if hasattr(self, "live_move_var"):
            self.live_move_var.set("OBSERVABLE")
            self.live_target_var.set("Sem recomendacao / no solver")
            self.live_route_var.set("Lendo tabuleiro e registrando jogadas")
            self.live_health_var.set("Somente observacao")
            self.live_slack_var.set("")
            self.live_confidence_var.set("")
            self.live_reason_var.set("Mude para Jogar para calcular a melhor posição.")
            self.live_gt1000_var.set("Win >1000: --")
            self.live_target_chance_var.set("Goal possible: --")
            self.live_route_chance_var.set("Route possible: --")
            self.live_p3_var.set("3 Lucky: --")
            self.live_expected_var.set("Expected pts: --")
        self.status.set("OBSERVABLE: tabuleiro e jogadas registrados, sem cálculos do solver.")

    def _change_operation_mode(self):
        new_mode = self.operation_var.get().strip().lower()
        if new_mode not in {"observe", "recommend"}:
            new_mode = "recommend"
            self.operation_var.set(new_mode)
        changed = self.app_cfg.operation_mode != new_mode
        self.app_cfg.operation_mode = new_mode
        self.app_cfg.save(self.app_cfg_path)
        if hasattr(self, "mode_status_var"):
            self.mode_status_var.set(
                "OBSERVABLE ativo: captura e histórico, sem solver"
                if new_mode == "observe"
                else "PLAY ativo: solver 3/4 e recomendações"
            )
        # Show the board immediately after changing modes; the selector itself
        # stays visible and accessible above the Game/ML notebook.
        if hasattr(self, "notebook"):
            self.notebook.select(0)
        # Generation barrier also discards stale WSL/process/fallback results.
        self._invalidate_recommendation("operation changed to " + new_mode)
        if new_mode == "observe":
            # Queued WSL decisions must not start while in Observable.
            self.solver_bridge._drop_pending_decisions()
            self._show_observe_state()
        else:
            self.status.set("JOGAR: solver e recomendações habilitados.")
            if self.running and self.board is not None and self.current_jewel is not None:
                self._recommend()
        if changed and self.board is not None:
            # Switching modes updates the episode's tracking metadata only;
            # it never destroys its placements or the historical SQLite DB.
            self.db.ensure_game(
                self.episode_id, "local-auto", self.session_id, self.board,
                self.box_color, self.app_cfg.operation_mode, self.app_cfg.risk_profile,
            )

    def _change_strategy(self):
        """V7.0.5 live play has one automatic planner.

        Kept as a compatibility hook for old integrations, but manual strategy
        switching is intentionally removed from the live UI.
        """
        self.app_cfg.set_risk_profile("auto")
        self.app_cfg.save(self.app_cfg_path)
        self.strategy_var.set("Win >1000 / Adaptive")
        self.status.set("Automatic Win >1000 / Adaptive planner is active.")


    # Backwards-compatible name used by older integrations/tests.
    def _update_phase(self, desync: bool = False, result_visible: bool = False):
        st = self.state_machine.update(
            has_board=self.board is not None,
            selected_count=self.selected_mask.bit_count(),
            current_jewel=self.current_jewel,
            result_visible=result_visible,
            desync=desync,
            complete_pending=self.completion_pending,
        )
        self.phase = st.phase
        phase_text = f"Phase: {st.phase.value}"
        # Tk variables are not thread-safe. The capture loop runs on a worker thread,
        # so marshal the UI update back to Tk's event loop.
        self._ui_call(self.phase_var.set, phase_text)
        return st

    def _checkpoint_state(self, kind: str, *, provisional_mask: int | None = None, payload: dict | None = None) -> None:
        """Persist state changes once without putting SQLite in the hot render path repeatedly."""
        try:
            state_version = self.live_coordinator.state.state_version if self.live_coordinator is not None else self.selected_mask.bit_count()
            signature = (
                self.episode_id, int(state_version), int(self.selected_mask),
                None if provisional_mask is None else int(provisional_mask), self.phase.value,
                self.current_jewel, self.app_cfg.mode, kind,
            )
            if signature == self._last_checkpoint_signature:
                return
            self._last_checkpoint_signature = signature
            active_target = self.last_recs[0].active_goal if self.last_recs else None
            merged = {"kind": kind}
            if payload:
                merged.update(payload)
            self.db.save_checkpoint(
                self.episode_id, int(state_version), int(self.selected_mask), provisional_mask,
                self.phase.value, self.current_jewel, self.app_cfg.mode, active_target, merged,
            )
        except Exception as exc:
            self._auto_log(f"Checkpoint persistence skipped: {exc}", None, "WARN")

    def _complete_from_visual_evidence(self, panel, final_mask: int, source: str) -> bool:
        """Close the current episode from a trusted recent raw 14/14 snapshot."""
        final_mask = int(final_mask)
        if self.episode_finalized or self.board is None or final_mask.bit_count() < 14:
            return False
        previous = self.selected_mask
        self.selected_mask = final_mask
        self.temporal.reset(final_mask)
        self.temporal_initialized = True
        self.current_jewel = None
        self.completion_pending = False
        self._update_phase()
        self._checkpoint_state("completion-evidence", payload={
            "source": source, "from_count": previous.bit_count(), "to_count": final_mask.bit_count(),
        })
        self.db.log_event(
            self.episode_id, "completion-evidence",
            f"Round completed from {source}; promoted recent raw mask to 14/14",
            GamePhase.COMPLETE.value,
            {"source": source, "previous_mask": int(previous), "final_mask": final_mask},
        )
        s_final = score_mask(final_mask)
        self._ui_call(self._invalidate_recommendation, "game complete from visual evidence", None)
        self._ui_call(self._show_game_complete_summary, s_final.lucky, s_final.normal, s_final.total)
        self._finalize_episode(panel, finalization_source=source)
        self._ui_call(self.current_label.set, "Current jewel: game complete")
        return True

    def _on_close(self):
        self.running = False
        try:
            self.waiting_diagnostics.record("app-close", session_id=self.session_id)
            self.waiting_diagnostics.stop(self._waiting_probe_marker)
        except Exception:
            pass
        self.auto_setup_cancel.set()
        self.overlay.hide()
        self._cancel_solver("application closing")
        try:
            self.solver_bridge.close()
        except Exception:
            pass
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=1.5)
        set_win32_sink(None)
        journal = getattr(self, "runtime_journal", None)
        if journal is not None:
            try:
                journal.record(
                    "app-close-complete", pending_maintenance=int(self._maintenance_pending_count),
                    worker_alive=bool(self.worker and self.worker.is_alive()),
                )
                journal.close(timeout=1.0)
            except Exception:
                pass
        # AUTO SETUP is daemonized and never owns Tk state directly, so we do
        # not block shutdown waiting for a computer-vision stage to finish.
        self.destroy()

    def _panel(self):
        if self.region is None:
            raise RuntimeError("Select the game area first")
        return capture_region(self.region)

    def _pause_monitor_for_setup(self) -> bool:
        if self.auto_setup_running:
            messagebox.showinfo(
                "AUTO SETUP is running",
                "Wait for AUTO SETUP to finish or click CANCEL AUTO SETUP before opening a manual calibration step.",
                parent=self,
            )
            return False
        was_running = self.running
        if not was_running:
            return True
        self.running = False
        self.start_btn.config(text="START", state="disabled")
        self.overlay.hide()
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=3.0)
        if self.worker and self.worker.is_alive():
            messagebox.showwarning(
                "Please wait",
                "The analyzer is still finishing a calculation. Wait a moment and try the setup step again.",
                parent=self,
            )
            return False
        self.status.set("Monitoring paused automatically for calibration.")
        return True

    def _append_auto_log_ui(self, line: str, progress: float | None = None):
        if hasattr(self, "auto_log_text"):
            try:
                self.auto_log_text.configure(state="normal")
                self.auto_log_text.insert("end", line + "\n")
                # Keep the on-screen log compact while the complete file log
                # remains on disk.
                lines = int(self.auto_log_text.index("end-1c").split(".")[0])
                if lines > 70:
                    self.auto_log_text.delete("1.0", "20.0")
                self.auto_log_text.see("end")
                self.auto_log_text.configure(state="disabled")
            except tk.TclError:
                pass
        if progress is not None and hasattr(self, "auto_progress_var"):
            self.auto_progress_var.set(float(max(0.0, min(100.0, progress))))
        if hasattr(self, "auto_progress_text"):
            clean = line.split("] ", 1)[-1]
            self.auto_progress_text.set(clean[:42])

    def _runtime_record(self, event: str, level: str = "INFO", **data: object) -> None:
        journal = getattr(self, "runtime_journal", None)
        if journal is None:
            return
        try:
            journal.record(event, level=level, **data)
        except Exception:
            pass

    def _runtime_tick(self) -> None:
        """Tk heartbeat + cheap resources for post-mortem lag analysis."""
        try:
            now = time.monotonic()
            interval = max(0.5, float(self.app_cfg.runtime_heartbeat_ms) / 1000.0)
            expected = float(getattr(self, "_runtime_tick_expected", now))
            drift_ms = max(0.0, (now - expected) * 1000.0)
            self._runtime_tick_expected = now + interval
            with self._live_state_lock:
                state = {
                    "running": bool(self.running),
                    "phase": self.phase.value,
                    "episode_id": self.episode_id,
                    "selected_count": int(self.selected_mask).bit_count(),
                    "current_jewel": self.current_jewel,
                    "monitor_age_ms": None if not self._diag_monitor_heartbeat else int((now - self._diag_monitor_heartbeat) * 1000),
                    "maintenance_active": bool(self.maintenance_thread and self.maintenance_thread.is_alive()),
                    "maintenance_pending_games": int(self._maintenance_pending_count),
                    "solver_bridge_available": bool(self.solver_bridge.available),
                    "solver_bridge_status": str(self.solver_bridge.status),
                    "solver_decision_lane": self.solver_bridge.decision_snapshot(),
                }
            self._runtime_record("ui-heartbeat", drift_ms=round(drift_ms, 2), **state)
            if drift_ms >= 400.0:
                self._runtime_record("ui-lag", level="WARN", drift_ms=round(drift_ms, 2), **state)

            resource_interval = max(1.0, float(self.app_cfg.runtime_resource_sample_ms) / 1000.0)
            if now - float(getattr(self, "_runtime_last_resource_sample", 0.0)) >= resource_interval:
                self._runtime_last_resource_sample = now
                gui_res = collect_gui_resources()
                vitals = collect_process_vitals()
                self._runtime_record("resource-sample", gui=gui_res, process=vitals, **state)
        except Exception as exc:
            self._runtime_record("runtime-tick-error", level="WARN", error=f"{type(exc).__name__}: {exc}")
        finally:
            try:
                if self.winfo_exists():
                    self.after(max(500, int(self.app_cfg.runtime_heartbeat_ms)), self._runtime_tick)
            except Exception:
                pass

    def _auto_log(self, message: str, progress: float | None = None, level: str = "INFO"):
        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {level}: {message}"
        self._runtime_record("app-log", level=level, message=str(message), progress=progress)
        if self.app_cfg.save_runtime_logs:
            try:
                with self.auto_setup_log_path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except Exception:
                pass
        self._ui_call(self._append_auto_log_ui, line, progress)

    def _set_auto_setup_active(self, active: bool):
        self.auto_setup_running = bool(active)
        if hasattr(self, "auto_setup_btn"):
            self.auto_setup_btn.configure(text="CANCEL AUTO SETUP" if active else "AUTO SETUP")
        if not active and hasattr(self, "auto_progress_text") and self.auto_progress_var.get() < 100:
            self.auto_progress_text.set("Stopped")
        self._refresh_setup_state()

    @staticmethod
    def _crop_screen_region(screen, mon: dict, region: ScreenRegion):
        x = int(region.left - mon["left"]); y = int(region.top - mon["top"])
        w = int(region.width); h = int(region.height)
        sh, sw = screen.shape[:2]
        if x < 0 or y < 0 or x + w > sw or y + h > sh or w <= 0 or h <= 0:
            return None
        return screen[y:y+h, x:x+w].copy()

    def auto_setup(self):
        """Run automatic calibration without blocking Tk's event loop.

        V5 originally performed a full-screen combinatorial grid search directly
        on the Tk thread. On cluttered desktops that could take tens of seconds
        or longer, making Windows report "Not Responding" while no progress was
        visible. V5.1 captures one clean screenshot on the UI thread, then moves
        all expensive computer vision to a daemon worker with bounded search,
        progress, cancellation and a persistent log.
        """
        if self.auto_setup_running:
            self.auto_setup_cancel.set()
            self._auto_log("Cancel requested; finishing the current bounded vision step...", None, "WARN")
            self._setup_feedback("Cancelling AUTO SETUP...", "warn")
            return
        if not self._pause_monitor_for_setup():
            return

        self.auto_setup_cancel.clear()
        self._set_auto_setup_active(True)
        self.auto_progress_var.set(2.0)
        self.auto_progress_text.set("Capturing clean screen")
        self._setup_feedback("AUTO SETUP started. Follow the live stage log below; the window should remain responsive.", "info")
        self._auto_log("AUTO SETUP started", 2)

        try:
            # Hide only long enough to capture a clean frame. All expensive work
            # happens afterwards, while the Assistant is visible and responsive.
            self.withdraw()
            self.update_idletasks()
            time.sleep(0.12)
            screen, mon = screenshot_all()
        except Exception as e:
            try:
                self.deiconify(); self.lift(); self.attributes("-topmost", True)
            except Exception:
                pass
            self._set_auto_setup_active(False)
            self._setup_feedback(f"AUTO SETUP capture failed: {e}", "error")
            self._auto_log(f"Screen capture failed: {e}", 0, "ERROR")
            return
        finally:
            try:
                self.deiconify(); self.lift(); self.attributes("-topmost", True)
            except Exception:
                pass

        previous = None if self.region is None else ScreenRegion(
            self.region.left, self.region.top, self.region.width, self.region.height
        )
        self._auto_log(f"Captured desktop {screen.shape[1]}x{screen.shape[0]}", 8)
        self.auto_setup_thread = threading.Thread(
            target=self._auto_setup_worker, args=(screen, dict(mon), previous), daemon=True, name="auto-setup"
        )
        self.auto_setup_thread.start()

    def _auto_setup_worker(self, screen, mon: dict, previous: ScreenRegion | None):
        started = time.perf_counter()
        try:
            def cancelled() -> bool:
                return self.auto_setup_cancel.is_set()

            if cancelled():
                self._ui_call(self._finish_auto_setup, {"cancelled": True})
                return

            region: ScreenRegion | None = None
            panel = None
            board_det = None
            panel_score = 0.0
            source = ""
            needs_confirmation = False

            # Fast path: a saved region is usually correct even if the Assistant
            # window itself moved. Moving this app does not change MU coordinates.
            if previous is not None:
                self._auto_log("1/5 Checking saved Jewel Bingo area...", 16)
                candidate = self._crop_screen_region(screen, mon, previous)
                if candidate is not None:
                    t = time.perf_counter()
                    d = detect_board_grid(candidate)
                    self._auto_log(
                        f"Saved-area board scan finished in {time.perf_counter()-t:.2f}s" +
                        (f"; board={d.score:.3f}, MU={d.mu_anchor_score:.3f}" if d else "; no board"),
                        26,
                    )
                    if d is not None and d.score >= 0.72 and d.mu_anchor_score >= 0.18:
                        region = previous; panel = candidate; board_det = d; source = "saved area"

                # The common case after setup is that the user simply moved the
                # MU/Jewel Bingo window. V5.1 skipped anchor relocation here and
                # jumped straight to generic desktop detection, which could
                # then demand manual panel selection. Try the saved title anchor
                # automatically before any generic search.
                if region is None and self.anchor_path.exists():
                    self._auto_log("1b/5 Saved coordinates changed; trying saved panel anchor...", 29)
                    found, anchor_score = relocate_region_by_anchor_in_image(
                        self.anchor_path, previous, screen, mon, threshold=0.55
                    )
                    self._auto_log(f"Anchor relocation match={anchor_score:.3f}", 31)
                    if found is not None:
                        candidate = self._crop_screen_region(screen, mon, found)
                        if candidate is not None:
                            d = detect_board_grid(candidate)
                            if d is not None and d.score >= 0.64 and d.mu_anchor_score >= 0.50:
                                region = found; panel = candidate; board_det = d; source = f"saved anchor ({anchor_score:.2f})"

            if cancelled():
                self._ui_call(self._finish_auto_setup, {"cancelled": True})
                return

            # If the panel moved or there is no calibration, use the bounded MU-
            # anchored detector. It intentionally never falls back to the old
            # exhaustive full-desktop combinatorial search.
            if region is None:
                self._auto_log("2/5 Searching the desktop for the blue MU center and 5x5 grid...", 34)
                t = time.perf_counter()
                detected = detect_panel_from_screen(screen)
                elapsed = time.perf_counter() - t
                if detected is None:
                    self._auto_log(f"Desktop search finished in {elapsed:.2f}s; no safe panel found", 45, "WARN")
                else:
                    self._auto_log(
                        f"Desktop search finished in {elapsed:.2f}s; panel={detected.score:.3f}, "
                        f"board={detected.board.score:.3f}, MU={detected.board.mu_anchor_score:.3f}",
                        45,
                    )
                if detected is not None:
                    # High-confidence candidates are committed automatically. A
                    # medium candidate is still much better than asking the user
                    # to redraw a panel: show one preview/confirmation instead.
                    acceptable_medium = (
                        detected.score >= 0.66
                        and detected.board.score >= 0.64
                        and detected.board.mu_anchor_score >= 0.50
                        and detected.panel_check.ok
                    )
                    if detected.high_confidence or acceptable_medium:
                        x, y, w, h = detected.roi
                        panel = screen[y:y+h, x:x+w].copy()
                        board_det = detect_board_grid(panel)
                        # Refine once more from the precise panel-local 5x5. The
                        # first full-screen MU candidate can be a smaller inner
                        # grid and produce a crop that clips the title/current
                        # row even though the board itself is found correctly.
                        if board_det is not None:
                            gx = x + board_det.roi[0]; gy = y + board_det.roi[1]
                            refined = infer_panel_roi_from_board(screen, (gx, gy, board_det.roi[2], board_det.roi[3]))
                            rx, ry, rw, rh = refined
                            refined_panel = screen[ry:ry+rh, rx:rx+rw].copy()
                            refined_board = detect_board_grid(refined_panel)
                            if refined_board is not None and refined_board.score >= board_det.score - 0.04:
                                x, y, w, h = refined
                                panel = refined_panel
                                board_det = refined_board
                        region = ScreenRegion(mon["left"] + x, mon["top"] + y, w, h)
                        panel_score = float(validate_game_panel(panel).score)
                        source = "automatic desktop detection"
                        needs_confirmation = not detected.high_confidence

            if cancelled():
                self._ui_call(self._finish_auto_setup, {"cancelled": True})
                return

            if region is None or panel is None or board_det is None:
                self._ui_call(self._finish_auto_setup, {
                    "ok": False,
                    "manual_panel": True,
                    "error": "I could not locate Jewel Bingo safely. Use 'Select panel' once; after that AUTO SETUP handles the board and jewels automatically.",
                    "elapsed": time.perf_counter() - started,
                })
                return

            self._auto_log(
                f"3/5 Board candidate: score={board_det.score:.3f}, coverage={board_det.coverage*100:.0f}%, "
                f"MU={board_det.mu_anchor_score:.3f}, cell≈{board_det.cell_size:.1f}px",
                58,
            )
            board_check = validate_board_roi(panel, board_det.roi)
            min_board = 0.64 if needs_confirmation else 0.72
            min_mu = 0.50 if needs_confirmation else 0.18
            if board_det.score < min_board or board_det.mu_anchor_score < min_mu or not board_check.ok:
                reasons = "; ".join(board_check.messages) if board_check.messages else "confidence below threshold"
                self._ui_call(self._finish_auto_setup, {
                    "ok": False,
                    "manual_board": True,
                    "error": f"A board candidate was found but rejected: {reasons}",
                    "elapsed": time.perf_counter() - started,
                })
                return

            if cancelled():
                self._ui_call(self._finish_auto_setup, {"cancelled": True})
                return

            current_roi = infer_current_jewel_roi(board_det.roi, panel.shape)
            self._auto_log(
                "4/5 Current-jewel SEARCH BAND inferred from the board. "
                "V5.2 will locate the actual icon dynamically when a box is open.", 70
            )

            self._auto_log("5/5 Looking for the six colored x4 reference jewels...", 80)
            ref = detect_reference_jewels_detailed(panel, board_det.roi)
            if ref is None:
                self._auto_log("Colored x4 references are not visible/reliable in this frame", 90, "WARN")
            else:
                self._auto_log(f"Six x4 references detected; confidence={ref.confidence:.3f}", 90)

            self._ui_call(self._finish_auto_setup, {
                "ok": True,
                "region": region,
                "panel": panel,
                "board_det": board_det,
                "board_check": board_check,
                "panel_score": panel_score,
                "current_roi": current_roi,
                "reference": ref,
                "source": source,
                "needs_confirmation": needs_confirmation,
                "elapsed": time.perf_counter() - started,
            })
        except Exception as e:
            self._auto_log(f"AUTO SETUP worker failed: {type(e).__name__}: {e}", None, "ERROR")
            self._ui_call(self._finish_auto_setup, {
                "ok": False, "error": f"{type(e).__name__}: {e}", "elapsed": time.perf_counter() - started
            })

    def _finish_auto_setup(self, result: dict):
        if result.get("cancelled") or self.auto_setup_cancel.is_set():
            self._set_auto_setup_active(False)
            self.auto_progress_var.set(0.0)
            self.auto_progress_text.set("Cancelled")
            self._setup_feedback("AUTO SETUP cancelled. No partial detection was saved.", "warn")
            self._auto_log("AUTO SETUP cancelled; no partial result committed", 0, "WARN")
            return

        if not result.get("ok"):
            self._set_auto_setup_active(False)
            self.auto_progress_var.set(0.0)
            self.auto_progress_text.set("Needs attention")
            msg = result.get("error", "Automatic setup failed")
            self._setup_feedback(msg, "error")
            self._auto_log(f"AUTO SETUP stopped: {msg}", 0, "ERROR")
            if result.get("manual_panel"):
                self._open_repair_dialog()
                messagebox.showwarning(
                    "Automatic panel detection needs help",
                    msg + "\n\nRepair / recovery has been opened. Use Manual panel only once; do NOT manually select the 5x5 board unless the app explicitly says the board detector also failed.",
                    parent=self,
                )
            elif result.get("manual_board"):
                self._open_repair_dialog()
                messagebox.showwarning("Automatic board detection needs review", msg, parent=self)
            self._refresh_setup_state()
            return

        region: ScreenRegion = result["region"]
        panel = result["panel"]
        board_det = result["board_det"]
        board_check = result["board_check"]
        ref = result.get("reference")

        if result.get("needs_confirmation"):
            self._setup_feedback(
                "AUTO SETUP found a medium-confidence candidate. Check the 5x5 preview once; you do not need to redraw anything.",
                "warn",
            )
            bimg = crop_roi(panel, board_det.roi)
            if not confirm_board_alignment_dialog(self, bimg, board_check):
                self._set_auto_setup_active(False)
                self.auto_progress_var.set(0.0)
                self.auto_progress_text.set("Candidate not confirmed")
                self._setup_feedback(
                    "Automatic candidate was not confirmed. Run AUTO SETUP again or open Repair / recovery only if repeated automatic attempts fail.",
                    "warn",
                )
                return

        self.region = region
        save_region_anchor(panel, self.anchor_path)
        self.app_cfg.screen_region = region.__dict__
        self.app_cfg.save(self.app_cfg_path)
        panel_score = float(result.get("panel_score") or validate_game_panel(panel).score)
        self.vision_cfg.game_area_geometry_score = panel_score

        self.vision_cfg.board_roi = board_det.roi
        self.vision_cfg.board_roi_validated = True
        self.vision_cfg.board_geometry_score = float(0.72 * board_det.score + 0.28 * board_check.score)

        # V5.2 intentionally does not persist a static current-jewel ROI. The
        # icon is located dynamically from the user-specific x4 references on
        # every live frame, so window scaling cannot leave a stale crop behind.
        self.vision_cfg.current_roi = None
        self.vision_cfg.current_roi_validated = False
        self.vision_cfg.current_geometry_score = 0.0
        self.vision_cfg.save(self.vision_cfg_path)

        if ref is not None and ref.confidence >= 0.72:
            self._commit_template_batch(
                ref.templates, f"V5.2 automatic x4 references ({ref.confidence:.2f})",
                geometry_score=ref.confidence, silent=True,
            )
            self.vision_cfg.templates_validated = True
            self.vision_cfg.template_geometry_score = float(ref.confidence)
            self.vision_cfg.save(self.vision_cfg_path)

        self._invalidate_vision_test()
        self._refresh_setup_state()
        elapsed = float(result.get("elapsed", 0.0))
        self.auto_progress_var.set(100.0)
        self.auto_progress_text.set(f"Complete in {elapsed:.1f}s")
        self._set_auto_setup_active(False)

        if not self._templates_ready() or not self.vision_cfg.templates_validated:
            msg = (
                f"Geometry complete from {result.get('source','automatic detection')} in {elapsed:.1f}s: "
                f"board={board_det.score:.2f}, MU={board_det.mu_anchor_score:.2f}. "
                "The six current-jewel references are still missing. Return once to the idle screen "
                "where the colored x4 list is visible and run AUTO SETUP again; no manual board labeling is needed."
            )
            self._setup_feedback(msg, "warn")
            self._auto_log(msg, 100, "WARN")
            return

        msg = (
            f"AUTO SETUP complete in {elapsed:.1f}s ✅ source={result.get('source','auto')}; "
            f"board={board_det.score:.2f}; MU={board_det.mu_anchor_score:.2f}; "
            f"coverage={board_det.coverage*100:.0f}%; six current-jewel references ready. "
            "Open/start a box and press START; current-jewel location is automatic."
        )
        self._setup_feedback(msg, "ok")
        self.status.set("Automatic calibration saved. START validates the board and then waits for/detects the current jewel automatically.")
        self._auto_log(msg, 100)

    def select_area(self, continue_manual: bool = True):
        if not self._pause_monitor_for_setup():
            return
        try:
            self.withdraw()
            self.update_idletasks()
            time.sleep(0.20)
            img, mon = screenshot_all()
            self.deiconify()
            self.lift()
            self.attributes("-topmost", True)

            while True:
                roi = select_roi_dialog(
                    self, img, "Step 1/5 – Select Jewel Bingo panel",
                    "Select the Jewel Bingo working panel: include the title, current-jewel row, 5x5 board and right-side jewel list. The bottom instruction text is optional. STOP before Event Inventory.",
                )
                if roi is None:
                    self.status.set("Game-area selection cancelled. Nothing changed.")
                    self._setup_feedback("Step 1 cancelled. Your previous game area was kept.", "warn")
                    self._refresh_setup_state()
                    return
                x, y, w, h = roi
                candidate = img[y:y+h, x:x+w]
                check = validate_game_panel(candidate)
                accepted = confirm_geometry_dialog(
                    self, candidate, "Validate Jewel Bingo panel", check, "game",
                    "The preview should contain the Jewel Bingo title, current-jewel row, 5×5 board and right-side jewel list. "
                    "Event Inventory and surrounding game UI must stay outside.",
                    accept_text="Use this game area",
                )
                if accepted and check.ok:
                    break
                self._setup_feedback(
                    f"Step 1 not saved. Geometry {check.score:.2f}/{quality_band(check.score)}; adjust the rectangle and review the metric breakdown.",
                    "warn" if check.ok else "error",
                )

            old = self.region
            self.region = ScreenRegion(mon["left"] + x, mon["top"] + y, w, h)
            panel = capture_region(self.region)
            save_region_anchor(panel, self.anchor_path)
            self.app_cfg.screen_region = self.region.__dict__
            self.app_cfg.save(self.app_cfg_path)
            self.vision_cfg.game_area_geometry_score = check.score
            self.vision_cfg.save(self.vision_cfg_path)
            self._invalidate_vision_test()

            size_changed = old and (
                abs(old.width - w) / max(1, old.width) > 0.08
                or abs(old.height - h) / max(1, old.height) > 0.08
            )
            if size_changed:
                self.vision_cfg.board_roi = None
                self.vision_cfg.current_roi = None
                self.vision_cfg.remaining_roi = None
                self.vision_cfg.result_lucky_roi = None
                self.vision_cfg.result_normal_roi = None
                self.vision_cfg.result_jewel_roi = None
                self.vision_cfg.result_total_roi = None
                self.vision_cfg.board_roi_validated = False
                self.vision_cfg.current_roi_validated = False
                self.vision_cfg.templates_validated = False
                self.vision_cfg.board_geometry_score = 0.0
                self.vision_cfg.template_geometry_score = 0.0
                self.vision_cfg.current_geometry_score = 0.0
                self.vision_cfg.save(self.vision_cfg_path)
                self.status.set("Game area validated and saved. Its size changed, so dependent calibration was invalidated. Next: select the 5x5 board.")
            else:
                self.status.set(f"Game area validated (score {check.score:.2f}) and saved. Next: select/validate the 5x5 board.")
            self._refresh_setup_state()
            if continue_manual:
                self._ask_continue_setup(
                    1,
                    f"Game area accepted and saved (validation score {check.score:.2f}). Step 1 is complete. Next: select only the 5x5 board.",
                    self.calibrate_board_roi,
                )
        except Exception as e:
            try:
                self.deiconify(); self.lift(); self.attributes("-topmost", True)
            except Exception:
                pass
            self._setup_feedback(f"Step 1 failed: {e}", "error")
            messagebox.showerror("Area selection", str(e), parent=self)
            self._refresh_setup_state()

    def find_moved_area(self, quiet: bool = False) -> bool:
        if self.region is None or not self.anchor_path.exists():
            if not quiet:
                messagebox.showinfo("Find panel", "Run AUTO SETUP once first.", parent=self)
            return False
        try:
            # Use a permissive anchor threshold, then require a real 5x5+MU
            # geometry check before committing. This is safer and more useful
            # than rejecting a moved panel solely because its title bar changed
            # slightly between idle/playing states.
            found, score = relocate_region_by_anchor(self.anchor_path, self.region, threshold=0.50)
            if found is None:
                if not quiet:
                    messagebox.showwarning("Find panel", f"Saved panel anchor not found (match {score:.2f}). Run AUTO SETUP.", parent=self)
                return False
            candidate = capture_region(found)
            fresh = detect_board_grid(candidate)
            if fresh is None or fresh.score < 0.64 or fresh.mu_anchor_score < 0.50:
                if not quiet:
                    messagebox.showwarning(
                        "Find panel",
                        f"An anchor candidate was found ({score:.2f}) but the 5x5/MU check rejected it. Run AUTO SETUP.",
                        parent=self,
                    )
                return False
            old = self.region
            moved = (found.left, found.top, found.width, found.height) != (old.left, old.top, old.width, old.height)
            self.region = found
            self.app_cfg.screen_region = found.__dict__
            self.app_cfg.save(self.app_cfg_path)
            self.vision_cfg.board_roi = fresh.roi
            self.vision_cfg.board_roi_validated = True
            self.vision_cfg.board_geometry_score = fresh.score
            self.vision_cfg.current_roi = None
            self.vision_cfg.current_roi_validated = False
            self.vision_cfg.current_geometry_score = 0.0
            self.vision_cfg.save(self.vision_cfg_path)
            save_region_anchor(candidate, self.anchor_path)
            self._invalidate_vision_test()
            self.status.set(
                f"Panel {'relocated' if moved else 'confirmed'} automatically: anchor={score:.2f}, "
                f"board={fresh.score:.2f}, MU={fresh.mu_anchor_score:.2f}."
            )
            self._refresh_setup_state()
            return True
        except Exception as e:
            if not quiet:
                messagebox.showerror("Find panel", str(e), parent=self)
            return False

    def calibrate_board_roi(self):
        if not self._pause_monitor_for_setup():
            return
        try:
            panel = self._panel()
            while True:
                roi = select_roi_dialog(
                    self, panel, "Step 2/5 – Select 5x5 board",
                    "Select ONLY the square 5x5 Jewel Bingo grid, including the MU center. Do not include the right-side jewel counters, title, or bottom text.",
                )
                if roi is None:
                    self.status.set("Board-area selection cancelled. Nothing changed.")
                    self._setup_feedback("Step 2 cancelled. Select only the 5x5 grid when ready.", "warn")
                    return
                check = validate_board_roi(panel, roi)
                bimg = crop_roi(panel, roi)
                accepted = confirm_board_alignment_dialog(
                    self, bimg, check
                )
                if accepted and check.ok:
                    break
                if not check.ok:
                    self._setup_feedback("Step 2 was NOT accepted. The 5x5 geometry check failed; adjust the crop.", "error")
                    messagebox.showwarning(
                        "Board crop rejected",
                        "Automatic validation blocked this board crop. Select again more tightly around the 25 cells.\n\n" +
                        "\n".join(f"• {m}" for m in check.messages),
                        parent=self,
                    )
                # If validation passed but user chose Select again, simply loop.

            old_roi = self.vision_cfg.board_roi
            self.vision_cfg.board_roi = roi
            self.vision_cfg.board_roi_validated = True
            self.vision_cfg.board_geometry_score = check.score
            # Board-cell appearance changes when the crop changes; force template
            # re-validation so an old bad V4.3 batch can never silently survive.
            if old_roi != roi:
                self.vision_cfg.templates_validated = False
                self.vision_cfg.template_geometry_score = 0.0
            self.vision_cfg.save(self.vision_cfg_path)
            self._invalidate_vision_test()
            self.status.set(
                f"5x5 board saved after validation: {check.score:.2f}/{quality_band(check.score)}. Next: learn the six jewels."
            )
            self._refresh_setup_state()
            self._ask_continue_setup(
                2,
                f"5x5 board accepted after review (geometry {check.score:.2f}/{quality_band(check.score)}). Step 2 is complete. Next: learn/verify the six jewel types.",
                self.calibrate_templates,
            )
        except Exception as e:
            self._setup_feedback(f"Step 2 failed: {e}", "error")
            messagebox.showerror("Board calibration", str(e), parent=self)
            self._refresh_setup_state()

    def _commit_template_batch(self, batch: dict[str, object], source: str, geometry_score: float = 1.0, silent: bool = False) -> None:
        """Replace current-jewel/x4 references with one trusted six-class batch.

        AUTO SETUP is the explicit repair boundary: old adaptive references are
        archived instead of silently surviving forever across versions.
        """
        self.icon_matcher.replace_batch(batch)
        self.vision_cfg.templates_validated = self.icon_matcher.ready()
        self.vision_cfg.template_geometry_score = float(geometry_score)
        self.vision_cfg.save(self.vision_cfg_path)
        self._invalidate_vision_test()
        self.status.set(
            f"Six current-jewel references saved ({source}). The current jewel will now be found automatically when a box is open."
        )
        self._refresh_setup_state()
        if silent:
            return
        self._setup_feedback(
            "Jewel references are ready. No current-jewel ROI is required in V5.2; open/start a box and press START.",
            "ok",
        )

    def calibrate_templates(self):
        """Refresh the six current-jewel references from the game's x4 list.

        V5.2 deliberately does not ask the user to label board cells here: board
        classification has its own board-domain seed/learning path, while the
        x4 sprites are the correct domain for locating the top current jewel.
        """
        if not self._pause_monitor_for_setup():
            return
        try:
            panel = self._panel()
            if self.vision_cfg.board_roi is None or not self.vision_cfg.board_roi_validated:
                raise RuntimeError("The board geometry is not ready. Run AUTO SETUP first.")
            auto = detect_reference_jewels_detailed(panel, self.vision_cfg.board_roi)
            if auto is None:
                self._setup_feedback(
                    "The colored x4 reference list is not visible. Finish/close the current box so the six colored x4 icons appear, then run AUTO SETUP again. No board-cell labeling is required.",
                    "warn",
                )
                messagebox.showinfo(
                    "Colored x4 references not visible",
                    "I could not see the six colored x4 icons on the right side.\n\n"
                    "Return Jewel Bingo to the idle screen where all six icons show x4, then click AUTO SETUP. "
                    "V5.2 no longer asks you to teach the six jewels by clicking board cells.",
                    parent=self,
                )
                return
            self._commit_template_batch(
                auto.templates,
                f"automatic x4 reference detection, confidence {auto.confidence:.2f}/{quality_band(auto.confidence)}",
                geometry_score=auto.confidence,
            )
        except Exception as e:
            messagebox.showerror("Learn jewels", str(e), parent=self)
            self._refresh_setup_state()

    def _learn_from_trusted_board(self, panel, detail) -> int:
        """Add diverse board-cell examples only from exceptionally clean reads.

        This is intentionally conservative to avoid self-training on an early
        classification mistake. Confirmed placements remain the stronger source
        of visual truth during normal play.
        """
        if (
            detail.board_confidence < 0.55
            or detail.min_assigned_probability < 0.25
            or len(detail.ambiguous_cells) != 0
            or self.vision_cfg.board_roi is None
        ):
            return 0
        bimg = crop_roi(panel, self.vision_cfg.board_roi)
        learned = 0
        for i, patch in board_cell_rois(bimg):
            if i == CENTER:
                continue
            label = detail.board.cells[i]
            prob = detail.assigned_probabilities[i]
            if label not in JEWELS or prob < 0.50:
                continue
            try:
                if self.clf.add_template_if_novel(label, patch, min_distance=0.20) is not None:
                    learned += 1
            except Exception:
                pass
        return learned

    def _accept_board(self, board: Board, mask: int, confidence: float):
        # Never promote the first screen-derived selected mask directly to a
        # durable checkpoint. Hover/glow on that one frame used to poison the
        # whole round (e.g. false 1/14 followed by ignored 0/14 regression).
        # Bootstrap from the physical invariant 0/14 and let TemporalStateFilter
        # confirm/catch up from repeated frames if the app started mid-round.
        initial_mask = 0
        with self._live_state_lock:
            self.board = board
            self.board_confidence = confidence
            self.selected_mask = initial_mask
            # The icon observed on the same first frame is only a visual hint.
            # Do not publish a recommendation until mask+jewel have temporal
            # confirmation from the live monitor.
            self.current_jewel = None
            self.current_confidence = 0.0
            self.current_margin = 0.0
            self.current_entropy = 1.0
        self.temporal.reset(initial_mask)
        self.temporal_initialized = True
        self._transition_fast_until = time.monotonic() + 2.0
        self.db.ensure_game(
            self.episode_id, "local-auto", self.session_id, board, self.box_color,
            self.app_cfg.operation_mode, self.app_cfg.risk_profile,
        )
        h = board_hash(board)
        repeated = {r["board_hash"]: r["games"] for r in self.db.repeated_layouts(2)}
        seen = repeated.get(h, 1)
        meta = f"Board hash {h} | vision {confidence:.2f} | seen in DB: {seen}x"
        self._ui_call(self.board_meta.set, meta)
        self._update_phase()
        # Stability-first root fix: do not launch six speculative WSL jobs
        # before the current jewel is even known.

    def read_once(self, show_report: bool = True) -> bool:
        """Validate the live board and opportunistically locate current jewel.

        V5.2 treats current-jewel detection as dynamic runtime state, not a
        calibration prerequisite. A valid board can therefore start monitoring
        while the app waits for a visible current jewel.
        """
        if not self._calibration_ready():
            missing = ", ".join(self._calibration_missing())
            self._setup_feedback(f"Cannot test live vision yet. Missing: {missing}.", "error")
            if show_report:
                messagebox.showwarning("Setup incomplete", f"Complete AUTO SETUP first. Missing: {missing}", parent=self)
            return False
        if not self._pause_monitor_for_setup():
            return False
        self.vision_test_passed = False
        try:
            panel = self._panel()
            self.last_panel = panel

            board_check = validate_board_roi(panel, self.vision_cfg.board_roi)
            if not board_check.ok:
                # Before blaming the user, try a fresh automatic board fit inside
                # the already-known panel. Moving/resizing the window should not
                # force a manual board selection.
                fresh = detect_board_grid(panel)
                if fresh is not None and fresh.score >= 0.64 and fresh.mu_anchor_score >= 0.50:
                    self.vision_cfg.board_roi = fresh.roi
                    self.vision_cfg.board_roi_validated = True
                    self.vision_cfg.board_geometry_score = fresh.score
                    self.vision_cfg.save(self.vision_cfg_path)
                    board_check = validate_board_roi(panel, fresh.roi)
            if not board_check.ok:
                self.vision_cfg.board_roi_validated = False
                self.vision_cfg.save(self.vision_cfg_path)
                self._refresh_setup_state()
                msg = "The live 5x5 geometry could not be recovered automatically. Run AUTO SETUP again."
                self._setup_feedback(msg, "error")
                if show_report:
                    messagebox.showerror("Board geometry", msg, parent=self)
                return False

            detail = read_board_detailed(panel, self.vision_cfg, self.clf)
            board_ok = (
                detail.board_confidence >= self.vision_cfg.board_min_confidence
                and detail.min_assigned_probability >= self.vision_cfg.board_min_cell_probability
                and len(detail.ambiguous_cells) <= self.vision_cfg.board_max_ambiguous_cells
            )

            icon = self.icon_matcher.locate(panel, self.vision_cfg.board_roi)
            current_ok = bool(icon is not None and icon.reliable)
            if icon is not None:
                self.vision_cfg.current_roi = icon.roi
                self.vision_cfg.current_roi_validated = current_ok
                self.vision_cfg.current_geometry_score = icon.score
                if current_ok:
                    self.current_jewel = icon.label
                    self.current_confidence = icon.score
                    self.current_margin = icon.margin
                    self.current_entropy = 0.0
                    self.last_current_patch = icon.patch.copy()
                    self.last_current_confidence = icon.score
                    self.current_label.set(
                        f"Current jewel: {icon.label} / {JEWEL_NAMES[icon.label]} "
                        f"(match {icon.score:.2f}, margin {icon.margin:.2f})"
                    )
                    self.vision_label.set(f"Current icon match {icon.score:.2f} | margin {icon.margin:.2f}")
                else:
                    self.current_jewel = None
                    self.current_label.set(
                        f"Current jewel: waiting/uncertain ({icon.label}? match {icon.score:.2f}, margin {icon.margin:.2f})"
                    )
            else:
                self.vision_cfg.current_roi = None
                self.vision_cfg.current_roi_validated = False
                self.vision_cfg.current_geometry_score = 0.0
                self.current_jewel = None
                self.current_label.set("Current jewel: waiting for a visible box jewel")

            # A current jewel is runtime state, not setup. The board is the only
            # mandatory live validation for START; monitoring can wait for icon.
            self.vision_test_passed = bool(board_ok)
            live_rows = [
                (
                    "Board confidence", detail.board_confidence,
                    f">= {self.vision_cfg.board_min_confidence:.2f}",
                    detail.board_confidence >= self.vision_cfg.board_min_confidence,
                    "Board-domain classifier confidence after enforcing exactly four of each jewel.",
                ),
                (
                    "Weakest board-cell probability", detail.min_assigned_probability,
                    f">= {self.vision_cfg.board_min_cell_probability:.2f}",
                    detail.min_assigned_probability >= self.vision_cfg.board_min_cell_probability,
                    "The least-certain assigned board cell must still have enough support.",
                ),
                (
                    "Ambiguous board cells", float(len(detail.ambiguous_cells)),
                    f"<= {self.vision_cfg.board_max_ambiguous_cells}",
                    len(detail.ambiguous_cells) <= self.vision_cfg.board_max_ambiguous_cells,
                    "Fewer ambiguous cells means a more trustworthy 5x5 reconstruction.",
                ),
            ]
            if icon is not None:
                live_rows.extend([
                    (
                        "Current icon template match", icon.score, ">= 0.76", icon.score >= 0.76,
                        "The top jewel is matched directly against references learned from the game's x4 list.",
                    ),
                    (
                        "Current icon label margin", icon.margin, ">= 0.035", icon.margin >= 0.035,
                        "Difference between the best and second-best jewel sprite matches.",
                    ),
                ])
            geometry_rows = [
                ("Game area", self.vision_cfg.game_area_geometry_score, "outer Jewel Bingo panel crop"),
                ("5x5 board", self.vision_cfg.board_geometry_score, "auto-detected repeated-cell geometry + MU anchor"),
                ("Six current-jewel references", self.vision_cfg.template_geometry_score, "colored x4 list; separate from board-cell classifier"),
            ]
            if show_report:
                show_vision_validation_dialog(self, self.vision_test_passed, live_rows, geometry_rows)

            if self.vision_test_passed:
                learned_now = self._learn_from_trusted_board(panel, detail)
                self._accept_board(detail.board, detail.selected_mask, detail.board_confidence)
                self._refresh_grid()
                self._update_phase()
                self._refresh_setup_state()
                if self.current_jewel:
                    self._recommend()
                    current_msg = f"current={self.current_jewel} match={icon.score:.2f}"
                else:
                    current_msg = "current jewel not visible yet; monitoring will wait automatically"
                self.status.set(
                    f"LIVE BOARD OK ✅ board={detail.board_confidence:.2f}, min-cell={detail.min_assigned_probability:.2f}; "
                    f"{current_msg}. Board visual memory +{learned_now}."
                )
                self._setup_feedback(
                    "Live board validation passed. Current-jewel detection is automatic; no manual Current jewel step is required.",
                    "ok",
                )
                return True

            self.board = detail.board
            self.selected_mask = detail.selected_mask
            self.board_confidence = detail.board_confidence
            self.board_meta.set(
                f"PREVIEW ONLY — board recognition failed | vision {detail.board_confidence:.3f} | no data saved"
            )
            self._refresh_grid()
            self._refresh_setup_state()
            self.status.set(
                f"BOARD RECOGNITION FAILED: conf={detail.board_confidence:.3f}, "
                f"min-cell={detail.min_assigned_probability:.3f}, ambiguous={len(detail.ambiguous_cells)}. "
                "This is a classifier issue, not a reason to redraw the board geometry."
            )
            self._setup_feedback(
                "Board geometry is valid but jewel recognition is uncertain. Do NOT reselect the board; run AUTO SETUP/diagnostics instead.",
                "error",
            )
            return False
        except Exception as e:
            self.vision_test_passed = False
            self._refresh_setup_state()
            if show_report:
                messagebox.showerror("Test vision", str(e), parent=self)
            else:
                self.status.set(f"Live vision preparation failed: {e}")
            return False

    @staticmethod
    def _move_visual_status(top: Recommendation, rec: Recommendation, rank: int) -> str:
        return move_visual_status(top, rec, rank)

    @staticmethod
    def _visual_status_style(status: str) -> tuple[str, str, str]:
        return visual_status_style(status)

    def _paint_ranked_cells(self, mask: int, recs: list[Recommendation]) -> None:
        if not recs:
            return
        top = recs[0]
        for rank, rec in enumerate(recs[:4]):
            if (mask >> rec.position) & 1:
                continue
            r0, c0 = divmod(rec.position, 5)
            _v, btn = self.cell_vars[r0][c0]
            status = self._move_visual_status(top, rec, rank)
            _label, color, _tag = self._visual_status_style(status)
            btn.config(bg=color, relief="solid", bd=3 if rank == 0 else 2)

    def _refresh_grid(self):
        if self.board is None:
            return
        for r in range(5):
            for c in range(5):
                i = r * 5 + c
                v, b = self.cell_vars[r][c]
                if i == CENTER:
                    v.set("MU")
                    b.config(bg="SystemButtonFace", relief="raised", bd=1)
                    continue
                jewel = self.board.cells[i]
                v.set(f"{jewel}\nL{r+1}C{c+1}")
                selected = (self.selected_mask >> i) & 1
                b.config(bg="#70a7ff" if selected else "SystemButtonFace", relief="raised", bd=1)
        if self._ranked_grid_mask == int(self.selected_mask) and self._ranked_grid_recs:
            self._paint_ranked_cells(int(self.selected_mask), self._ranked_grid_recs)
        elif self._preview_grid_mask == int(self.selected_mask) and self._preview_grid_pos is not None:
            pos = int(self._preview_grid_pos)
            if pos != CENTER and not ((int(self.selected_mask) >> pos) & 1):
                rr, cc = divmod(pos, 5)
                _var, button = self.cell_vars[rr][cc]
                button.config(bg="#ffe0a0", relief="solid", bd=3)

    def _monitor_refresh_grid_if_changed(self) -> None:
        """Avoid repainting 25 Tk cells on every identical camera frame.

        Explicit recommendation and mode changes still repaint immediately.
        """
        if self.board is None:
            return
        signature = (self.episode_id, board_hash(self.board), int(self.selected_mask))
        if signature == self._last_monitor_grid_signature:
            return
        self._last_monitor_grid_signature = signature
        self._refresh_grid()

    def toggle_cell(self, idx: int):
        if idx == CENTER:
            return
        self.selected_mask ^= 1 << idx
        self.temporal.reset(self.selected_mask)
        self._refresh_grid()
        self._recommend()

    def _learned_scores(self, mask: int, jewel: str, positions: tuple[int, ...]) -> dict[int, float]:
        if self.board is None or not self.app_cfg.ml_enabled:
            return {}
        # Exact full-board + chronological jewel/position prefix only.
        # No future jewels or scores are supplied by a live request.
        return self.experience.predict(
            self.board, mask, jewel, positions,
            observed_jewels=tuple(self.game_sequence),
            observed_positions=tuple(int(a["action_pos"]) for a in self.episode_actions),
        )

    def _advisor(self) -> Advisor:
        assert self.board is not None
        hist = self.history.load()
        weights = learned_bias_weights(
            hist,
            blend=self.app_cfg.empirical_bias_blend,
            min_games=self.app_cfg.rng_min_games,
            local_only=True,
        )
        return Advisor(
            self.board,
            mode=self.app_cfg.mode,
            exact_horizon=self.app_cfg.exact_horizon,
            rollout_exact_horizon=self.app_cfg.rollout_exact_horizon,
            rollouts=self.app_cfg.rollouts,
            bias_weights=weights,
            rng_seed=self.app_cfg.rng_seed,
            learned_action_scores=self._learned_scores if self.app_cfg.ml_enabled else None,
            p3_min=self.app_cfg.p3_min,
            p2l1n_min=self.app_cfg.p2l1n_min,
            p1l2n_min=self.app_cfg.p1l2n_min,
            p1l1n_min=self.app_cfg.p1l1n_min,
            fallback_override_pp=self.app_cfg.fallback_override_pp,
            p3_near_best_tolerance=self.app_cfg.p3_near_best_tolerance,
            ml_near_tie_probability=self.app_cfg.ml_near_tie_probability,
            ml_near_tie_score=self.app_cfg.ml_near_tie_score,
        )

    def _decision_image_path(self, mask: int) -> str | None:
        if not self.app_cfg.save_game_images or self.last_panel is None:
            return None
        step = mask.bit_count()
        path = self.decision_image_dir / f"{self.episode_id}_step{step:02d}.jpg"
        if not path.exists():
            cv2.imwrite(str(path), self.last_panel, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        return str(path)

    def _persist_counterfactuals(self, mask: int, jewel: str, recs: list[Recommendation]) -> None:
        if self.board is None:
            return
        key = (self.episode_id, mask, jewel)
        if key == self.last_decision_key:
            return
        self.last_decision_key = key
        image_path = self._decision_image_path(mask)
        self.last_decision_image = image_path
        rows = []
        for rec in recs:
            u = rec.utility
            rows.append({
                "position": rec.position,
                "method": rec.method,
                "p3": u.p3,
                "p2l1n": u.p2l1n,
                "p1l2n": u.p1l2n,
                "p1l1n": u.p1l1n,
                "p_gt1000": u.p_gt1000,
                "p_ge999": u.p_ge999,
                "expected_score": u.expected_score,
                "solver_target": utility_learning_target(u),
                "active_goal": rec.active_goal,
                "samples": rec.samples,
                "stderr_p3": rec.stderr_p3,
                "stderr_gt1000": rec.stderr_gt1000,
                "stderr_score": rec.stderr_score,
                "solver_regret": rec.solver_regret,
                "rng_seed": int(self.app_cfg.rng_seed),
                "scenario_set": f"seed={self.app_cfg.rng_seed}|board={board_hash(self.board)}|mask={mask}|jewel={jewel}|samples={rec.samples}",
                "routes": [
                    {
                        "target": route.target, "probability": route.probability,
                        "lucky_lines": list(route.lucky_lines), "normal_lines": list(route.normal_lines),
                        "needs": list(route.needs), "missing_cells": route.missing_cells,
                    } for route in rec.routes
                ],
                "primary_stat_tie": rec.primary_stat_tie,
                "tie_break_metric": rec.tie_break_metric,
            })
        self.db.save_decision_samples(
            self.episode_id, mask.bit_count(), mask, jewel, self.board, rows, image_path
        )

    def _compute_recommendations(self, mask: int, jewel: str, persist: bool = True) -> list[Recommendation]:
        if self.app_cfg.operation_mode != "recommend" or self.board is None:
            return []
        with self._solver_lock:
            key = (board_hash(self.board), int(mask), jewel)
            if key == self.cached_recommend_key and self.last_recs:
                return list(self.last_recs)
            adv = self._advisor()
            recs = adv.recommend(mask, jewel)
            self.cached_recommend_key = key
            self.last_recs = list(recs)
            if persist:
                self._persist_counterfactuals(mask, jewel, recs)
            return recs

    def _solver_service_status(self, ok: bool, status: str) -> None:
        if ok:
            self._auto_log(f"V18 persistent solver ready: {status}", None, "INFO")
            if not self.running:
                self.status.set("Persistent solver ready (" + self.app_cfg.solver_backend + ").")
        else:
            self._auto_log(f"V18 persistent solver unavailable: {status}; local fallback remains enabled", None, "WARN")
        self._refresh_system_panel()

    def _live_game_state(self, jewel: str | None = None) -> LiveGameState | None:
        if self.board is None:
            return None
        bh = board_hash(self.board)
        phase = self.phase.value
        strategy = self.app_cfg.mode
        coord = self.live_coordinator
        if (
            coord is None
            or coord.state.episode_id != self.episode_id
            or coord.state.board_hash != bh
            or coord.state.board_cells != tuple(self.board.cells)
        ):
            state = LiveGameState(
                episode_id=self.episode_id, state_version=0, strategy_version=0,
                board_hash=bh, board_cells=tuple(self.board.cells),
                accepted_mask=int(self.selected_mask), current_jewel=jewel,
                strategy=strategy, phase=phase,
            )
            self.live_coordinator = LiveCoordinator(state)
            return state
        if coord.state.accepted_mask != int(self.selected_mask) or coord.state.phase != phase:
            coord.update_board(accepted_mask=int(self.selected_mask), phase=phase)
        if coord.state.strategy != strategy:
            coord.update_strategy(strategy)
        if coord.state.current_jewel != jewel:
            coord.update_jewel(jewel)
        return coord.state

    def _protocol_solver_request(
        self, request_id: int, mask: int, jewel: str, rollouts: int, *, purpose: str = "decision"
    ) -> SolverRequest | None:
        state = self._live_game_state(jewel)
        if state is None:
            return None
        legacy = self._solver_request(mask, jewel, rollouts)
        return SolverRequest.from_legacy(request_id, state, legacy, purpose=purpose)

    def _start_remote_solver(
        self, request_id: int, key: tuple, mask: int, jewel: str, *, stage: str = "preview",
    ) -> None:
        """Two-step persistent solve: publish PREVIEW, then confirm FINAL.

        Both requests run through the same single latest-state lane and hot
        WSL worker. Only FINAL can commit an auto plan, persist ML or click.
        """
        if self.app_cfg.operation_mode != "recommend":
            return
        if request_id != self.recommend_request_id or self._recommend_state_key() != key:
            return
        # Progressive decisions give early visible feedback, then a final pass.
        final = (stage == "final" or not self.app_cfg.progressive_live_decisions)
        budget = self.app_cfg.live_rollouts if final else min(
            int(self.app_cfg.preview_rollouts), int(self.app_cfg.live_rollouts)
        )
        request = self._protocol_solver_request(
            request_id, mask, jewel, budget, purpose="live-final" if final else "live-preview"
        )
        if request is None:
            return
        self.remote_solver_active_key = key
        started = time.perf_counter()

        def on_result(response):
            self._ui_call(
                self._handle_remote_solver_result, request_id, key, mask, jewel, request,
                response, started, "final" if final else "preview",
            )

        def on_error(exc):
            self._ui_call(
                self._handle_remote_solver_error, request_id, key, mask, jewel, exc
            )

        submit_info = self.solver_bridge.solve_async(request, on_result, on_error)
        self._runtime_record(
            "solver-decision-submit",
            request_id=int(request_id), stage="final" if final else "preview",
            rollouts=int(budget), jewel=str(jewel),
            selected_count=int(mask).bit_count(),
            accepted=bool(submit_info.get("accepted", False)),
            generation=int(submit_info.get("generation", 0)),
            superseded=bool(submit_info.get("superseded", False)),
        )
        if submit_info.get("superseded"):
            self._runtime_record(
                "solver-decision-superseded", level="INFO",
                newest_request_id=int(request_id), stage=stage, jewel=str(jewel),
                selected_count=int(mask).bit_count(),
            )

    def _adopt_auto_plan(self, recs: list[Recommendation]) -> None:
        if not recs or self.app_cfg.mode != "auto" or self.live_coordinator is None:
            return
        top = recs[0]
        if not top.plan_target:
            return
        before = self.live_coordinator.state
        after = self.live_coordinator.update_plan(
            target=top.plan_target,
            lucky_lines=top.plan_lucky_lines,
            normal_lines=top.plan_normal_lines,
        )
        if before.strategy_version != after.strategy_version:
            route_text = " + ".join([*(f"Lucky {x}" for x in top.plan_lucky_lines), *(f"Normal {x}" for x in top.plan_normal_lines)]) or "best score"
            self.db.log_event(
                self.episode_id, "strategy-plan-change",
                f"target={top.plan_target}; route={route_text}", self.phase.value,
                {
                    "target": top.plan_target,
                    "lucky_lines": list(top.plan_lucky_lines),
                    "normal_lines": list(top.plan_normal_lines),
                    "reason": top.plan_reason,
                    "probability": top.plan_probability,
                },
            )
            self._checkpoint_state(
                "strategy-plan-change",
                payload={
                    "target": top.plan_target, "lucky_lines": list(top.plan_lucky_lines),
                    "normal_lines": list(top.plan_normal_lines), "reason": top.plan_reason,
                },
            )

    def _handle_remote_solver_result(
        self, request_id: int, key: tuple, mask: int, jewel: str,
        request: SolverRequest, response, started: float, stage: str = "final",
    ) -> None:
        if self.app_cfg.operation_mode != "recommend":
            return
        coord = self.live_coordinator
        if (
            request_id != self.recommend_request_id
            or self._recommend_state_key() != key
            or self.phase == GamePhase.COMPLETE
            or self.selected_mask.bit_count() >= 14
            or coord is None
            or not coord.accepts(response)
        ):
            self._auto_log("V18 discarded stale persistent-solver result", None, "INFO")
            return
        recs = list(response.recommendations)
        exact = bool(recs and recs[0].method in {"immediate-lucky-fastpath", "reward-tier-exact"})
        stage_ms = (time.perf_counter() - started) * 1000.0
        overall_ms = ((time.perf_counter() - self._decision_started_at) * 1000.0
                      if getattr(self, "_decision_started_key", None) == key else None)
        self._runtime_record(
            "solver-stage-result", request_id=int(request_id), stage=stage,
            roundtrip_ms=round(stage_ms, 2), compute_ms=round(float(response.elapsed_ms), 2),
            since_ui_request_ms=round(overall_ms, 2) if overall_ms is not None else None,
            cache_hit=bool(response.cache_hit), state_mask=int(mask), jewel=str(jewel),
        )
        if stage == "preview" and not exact:
            # Never cache, log a game decision, update the strategy or permit
            # acting on this statistically provisional ranking.
            self._render_recommendations(mask, jewel, recs, refined=False)
            self._start_remote_solver(request_id, key, mask, jewel, stage="final")
            return

        if self.remote_solver_active_key == key:
            self.remote_solver_active_key = None
        self.last_recs = recs
        self.cached_recommend_key = key
        if not exact:
            try:
                self._persist_counterfactuals(mask, jewel, recs)
            except Exception as exc:
                self._auto_log(f"Could not persist V18 solver samples: {exc}", None, "WARN")
        self._adopt_auto_plan(recs)
        self._render_recommendations(mask, jewel, recs, refined=True)
        self._schedule_speculative_best_child(mask, recs)
        self._last_solver_roundtrip_ms = stage_ms
        self._last_solver_compute_ms = float(response.elapsed_ms)
        try:
            self.db.save_telemetry(self.episode_id, "decision-roundtrip", stage_ms, response.cache_hit, {"jewel": jewel})
            self.db.save_telemetry(self.episode_id, "solver-compute", response.elapsed_ms, response.cache_hit, {"jewel": jewel})
        except Exception:
            pass
        hot = "CACHE" if response.cache_hit else "COMPUTE"
        self._auto_log(
            f"V18 {hot} final {response.elapsed_ms:.1f} ms solver / {stage_ms:.1f} ms round-trip",
            None, "INFO",
        )
        self._refresh_system_panel()

    def _handle_remote_solver_error(
        self, request_id: int, key: tuple, mask: int, jewel: str, exc: BaseException
    ) -> None:
        if self.app_cfg.operation_mode != "recommend":
            return
        if self.remote_solver_active_key == key:
            self.remote_solver_active_key = None
        if request_id != self.recommend_request_id or self._recommend_state_key() != key:
            return
        self._auto_log(
            f"Native/persistent solver unavailable ({type(exc).__name__}: {exc}); using local compatibility fallback",
            None, "WARN",
        )
        self.status.set("Persistent solver unavailable; using bounded local fallback...")
        # A service failure must not turn into the old 60 -> 1600 rollout live
        # refinement. Publish one bounded FAST-quality answer and restart service
        # in parallel; the next state will return to the persistent solver.
        self._start_solver_stage(request_id, key, mask, jewel, "fallback-final", self.app_cfg.fast_rollouts)
        self.solver_bridge.restart_async(lambda ok, status: self._ui_call(self._solver_service_status, ok, status))

    def _schedule_speculative_best_child(self, mask: int, recs: list[Recommendation]) -> None:
        """Precompute all six next jewels for the move the UI just recommended.

        Disabled by default in the state-integrity root-fix because six CPU-bound
        speculative workers were measured consuming several cores during live UI.
        """
        if self.app_cfg.operation_mode != "recommend":
            return
        if not getattr(self.app_cfg, "live_speculative_precompute", False):
            return
        """Legacy description retained for research mode.

        This begins before the user clicks, so human reaction time becomes useful
        compute time. If the recommended move is followed, the next live lookup
        is normally hot even when the board-transition-to-jewel delay is short.
        """
        if self.app_cfg.solver_backend == "local" or self.board is None or not recs:
            return
        predicted_mask = int(mask) | (1 << int(recs[0].position))
        if predicted_mask.bit_count() >= 14:
            return
        base = self._live_game_state(None)
        if base is None:
            return
        # State identity is intentionally not part of the solver cache key. The
        # anticipated mask is therefore reusable when vision later confirms it.
        from dataclasses import replace as _dc_replace
        predicted = _dc_replace(base, accepted_mask=predicted_mask, current_jewel=None, state_version=base.state_version + 1)
        requests: list[SolverRequest] = []
        seed = max(1, self.recommend_request_id + 1000)
        for i, next_jewel in enumerate(JEWELS):
            if not self.board.legal_positions(next_jewel, predicted_mask):
                continue
            legacy = self._solver_request(predicted_mask, next_jewel, self.app_cfg.live_rollouts)
            requests.append(SolverRequest.from_legacy(
                -(seed * 10 + i + 1), predicted.with_jewel(next_jewel), legacy,
                purpose="speculative-best-child",
            ))
        self.solver_bridge.precompute_async(requests)

    def _schedule_precompute(self) -> None:
        if self.app_cfg.operation_mode != "recommend":
            return
        if not getattr(self.app_cfg, "live_speculative_precompute", False):
            return
        if self.app_cfg.solver_backend == "local" or self.board is None:
            return
        if self.phase == GamePhase.COMPLETE or self.selected_mask.bit_count() >= 14:
            return
        base = self._live_game_state(None)
        if base is None:
            return
        requests: list[SolverRequest] = []
        seed = max(1, self.recommend_request_id + 1)
        for i, jewel in enumerate(JEWELS):
            if not self.board.legal_positions(jewel, int(self.selected_mask)):
                continue
            state = base.with_jewel(jewel)
            legacy = self._solver_request(int(self.selected_mask), jewel, self.app_cfg.live_rollouts)
            requests.append(SolverRequest.from_legacy(-(seed * 10 + i + 1), state, legacy, purpose="next-jewel-precompute"))
        if not requests:
            return
        self.solver_bridge.precompute_async(
            requests,
            lambda info: self._ui_call(
                self._auto_log,
                f"V18 precompute: scheduled={info.get('scheduled', 0)}, cache={info.get('cached', 0)}, pending={info.get('pending', 0)}",
                None, "INFO",
            ),
        )

    def _recommend_state_key(self) -> tuple | None:
        with self._live_state_lock:
            board = self.board
            jewel = self.current_jewel
            mask = int(self.selected_mask)
            episode_id = self.episode_id
        if board is None or jewel is None:
            return None
        return (
            board_hash(board), mask, jewel,
            self.app_cfg.operation_mode, self.app_cfg.risk_profile, episode_id,
        )

    def _solver_request(self, mask: int, jewel: str, rollouts: int) -> dict:
        if self.app_cfg.operation_mode != "recommend":
            raise RuntimeError("Solver request is disabled in Observable mode")
        assert self.board is not None
        self._reload_experience_if_ready()
        hist = self.history.load()
        weights = learned_bias_weights(
            hist,
            blend=self.app_cfg.empirical_bias_blend,
            min_games=self.app_cfg.rng_min_games,
            local_only=True,
        )
        positions = self.board.legal_positions(jewel, mask)
        learned = self._learned_scores(mask, jewel, positions)
        return {
            "board_cells": list(self.board.cells),
            "mask": int(mask),
            "jewel": jewel,
            "mode": self.app_cfg.mode,
            "exact_horizon": self.app_cfg.exact_horizon,
            # Live latency budget: preserve full endgame Expectimax; allow
            # rollout depths 3/4 only under bounded decision conditions.
            "rollout_exact_horizon": self.app_cfg.live_horizon(
                mask, len(positions), rollouts,
            ),
            "rollouts": int(rollouts),
            "rng_seed": int(self.app_cfg.rng_seed),
            "bias_weights": list(weights),
            "learned_scores": learned,
            "p3_min": self.app_cfg.p3_min,
            "p2l1n_min": self.app_cfg.p2l1n_min,
            "p1l2n_min": self.app_cfg.p1l2n_min,
            "p1l1n_min": self.app_cfg.p1l1n_min,
            "fallback_override_pp": self.app_cfg.fallback_override_pp,
            "adaptive_score_min": self.app_cfg.adaptive_score_min,
            "p3_near_best_tolerance": self.app_cfg.p3_near_best_tolerance,
            "ml_near_tie_probability": self.app_cfg.ml_near_tie_probability,
            "ml_near_tie_score": self.app_cfg.ml_near_tie_score,
            "auto_p3_floor": self.app_cfg.auto_p3_floor,
            "auto_fallback_ratio": self.app_cfg.auto_fallback_ratio,
            "route_switch_ratio": self.app_cfg.route_switch_ratio,
            "plan_target": self.live_coordinator.state.plan_target if self.live_coordinator is not None else "",
            "plan_lucky_lines": self.live_coordinator.state.plan_lucky_lines if self.live_coordinator is not None else (),
            "plan_normal_lines": self.live_coordinator.state.plan_normal_lines if self.live_coordinator is not None else (),
        }

    def _cancel_solver(self, reason: str = "state changed") -> None:
        if self.solver_poll_job is not None:
            try:
                self.after_cancel(self.solver_poll_job)
            except Exception:
                pass
            self.solver_poll_job = None
        proc = self.solver_process
        if proc is not None:
            try:
                if proc.is_alive():
                    proc.terminate()
                proc.join(timeout=0.15)
            except Exception:
                pass
        q = self.solver_queue
        if q is not None:
            try:
                q.cancel_join_thread()
                q.close()
            except Exception:
                pass
        sub = self.solver_subprocess
        if sub is not None:
            try:
                if sub.poll() is None:
                    sub.terminate()
                    try:
                        sub.wait(timeout=0.2)
                    except Exception:
                        sub.kill()
            except Exception:
                pass
        if self.solver_subprocess_paths is not None:
            for path in self.solver_subprocess_paths:
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    pass
        if proc is not None and reason:
            self._auto_log(f"SOLVER cancelled: {reason}", None, "INFO")
        self.solver_process = None
        self.solver_queue = None
        self.solver_subprocess = None
        self.solver_subprocess_paths = None
        self.solver_stage = None
        # A fallback thread cannot be force-killed safely. Clearing this token
        # makes any late result stale; request-id validation discards it.
        self.solver_fallback_key = None
        self.solver_fallback_stage = None
        self.remote_solver_active_key = None

    def _clear_decision_visuals(self, message: str | None = None) -> None:
        self.last_recs = []
        self.cached_recommend_key = None
        self.recommend_active_key = None
        self._ranked_grid_mask = None
        self._ranked_grid_recs = []
        self.overlay.hide()
        self._refresh_grid()
        if hasattr(self, "rec_text"):
            self.rec_text.config(state="normal")
            self.rec_text.delete("1.0", "end")
            if message:
                self.rec_text.insert("end", message)
            self.rec_text.config(state="disabled")
        if hasattr(self, "target_var"):
            self.target_var.set("Target: waiting")
        if hasattr(self, "live_move_var"):
            self.live_move_var.set("WAITING")
            self.live_target_var.set("Target: waiting")
            self.live_route_var.set("Route: waiting")
            self.live_health_var.set("Plan: --")
            self.live_slack_var.set("Slack: --")
            self.live_confidence_var.set("Confidence: --")
            self.live_reason_var.set("")
            self.live_gt1000_var.set(">1000: --")
            self.live_p3_var.set("3 Lucky: --")
            self.live_expected_var.set("Expected score: --")
            if hasattr(self, "live_rank_tree"):
                for iid in self.live_rank_tree.get_children():
                    self.live_rank_tree.delete(iid)

    def _invalidate_recommendation(self, reason: str, message: str | None = None) -> None:
        """Hard invalidation barrier for a changed board/round state.

        Remote WSL decisions use a latest-state-wins lane.  Clearing the GUI
        active key here allows the newest screen state to be submitted at once;
        the bridge collapses any intermediate states instead of building a queue.
        """
        self.recommend_request_id += 1
        self.remote_solver_active_key = None
        self._cancel_solver(reason)
        self._runtime_record(
            "recommendation-invalidated", reason=str(reason),
            request_generation=int(self.recommend_request_id),
            selected_count=int(self.selected_mask).bit_count(),
            current_jewel=self.current_jewel,
        )
        self._clear_decision_visuals(message)

    def _show_game_complete_summary(self, lucky: int, normal: int, total: int) -> None:
        self._clear_decision_visuals(None)
        self.target_var.set("Target: complete")
        if hasattr(self, "live_move_var"):
            self.live_move_var.set("GAME COMPLETE")
            self.live_target_var.set(f"Result: {lucky} Lucky + {normal} Normal")
            self.live_route_var.set(f"Final score: {total} pts")
            self.live_health_var.set("Saved automatically")
            self.live_slack_var.set("")
            self.live_confidence_var.set("")
            self.live_reason_var.set("Waiting automatically for the next round.")
            self.target_change_var.set("")
            self._ui_last_plan_target = None
        auto_verified = bool(
            total > 1000 and getattr(self.app_cfg, "auto_verify_computed_gt1000", True)
        )
        verification_text = (
            "Verification: VERIFIED automatically from trusted 14/14 (>1000)"
            if auto_verified
            else "Verification: completed; automatic verification is reserved for >1000"
        )
        if not hasattr(self, "rec_text"):
            return
        self.rec_text.config(state="normal")
        self.rec_text.delete("1.0", "end")
        self.rec_text.insert(
            "end",
            "GAME COMPLETE\n\n"
            f"Final result: {lucky} Lucky + {normal} Normal | {total} pts\n"
            "Saved automatically ✓\n"
            f"{verification_text}\n"
            "ML + backup: deferred until STOP while live monitoring is active\n\n"
            "Waiting automatically for the next Jewel Bingo round...",
        )
        self.rec_text.config(state="disabled")

    def _render_legal_positions(self, mask: int, jewel: str) -> None:
        if self.board is None:
            return
        positions = self.board.legal_positions(jewel, mask)
        self._ranked_grid_mask = None
        self._ranked_grid_recs = []
        self._preview_grid_mask = None
        self._preview_grid_pos = None
        self._refresh_grid()
        for pos in positions:
            r0, c0 = divmod(pos, 5)
            _v, btn = self.cell_vars[r0][c0]
            if not ((mask >> pos) & 1):
                # Legal-but-unranked cells stay neutral. V5.6.2 painted FAST
                # colors and could visibly change its mind during refinement.
                btn.config(relief="solid", bd=2)
        coords = [f"L{Board.rc(p)[0]}C{Board.rc(p)[1]}" for p in positions]
        strategy_name = {"auto": "Win >1000 / Adaptive", "adaptive": "Adaptive (legacy)", "score": "Score >1000 (legacy)", "lucky": "3 Lucky (legacy)"}.get(
            self.app_cfg.risk_profile, self.app_cfg.risk_profile
        )
        if getattr(self, "compact_game_ui", False):
            self.target_var.set("Target: calculating")
            self.live_move_var.set(f"{jewel}  →  CALCULATING…")
            return
        self.rec_text.config(state="normal")
        self.rec_text.delete("1.0", "end")
        self.rec_text.insert(
            "end",
            f"Current jewel: {jewel} / {JEWEL_NAMES[jewel]}\n"
            f"Strategy: {strategy_name}\n"
            f"Legal positions now: {', '.join(coords) if coords else 'none'}\n\n"
            "Analyzing... Legal cells are outlined neutrally. Ranking colors appear only after the final REFINED decision."
        )
        self.rec_text.config(state="disabled")
        self.target_var.set("Target: calculating")
        if hasattr(self, "live_move_var"):
            self.live_move_var.set(f"{jewel}  →  CALCULATING…")
            self.live_target_var.set("Target: calculating")
            self.live_reason_var.set("Legal cells detected. Waiting for the final refined decision.")

    def _start_solver_stage(
        self, request_id: int, key: tuple, mask: int, jewel: str, stage: str, rollouts: int, attempt: int = 1
    ) -> None:
        if self.app_cfg.operation_mode != "recommend":
            return
        if request_id != self.recommend_request_id or self._recommend_state_key() != key:
            return
        if self.phase == GamePhase.COMPLETE or self.selected_mask.bit_count() >= 14:
            return
        self._cancel_solver("")
        request = self._solver_request(mask, jewel, rollouts)
        started_at = time.perf_counter()
        try:
            # Commit parent-side solver state only AFTER Windows successfully
            # starts the child. This avoids the half-started state seen when
            # multiprocessing raised "Cannot open console input buffer...".
            proc, q = start_solver_process(self._solver_ctx, request, stage, mask.bit_count())
        except Exception as exc:
            self.solver_process = None
            self.solver_queue = None
            self.solver_stage = None
            self.solver_poll_job = None
            self._auto_log(
                f"SOLVER {stage} spawn failed (attempt {attempt}/2): {type(exc).__name__}: {exc}",
                None, "ERROR",
            )
            if request_id != self.recommend_request_id or self._recommend_state_key() != key:
                self.solver_fallback_key = None
                return
            if attempt < 2:
                self.solver_fallback_key = key
                self.solver_fallback_stage = f"{stage}-spawn-retry"
                self.status.set(f"Solver process failed to start; retrying {stage} safely...")
                self.after(150, self._start_solver_stage, request_id, key, mask, jewel, stage, rollouts, attempt + 1)
                return
            self.status.set("Multiprocessing unavailable; switching to isolated subprocess fallback...")
            self._start_solver_subprocess_fallback(request_id, key, mask, jewel, stage, request, started_at)
            return

        self.solver_queue = q
        self.solver_process = proc
        self.solver_stage = stage
        self.solver_started_at = started_at
        self.solver_fallback_key = None
        self.solver_fallback_stage = None
        self.solver_poll_job = self.after(40, self._poll_solver, request_id, key, mask, jewel, stage)

    def _start_solver_subprocess_fallback(
        self, request_id: int, key: tuple, mask: int, jewel: str, stage: str, request: dict, started_at: float
    ) -> None:
        """Use a plain child Python process when multiprocessing spawn is broken."""
        try:
            proc, req_path, out_path = start_solver_subprocess(
                request, self.root_dir, stage, mask.bit_count()
            )
        except Exception as exc:
            self._auto_log(
                f"SOLVER {stage} subprocess fallback failed: {type(exc).__name__}: {exc}; using thread fallback",
                None, "ERROR",
            )
            self.status.set("Process fallback also failed; using last-resort background thread.")
            self._start_solver_thread_fallback(request_id, key, mask, jewel, stage, request, started_at)
            return
        self.solver_subprocess = proc
        self.solver_subprocess_paths = (req_path, out_path)
        self.solver_fallback_key = key
        self.solver_fallback_stage = stage
        self.solver_started_at = started_at
        self.solver_poll_job = self.after(
            50, self._poll_solver_subprocess, request_id, key, mask, jewel, stage
        )

    def _poll_solver_subprocess(
        self, request_id: int, key: tuple, mask: int, jewel: str, stage: str
    ) -> None:
        self.solver_poll_job = None
        if request_id != self.recommend_request_id or self._recommend_state_key() != key or self.phase == GamePhase.COMPLETE:
            self._cancel_solver("subprocess result became stale")
            return
        proc = self.solver_subprocess
        paths = self.solver_subprocess_paths
        if proc is None or paths is None:
            return
        req_path, out_path = paths
        if proc.poll() is None:
            self.solver_poll_job = self.after(
                50, self._poll_solver_subprocess, request_id, key, mask, jewel, stage
            )
            return

        elapsed = time.perf_counter() - self.solver_started_at
        try:
            if out_path.exists():
                kind, payload = read_solver_subprocess_result(out_path)
            else:
                kind, payload = "error", f"solver subprocess exited with code {proc.returncode} without a result"
        except Exception as exc:
            kind, payload = "error", f"could not read subprocess result: {type(exc).__name__}: {exc}"
        for path in (req_path, out_path):
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
        self.solver_subprocess = None
        self.solver_subprocess_paths = None
        self.solver_fallback_key = None
        self.solver_fallback_stage = None
        self._handle_solver_payload(
            request_id, key, mask, jewel, stage, kind, payload, elapsed, "subprocess-fallback"
        )

    def _start_solver_thread_fallback(
        self, request_id: int, key: tuple, mask: int, jewel: str, stage: str, request: dict, started_at: float
    ) -> None:
        self.solver_fallback_key = key
        self.solver_fallback_stage = stage

        def work():
            try:
                payload = solve_request(request)
                kind = "ok"
            except BaseException as exc:
                kind, payload = "error", f"{type(exc).__name__}: {exc}"
            elapsed = time.perf_counter() - started_at
            self._ui_call(
                self._handle_solver_payload,
                request_id, key, mask, jewel, stage, kind, payload, elapsed, "thread-fallback",
            )

        self.solver_fallback_thread = threading.Thread(
            target=work, name=f"solver-fallback-{stage}", daemon=True
        )
        self.solver_fallback_thread.start()

    def _recommend(self):
        """Latest-state-wins staged solver: never compute in Observable."""
        if self.app_cfg.operation_mode != "recommend":
            self._show_observe_state()
            return
        if self.board is None or self.current_jewel is None or self.selected_mask.bit_count() >= 14:
            return
        if self.phase == GamePhase.COMPLETE:
            return
        # Keep the authoritative coordinator (including the locked automatic
        # route plan) alive even when WSL is temporarily unavailable and the
        # local compatibility solver is used.
        self._live_game_state(str(self.current_jewel))
        key = self._recommend_state_key()
        if key is None:
            return

        # Repeated monitor frames for the SAME state must not restart work.
        if self.solver_process is not None and self.solver_process.is_alive() and self.recommend_active_key == key:
            return
        if self.solver_fallback_key == key or self.remote_solver_active_key == key:
            return
        if self.cached_recommend_key == key and self.last_recs:
            self._render_recommendations(int(self.selected_mask), str(self.current_jewel), self.last_recs, refined=True)
            return

        if self.solver_process is not None:
            self._cancel_solver("newer screen state won")

        self.recommend_request_id += 1
        request_id = self.recommend_request_id
        self.recommend_active_key = key
        mask = int(self.selected_mask)
        jewel = str(self.current_jewel)
        self._render_legal_positions(mask, jewel)
        self._decision_started_at = time.perf_counter()
        self._decision_started_key = key
        self._runtime_record("solver-request-ui", request_id=int(request_id),
                             selected_count=int(mask).bit_count(), jewel=jewel,
                             backend=str(self.app_cfg.solver_backend))
        self.status.set(f"ANALYZING: {jewel} — preview first, then confirmed final decision.")
        if self.app_cfg.solver_backend != "local":
            self._start_remote_solver(request_id, key, mask, jewel)
        else:
            if self.app_cfg.progressive_live_decisions:
                self._start_solver_stage(request_id, key, mask, jewel, "fast", min(self.app_cfg.preview_rollouts, self.app_cfg.live_rollouts))
            else:
                self._start_solver_stage(request_id, key, mask, jewel, "refine", self.app_cfg.live_rollouts)

    def _poll_solver(self, request_id: int, key: tuple, mask: int, jewel: str, stage: str) -> None:
        self.solver_poll_job = None
        if request_id != self.recommend_request_id or self._recommend_state_key() != key or self.phase == GamePhase.COMPLETE:
            self._cancel_solver("result became stale before completion")
            if self.running and self.current_jewel and self.phase != GamePhase.COMPLETE:
                self.after(10, self._recommend)
            return

        proc, q = self.solver_process, self.solver_queue
        if proc is None or q is None:
            return
        try:
            kind, payload = q.get_nowait()
        except queue.Empty:
            if proc.is_alive():
                self.solver_poll_job = self.after(40, self._poll_solver, request_id, key, mask, jewel, stage)
                return
            kind, payload = "error", "solver child exited without returning a result"

        elapsed = time.perf_counter() - self.solver_started_at
        try:
            proc.join(timeout=0.15)
        except Exception:
            pass
        try:
            q.cancel_join_thread(); q.close()
        except Exception:
            pass
        self.solver_process = None
        self.solver_queue = None
        self.solver_stage = None
        self._handle_solver_payload(request_id, key, mask, jewel, stage, kind, payload, elapsed, "process")

    def _handle_solver_payload(
        self, request_id: int, key: tuple, mask: int, jewel: str, stage: str,
        kind: str, payload, elapsed: float, source: str,
    ) -> None:
        if self.app_cfg.operation_mode != "recommend":
            return
        if source == "thread-fallback" and self.solver_fallback_key == key:
            self.solver_fallback_key = None
            self.solver_fallback_stage = None

        # COMPLETE is an absolute barrier: no late FAST/refined callback may
        # resurrect an old BEST MOVE after the round has ended.
        if (
            request_id != self.recommend_request_id
            or self._recommend_state_key() != key
            or self.phase == GamePhase.COMPLETE
            or self.selected_mask.bit_count() >= 14
        ):
            self._auto_log(f"SOLVER discarded stale {stage} result after {elapsed:.2f}s ({source})", None, "INFO")
            return

        if kind != "ok":
            self._auto_log(f"SOLVER {stage} error after {elapsed:.2f}s ({source}): {payload}", None, "ERROR")
            self.status.set(f"Recommendation error: {payload}. The monitor remains active.")
            return

        recs = list(payload)
        self._auto_log(f"SOLVER {stage} ready in {elapsed:.2f}s for {jewel} ({source})", None, "INFO")
        self._runtime_record("solver-local-stage-result", stage=stage, backend=source,
                             elapsed_ms=round(elapsed * 1000, 2), jewel=jewel)

        future_after_move = 14 - (mask.bit_count() + 1)
        exact_now = future_after_move <= self.app_cfg.exact_horizon
        immediate = bool(recs and recs[0].method in {"immediate-lucky-fastpath", "reward-tier-exact"})
        if stage == "fast" and not immediate and not exact_now and self.app_cfg.progressive_live_decisions and self.app_cfg.live_rollouts > self.app_cfg.preview_rollouts:
            # Present a clearly provisional recommendation.
            self._render_recommendations(mask, jewel, recs, refined=False)
            self.after(25, self._start_solver_stage, request_id, key, mask, jewel, "refine", self.app_cfg.live_rollouts)
            return

        # Only a final/exact decision becomes the authoritative recommendation
        # used by the UI and by action logging. Provisional FAST results are not
        # allowed to leak into recorded recommendations.
        self.last_recs = recs
        self.cached_recommend_key = key
        if not immediate:
            try:
                self._persist_counterfactuals(mask, jewel, recs)
            except Exception as exc:
                self._auto_log(f"Could not persist solver samples: {exc}", None, "WARN")
        self._adopt_auto_plan(recs)
        self._render_recommendations(mask, jewel, recs, refined=True)

    def _monitor_sleep_seconds(self, transition_hint: bool = False) -> float:
        fast = transition_hint or time.monotonic() < self._transition_fast_until
        if fast:
            return max(0.05, self.app_cfg.transition_poll_ms / 1000.0)
        return max(0.15, self.app_cfg.poll_ms / 1000.0)

    @staticmethod
    def _format_needs(needs) -> str:
        return " + ".join(f"{j}×{n}" for j, n in needs) if needs else "nothing"

    def _candidate_line_details(self, mask: int, position: int) -> list[str]:
        if self.board is None:
            return []
        after = mask | (1 << position)
        feasibility = {(x.kind, x.name): x for x in line_feasibilities(self.board, after)}
        rows = []
        for impact in move_line_impacts(mask, position):
            f = feasibility[(impact.kind, impact.name)]
            rows.append(
                f"{impact.kind} {impact.name}: {impact.after}/{impact.size} | "
                f"receive-feasibility {f.probability*100:.1f}% | needs {self._format_needs(f.needs)}"
            )
        return rows

    @staticmethod
    def _plain_plan_health(probability: float) -> str:
        if probability >= 0.15:
            return "STRONG"
        if probability >= 0.07:
            return "HEALTHY"
        if probability >= 0.03:
            return "RISKY"
        if probability > 0:
            return "CRITICAL"
        return "IMPOSSIBLE"

    @staticmethod
    def _plain_confidence(rec: Recommendation) -> str:
        if rec.tie_kind == "exact":
            return "EQUAL OPTIONS"
        if rec.tie_kind in {"statistical", "target"}:
            return "CLOSE CALL"
        if rec.confidence == "LOW_SAMPLE":
            return "LOW SAMPLE"
        return "HIGH"

    def _update_live_summary(self, mask: int, jewel: str, recs: list[Recommendation]) -> None:
        if not recs or not hasattr(self, "live_move_var"):
            return
        top = recs[0]
        rr, cc = Board.rc(top.position)
        self.live_move_var.set(f"{jewel}  →  L{rr}C{cc}")
        target = top.plan_target or top.active_goal or "best score"
        parts = [*(f"Lucky {x}" for x in top.plan_lucky_lines), *(f"Normal {x}" for x in top.plan_normal_lines)]
        route_name = " + ".join(parts) if parts else "best available score"
        if top.method == "reward-tier-exact":
            if top.reward_guarantee == ">1000 guaranteed":
                self.live_target_var.set("BOX >1000 GUARANTEED")
            elif top.reward_guarantee == "900+ guaranteed":
                self.live_target_var.set("BOX 900+ GUARANTEED")
            else:
                self.live_target_var.set(f"Best box chance: >=900 {top.reward_p_ge900*100:.1f}%")
            self.live_route_var.set(
                f"Worst case with the remaining draws: {top.reward_min_final} points"
            )
            self.live_health_var.set("Target health: SECURED" if top.reward_guarantee != "not guaranteed" else "Target health: AT RISK")
        elif target == "expected score":
            self.live_target_var.set("Target: Best achievable score")
            self.live_route_var.set("Route: no structured route remains")
            self.live_health_var.set("Target health: SALVAGE")
        else:
            if target == ">=999" and top.plan_target == ">=999":
                # Hypergeometric probability applies to this ONE concrete
                # route's jewel requirements, not the chance of any >=999 win.
                self.live_target_var.set("Goal: >=999  ·  proven-scoring geometry")
                self.live_route_var.set(
                    f"Current route: {route_name}  ·  stone availability {top.plan_probability*100:.1f}%"
                )
            else:
                self.live_target_var.set(f"Goal: {target}  ·  possible {top.target_probability*100:.1f}%")
                self.live_route_var.set(f"Current route: {route_name}  ·  possible {top.plan_probability*100:.1f}%")
            self.live_health_var.set(f"Target health: {self._plain_plan_health(top.target_probability)}")
        if top.method == "reward-tier-exact":
            self.live_slack_var.set("Reward tier priority: >1000 secured > 900+ secured > speculative score")
        elif target == ">=999" and top.plan_target == ">=999":
            self.live_slack_var.set(
                f"999 route: {top.route_off_route} off-route / "
                f"{top.route_slack_remaining} spare · floor {top.route_score_floor}"
            )
        elif top.lucky_slack is not None and top.lucky_slack.viable:
            self.live_slack_var.set(f"Slack: {top.lucky_slack.slack_remaining}/2")
        elif target == "3 Lucky":
            self.live_slack_var.set("Slack: none")
        else:
            self.live_slack_var.set("Slack: --")
        self.live_confidence_var.set(f"Confidence: {self._plain_confidence(top)}")
        if top.method == "reward-tier-exact":
            reason = (
                f"Guaranteed box: at least {top.reward_min_final} final points. "
                "Avoid risking 900+ for only a chance of 999 (same reward category)."
                if top.reward_guarantee != "not guaranteed"
                else f"No box is guaranteed; maximum exact chance of >=900 is {top.reward_p_ge900*100:.1f}%."
            )
        elif top.method == "immediate-lucky-fastpath":
            reason = "EXACT LUCKY: closes a line now without Monte Carlo; the active route was checked."
        elif top.route_switched:
            reason = "ROUTE CHANGED: the previous route no longer survived this jewel; the target is still viable."
        elif top.forced_slack:
            reason = "FORCED SLACK: this jewel has no free cell in the locked route; best off-route move without sacrificing the target."
        elif top.lucky_clears_gained:
            reason = "CLOSES LUCKY NOW: same plan chance, so banking the Lucky line gets priority."
        elif top.normal_clears_gained:
            reason = "CLOSES NORMAL NOW: same plan chance, so banking the completed line gets priority."
        elif top.plan_on_route:
            reason = "Advances the locked route."
        elif top.tie_kind:
            reason = "Several options preserve the target; secondary structure selected this move."
        else:
            reason = "Best move for the current automatic target."
        self.live_reason_var.set(reason)
        if top.method == "immediate-lucky-fastpath" and top.utility.p_gt1000 < 1.0:
            self.live_gt1000_var.set("Win >1000: -- (no simulation)")
        else:
            self.live_gt1000_var.set(f"Win >1000: {top.utility.p_gt1000*100:.1f}%")
        if top.method == "reward-tier-exact":
            self.live_target_chance_var.set(f"900+ reward: {min(100., top.reward_p_ge900*100):.1f}% exact")
            self.live_route_chance_var.set(f"Minimum final score: {top.reward_min_final} (exact worst case)")
        elif target == "expected score":
            self.live_target_chance_var.set("Goal possible: --")
            self.live_route_chance_var.set("Route possible: --")
        elif target == ">=999" and top.plan_target == ">=999":
            self.live_target_chance_var.set("Goal: >=999 / route verified by scoring rules")
            self.live_route_chance_var.set(f"Route stones available: {top.plan_probability*100:.1f}%")
        else:
            self.live_target_chance_var.set(f"Goal possible: {top.target_probability*100:.1f}%")
            self.live_route_chance_var.set(f"Route possible: {top.plan_probability*100:.1f}%")
        if top.method == "immediate-lucky-fastpath" and top.utility.p3 < 1.0:
            self.live_p3_var.set("3 Lucky: -- (no simulation)")
        else:
            self.live_p3_var.set(f"3 Lucky: {top.utility.p3*100:.1f}%")
        self.live_expected_var.set(f"Expected pts: {top.utility.expected_score:.0f}")

        previous = self._ui_last_plan_target
        if top.method == "reward-tier-exact":
            self.target_change_var.set(
                "REWARD FIRST: guarantee the available box; higher points in the same tier are not worth the gamble."
                if top.reward_guarantee != "not guaranteed"
                else "REWARD FIRST: exact endgame ranks the chance of reaching a reward box."
            )
        elif previous and previous != target:
            self.target_change_var.set(
                f"TARGET CHANGED  {previous}  →  {target}   |   current jewel made the previous target too weak/impossible"
            )
        elif top.route_switched:
            self.target_change_var.set(
                f"ROUTE CHANGED inside {target}   |   previous locked route became impossible or materially worse"
            )
        elif not previous:
            self.target_change_var.set("")
        else:
            self.target_change_var.set("")
        self._ui_last_plan_target = target

        if hasattr(self, "live_rank_tree"):
            for iid in self.live_rank_tree.get_children():
                self.live_rank_tree.delete(iid)
            for rank, rec in enumerate(recs[:4], 1):
                r, c = Board.rc(rec.position)
                status = self._move_visual_status(top, rec, rank - 1)
                status_label, _bg, status_tag = self._visual_status_style(status)
                self.live_rank_tree.insert("", "end", values=(
                    f"#{rank} L{r}C{c}", status_label, f"{rec.target_probability*100:.1f}%",
                    f"{rec.plan_probability*100:.1f}%", f"{rec.utility.p_gt1000*100:.1f}%",
                    f"{rec.utility.p3*100:.1f}%",
                ), tags=(status_tag,))

    def _render_recommendations(self, mask: int, jewel: str, recs: list[Recommendation], refined: bool = True) -> None:
        """Render a decision-first ranking with probabilities, ties and line impact."""
        # Keep the authoritative decision separate from provisional MC preview.
        # _record_action() consumes last_recs: never feed them FAST.
        if refined:
            self.last_recs = list(recs)
        s = score_mask(mask)

        # OBSERVE is a strict no-advice mode. Do not leak recommendations through
        # board colors, ranking rows, probabilities or stale prior decisions.
        if self.app_cfg.operation_mode == "observe":
            self._show_observe_state()
            return

        # Only final results paint ranked cells. Preliminary directions remain
        # explicitly labelled PREVIEW and cannot commit an automatic click.
        if refined:
            self._preview_grid_mask = None
            self._preview_grid_pos = None
            self._ranked_grid_mask = int(mask)
            self._ranked_grid_recs = list(recs)
        elif recs:
            self._preview_grid_mask = int(mask)
            self._preview_grid_pos = int(recs[0].position)
        self._refresh_grid()
        if refined:
            self.update_idletasks()

        if getattr(self, "compact_game_ui", False):
            self._show_compact_decision(mask, jewel, recs, refined)
            return

        strategy_name = {"auto": "Win >1000 / Adaptive", "adaptive": "Adaptive (legacy)", "score": "Score >1000 (legacy)", "lucky": "3 Lucky (legacy)"}.get(
            self.app_cfg.risk_profile, self.app_cfg.risk_profile
        )
        active_target = recs[0].active_goal if recs else "waiting"
        stage_name = "REFINED" if refined else "FAST / PROVISIONAL"

        self.rec_text.config(state="normal")
        self.rec_text.delete("1.0", "end")

        def put(text: str, tag: str | None = None):
            if tag:
                self.rec_text.insert("end", text, tag)
            else:
                self.rec_text.insert("end", text)

        put(f"{stage_name} DECISION\n", "title")
        put("COLOR GUIDE  GREEN=best/equal | YELLOW=close | ORANGE=less flex | AMBER=forced slack | PURPLE=fallback | BLUE=already placed\n", "muted")
        put(
            f"CURRENT  {jewel} / {JEWEL_NAMES[jewel]} | {mask.bit_count()}/14 placed | "
            f"{s.lucky} Lucky + {s.normal} Normal | {s.total} pts\n"
        )
        put(f"STRATEGY {strategy_name}  ->  TARGET {active_target}\n")
        if recs and recs[0].plan_target:
            _plan = recs[0]
            _route_parts = [*(f"Lucky {x}" for x in _plan.plan_lucky_lines), *(f"Normal {x}" for x in _plan.plan_normal_lines)]
            _route_text = " + ".join(_route_parts) if _route_parts else "best score"
            put(f"ACTIVE PLAN  {_plan.plan_target} | LOCKED ROUTE  {_route_text}\n")
            put(
                f"TARGET CHANCE  {_plan.target_probability*100:.1f}% | ROUTE CHANCE  {_plan.plan_probability*100:.1f}% | "
                f"{_plan.plan_reason}\n\n", "tie"
            )
        else:
            put("\n")

        if recs:
            top = recs[0]
            rr, cc = Board.rc(top.position)
            put(f"BEST MOVE  {jewel} -> L{rr}C{cc}\n", "best")
            tied_coords = [f"L{Board.rc(p)[0]}C{Board.rc(p)[1]}" for p in top.tied_with]
            if top.tie_kind == "exact":
                put(f"CONFIDENCE EXACT TIE with {', '.join(tied_coords)} — mathematically identical ranking.\n", "tie")
            elif top.tie_kind == "target":
                put(f"CONFIDENCE TARGET TIE with {', '.join(tied_coords)} — same primary objective; secondary tie-break selected #1.\n", "tie")
            elif top.tie_kind == "statistical":
                put(f"CONFIDENCE STATISTICAL TIE with {', '.join(tied_coords)} — 95% paired uncertainty overlaps.\n", "tie")
            elif not refined:
                put("CONFIDENCE PROVISIONAL — wait for REFINED before treating #1 as a clear winner.\n", "tie")
            elif top.confidence == "LOW_SAMPLE":
                put(
                    "CONFIDENCE LOW-SAMPLE LEAD — current Monte Carlo ranks #1 first, but the live sample "
                    "is still small; exact structural tie-breaks are shown below.\n",
                    "tie",
                )
            else:
                put("CONFIDENCE CLEAR LEAD under the current solver target.\n")

            u = top.utility
            put(
                f"KEY PROBABILITIES / POLICY OUTCOMES  >1000 {u.p_gt1000*100:5.1f}% | >=999 {u.p_ge999*100:5.1f}% | "
                f"3 Lucky|policy {u.p3*100:5.1f}% | E[score] {u.expected_score:6.1f}\n"
            )
            if active_target == ">1000":
                put(
                    f"STRUCTURAL SCORE TIE-BREAK  >1000-route potential {top.score_route_potential:.3f} | "
                    f"Normal-line leverage {top.normal_leverage:.3f}\n"
                )
            route = route_by_name(top.routes, active_target)
            if route is not None:
                line_names = ", ".join(route.lines) if route.lines else "already satisfied"
                put(
                    f"ROUTE FEASIBILITY  {active_target}: {route.probability*100:.1f}% exact inventory feasibility\n"
                    f"BEST ROUTE  {line_names} | needs {self._format_needs(route.needs)}\n"
                )
            if top.primary_stat_tie and top.tie_break_metric:
                margin = top.tie_break_margin
                if top.tie_break_metric == "E[score]":
                    unit = " pts"; shown = margin
                elif top.tie_break_metric == "3L slack remaining":
                    unit = " placement(s)"; shown = margin
                elif top.tie_break_metric in {">1000 structural route potential", "Normal-line leverage"}:
                    unit = " structural"; shown = margin
                else:
                    unit = " pp"; shown = margin * 100.0
                put(
                    f"HIERARCHICAL TIE-BREAK  primary criterion overlaps; {top.tie_break_metric} separates #1 "
                    f"by {shown:+.1f}{unit}.\n", "tie"
                )
            put(f"WHY THIS CELL  {compact_move_line_summary(mask, top.position)}\n")
            if top.lucky_slack is not None:
                slack = top.lucky_slack
                trio = " + ".join(slack.lucky_lines)
                status = "VIABLE" if slack.viable else "IMPOSSIBLE"
                put(
                    f"3-LUCKY FLEXIBILITY  trio {trio} | slack used {slack.slack_used}/2 | "
                    f"remaining {slack.slack_remaining} | {status} | structural feasibility {slack.probability*100:.1f}%\n"
                )
                if slack.viable:
                    put(f"  Trio needs {self._format_needs(slack.needs)}\n")
            for detail in self._candidate_line_details(mask, top.position):
                put(f"  {detail}\n")
            if top.routes:
                put("STRUCTURAL ROUTES (exact remaining-inventory feasibility)\n", "title")
                shown_targets = (
                    ">=999", "3 Lucky + 1 Normal", "3 Lucky", "2 Lucky + 2 Normal",
                    "2 Lucky + 1 Normal", "1 Lucky + 2 Normal", "1 Lucky + 1 Normal",
                )
                for target_name in shown_targets:
                    route_item = route_by_name(top.routes, target_name)
                    if route_item is None:
                        continue
                    lines = ", ".join(route_item.lines) if route_item.lines else "already satisfied"
                    put(
                        f"  {target_name:<20} {route_item.probability*100:5.1f}% | "
                        f"{lines} | needs {self._format_needs(route_item.needs)}\n"
                    )
            put("\n")

        if self.app_cfg.risk_profile in {"auto", "adaptive"} and recs:
            best_p3_route = max(route_probability(r.routes, "3 Lucky") for r in recs)
            best_gt = max(r.utility.p_gt1000 for r in recs)
            p3_viability = viability_label(best_p3_route, self.app_cfg.p3_min, 0.25)
            score_viability = viability_label(best_gt, self.app_cfg.adaptive_score_min, 0.50)
            put(
                f"AUTO PLANNER STATUS  best 3-Lucky route {p3_viability} {best_p3_route*100:.1f}% | "
                f">1000 under policy {score_viability} {best_gt*100:.1f}%\n", "title"
            )
            put(f"Decision: target the strongest remaining concrete route -> {active_target}.\n\n")

        put("RANKED OPTIONS\n", "title")
        route_column_target = active_target if route_by_name(recs[0].routes, active_target) is not None else "3 Lucky" if recs else "3 Lucky"
        route_column_label = (route_column_target[:10] + " route") if route_column_target != "3 Lucky" else "3L route"
        put(f"Color     Move    {route_column_label:<14} >1000   >=999   3L|policy  Slack  Avg pts    Line impact\n", "muted")
        put("-" * 100 + "\n", "muted")
        for rank, rec in enumerate(recs, 1):
            r, c = Board.rc(rec.position)
            u = rec.utility
            status = self._move_visual_status(recs[0], rec, rank - 1)
            status_label, _bg, status_tag = self._visual_status_style(status)
            tied_to_top = rec.position == recs[0].position or (recs[0].tie_kind == "exact" and rec.position in set(recs[0].tied_with))
            label = "TIE#1" if tied_to_top and rec.position != recs[0].position else f"#{rank}"
            prefix = f"{status_label:<12} {label:<5} L{r}C{c:<2} "
            put(prefix, status_tag)
            route_p = route_probability(rec.routes, route_column_target) if rec.routes else 0.0
            slack_text = "-" if rec.lucky_slack is None or not rec.lucky_slack.viable else str(rec.lucky_slack.slack_remaining)
            put(
                f"{route_p*100:6.1f}%  {u.p_gt1000*100:6.1f}%  {u.p_ge999*100:6.1f}%  "
                f"{u.p3*100:8.1f}%  {slack_text:^5}  {u.expected_score:8.1f}   {compact_move_line_summary(mask, rec.position)}\n"
            )
            if rec.samples:
                ci_goal = 1.96 * rec.stderr_goal
                if active_target == "expected score":
                    ci_text = f"±{ci_goal:.1f} score"
                else:
                    ci_text = f"±{ci_goal*100:.1f} pp on {active_target}"
                put(
                    f"              MC {rec.samples}/action | 95% {ci_text} | "
                    f"2L+1N {u.p2l1n*100:.1f}% | 1L+2N {u.p1l2n*100:.1f}% | 1L+1N {u.p1l1n*100:.1f}%\n",
                    "muted",
                )

        self.target_var.set(f"Target: {active_target}")
        self._update_live_summary(mask, jewel, recs)
        self.rec_text.config(state="disabled")

        if recs:
            top = recs[0]
            rr, cc = Board.rc(top.position)
            if not refined:
                self.status.set(f"PREVIEW: {jewel} -> L{rr}C{cc}; refining final decision...")
            elif top.tie_kind:
                ties = "/".join(f"L{Board.rc(p)[0]}C{Board.rc(p)[1]}" for p in top.tied_with)
                prefix = "EXACT TIE" if top.tie_kind == "exact" else ("STAT TIE" if top.tie_kind == "statistical" else "TARGET TIE")
                self.status.set(f"{prefix}: {jewel} L{rr}C{cc} / {ties} | target {top.active_goal}")
            else:
                self.status.set(f"BEST: {jewel} -> L{rr}C{cc} | target {top.active_goal}")
            if refined and self.app_cfg.overlay_enabled and self.region and self.vision_cfg.board_roi:
                self.overlay.show_cell(self.region, self.vision_cfg.board_roi, rr, cc, f"{jewel} -> {rr},{cc}")
            else:
                self.overlay.hide()

    def _persist_box_color_async(self):
        if self.board is None:
            return
        board = self.board
        color = self.box_color
        episode_id = self.episode_id
        session_id = self.session_id
        operation_mode = self.app_cfg.operation_mode
        risk_profile = self.app_cfg.risk_profile

        def work():
            try:
                self.db.ensure_game(
                    episode_id, "local-auto", session_id, board, color,
                    operation_mode, risk_profile,
                )
                self._ui_call(self._auto_log, f"Box label saved: {color or 'not set'}", None, "INFO")
            except Exception as exc:
                self._ui_call(self._auto_log, f"Could not save box label yet: {exc}", None, "WARN")

        threading.Thread(target=work, name="box-label-save", daemon=True).start()

    def _on_box_color_change(self, _event=None):
        raw = self.box_color_var.get().strip().lower()
        # Only the three Jewel Bingo box colors are valid research labels.
        self.box_color = raw if raw in {"green", "blue", "red"} else None
        self.status.set(f"Box label: {self.box_color or 'not set'} (optional research metadata)")
        self._persist_box_color_async()

    def _clear_box_color(self):
        self.box_color_var.set("")
        self.box_color = None
        self.status.set("Box label cleared (optional research metadata)")
        self._persist_box_color_async()

    def correct_current(self):
        if self.last_current_patch is None:
            messagebox.showinfo("Correction", "Read/capture the game first")
            return
        label = simpledialog.askstring("Correct jewel", "Enter BL, SO, LI, CR, HA or CH", parent=self)
        if label:
            label = label.strip().upper()
            if label not in JEWELS:
                messagebox.showerror("Correction", "Invalid jewel label")
                return
            self.icon_matcher.add_template(label, self.last_current_patch)
            with self._live_state_lock:
                self.current_jewel = label
                self.current_confidence = 1.0
                self.current_margin = 1.0
                self.current_entropy = 0.0
            # A manual label is explicit ground truth for the current frame.
            self.temporal.accepted_jewel = label
            self.current_label.set(f"Current jewel corrected: {label} / {JEWEL_NAMES[label]}")
            self._recommend()

    def _waiting_diagnostic_snapshot(self) -> dict:
        now = time.monotonic()
        with self._live_state_lock:
            live = {
                "episode_id": self.episode_id,
                "mask": int(self.selected_mask),
                "mask_count": int(self.selected_mask).bit_count(),
                "current_jewel": self.current_jewel,
                "current_confidence": float(self.current_confidence),
                "current_margin": float(self.current_margin),
                "current_entropy": float(self.current_entropy),
                "board_confidence": float(self.board_confidence),
                "completion_pending": bool(self.completion_pending),
                "episode_finalized": bool(self.episode_finalized),
            }
        bridge = self.solver_bridge.admission.snapshot()
        return {
            **live,
            "session_id": self.session_id,
            "running": bool(self.running),
            "monitor_thread_alive": bool(self.worker and self.worker.is_alive()),
            "monitor_heartbeat": float(self._diag_monitor_heartbeat),
            "monitor_heartbeat_age_ms": None if not self._diag_monitor_heartbeat else int((now - self._diag_monitor_heartbeat) * 1000),
            "last_transition_age_ms": None if not self._diag_last_transition else int((now - self._diag_last_transition) * 1000),
            "last_confirm_age_ms": None if not self._diag_last_confirm else int((now - self._diag_last_confirm) * 1000),
            "last_stable_reason": self._diag_last_stable_reason,
            "bridge_admission": bridge,
            "solver_bridge_available": bool(self.solver_bridge.available),
            "solver_bridge_status": str(self.solver_bridge.status),
            "solver_exclusive_token_held": bool(getattr(self.solver_bridge, "_exclusive_token", None)),
            "solver_backend": str(self.app_cfg.solver_backend),
            "solver_host": str(self.app_cfg.solver_host),
            "solver_port": int(self.app_cfg.solver_port),
            "solver_wsl_distro": str(self.app_cfg.solver_wsl_distro),
        }

    def _set_live_compact_mode(self, active: bool) -> None:
        """Keep setup progress out of the way while monitoring the game."""
        try:
            if active:
                self.setup_progress_row.pack_forget()
            elif not self.setup_progress_row.winfo_manager():
                self.setup_progress_row.pack(fill="x", before=self.controls_frame)
        except tk.TclError:
            pass

    def toggle(self):
        if not self.running:
            if self.maintenance_thread is not None and self.maintenance_thread.is_alive():
                self.status.set("START waiting: background learning/backup is still finishing. Try again in a moment.")
                self._runtime_record("monitor-start-blocked", reason="maintenance-active")
                return
            if not self._calibration_ready():
                missing = self._calibration_missing()
                messagebox.showwarning(
                    "Setup incomplete",
                    "Automatic setup is not complete yet. Missing:\n\n• " + "\n• ".join(missing) +
                    "\n\nKeep Jewel Bingo visible and click AUTO SETUP.",
                    parent=self,
                )
                self._refresh_setup_state()
                return
            if self.app_cfg.auto_relocate:
                self.find_moved_area(quiet=True)
            # V5.3 does not abort START because of a single low-confidence board
            # frame. Geometry and x4 references were already validated by AUTO
            # SETUP; live board recognition now stabilizes across several frames
            # in the monitor thread. This avoids the contradictory V5.2 flow
            # where AUTO SETUP succeeded and START immediately told the user to
            # run manual setup again because one classifier frame was weak.
            self.vision_test_passed = False
            self.board = None
            self.board_consensus.reset()
            self._live_board_failures = 0
            self._live_board_last_log = 0.0
            self.running = True
            self._set_live_compact_mode(True)
            self.start_btn.config(text="STOP", state="normal")
            self.worker = threading.Thread(target=self._loop, name="bingo-monitor", daemon=True)
            self.worker.start()
            self._runtime_record(
                "monitor-start", episode_id=self.episode_id, selected_count=int(self.selected_mask).bit_count(),
                board_confidence=float(self.board_confidence),
            )
            self._setup_feedback(
                "START active. Stabilizing the live 5x5 board automatically; no manual board/current-jewel step is required.",
                "ok",
            )
            self.status.set("START active — validating the live board across multiple frames. Keep Jewel Bingo visible for a moment.")
        else:
            self.running = False
            self._runtime_record(
                "monitor-stop", episode_id=self.episode_id, selected_count=int(self.selected_mask).bit_count(),
                pending_maintenance=int(self._maintenance_pending_count),
            )
            self._set_live_compact_mode(False)
            self.start_btn.config(text="START", state="normal")
            self.overlay.hide()
            self.status.set("Monitoring stopped. Setup and saved learning were preserved. Deferred learning/backup can run now.")
            self._refresh_setup_state()
            if self._maintenance_pending_episode_id:
                self.after(2000, self._flush_deferred_maintenance)

    def _save_snapshot(self, panel, suffix: str) -> str | None:
        if not self.app_cfg.save_game_images:
            return None
        path = self.image_dir / f"{self.episode_id}_{suffix}.png"
        cv2.imwrite(str(path), panel)
        return str(path)

    def _recommendation_payload(self, recs: list[Recommendation]) -> dict:
        out = []
        for r in recs:
            u = r.utility
            out.append({
                "position": r.position, "active_goal": r.active_goal, "method": r.method,
                "p3": u.p3, "p2l1n": u.p2l1n, "p1l2n": u.p1l2n, "p1l1n": u.p1l1n,
                "p_gt1000": u.p_gt1000, "p_ge999": u.p_ge999,
                "expected_score": u.expected_score, "learned_score": r.learned_score,
                "samples": r.samples, "stderr_p3": r.stderr_p3,
                "stderr_gt1000": r.stderr_gt1000, "stderr_score": r.stderr_score,
                "solver_regret": r.solver_regret,
                "plan_target": r.plan_target,
                "plan_lucky_lines": list(r.plan_lucky_lines),
                "plan_normal_lines": list(r.plan_normal_lines),
                "plan_probability": r.plan_probability,
                "plan_on_route": r.plan_on_route,
                "plan_reason": r.plan_reason,
                "lucky_slack": None if r.lucky_slack is None else {
                    "lucky_lines": list(r.lucky_slack.lucky_lines),
                    "slack_used": r.lucky_slack.slack_used,
                    "slack_remaining": r.lucky_slack.slack_remaining,
                    "viable": r.lucky_slack.viable,
                    "probability": r.lucky_slack.probability,
                    "needs": list(r.lucky_slack.needs),
                },
            })
        return {"candidates": out}

    def _record_action(
        self, mask_before: int, mask_after: int, predicted_jewel: str | None, current_patch,
        current_conf: float, current_margin: float = 0.0, current_entropy: float = 1.0, panel=None,
    ):
        delta = mask_after & ~mask_before
        if delta.bit_count() != 1 or self.board is None:
            return
        pos = delta.bit_length() - 1
        actual_jewel = self.board.cells[pos]
        if actual_jewel not in JEWELS:
            return

        # The accepted destination is ground truth in TWO visual domains:
        # the top icon teaches the current-icon matcher, while the destination
        # cell teaches the board-cell classifier. Keeping them separate avoids
        # the V5.1 domain-mismatch failure.
        if current_patch is not None:
            try:
                self.icon_matcher.add_template_if_novel(actual_jewel, current_patch)
            except Exception:
                pass
        if panel is not None:
            try:
                destination_patch = board_patch(panel, self.vision_cfg, pos)
                self.clf.add_template_if_novel(actual_jewel, destination_patch, min_distance=0.18)
            except Exception:
                pass

        if predicted_jewel != actual_jewel:
            self.db.log_event(
                self.episode_id, "vision-correction",
                f"predicted {predicted_jewel}, placement proved {actual_jewel}",
                self.phase.value,
                {"predicted": predicted_jewel, "actual": actual_jewel, "position": pos},
            )
            if self.app_cfg.operation_mode == "recommend":
                try:
                    recs = self._compute_recommendations(mask_before, actual_jewel, True)
                except Exception:
                    recs = []
            else:
                recs = []
        else:
            recs = list(self.last_recs) if self.app_cfg.operation_mode == "recommend" else []

        chosen_rec = next((r for r in recs if r.position == pos), None)
        recommended_pos = recs[0].position if recs else None
        regret = chosen_rec.solver_regret if chosen_rec is not None else None

        self.episode_actions.append({"mask_before": mask_before, "jewel": actual_jewel, "action_pos": pos})
        self.game_sequence.append(actual_jewel)
        self.db.save_action(
            self.episode_id, mask_before.bit_count(), mask_before, actual_jewel, pos,
            self._recommendation_payload(recs), self.last_decision_image, current_conf, self.board_confidence,
            current_margin=current_margin, current_entropy=current_entropy, phase=self.phase.value,
            strategy=self.app_cfg.risk_profile, recommended_pos=recommended_pos, solver_regret=regret,
        )

    @staticmethod
    def _score_dict(s) -> dict:
        return {
            "lucky": s.lucky,
            "normal": s.normal,
            "jewel_cells": s.jewel_cells,
            "total": s.total,
            "lucky_names": list(s.lucky_names),
            "normal_names": list(s.normal_names),
        }

    @staticmethod
    def _verified_to_score(v: VerifiedResult) -> dict:
        lucky = (v.lucky_score // 312) if v.lucky_score is not None and v.lucky_score % 312 == 0 else None
        normal = (v.normal_score // 240) if v.normal_score is not None and v.normal_score % 240 == 0 else None
        jewel_cells = (v.jewel_score // 45) if v.jewel_score is not None and v.jewel_score % 45 == 0 else None
        return {
            "lucky": lucky,
            "normal": normal,
            "jewel_cells": jewel_cells,
            "total": v.total_score,
            "raw_lucky_score": v.lucky_score,
            "raw_normal_score": v.normal_score,
            "raw_jewel_score": v.jewel_score,
            "source": v.source,
        }

    def _rebuild_learning_history_async(self) -> None:
        """Rebuild ML from the persistent DB, including retrospective rewards.

        Training is a full deterministic rebuild from stored games, not an
        incremental "reward click". Therefore opening/replaying the same game
        repeatedly can never duplicate its influence. Heavy rebuilds are never
        started while live monitoring is active.
        """
        if self.running:
            if hasattr(self, "learning_hindsight_var"):
                self.learning_hindsight_var.set("Hindsight reward model: STOP monitoring before rebuilding learning.")
            self._runtime_record("learning-rebuild-deferred", reason="live-monitoring")
            return
        if self.maintenance_thread is not None and self.maintenance_thread.is_alive():
            if hasattr(self, "learning_hindsight_var"):
                self.learning_hindsight_var.set("Hindsight reward model: training already in progress...")
            return
        db_path = self.data_dir / "bingo.sqlite3"
        model_path = self.data_dir / "policy_model_v4.joblib"
        episodes_path = self.data_dir / "episodes.json"
        holdout = self.app_cfg.ml_holdout_fraction
        if hasattr(self, "learning_hindsight_var"):
            self.learning_hindsight_var.set("Hindsight reward model: REBUILDING from stored games...")

        def apply_result(info):
            try:
                self.experience.reload()
                self._maintenance_reload_pending = False
                self._refresh_learning_panel()
                self._auto_log(
                    f"LEARNING rebuild complete: CF={info.counterfactual_samples}, "
                    f"outcomes={info.outcome_samples}, hindsight={info.hindsight_samples} "
                    f"from {info.hindsight_games} games",
                    None, "INFO",
                )
            except Exception as exc:
                if hasattr(self, "learning_hindsight_var"):
                    self.learning_hindsight_var.set(f"Hindsight reward model: reload error {exc}")

        def work():
            try:
                local_exp = ExperiencePolicy(db_path, model_path, episodes_path)
                info = local_exp.train(
                    holdout_fraction=holdout,
                    min_validation_games=self.app_cfg.ml_min_validation_games,
                )
                self._ui_call(apply_result, info)
            except Exception as exc:
                self._ui_call(
                    self.learning_hindsight_var.set if hasattr(self, "learning_hindsight_var") else self.status.set,
                    f"Hindsight reward model: rebuild error {exc}",
                )

        self.maintenance_thread = threading.Thread(
            target=work, name="history-learning-rebuild", daemon=True
        )
        self.maintenance_thread.start()

    def _audit_finished_episode_async(self, game_id: str) -> None:
        """Idempotent offline replay on a separate worker, after finalization."""
        def work():
            try:
                from .winning_memory import validated_episodes
                from .retrospective_learning import cached_audit
                local_db = BingoDB(self.data_dir / "bingo.sqlite3")
                episode = next((e for e in validated_episodes(local_db) if e.game_id == game_id), None)
                if episode is None:
                    local_db.log_event(game_id, "hindsight-rejected", "incomplete or inconsistent history")
                    return
                report = cached_audit(local_db, episode)
                if not report or not report.get("valid"):
                    return
                local_db.log_event(game_id, "hindsight-audited", "complete offline replay",
                                   payload={"actual": report["observed_total"],
                                            "possible_best": report["best_total"],
                                            "source": "counterfactual-not-real-win"})
                self._ui_call(self._refresh_learning_panel)
            except Exception as exc:
                self._runtime_record("hindsight-audit-error", level="WARN", error=str(exc), episode_id=game_id)
        threading.Thread(target=work, name="finished-game-hindsight", daemon=True).start()

    def _post_game_maintenance_async(self, completed_episode_id: str, *, already_queued: bool = False) -> None:
        """Train/backup without competing with live vision/input.

        Full-history training gets more expensive as games accumulate. Running it
        after every completed round caused a growing CPU/memory burst that could
        overlap the next round. Live monitoring now only queues one maintenance
        request; STOP flushes the newest request and one rebuild incorporates all
        games collected in the meantime.
        """
        self._maintenance_pending_episode_id = str(completed_episode_id)
        if not already_queued:
            self._maintenance_pending_count += 1

        if self.running and not bool(self.app_cfg.maintenance_during_live):
            self.db.log_event(
                completed_episode_id, "maintenance-queued",
                "ML/backup deferred until live monitoring stops", self.phase.value,
                {"pending_games": int(self._maintenance_pending_count)},
            )
            self._runtime_record(
                "maintenance-queued", episode_id=completed_episode_id,
                pending_games=int(self._maintenance_pending_count), reason="live-monitoring",
            )
            return

        if self.maintenance_thread is not None and self.maintenance_thread.is_alive():
            self.db.log_event(
                completed_episode_id, "maintenance-deferred",
                "previous background maintenance is still running", self.phase.value,
            )
            self._runtime_record(
                "maintenance-deferred", episode_id=completed_episode_id,
                pending_games=int(self._maintenance_pending_count), reason="worker-active",
            )
            return

        # One rebuild sees every game currently persisted, so a single pass drains
        # all queued games.
        self._maintenance_pending_episode_id = None
        queued_games = int(self._maintenance_pending_count)
        self._maintenance_pending_count = 0

        db_path = self.data_dir / "bingo.sqlite3"
        model_path = self.data_dir / "policy_model_v4.joblib"
        episodes_path = self.data_dir / "episodes.json"
        backup_dir = self.backup_dir
        backup_every = self.app_cfg.backup_every_games
        keep_backups = self.app_cfg.keep_backups
        holdout = self.app_cfg.ml_holdout_fraction

        def work():
            started = time.perf_counter()
            self._runtime_record(
                "maintenance-start", episode_id=completed_episode_id, queued_games=queued_games
            )
            try:
                local_db = BingoDB(db_path)
                completed = local_db.completed_game_count()
                if backup_every > 0 and completed % backup_every == 0:
                    local_db.backup(backup_dir, keep_backups)
                local_exp = ExperiencePolicy(db_path, model_path, episodes_path)
                info = local_exp.train(
                    holdout_fraction=holdout,
                    min_validation_games=self.app_cfg.ml_min_validation_games,
                )
                self._maintenance_reload_pending = True
                local_db.log_event(
                    completed_episode_id, "maintenance-complete",
                    f"background ML/backup complete; CF={info.counterfactual_samples}, outcomes={info.outcome_samples}, hindsight={info.hindsight_samples}",
                    GamePhase.COMPLETE.value,
                )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                self._runtime_record(
                    "maintenance-complete", episode_id=completed_episode_id, queued_games=queued_games,
                    elapsed_ms=round(elapsed_ms, 2), completed_games=int(completed),
                    counterfactual_samples=int(info.counterfactual_samples),
                    outcome_samples=int(info.outcome_samples), hindsight_samples=int(info.hindsight_samples),
                )
                self._ui_call(
                    self._auto_log,
                    f"BACKGROUND maintenance complete in {elapsed_ms/1000.0:.1f}s: "
                    f"CF={info.counterfactual_samples}, outcomes={info.outcome_samples}, hindsight={info.hindsight_samples}",
                    None, "INFO"
                )
            except Exception as exc:
                self._runtime_record(
                    "maintenance-error", level="WARN", episode_id=completed_episode_id,
                    queued_games=queued_games, elapsed_ms=round((time.perf_counter() - started) * 1000.0, 2),
                    error=f"{type(exc).__name__}: {exc}",
                )
                try:
                    BingoDB(db_path).log_event(
                        completed_episode_id, "maintenance-error", str(exc), GamePhase.COMPLETE.value
                    )
                except Exception:
                    pass
                self._ui_call(self._auto_log, f"BACKGROUND maintenance error: {exc}", None, "WARN")

        self.maintenance_thread = threading.Thread(target=work, name="post-game-maintenance", daemon=True)
        self.maintenance_thread.start()

    def _flush_deferred_maintenance(self) -> None:
        episode_id = self._maintenance_pending_episode_id
        if not episode_id:
            return
        self._runtime_record(
            "maintenance-flush-request", episode_id=episode_id,
            pending_games=int(self._maintenance_pending_count),
        )
        self._post_game_maintenance_async(episode_id, already_queued=True)

    def _reload_experience_if_ready(self) -> None:
        if not self._maintenance_reload_pending:
            return
        if self.solver_process is not None and self.solver_process.is_alive():
            return
        if self.solver_fallback_thread is not None and self.solver_fallback_thread.is_alive():
            return
        try:
            self.experience.reload()
            self._ui_call(self._refresh_learning_panel)
        finally:
            self._maintenance_reload_pending = False

    def _start_detected_round(self, board: Board, mask: int, confidence: float) -> None:
        previous_episode = self.episode_id
        if self.completion_pending and not self.episode_finalized:
            self.db.finalize_partial_game(
                previous_episode, self.selected_mask, self.last_verified_result,
                source="new-round-after-result-screen",
                notes="Reward/result screen was detected but the final 14-cell mask was not recoverable before the next round.",
            )
            self.db.log_event(
                previous_episode, "episode-closed-partial",
                "new round confirmed; previous result-screen episode closed without inventing a missing final mask",
                GamePhase.COMPLETE_PENDING.value,
            )
        if self.last_verified_result is None:
            self.db.log_event(
                previous_episode, "result-pending",
                "new round appeared before a verified result was captured; computed score remains authoritative pending manual verification",
                self.phase.value,
            )

        initial_mask = 0
        with self._live_state_lock:
            self.board = board
            self.board_confidence = float(confidence)
            self.selected_mask = initial_mask
            self.current_jewel = None
            self.current_confidence = 0.0
            self.current_margin = 0.0
            self.current_entropy = 1.0
            self.episode_finalized = False
            self.completion_pending = False
            self.episode_id = self._new_episode_id()
        self.game_sequence = []
        self.episode_actions = []
        self.one_left_saved = False
        self.one_left_path = None
        self._last_checkpoint_signature = None
        self.awaiting_result_until = 0.0
        self.last_verified_result = None
        self.last_recs = []
        self.cached_recommend_key = None
        self.last_decision_key = None
        self.last_decision_image = None
        self.last_current_patch = None
        self.last_current_confidence = 0.0
        with self._live_state_lock:
            self.last_state_warning = None
        self.temporal.reset(initial_mask)
        self.temporal_initialized = True
        self._transition_fast_until = time.monotonic() + 2.0
        self.board_consensus.reset()
        self.state_machine.reset()
        self.new_round_detector.reset()
        self.completion_detector.reset()
        self._ui_call(
            self._invalidate_recommendation,
            "new round detected",
            "New round detected. Reconfirming selected-mask from repeated frames before play...",
        )

        # Box color is research metadata for one round; do not silently copy the
        # previous round's label onto a newly detected game.
        self.box_color = None
        self._ui_call(self.box_color_var.set, "")
        self.db.ensure_game(
            self.episode_id, "local-auto", self.session_id, board, self.box_color,
            self.app_cfg.operation_mode, self.app_cfg.risk_profile,
        )
        self.db.log_event(
            self.episode_id, "new-round-auto",
            "new round detected; selected-mask bootstrap reset to 0 pending temporal confirmation", GamePhase.READY.value,
            {"previous_episode": previous_episode, "screen_mask": int(mask), "initial_mask": 0},
        )
        self._update_phase()
        self._ui_call(self._refresh_grid)
        self._ui_call(self.board_meta.set, f"Board hash {board_hash(board)} | vision {confidence:.2f} | auto new round")
        self._ui_call(self.turn_label.set, "Placed: 0/14 accepted | NEW GAME DETECTED; confirming screen state")
        self._ui_call(self.current_label.set, "Current jewel: waiting for first/next jewel")
        self._ui_call(self.target_var.set, "Target: waiting")
        self._ui_call(
            self.status.set,
            f"NEW GAME DETECTED ✅ Tracking continued automatically from {mask.bit_count()}/14; no Reset/new game needed."
        )
        self._reload_experience_if_ready()

    def _try_auto_new_round(self, panel) -> bool:
        """Detect a fresh playable board after a completed round.

        Result overlays and animation frames are rejected by requiring a valid
        board decode plus the same low selected-count state on two observations.
        """
        try:
            detail = read_board_detailed(panel, self.vision_cfg, self.clf)
            valid_board = not (
                detail.board_confidence < self.vision_cfg.board_min_confidence
                or detail.min_assigned_probability < self.vision_cfg.board_min_cell_probability
                or len(detail.ambiguous_cells) > self.vision_cfg.board_max_ambiguous_cells
            )
            raw_mask, _ = read_selected_mask(panel, self.vision_cfg)
            if not self.new_round_detector.observe(board_hash(detail.board), int(raw_mask), valid_board):
                return False
            self._start_detected_round(detail.board, int(raw_mask), float(detail.board_confidence))
            return True
        except Exception:
            self.new_round_detector.reset()
            return False

    def _finalize_episode(self, panel, finalization_source: str = "board-complete"):
        if self.episode_finalized or self.board is None or self.selected_mask.bit_count() < 14:
            return
        with self._live_state_lock:
            self.episode_finalized = True
        self._ui_call(self.overlay.hide)
        final_img = self._save_snapshot(panel, "final_board")
        s = score_mask(self.selected_mask)
        counts = list(self.board.jewel_counts_in_mask(self.selected_mask))
        score_dict = self._score_dict(s)
        auto_verified = computed_gt1000_verification(
            score_dict,
            selected_count=int(self.selected_mask).bit_count(),
            enabled=bool(self.app_cfg.auto_verify_computed_gt1000),
            finalization_source=str(finalization_source),
        )
        if auto_verified is not None:
            self.last_verified_result = auto_verified

        self.db.finalize_game(
            self.episode_id,
            self.board,
            counts,
            list(self.game_sequence),
            self.selected_mask,
            score_dict,
            self.one_left_path,
            final_img,
            auto_verified,
        )
        self.db.mark_finalization_source(self.episode_id, finalization_source)
        self._checkpoint_state("episode-finalized", payload={"source": finalization_source, "total": s.total})
        self.history.append({
            "source": f"local-{self.app_cfg.operation_mode}",
            "episode_id": self.episode_id,
            "counts": counts,
            "board": list(self.board.cells),
            "selected_mask": self.selected_mask,
            "sequence": list(self.game_sequence),
            "computed_score": score_dict,
        })
        self.awaiting_result_until = 0.0
        self._update_phase(result_visible=auto_verified is not None)
        self.db.log_event(
            self.episode_id, "game-finalized", f"computed total {s.total}", self.phase.value, score_dict
        )
        if auto_verified is not None:
            self.db.log_event(
                self.episode_id, "result-auto-verified",
                f"trusted final 14-cell board computed total {s.total} (>1000)",
                self.phase.value, auto_verified,
            )
            self._runtime_record(
                "result-auto-verified", episode_id=self.episode_id, total=int(s.total),
                lucky=int(s.lucky), normal=int(s.normal), source=str(finalization_source),
            )
            self._ui_call(self._refresh_learning_panel)
        completed_episode_id = self.episode_id
        # Analyze the just-finished game without waiting for a full-history ML
        # rebuild; a separate low-priority thread never blocks screen capture.
        self._audit_finished_episode_async(completed_episode_id)
        self._post_game_maintenance_async(completed_episode_id)
        self.new_round_detector.reset()
        self.completion_detector.reset()
        if auto_verified is not None:
            self._set_status(
                f"Game saved + VERIFIED automatically ✅ {s.lucky} Lucky, {s.normal} Normal, {s.total}. "
                "Verification basis: trusted 14-cell final board. Learning/backup are deferred while live monitoring is active."
            )
        else:
            self._set_status(
                f"Game saved ✅ {s.lucky} Lucky, {s.normal} Normal, {s.total}. "
                "Watching for the next round. Offline hindsight will check alternatives after monitoring stops."
            )

    def _loop(self):
        while self.running:
            self._diag_monitor_heartbeat = time.monotonic()
            if not self.running:
                break
            try:
                panel = self._panel()
                self.last_panel = panel

                if self.episode_finalized:
                    if self._try_auto_new_round(panel):
                        time.sleep(0.05)
                        continue
                    time.sleep(max(0.15, self.app_cfg.poll_ms / 1000.0))
                    continue

                if self.completion_pending:
                    # No OCR runs in the live path. Freeze decisions until either
                    # the trusted final 14-cell state is reconstructed or a fresh
                    # round is detected. Never invent the missing placement.
                    if self._try_auto_new_round(panel):
                        time.sleep(0.05)
                        continue
                    time.sleep(max(0.10, self.app_cfg.transition_poll_ms / 1000.0))
                    continue

                # Decode the board with the board-domain classifier. V5.2 ships
                # numeric seed descriptors and then learns user-specific cell
                # appearances from trusted reads/placements.
                if self.board is None:
                    detail = read_board_detailed(panel, self.vision_cfg, self.clf)
                    h = board_hash(detail.board)
                    board_stable, seen = self.board_consensus.observe(h)
                    too_uncertain = (
                        detail.board_confidence < self.vision_cfg.board_min_confidence
                        or detail.min_assigned_probability < self.vision_cfg.board_min_cell_probability
                        or len(detail.ambiguous_cells) > self.vision_cfg.board_max_ambiguous_cells
                    )

                    # Always let temporal consensus accumulate, even when the
                    # raw classifier confidence is temporarily low. V5.2 threw
                    # low-confidence frames away before consensus, which made a
                    # fresh install get stuck forever despite perfect geometry.
                    if too_uncertain or not board_stable:
                        self._live_board_failures += 1
                        now = time.monotonic()
                        if now - self._live_board_last_log >= 1.0:
                            self._live_board_last_log = now
                            msg = (
                                f"LIVE BOARD warm-up: conf={detail.board_confidence:.3f}, "
                                f"min-cell={detail.min_assigned_probability:.3f}, "
                                f"ambiguous={len(detail.ambiguous_cells)}, "
                                f"same-layout={seen}/{self.app_cfg.board_consensus_required}. "
                                f"model seeds={self.clf.seed_examples}, personal templates={sum(self.clf.template_counts().values())}."
                            )
                            self._ui_call(self.status.set, msg)
                            self._ui_call(self._auto_log, msg, None, "INFO")

                        # If several frames are weak, first re-fit the 5x5 inside
                        # the already validated panel. This is automatic geometry
                        # recovery, not a request for the user to redraw anything.
                        if self._live_board_failures % 8 == 0:
                            fresh = detect_board_grid(panel)
                            if fresh is not None and fresh.score >= 0.72 and fresh.mu_anchor_score >= 0.70:
                                self.vision_cfg.board_roi = fresh.roi
                                self.vision_cfg.board_roi_validated = True
                                self.vision_cfg.board_geometry_score = fresh.score
                                self.vision_cfg.save(self.vision_cfg_path)

                        time.sleep(max(0.15, self.app_cfg.poll_ms / 1000.0))
                        continue

                    self._live_board_failures = 0
                    self.vision_test_passed = True
                    self._accept_board(detail.board, detail.selected_mask, detail.board_confidence)
                    self._ui_call(self._refresh_grid)
                    self._ui_call(self._refresh_setup_state)
                    self._ui_call(
                        self._setup_feedback,
                        f"Live board validated automatically: confidence {detail.board_confidence:.2f}, "
                        f"min-cell {detail.min_assigned_probability:.2f}, ambiguous {len(detail.ambiguous_cells)}. "
                        "Current jewel detection is automatic.",
                        "ok",
                    )
                    self._ui_call(
                        self._auto_log,
                        f"LIVE BOARD accepted ✅ conf={detail.board_confidence:.3f}; "
                        f"min-cell={detail.min_assigned_probability:.3f}; ambiguous={len(detail.ambiguous_cells)}; "
                        f"consensus={seen}/{self.app_cfg.board_consensus_required}",
                        None,
                        "INFO",
                    )

                raw_mask, _ = read_selected_mask(panel, self.vision_cfg)
                self._raw_selected_count = int(raw_mask.bit_count())
                icon = self.icon_matcher.locate(panel, self.vision_cfg.board_roi)
                raw_label = icon.label if (icon is not None and icon.reliable) else None
                raw_conf = icon.score if icon is not None else 0.0
                raw_patch = icon.patch.copy() if icon is not None else None

                # End-of-round detection stays entirely vision/state based in P5.
                # The live loop no longer invokes Tesseract/number OCR. A recent
                # real 14/14 frame followed by the result-screen disappearance is
                # enough to preserve the trusted final mask.
                completion = self.completion_detector.observe(
                    accepted_mask=self.temporal.accepted_mask if self.temporal_initialized else self.selected_mask,
                    raw_mask=raw_mask,
                    jewel_visible=bool(icon is not None and icon.reliable),
                    remaining_ui=None,
                    result_plausible=False,
                )
                if completion.confirmed:
                    if completion.final_mask is not None and int(completion.final_mask).bit_count() == 14:
                        if self._complete_from_visual_evidence(panel, int(completion.final_mask), completion.reason):
                            continue
                    else:
                        # Strong result evidence without a reconstructable 14-cell
                        # mask freezes decision-making rather than inventing the
                        # missing placement. Persist it for recovery/OCR.
                        if not self.completion_pending:
                            with self._live_state_lock:
                                self.completion_pending = True
                            self._update_phase()
                            self._ui_call(
                                self._invalidate_recommendation,
                                "complete pending final mask",
                                "RESULT SCREEN detected. Freezing this episode while final state/result verification completes...",
                            )
                            self.db.log_event(
                                self.episode_id, "complete-pending-mask", completion.reason, self.phase.value,
                                {"accepted_mask": int(self.selected_mask), "raw_mask": int(raw_mask), "remaining_ui": None},
                            )
                            self._checkpoint_state(
                                "complete-pending-mask", provisional_mask=self.temporal.provisional_mask,
                                payload={"reason": completion.reason, "remaining_ui": None},
                            )
                        time.sleep(self._monitor_sleep_seconds(transition_hint=True))
                        continue

                # Show what vision sees immediately. Recommendation still waits
                # for temporal mask/jewel consensus, but the user should never
                # stare at "No recommendation yet" without knowing whether the
                # top jewel was found.
                if icon is not None:
                    state = "detected" if icon.reliable else "uncertain"
                    self._ui_call(
                        self.current_label.set,
                        f"Current jewel {state}: {icon.label} / {JEWEL_NAMES[icon.label]} "
                        f"(match {icon.score:.2f}, margin {icon.margin:.2f})"
                    )
                    self._ui_call(
                        self.vision_label.set,
                        f"Current icon: {icon.label} | match {icon.score:.2f} | margin {icon.margin:.2f}"
                    )
                    log_key = (icon.label, round(icon.score, 2), round(icon.margin, 2), self._raw_selected_count)
                    now = time.monotonic()
                    if log_key != self._last_icon_log_key or now - self._last_icon_log_at >= 2.0:
                        self._last_icon_log_key = log_key
                        self._last_icon_log_at = now
                        self._ui_call(
                            self._auto_log,
                            f"CURRENT JEWEL {state}: {icon.label}/{JEWEL_NAMES[icon.label]} "
                            f"match={icon.score:.3f}, margin={icon.margin:.3f}; raw placed={self._raw_selected_count}/14",
                            None, "INFO"
                        )
                else:
                    self._ui_call(self.current_label.set, "Current jewel: not visible yet")
                    self._ui_call(self.vision_label.set, "Current icon: no reliable candidate in the top search band")

                if not self.temporal_initialized:
                    # Do not trust a single first frame. Bootstrap at 0/14 and
                    # let repeated observations establish/catch up the checkpoint.
                    self.temporal.reset(0)
                    self.temporal_initialized = True

                with self._live_state_lock:
                    old_mask = int(self.selected_mask)
                    old_jewel = self.current_jewel
                    old_patch = None if self.last_current_patch is None else self.last_current_patch.copy()
                    old_conf = float(self.current_confidence)
                    old_margin = float(self.current_margin)
                    old_entropy = float(self.current_entropy)

                stable = self.temporal.observe(raw_mask, raw_label, raw_conf)
                self._diag_last_stable_reason = str(stable.reason)
                wait_signature = (
                    bool(stable.stable), str(stable.reason), int(self.temporal.accepted_mask.bit_count()),
                    int(self._raw_selected_count), None if icon is None else icon.label,
                    False if icon is None else bool(icon.reliable),
                )
                if wait_signature != self._runtime_last_wait_signature:
                    self._runtime_last_wait_signature = wait_signature
                    self._runtime_record(
                        "vision-state",
                        stable=bool(stable.stable), reason=str(stable.reason),
                        accepted_count=int(self.temporal.accepted_mask.bit_count()),
                        raw_count=int(self._raw_selected_count),
                        icon_label=None if icon is None else icon.label,
                        icon_reliable=False if icon is None else bool(icon.reliable),
                        icon_score=None if icon is None else float(icon.score),
                        icon_margin=None if icon is None else float(icon.margin),
                        episode_id=self.episode_id,
                    )
                if not stable.stable:
                    is_desync = "desync" in stable.reason
                    self._update_phase(desync=is_desync)
                    if is_desync and stable.reason != self.last_state_warning:
                        with self._live_state_lock:
                            self.last_state_warning = stable.reason
                        self.db.log_event(self.episode_id, "desync", stable.reason, self.phase.value)
                    accepted_n = int(self.temporal.accepted_mask.bit_count())
                    provisional = self.temporal.provisional_mask
                    if provisional is not None:
                        self._checkpoint_state(
                            "provisional-mask", provisional_mask=provisional,
                            payload={"reason": stable.reason, "screen_mask": int(raw_mask)},
                        )
                    self._ui_call(
                        self.turn_label.set,
                        f"Placed: {accepted_n}/14 accepted | screen sees {self._raw_selected_count}/14 | {stable.reason}"
                    )
                    buffered = stable.jewel or (icon.label if icon is not None and icon.reliable else None)
                    if icon is not None and icon.reliable:
                        # Keep this as observation-only. If the mask is a genuine
                        # post-click +1, the visible icon can already be the NEXT
                        # jewel; publishing it as current_jewel before mask commit
                        # would misattribute the previous action.
                        self._ui_call(
                            self.current_label.set,
                            f"Current jewel BUFFERED: {icon.label} / {JEWEL_NAMES[icon.label]} "
                            f"(match {icon.score:.2f}); board checkpoint still pending"
                        )
                    current_hint = f" Current candidate buffered: {buffered}." if buffered else ""
                    invalid = stable.reason.startswith("invalid-mask-count:")
                    state_text = "rejecting impossible screen mask" if invalid else "stabilizing board-selection glow"
                    self._ui_call(
                        self.status.set,
                        f"STATE CHECK: {stable.reason}; {state_text}.{current_hint} "
                        "No click is committed until the board checkpoint is confirmed."
                    )
                    time.sleep(self._monitor_sleep_seconds(transition_hint=(self._raw_selected_count != accepted_n)))
                    continue
                with self._live_state_lock:
                    self.last_state_warning = None

                new_mask = stable.mask
                transitioned = new_mask != old_mask
                if transitioned:
                    self._last_transition_started_at = time.perf_counter()
                    # The next-current-jewel icon usually appears immediately after
                    # a placement. Temporarily switch capture cadence to ~35 ms so
                    # provisional commit/rollback settles quickly.
                    self._transition_fast_until = time.monotonic() + 2.0
                    is_false_plus_one_rollback = stable.reason.startswith("rollback-false-plus-one")
                    jump = (new_mask & ~old_mask).bit_count()
                    if is_false_plus_one_rollback:
                        payload = {
                            "from_mask": int(old_mask), "to_mask": int(new_mask),
                            "from_count": old_mask.bit_count(), "to_count": new_mask.bit_count(),
                            "reason": stable.reason,
                        }
                        self.db.log_event(
                            self.episode_id, "state-rollback",
                            "Rolled back a recently committed +1 after the screen returned to the previous checkpoint",
                            self.phase.value, payload,
                        )
                        self._ui_call(
                            self.status.set,
                            f"STATE ROLLBACK: false +1 removed; restored {new_mask.bit_count()}/14 confirmed placements."
                        )
                    elif jump == 1:
                        self._record_action(
                            old_mask, new_mask, old_jewel, old_patch, old_conf, old_margin, old_entropy, panel=panel
                        )
                    else:
                        recovered_positions = [i for i in range(25) if ((new_mask & ~old_mask) >> i) & 1]
                        recovered_jewels = [self.board.cells[i] for i in recovered_positions] if self.board else []
                        payload = {
                            "from_count": old_mask.bit_count(),
                            "to_count": new_mask.bit_count(),
                            "missed": jump,
                            "positions": recovered_positions,
                            "jewels": recovered_jewels,
                        }
                        self.db.log_event(
                            self.episode_id, "state-catch-up",
                            f"Recovered {jump} missed placements from stable screen mask",
                            self.phase.value, payload,
                        )
                        self._ui_call(
                            self.status.set,
                            f"STATE RECOVERED: missed {jump} screen transitions; synchronized to {new_mask.bit_count()}/14 automatically."
                        )
                    with self._live_state_lock:
                        self.current_jewel = None
                        self.current_confidence = 0.0
                        self.current_margin = 0.0
                        self.current_entropy = 1.0
                        self.selected_mask = int(new_mask)
                else:
                    with self._live_state_lock:
                        self.selected_mask = int(new_mask)
                if transitioned:
                    self._update_phase()
                    self._checkpoint_state(
                        "mask-rollback" if stable.reason.startswith("rollback-false-plus-one") else "mask-commit",
                        payload={"reason": stable.reason, "old_mask": int(old_mask), "new_mask": int(new_mask)},
                    )
                    self._ui_call(
                        self._invalidate_recommendation,
                        f"board advanced {old_mask.bit_count()}->{new_mask.bit_count()}",
                        f"Board updated to {new_mask.bit_count()}/14. Waiting for a stable current jewel...",
                    )
                    self._ui_call(self._schedule_precompute)
                self._ui_call(
                    self.turn_label.set,
                    f"Placed: {self.selected_mask.bit_count()}/14 | Remaining: {max(0, 14-self.selected_mask.bit_count())} | screen stable"
                )

                if self.selected_mask.bit_count() == 13 and not self.one_left_saved:
                    self.one_left_path = self._save_snapshot(panel, "one_left")
                    self.one_left_saved = True
                    self.db.log_event(self.episode_id, "one-left", "13/14 state captured", GamePhase.ONE_LEFT.value)

                if self.selected_mask.bit_count() == 14:
                    with self._live_state_lock:
                        self.current_jewel = None
                    self._update_phase()
                    s_final = score_mask(self.selected_mask)
                    self._ui_call(self._invalidate_recommendation, "game complete", None)
                    self._ui_call(self._show_game_complete_summary, s_final.lucky, s_final.normal, s_final.total)
                    self._finalize_episode(panel)
                    self._ui_call(self.current_label.set, "Current jewel: game complete")
                    continue

                reliable_current = bool(icon is not None and icon.reliable and stable.jewel == icon.label)
                if icon is not None:
                    self._ui_call(
                        self.vision_label.set,
                        f"Current icon match {icon.score:.2f} | margin {icon.margin:.2f}"
                    )
                else:
                    self._ui_call(self.vision_label.set, "Current jewel: waiting for visible icon")

                if reliable_current:
                    self._transition_fast_until = 0.0
                    if self._last_transition_started_at is not None:
                        try:
                            vision_ms = (time.perf_counter() - self._last_transition_started_at) * 1000.0
                            self.db.save_telemetry(self.episode_id, "transition-to-jewel", vision_ms, None, {"jewel": icon.label})
                        except Exception:
                            pass
                        self._last_transition_started_at = None
                    with self._live_state_lock:
                        self.current_jewel = icon.label
                        self.current_confidence = icon.score
                        self.current_margin = icon.margin
                        self.current_entropy = 0.0
                        self.last_current_patch = raw_patch.copy() if raw_patch is not None else None
                        self.last_current_confidence = icon.score
                    # Saved only as last-observed diagnostics; it is never a
                    # setup prerequisite and can move frame to frame.
                    self.vision_cfg.current_roi = icon.roi
                elif transitioned or self.current_jewel is None:
                    with self._live_state_lock:
                        self.current_jewel = None
                    if icon is None:
                        self._ui_call(self.current_label.set, "Current jewel: waiting for a visible jewel")
                    else:
                        self._ui_call(
                            self.current_label.set,
                            f"Current jewel uncertain: {icon.label}? match={icon.score:.2f}, margin={icon.margin:.2f}"
                        )

                self._update_phase()
                self._ui_call(self._monitor_refresh_grid_if_changed)
                if self.current_jewel:
                    self._ui_call(
                        self.current_label.set,
                        f"Current jewel ACCEPTED: {self.current_jewel} / {JEWEL_NAMES[self.current_jewel]} "
                        f"(match {self.current_confidence:.2f})"
                    )
                    self._ui_call(self._recommend)
                else:
                    self._ui_call(self.overlay.hide)
            except Exception as e:
                self._runtime_record(
                    "monitor-error", level="ERROR", error=f"{type(e).__name__}: {e}",
                    episode_id=self.episode_id, phase=self.phase.value,
                )
                self._ui_call(self.status.set, f"Monitor error: {e}")
            time.sleep(self._monitor_sleep_seconds())

    def verify_last_result(self):
        if not self.episode_finalized:
            messagebox.showinfo("Verify result", "Finish a game first.")
            return
        try:
            lucky_score = simpledialog.askinteger("Verify result", "Lucky Clear SCORE shown (e.g. 936)", parent=self)
            normal_score = simpledialog.askinteger("Verify result", "Normal Clear SCORE shown (e.g. 240 or 0)", parent=self)
            jewel_score = simpledialog.askinteger("Verify result", "Jewel Score shown (e.g. 90)", parent=self)
            total_score = simpledialog.askinteger("Verify result", "Total Score shown", parent=self)
            if None in (lucky_score, normal_score, jewel_score, total_score):
                return
            v = VerifiedResult(lucky_score, normal_score, jewel_score, total_score, "manual")
            verified = self._verified_to_score(v)
            self.db.verify_score(self.episode_id, verified)
            self.last_verified_result = verified
            self._update_phase(result_visible=True)
            self.db.log_event(self.episode_id, "result-verified-manual", f"manual total {total_score}", self.phase.value, verified)
            # Avoid blocking Tk/UI or the live solver with a full retrain.
            self._post_game_maintenance_async(self.episode_id)
            self._ui_call(self._refresh_learning_panel)
            self.status.set(f"Manual result verified: total {total_score}. Learning queued in background.")
        except Exception as e:
            messagebox.showerror("Verify result", str(e))

    def reset_state(self):
        """Start a fresh tracking episode and synchronize it to the visible board.

        V5.5 always forced the accepted mask to 0, which could immediately create
        an impossible 0/14-vs-4/14 desync when Reset was pressed mid-round. V5.6
        treats Reset as a tracking reset, not as proof that the screen is empty.
        """
        self._invalidate_recommendation("manual tracking reset", "Resetting tracking and synchronizing to the visible board...")

        synced_board = None
        synced_mask = 0
        synced_conf = 0.0
        sync_error = None
        if self.region is not None and self.vision_cfg.board_roi is not None:
            try:
                panel = self._panel()
                self.last_panel = panel
                detail = read_board_detailed(panel, self.vision_cfg, self.clf)
                raw_mask, _ = read_selected_mask(panel, self.vision_cfg)
                synced_board = detail.board
                synced_mask = int(raw_mask)
                synced_conf = float(detail.board_confidence)
            except Exception as exc:
                sync_error = str(exc)

        with self._live_state_lock:
            self.board = synced_board
            self.board_confidence = synced_conf
            self.selected_mask = synced_mask
            self.current_jewel = None
            self.current_confidence = 0.0
            self.current_margin = 0.0
            self.current_entropy = 1.0
            self.episode_finalized = False
            self.completion_pending = False
            self.episode_id = self._new_episode_id()
        self.game_sequence = []
        self.episode_actions = []
        self.one_left_saved = False
        self.one_left_path = None
        self.box_color = None
        if hasattr(self, "box_color_var"):
            self.box_color_var.set("")
        self.awaiting_result_until = 0.0
        self.last_verified_result = None
        self.last_recs = []
        self.cached_recommend_key = None
        self.last_decision_key = None
        self.last_decision_image = None
        self.last_current_patch = None
        self.last_current_confidence = 0.0
        self.temporal.reset(synced_mask)
        self.temporal_initialized = synced_board is not None
        self._transition_fast_until = time.monotonic() + 2.0 if synced_board is not None else 0.0
        self.board_consensus.reset()
        self._live_board_failures = 0
        self._live_board_last_log = 0.0
        self.state_machine.reset()
        self.phase = GamePhase.NO_BOARD
        self.phase_var.set("Phase: no-board")
        with self._live_state_lock:
            self.last_state_warning = None
        self.current_label.set("Current jewel: waiting")
        self.vision_label.set("Vision: waiting")

        if synced_board is not None:
            self.db.ensure_game(
                self.episode_id, "local-auto", self.session_id, synced_board, self.box_color,
                self.app_cfg.operation_mode, self.app_cfg.risk_profile,
            )
            h = board_hash(synced_board)
            self.board_meta.set(f"Board hash {h} | vision {synced_conf:.2f} | synchronized on reset")
            self.turn_label.set(
                f"Placed: {synced_mask.bit_count()}/14 | synchronized from current screen"
            )
            self._update_phase()
            self._refresh_grid()
            self.status.set(
                f"New tracking episode synchronized automatically to the visible round at {synced_mask.bit_count()}/14."
            )
        else:
            self.board_meta.set("Board: waiting for live reacquisition")
            self.turn_label.set("Placed: ?/14 | Waiting to synchronize current screen")
            suffix = f" ({sync_error})" if sync_error else ""
            self.status.set(
                "New tracking episode ready. The monitor will reacquire the visible board instead of assuming 0/14." + suffix
            )
        self._refresh_setup_state()



def run(root_dir: Path):
    app = BingoApp(root_dir)
    app.mainloop()
