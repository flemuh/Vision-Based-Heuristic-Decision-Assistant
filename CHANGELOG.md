# V18.0.26 STATE-INTEGRITY-P1 root fix

- temporal bootstrap no longer trusts a single selected-mask frame; hard <=14 invariant
- current-jewel observation buffered independently while board checkpoint is provisional
- current-jewel cross-label contamination filtering + AUTO SETUP trusted-batch replacement
- isolated clean mutable profile for real cross-version A/B
- live six-jewel speculative precompute disabled by default
- cursor verifier accepts physically reached waypoints within 5 px while preserving hard-failure aborts

# V18.0.26 WSL Stable — Phase 13 WAITING diagnostics

- Added a localhost-only in-memory runtime probe for Auto Play stalls.
- Added `diagnose_waiting.bat` and `tools/capture_waiting_diagnostics.py`.
- Captures commit phase, generation/token, board mask/jewel, monitor heartbeat and safe-point state, Windows bridge admission/IPC state, WSL global fence foreground/background/frozen state, Win32/Bézier events, process lists and Python thread stacks.
- Diagnostic script samples the live app for 12 seconds and emits a ZIP plus automatic classification (`AWAIT_CONFIRM_VISION_STALL`, `MONITOR_LOOP_STALLED`, `MONITOR_RELEASE_STUCK`, `SOLVER_FENCE_RELEASE_STUCK`, `CONCURRENCY_NOT_IDLE`, etc.).
- Auto Play hot path still performs no diagnostic disk I/O; the external script writes the bundle only when explicitly launched.
- No solver policy, scoring, pacing, fence semantics or mouse behavior changed in this phase.

# V18.0.26 WSL Stable — Phase 12B launcher hotfix

- Removes the requirement for `pip` to exist inside `.venv`.
- Creates `.venv` with `python -m venv --without-pip`.
- Validates the base Python installation's pip and requires pip's `--python` target-interpreter option.
- Installs/reinstalls requirements into `.venv` with `py -3.12 -m pip --python .venv\Scripts\python.exe ...`.
- Eliminates the failing `.venv\Scripts\python.exe -m ensurepip` bootstrap path.
- Keeps WSL/global fence/session/monitor/Bézier/strategy behavior unchanged.

# V18.0.26 WSL Stable — Phase 11 full regression checkpoint

- Executes the complete offline project suite: 233 tests across 55 test files, all passing in four timeout-safe blocks.
- Revalidates legacy and current Auto Play, WSL persistent solver/precompute, V18 planner/learning/history/release contracts, V56/V7 compatibility, vision and all Phase 5–10 fence/lifecycle/stress tests together.
- Final `tools/validate_release.py` passes all release contracts, including HA -> L3C2 immediate Lucky priority, 13/14 completion recovery and 2 Lucky + 1 Normal fallback.
- No production code changes were required in Phase 11; this checkpoint proves the Phase 5–10 refactor did not regress the rest of the project offline.

# V18.0.26 WSL Stable — Phase 10 stress checkpoint

- Adds a 1,000-cycle automated end-to-end stress regression and a 3,000-cycle extended validation over the real TCP solver server, `LiveSolverBridge`, `GlobalSolverFence`, `MonitorInputGate` and `InputExclusiveCoordinator`.
- Stress rotates `session_id` every 100 cycles, recreates `pending=6` every 50 cycles, mixes active Windows IPC / foreground / background work, and verifies stale tickets never cross a closed admission gate.
- Stress exposed a monitor handshake race: `release()` could return before the monitor actually left its safe point, allowing the next cycle to reuse handshake state too early.
- Replaces the monitor Event reuse with a condition-based synchronous safe-point lifecycle. Release now waits for physical monitor exit before the commit owner is reusable.
- Extended 3,000-cycle result: 3,000 acquire/release pairs, 29 session rotations, 60 pending=6 cycles, 3,000 stale-ticket rejections, 0 forbidden admissions, 0 leaks and 0 deadlocks.
- Focused accumulated Phase 5–10 validation: 39 tests passed.

# V18.0.25 — Auto Play input-exclusive stabilization

- Keeps `jewel_bingo/autoplay.py` byte-identical to the user-confirmed V18.0.11 Windows transport (`e9b9a4ce...d44bb2`).
- Restores atomic live-state locking removed by V18.0.24 so `episode/mask/current jewel/confidence` cannot be read as a mixed frame during Auto Play validation.
- Adds a generation-stamped `AutoPlayCommitCoordinator` with explicit `SCHEDULED -> PREPARING -> QUIESCING -> COMMITTING -> AWAIT_CONFIRM` ownership. Late/stale callbacks cannot commit.
- Disables speculative and next-jewel background precompute for the entire Auto Play run. Foreground decisions remain enabled.
- Adds a server-side WSL background gate and condition-based drain barrier. Input proceeds only when pending background work is truly idle; timeout aborts closed instead of assuming elapsed milliseconds are safe.
- Keeps the V18.0.24 monitor safe-point handshake as a second barrier after the solver reaches idle.
- Restores the V18.0.22 regression guard that prevents `_schedule_precompute()` after a board transition while Auto Play is enabled.
- Restores `save_runtime_logs` / `save_game_images` gates, so diagnostic disk I/O is not required by the input hot path.
- Adds V18.0.25 concurrency/idempotency/stale-generation/background-barrier tests, including 1,000 commit generations.
- No Bézier motion in V18.0.25. Curved trajectory remains deferred to V18.0.26 after real-game stability is confirmed.

# V18.0.20 — Auto Play input isolation

- Built from the user-confirmed V18.0.11 Windows input transport. `jewel_bingo/autoplay.py` is byte-identical to that working build.
- Keeps the V18.0.12/V18.0.13 solver corrections in `advisor.py`.
- When Auto Play is armed, speculative WSL best-child precompute is deferred until vision confirms the click changed the board. This prevents fresh/cold background solver workers from starting or computing while Win32 cursor movement is in progress.
- This patch deliberately does not add curved cursor motion. The goal is to isolate the post-V18.0.11 runtime interaction before changing cursor geometry again.

## V18.0.11

- Fixes Auto Play stopping with `Windows SetCursorPos failed` on systems where that cursor API is rejected.
- Adds a no-movement Windows mouse-control preflight before arming Auto Play.
- Keeps SetCursorPos as the primary visible-motion backend and falls back to absolute SendInput movement.
- Uses checked SendInput for the left click and reports a clear same-privilege/admin diagnostic if Windows blocks injected input.
- Logs which Windows input backend was used for each Auto Play click.
- Clarifies that Green/Blue/Red `Box (optional)` is research metadata and does not control the in-game box.
- No solver, ML, hindsight, strategy, pacing, database schema or vision changes.

## V18.0.10

- Added campaign pacing aimed at ~51.4 seconds per completed round, which is ~10 hours for 700 rounds when UI/solver throughput can keep up.
- Pacing never shortens the bounded UI-settle safety floor to catch up after a slow solver, animation or result transition.
- Increased conservative UI timing defaults: 900–1600 ms base settle + 200–700 ms extra settle + 220–380 ms visible cursor travel.
- Increased post-click board-transition watchdog to 6.5 s to reduce false stops during slower animations.
- Added a second full state/confidence/recommendation validation immediately before the click, after all pacing/settle time has elapsed.
- The click is still permitted only after a final/refined recommendation, and the next click still requires a confirmed board-mask transition.
- Existing bounded timing/position variability remains for UI robustness and is not an anti-cheat bypass or detectability guarantee.
- Solver, planner, hindsight learning, protocol, schema and ML contracts are unchanged.

## V18.0.9

- Added a separate bounded 50–500 ms operational settle backoff before Auto Play TEST clicks.
- Existing 550–950 ms pre-click wait remains intact, making total pre-click settling 600–1450 ms.
- Click audit events now store base delay, extra settle delay, total pre-click delay and cursor-move duration.
- Solver, planner, vision gates, hindsight learning, schema, protocol and ML contracts are unchanged.
- Timing variability remains for UI robustness/observability, not anti-cheat evasion.

## V18.0.8

- Auto Play TEST now moves the visible Windows cursor smoothly to the chosen cell instead of teleporting it.
- Click target uses bounded safe-center jitter so UI automation is not tied to one exact pixel.
- Pre-click wait and cursor-travel duration use bounded variability for UI robustness.
- Existing solver, planner, DB schema, hindsight learning and anti-duplicate/transition gates are unchanged.
- This is not an anti-cheat bypass or stealth mode; no process injection, hidden input or human-signature emulation is implemented.

## V18.0.7

- Added opt-in **Auto Play TEST** that executes the final #1 recommendation as one ordinary Windows click.
- Auto Play waits for accepted jewel/board confidence, exact final recommendation, and post-click visual confirmation before another click.
- Added per-run game limit (1..700), transition watchdog, assistant-window overlap protection, automatic stop on state warnings/errors, and DB click audit events.
- Auto Play contains no humanization/random mouse paths, stealth, process injection, memory access, or anti-cheat bypass. Timing is driven by solver/vision readiness plus a fixed settle delay.
- Existing solver, scoring, hindsight reward learning, protocol, database schema, and planner versions are unchanged.

## V18.0.6

- Added candidate-level hindsight reward learning from complete stored games.
- Retrospective training rewards both chosen and unchosen candidates according to whether the real draw composition still allowed 3 Lucky, >1000, fallback targets, and the best reachable score.
- Hindsight reward uses only live-state features at inference; future jewels never enter the live solver input.
- Hindsight learning remains a true-tie-only influence and cannot override a mathematically superior move.
- Rebuilding is deterministic from SQLite, so replaying/opening a game cannot double-count it.
- Added automatic one-time rebuild for pre-18.0.6 history and a manual rebuild control in Learning.
- Protocol 3, schema 9 and Strategy Planner 21 unchanged; ML model contract bumped to 6.

## V18.0.5

- Added a Game History / Replay window backed by the persistent SQLite database instead of exposing only the newest game.
- Historical replay now includes saved top candidates and live recommendation-vs-accepted-placement details.
- Added exact actual-sequence hindsight over all legal terminal allocations for the 14 jewels that actually arrived.
- Reports whether 3 Lucky and >1000 were achievable and the first irreversible hindsight loss, with preserving alternatives.
- Hindsight is explicitly retrospective and does not modify or judge the live solver using future information.
- No changes to protocol 3, schema 9, ML model 5 or Strategy Planner 21.

## V18.0.4

- Canonicalized structured route identities: completed lines remain part of the full target route instead of disappearing from the route identity.
- Fixed the real 7/14 Harmony regression: `HA -> L3C2` now outranks `L5C3` when both preserve the same 3-Lucky feasibility because L3C2 completes Lucky H immediately.
- Added deterministic tie-breaks for immediate Lucky clears, then immediate Normal clears, only after target/route feasibility is equal.
- Rejects shortened legacy locked routes so an old `V+\` identity cannot masquerade as a complete 3-Lucky plan.
- Added Live explanations for immediate line clears.
- Kept protocol 3, schema 9 and ML model 5 unchanged; Strategy Planner contract bumped to 21.

## V18.0.3

- Fixed Observe mode leaking recommendation colors while intentionally hiding move advice.
- Added an always-visible Live color legend and the same color guide to Analysis.
- Observe mode now clears stale ranking rows/probabilities and explicitly says to switch to recommend for advice.
- Ranking colors remain final/refined-only in Recommend mode.
- No changes to scoring, route feasibility, Expectimax, Monte Carlo, WSL, cache or planner contract.

# Changelog

## V18.0.2

- Added semantic ranking colors so primary-probability overlaps no longer appear as multiple green winners unless the full hierarchy is an exact tie.
- Added golden 12/14 Soul regression: L5C5 wins through Normal-line leverage while L3C5/L4C5 remain close alternatives.
- Added dynamic endgame route hysteresis: 1.35 early, 1.20 at 10+, 1.10 at 12+, 1.00 at 13+.
- Live table now shows four options with simple Goal / Route / Win / 3 Lucky percentages and a short reason/status column.
- Added Analysis scrollbar, Learning table scrollbar and a recent-log viewer under System; setup logs remain hidden from Live while monitoring.
- Kept protocol 3, schema 9 and ML model 5 unchanged; Strategy Planner contract bumped to 20.


## V18.0.1

- Planner now conditions target and locked-route selection on the jewel already visible on screen.
- Emergency same-target route switching happens before dropping to the fallback ladder.
- Target chance and locked-route chance are tracked separately in recommendations/UI.
- True target ties prefer Lucky contingency structure, then useful Normal structure, before incidental average points.
- Exact endgame expected value participates early in late-game tie-breaks.
- Forced off-route/slack moves are explicitly marked instead of looking like a fourth-Lucky objective.
- Live UI is more compact; expected points moved out of the primary live ranking.
- Strategy Planner contract bumped to 19; probability/scoring cores remain unchanged.

## V18.0.0

V18 consolidates the validated V7.0.5 mathematical/WSL baseline into a stabilized release.

### Phase 1 — Baseline & Contracts
- centralized app/protocol/schema/model/planner versions;
- protocol drift rejection;
- pre-migration SQLite backups;
- golden regression suite from real live-play failures.

### Phase 2 — Live UX
- simplified Live view for fast decision reading;
- detailed Analysis separated from normal play;
- explicit target-change banner and plain-language confidence;
- compact top-option presentation.

### Phase 3 — Learning & Pattern Lab
- Learning tab with sample/validation/trust metrics;
- `ml_min_validation_games` is now enforced;
- outcome model reports improvement versus baseline;
- RNG/pattern stages are visible but remain research-only by default.

### Phase 4 — Operations
- System health tab;
- persistent P50/P95/P99 latency and cache-hit telemetry;
- conservative current-round recovery;
- stored-game replay and research dataset export;
- automatic WSL restart attempt after remote solver failure.

### Phase 5 — Stabilization
- V18 final identity and user-facing strings;
- current README/Architecture/Research documentation;
- legacy V4/V5/V7 documents archived, not deleted;
- expanded final release validator and migration/release regression tests.

## V18.0.24-STABLE — explicit monitor/input handshake
- Based on the V18.0.21 solver/planner state, but removes the temporary Auto Play diagnostic JSONL hot path.
- Restores `jewel_bingo/autoplay.py` byte-for-byte from the known-good V18.0.11 Windows input implementation.
- Adds an explicit `MonitorInputGate`: Auto Play requests a safe point, the `bingo-monitor` acknowledges only between complete vision cycles, parks during cursor movement/click, then resumes to confirm `N/14 -> N+1/14`.
- The safe-point wait uses synchronization events rather than disk I/O or arbitrary extra milliseconds.
- Keeps speculative solver precompute deferred while Auto Play is active.
- Keeps the V18.0.12 endgame fix and V18.0.13 primary-3-Lucky planner fix.
- Keeps the existing ~51.4 s/game pacing target (~10 h for 700 rounds); pacing is independent of the race fix.

## V18.0.26 WSL Stable — Phase 6 checkpoint

- Added deterministic `pending=6` global-freeze integration coverage.
- The test uses six non-cancellable RUNNING Futures through the persistent TCP server path.
- Verified no exclusive token can be returned until all six admitted background jobs have physically left the solver fence.
- Verified foreground/background admission remains closed while freeze is pending.
- Verified release reopens admission with both counters at zero.
- No new runtime behavior was added in this phase; this phase validates the fence implemented previously.

## V18.0.26 WSL Stable — Phase 7 session takeover checkpoint

- Added deterministic restart/session-takeover tests with physically running old foreground and background jobs.
- Proved new-session ownership waits until both old-work counters reach zero.
- Proved old and new solver requests fail closed while rotation is pending.
- Proved stale old-session traffic is rejected after rotation.
- Proved a new session cannot steal an active exclusive token and only the owning session/token can release it.
- Added Windows-bridge validation that stale responses from a previous `session_id` are dropped before GUI callbacks.
- No production timing sleeps or disk-I/O synchronization were added.


## V18.0.26 WSL Stable — Phase 8 bridge lifecycle checkpoint

- Hardened `LiveSolverBridge.close()` so session detach is attempted even after a transient bridge failure marked it unavailable.
- Added a separate lifecycle/control TCP connection for `end_session`/shutdown, preventing a busy worker IPC socket from blocking server admission closure.
- Verified graceful close revokes the session's own exclusive token without leaving the persistent solver frozen.
- Added controlled recovery for a previous GUI crash that orphaned an exclusive token: fail closed, restart service, begin a clean new session.
- Added deterministic bridge lifecycle tests; focused V18.0.26 architecture suite now passes 28 tests.

## V18.0.26 WSL Stable — Phase 9 checkpoint

- Added complete input-exclusive end-to-end coverage using a real persistent TCP solver server, `LiveSolverBridge`, `MonitorInputGate` and the global WSL fence.
- Hardened release semantics so monitor cleanup cannot skip solver cleanup.
- Exclusive leases remain retryable when solver release is incomplete.
- Windows admission is now reopened only after the WSL server confirms exclusive-token release.
- Added independent control-connection fallback when the worker SolverClient socket fails during token release.
- Server release rejection is fail-closed: admission stays shut and the token remains owned for retry/recovery.
- Added failure injection for atomic validation, Bézier movement and click; every path proves cleanup of monitor and solver fences.
- Accumulated focused V18.0.26 architecture validation: 36 tests passed.

## V18.0.26 WSL Stable — Phase 12 architecture/package audit

- Added `LiveInputExecutor`; live GUI no longer calls raw Win32 movement directly.
- Added executable architecture audit (`tools/audit_architecture.py`).
- Added clean release builder (`tools/build_release_zip.py`).
- Updated Windows/WSL launchers from stale V18.0.11/V7 labels and dependency markers to V18.0.26.
- Release packaging now excludes `.venv`, Python caches, pip logs, WSL runtime logs and runtime SQLite files.
- Added clean-start/release audit regression tests.
- No solver policy, scoring, vision thresholds, pacing or global-fence semantics changed.


### Phase 12A Windows launcher hotfix
- Fixed clean-start bootstrap when an existing `.venv` contains a partially corrupted pip.
- `pip --version` is no longer considered sufficient health validation; the launcher now imports the real install command/vendor stack and executes `pip install --help`.
- Corrupt environments are deleted and recreated from the detected Python installation instead of being repaired in-place.
- Removed mandatory pip self-upgrade from normal startup to reduce the chance of corrupting pip during bootstrap.
- Core dependency imports are checked and, if stale, requirements are force-reinstalled once automatically.
- Added static regression checks for unique batch labels/control flow.


## Phase 12C short-path Windows runtime hotfix

The Windows virtual environment is now stored outside the extracted project tree at `%LOCALAPPDATA%\SpeedloraJewelBingo\v18026-py<version>`. This prevents NumPy/Pillow/OpenCV package files from exceeding legacy Windows MAX_PATH when release ZIPs are extracted under long Downloads paths. The project-local `.venv` is no longer used. Host pip still installs into the pip-less runtime venv via `pip --python`.
