# V18 Architecture

## Runtime split

**Windows / authoritative side**
- game capture and board/current-jewel vision;
- temporal filtering, provisional/confirmed masks and result-screen lifecycle;
- authoritative `episode_id`, `state_version` and `strategy_version`;
- strategy-plan persistence and UI;
- SQLite episodes, checkpoints, candidate decisions, outcomes and telemetry;
- learning/research reporting.

**Ubuntu/WSL / solver side**
- persistent service started explicitly in the configured Ubuntu distribution;
- worker pool;
- transposition/cache reuse;
- exact route feasibility and short-horizon Expectimax;
- bounded live Monte Carlo;
- six-jewel precompute and speculative child precompute.

Docker is deliberately outside the live hot path.

## State correctness

Every solver request/response is bound to the episode/state/strategy version. Stale work is discarded. A single visual `+1` transition is provisional until confirmed; a stable rollback can undo a false glow transition. Completed episodes are frozen and are not re-solved while the reward screen remains open.

## Strategy planner

The live strategy is `Win >1000 / Adaptive`. The planner stores a concrete active route across moves and uses hysteresis before changing it. Its fallback ladder is:

1. 3 Lucky
2. 2 Lucky + 1 Normal
3. 1 Lucky + 2 Normal
4. 1 Lucky + 1 Normal
5. best achievable score

Route feasibility, slack and on-route placement are used to avoid spending pieces on a fourth Lucky line when an equivalent placement advances the locked trio.

## Mathematical core

The scoring rules, line definitions, exact multivariate inventory feasibility, Expectimax semantics and Monte Carlo objective are regression-frozen in V18. They are changed only when a reproducible test demonstrates a defect.

## Persistence

The persistent database lives under `%LOCALAPPDATA%\Speedlora\JewelBingo`. Schema migrations create a pre-migration backup. The database stores games, actions, candidate decisions, state events/checkpoints and latency telemetry.

## Compatibility contracts

`vision_engine/versioning.py` is the single source of truth for:
- application version;
- solver protocol version;
- SQLite schema version;
- local model version;
- strategy-planner version.

## Observability

The System tab exposes WSL health, database health, vision status, decision latency percentiles, cache hit rate and backup state. The Learning tab exposes trust/validation state rather than hiding model behavior behind logs.

## Current-jewel-aware planner (18.0.1)

The strategy planner selects its live target from the child states reachable by the **known current jewel**, not from the pre-draw mask. Locked-route hysteresis remains active while the route survives. If the current jewel makes that route impossible, the planner performs an emergency same-target route switch before descending the fallback ladder. Recommendation payloads expose both global target feasibility and concrete locked-route feasibility.

## V18.0.25 input-exclusive Auto Play boundary
Auto Play uses two explicit barriers and one generation-stamped commit owner:
`FINAL recommendation -> normal pacing -> PREPARING -> disable/drain WSL background pool -> QUIESCING -> monitor safe-point ACK -> atomic state revalidation -> COMMITTING -> V18.0.11 Win32 move/click -> AWAIT_CONFIRM -> release monitor -> visual N->N+1 confirmation -> next generation`.

While Auto Play is enabled, speculative-best-child and general next-jewel precompute are suppressed. This is deliberately conservative: V18.0.25 optimizes for deterministic input stability, not solver cache throughput. Background precompute is re-enabled only when Auto Play is turned off.

The input path does not depend on diagnostic file writes or arbitrary synchronization sleeps. Timeouts exist only as fail-closed safety limits around state conditions.

## V18.0.24 input transaction boundary
Auto Play no longer relies on diagnostic file I/O to perturb thread scheduling. The input critical section is explicit:
`FINAL recommendation -> normal pacing -> state validation -> request monitor safe point -> monitor ACK/parked -> second validation -> known-good V18.0.11 cursor move/click -> release monitor -> visual N->N+1 confirmation`.


## V18.0.26 bridge/session lifecycle

The Windows bridge separates **worker IPC** from **lifecycle control**. Solver requests use the admission-gated worker client/pool. `end_session` and controlled service shutdown use a separate TCP client so a blocked solve cannot prevent the server from closing admission during application shutdown.

Graceful shutdown is: `close Windows admission -> best-effort drain worker IPC -> end_session on control connection -> revoke same-session exclusive token -> close worker socket`. If this Windows process launched the WSL service, it also requests service shutdown. If the service was already persistent, it remains alive but detached with admission closed until the next `begin_session`.

Crash recovery is fail-closed. A new GUI never steals an orphaned exclusive token. If `begin_session` times out because the prior GUI died while owning the token, the bridge requests a controlled solver-service restart and begins its new `session_id` only on the clean replacement service.

## Phase 9 — Complete input-exclusive transaction

The physical input window is now modeled as one fail-closed transaction:

```text
close Windows bridge admission
        ↓
drain active IPC to zero
        ↓
GLOBAL FREEZE persistent WSL solver
        ↓
foreground = 0 / background = 0
        ↓
exclusive token
        ↓
monitor safe-point / QUIESCED
        ↓
atomic live-state validation
        ↓
conservative Bézier movement + click
        ↓
monitor release
        ↓
WSL release(exclusive token)
        ↓
Windows bridge admission open
```

Cleanup is intentionally asymmetric toward safety: Windows admission is never reopened merely because a release was attempted. The server must confirm the exclusive token ended. A broken worker socket retries release on an independent lifecycle connection; a rejected release remains fail-closed and retryable.

## Phase 10 — Repeated transaction stress and monitor release acknowledgement

The monitor handshake is now synchronous across commit boundaries. `MonitorInputGate.release()` clears the request and waits for the monitor thread to leave its current safe point before releasing commit ownership. This prevents a fast following commit from reusing handshake state while the previous monitor cycle is still unwinding.

The Phase 10 stress harness (`vision_engine/diagnostics/input_exclusive_stress.py`) keeps one real TCP solver service alive while repeatedly exercising Windows admission, server foreground/background counters, global freeze/release, monitor quiescence and session rotation. It intentionally creates stale pre-freeze tickets and periodic six-job background loads. The physical cursor is not moved by this offline harness; the input window is represented by an assertion that no solver admission can occur while the exclusive token and monitor safe point are held.

## Phase 12 — audited live input boundary

The live GUI no longer imports the raw movement function. Physical input follows:

`BingoApp -> InputExclusiveCoordinator.acquire() -> LiveInputExecutor -> input/win32.py`.

`LiveInputExecutor` rejects execution when the exclusive lease has already been released, the monitor is not quiesced, or a remote-solver lease has no global solver token. This keeps the raw Win32 transport behind a lease-aware live boundary while preserving `autoplay.py` only as a backwards-compatible facade.

Release packaging is built with `tools/build_release_zip.py`; local `.venv`, caches, pip/WSL logs and runtime SQLite files are excluded by construction.

## Phase 13 — post-click WAITING diagnostics

The diagnostic path is deliberately outside the Auto Play disk hot path:

```text
BingoApp / monitor / bridge / Win32
        -> in-memory WaitingDiagnostics + Win32 ring buffers
        -> localhost probe thread (127.0.0.1 only)
        -> diagnose_waiting.bat (explicit user action)
        -> 12 s sampling + independent WSL solver_status
        -> REPORT.txt + JSON + process lists + ZIP
```

The probe exposes read-only snapshots. It does not release fences, click, restart the solver, or mutate game state. Deep snapshots include Python thread stacks so a stalled Tk/monitor/bridge thread can be distinguished from a clean `AWAIT_CONFIRM` vision stall.
