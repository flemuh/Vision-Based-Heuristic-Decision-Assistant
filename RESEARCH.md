# V18 Research & Learning

V18 treats deterministic/exact game mathematics as the primary decision system. Local ML is deliberately conservative and cannot override a mathematically separated recommendation; it can help only in genuine mathematical/statistical ties.

## Counterfactual model

Each solver decision records candidate cells, not only the chosen move. This produces multiple training examples per turn. The model estimates solver-related outcomes for state/action pairs and must satisfy sample, validation-game and MAE gates before being marked `TRUSTED`.

## Real-outcome residual model

Completed games provide real outcomes. A second model learns residual error relative to the mathematical solver and must beat a zero-residual baseline before being trusted. `ml_min_validation_games` is the common validation-game gate used by V18.

## Pattern/RNG research

Observed jewel frequencies, complete count vectors, repeated layouts and related signatures are collected for analysis. Pattern status is:

- 0-49 local complete games: `COLLECTING`;
- 50-99: `ANALYZING`;
- 100+: `ELIGIBLE FOR VALIDATION`.

`empirical_bias_blend` remains zero by default, so observed frequency noise does not silently alter solver probabilities. Any future RNG adaptation should require holdout validation, confidence intervals and stability across time blocks.

## Replay and export

Stored games can be replayed from the database. Research export produces episodes, moves, candidates, events, latency metrics, model metrics and pattern analysis in portable CSV/JSON form.

## Regression philosophy

Real failures discovered during live play become golden regression cases. V18 preserves tests for temporal glow errors, result-screen completion, route ranking, route lock, automatic fallback and historical exact-decision states.

## V18.0.26 Phase 10 stress evidence

A deterministic 3,000-cycle architecture stress run completed with 0 failures/deadlocks, 29 session rotations, 60 explicit `pending=6` cycles, 3,000 stale bridge ticket rejections, and final foreground/background/bridge-active/bridge-queued counts all equal to zero. The stress run discovered one monitor handshake reuse race, which was fixed by making monitor release acknowledgement synchronous before the next commit owner can acquire the gate.
