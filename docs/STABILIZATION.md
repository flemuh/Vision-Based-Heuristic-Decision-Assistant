# V18 stabilization gates

A V18 release must preserve the frozen mathematical behavior and pass:

- unit/integration tests;
- golden real-session regressions;
- protocol/version checks;
- schema migration + backup checks;
- state/result-screen lifecycle checks;
- Strategy Planner route-lock/fallback checks;
- Learning/Pattern trust-gate checks;
- replay/export/telemetry checks;
- release validator from an extracted ZIP.

UI-only changes must not change solver output for the same state/seed.
