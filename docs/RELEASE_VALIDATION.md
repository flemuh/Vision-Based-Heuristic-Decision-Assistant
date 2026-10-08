# V18 Final Release Validation

Validated after the five stabilization phases.

- Collected tests: **139**.
- Complete regression executed in three fresh pytest processes: all passed.
- Final offline release validator: passed.
- Python compile check for `vision_engine` and `tools`: passed.
- Final local live-architecture benchmark in the release environment:
  - cold decision: 1198.4 ms;
  - hot cache: 0.97 ms;
  - six-jewel precompute ready: 1255.0 ms;
  - next-jewel cache: 0.82 ms.

These benchmark numbers are environment-specific and are not a promise of Windows/WSL screen-to-screen latency. V18 also records real live P50/P95/P99 telemetry during play so the installed system can be judged from its own sessions.

The release ZIP is additionally extracted into a clean directory and checked with the release validator and golden/final regressions before publication.
