from __future__ import annotations

import argparse
import json

from _bootstrap import ROOT  # noqa: F401
from vision_engine.diagnostics.fence_simulation import run_fence_simulation


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress the V18.0.26 global solver/input fence")
    parser.add_argument("--cycles", type=int, default=1000)
    args = parser.parse_args()
    report = run_fence_simulation(args.cycles)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    if report.failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
