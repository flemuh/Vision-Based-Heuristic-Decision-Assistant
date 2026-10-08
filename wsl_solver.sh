#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
PORT="${JEWEL_SOLVER_PORT:-57641}"
WORKERS="${JEWEL_SOLVER_WORKERS:-6}"
exec python3 -m vision_engine.solver.service --host 0.0.0.0 --port "$PORT" --workers "$WORKERS"
