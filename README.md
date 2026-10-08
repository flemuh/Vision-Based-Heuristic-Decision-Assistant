# Vision-Based Heuristic Decision Assistant

A desktop research project that combines **computer vision, mathematical optimization, and probabilistic decision-making** to analyze a visual 5×5 board and recommend strategic moves. Built around a game-based case study.

## Technologies

- **Python 3.11+**, OpenCV, NumPy, MSS, Pillow, Tkinter
- **SQLite** for game history, decisions, and replay
- **Ubuntu/WSL** for the persistent solver; Docker for optional offline benchmarks

## Core Concepts

- **Computer Vision:** screen capture, board calibration, feature matching, and state recognition
- **Heuristic Search & Optimization:** route feasibility, constrained allocation, and move ranking
- **Probabilistic Modeling:** Monte Carlo simulation and Expectimax planning
- **State Consistency:** confidence checks, stale-result rejection, and transition validation
- **Applied Machine Learning:** auxiliary local learning from historical decisions and retrospective outcomes

## Run Locally (Windows)

1. Install **Python 3.11+** and **WSL with Ubuntu**.
2. Run `setup_windows.bat` to configure dependencies; follow any WSL setup prompts.
3. Launch using `run_windows.bat`.
4. In the application, complete **Auto Setup**, start monitoring, and inspect move recommendations in **Live**.

The solver runs through WSL. Docker is **not required** for normal operation. Optional Auto Play is experimental and should not be treated as a guaranteed stable feature.

## Repository structure

- `vision_engine/` — internal Python package for vision and decision logic
- `tools/` and `tests/` — research utilities and regression checks
- `data/` and `assets/` — sample board configurations and vision bootstrap data
- `docs/` — technical notes and experiment history

The internal package was renamed to `vision_engine`; entry points and imports have been updated.

## Architecture

`Windows screen capture → visual state recognition → WSL mathematical solver → ranked recommendation → UI / SQLite history`

See [ARCHITECTURE.md](ARCHITECTURE.md) for the internal design and [RESEARCH.md](RESEARCH.md) for experiment details.
