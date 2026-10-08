# Vision-Based Heuristic Decision Assistant

A Python desktop application that combines **computer vision, heuristic optimization, and probabilistic modeling** to analyze visual board states and recommend strategic decisions.

Developed as an applied research project using a 5×5 game-based environment to explore visual recognition, mathematical search, decision optimization, and automated analysis.

## Key Features

- **Visual Recognition** — Screen capture, board detection, calibration, and state recognition.
- **Heuristic Optimization** — Evaluate possible moves using constraints, priorities, and strategic objectives.
- **Probabilistic Analysis** — Monte Carlo simulations and Expectimax-based decision planning.
- **Decision Support** — Rank potential moves and present recommendations through a desktop interface.
- **Experiment Tracking** — Store game states, decisions, and outcomes for analysis and replay.
- **State Validation** — Confidence checks and consistency verification between observations and decisions.

## Tech Stack

- **Language:** Python 3.11+
- **Computer Vision:** OpenCV, NumPy, Pillow, MSS
- **Desktop Interface:** Tkinter
- **Database:** SQLite
- **Environment:** Windows, Ubuntu/WSL
- **Research Tools:** Docker (optional)

## Core Concepts

Computer Vision · Heuristic Search · Combinatorial Optimization · Monte Carlo Simulation · Expectimax · Probability · State Management · Applied Machine Learning

The project primarily uses mathematical optimization and heuristic decision-making, with auxiliary learning experiments based on historical and synthetic data.

## Getting Started

**Requirements:** Windows, Python 3.11+, and WSL with Ubuntu.

1. Clone or download the repository.
2. Run `setup_windows.bat` to install dependencies and configure the environment.
3. Launch the application using `run_windows.bat`.
4. Complete **Auto Setup** to calibrate the visual board.
5. Start monitoring and explore recommendations in **Live** mode.

The solver operates through WSL. Docker is not required for normal use.

## Project Structure

- `vision_engine/` — Computer vision, decision logic, and optimization engine.
- `tools/` — Research scripts, simulations, and evaluation utilities.
- `tests/` — Automated tests and regression checks.
- `data/` and `assets/` — Board configurations and visual reference data.
- `docs/` — Additional technical documentation.

## Architecture

**Screen Capture → Visual Recognition → Mathematical Solver → Decision Ranking → Desktop Interface & SQLite**

## Research & Development

This project explores how visual perception, mathematical optimization, and probabilistic reasoning can support decisions in constrained environments.

It is an experimental research application. Automated interaction features, including Auto Play, are not guaranteed to operate reliably in every environment.

For more details, see [Architecture](ARCHITECTURE.md) and [Research](RESEARCH.md).
