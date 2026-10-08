# Docker solver (offline/benchmark only)

The V7 live path uses Ubuntu/WSL directly to minimize IPC layers. Docker is kept
for reproducible replay, CI and benchmark runs. It is not required for MU live play.
