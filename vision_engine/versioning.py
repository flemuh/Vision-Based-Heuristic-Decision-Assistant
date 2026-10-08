"""Central compatibility/version contract for the V18 line.

These constants deliberately live in one module so UI, persistence and the WSL
solver cannot silently drift apart during future patches.
"""
APP_VERSION = "18.0.26-STATE-INTEGRITY-P1"
PROTOCOL_VERSION = 4
SCHEMA_VERSION = 9
MODEL_VERSION = 6
STRATEGY_PLANNER_VERSION = 22
