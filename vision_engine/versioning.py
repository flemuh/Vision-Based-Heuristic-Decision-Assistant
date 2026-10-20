"""Central compatibility/version contract for the V18 line.

These constants deliberately live in one module so UI, persistence and the WSL
solver cannot silently drift apart during future patches.
"""
APP_VERSION = "18.0.26-ML999-R3.9-NATIVE-COMPILED-PLAN"
PROTOCOL_VERSION = 5
SCHEMA_VERSION = 10
MODEL_VERSION = 6
STRATEGY_PLANNER_VERSION = 24
