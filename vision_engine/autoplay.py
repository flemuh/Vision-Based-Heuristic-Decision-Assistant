"""Compatibility facade for Auto Play helpers.

V18.0.26 moved geometry, pacing, trajectory and Win32 input into focused modules.
The public imports remain stable for older callers/tests.
"""
from __future__ import annotations

import sys

from .input.geometry import AutoPlayPoint, ScreenRegionLike, cell_center, safe_cell_point
from .input.motion import bezier_motion_points, linear_motion_points, motion_points
from .input.pacing import bounded_random_ms, paced_preclick_delay_ms
from .input.win32 import (
    _sendinput_left_click,
    _sendinput_move,
    _windows_user32,
    windows_input_preflight,
    windows_left_click,
    windows_move_and_left_click,
)

# Compatibility strings intentionally remain visible here for diagnostics/tests:
# backend: "SendInput fallback"
# privilege guidance: "same privilege level as MU"

__all__ = [
    "AutoPlayPoint",
    "ScreenRegionLike",
    "cell_center",
    "safe_cell_point",
    "bounded_random_ms",
    "paced_preclick_delay_ms",
    "linear_motion_points",
    "bezier_motion_points",
    "motion_points",
    "windows_input_preflight",
    "windows_move_and_left_click",
    "windows_left_click",
]
