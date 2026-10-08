from __future__ import annotations

import math
import sys
import time

from vision_engine.diagnostics.win32 import record_win32
from vision_engine.diagnostics.win32_context import capture_boundary_snapshot

from .geometry import AutoPlayPoint
from .motion import motion_points

CURSOR_TOLERANCE_PX = 5.0


def _windows_user32():
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
    user32.GetCursorPos.restype = wintypes.BOOL
    user32.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
    user32.SetCursorPos.restype = wintypes.BOOL
    user32.GetSystemMetrics.argtypes = [ctypes.c_int]
    user32.GetSystemMetrics.restype = ctypes.c_int
    user32.SendInput.argtypes = [wintypes.UINT, ctypes.c_void_p, ctypes.c_int]
    user32.SendInput.restype = wintypes.UINT
    return user32


def _input_types():
    import ctypes
    from ctypes import wintypes

    ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", wintypes.LONG),
            ("dy", wintypes.LONG),
            ("mouseData", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ULONG_PTR),
        ]

    class INPUTUNION(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT)]

    class INPUT(ctypes.Structure):
        _anonymous_ = ("u",)
        _fields_ = [("type", wintypes.DWORD), ("u", INPUTUNION)]

    return MOUSEINPUT, INPUT


def _sendinput_move(user32, x: int, y: int) -> bool:
    import ctypes

    MOUSEINPUT, INPUT = _input_types()
    vx = int(user32.GetSystemMetrics(76))
    vy = int(user32.GetSystemMetrics(77))
    vw = max(1, int(user32.GetSystemMetrics(78)))
    vh = max(1, int(user32.GetSystemMetrics(79)))
    nx = max(0, min(65535, int(round((int(x) - vx) * 65535.0 / max(1, vw - 1)))))
    ny = max(0, min(65535, int(round((int(y) - vy) * 65535.0 / max(1, vh - 1)))))
    inp = INPUT(
        type=0,
        mi=MOUSEINPUT(nx, ny, 0, 0x0001 | 0x8000 | 0x4000, 0, 0),
    )
    return int(user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))) == 1


def _sendinput_left_click(user32) -> bool:
    import ctypes

    MOUSEINPUT, INPUT = _input_types()
    arr = (INPUT * 2)(
        INPUT(type=0, mi=MOUSEINPUT(0, 0, 0, 0x0002, 0, 0)),
        INPUT(type=0, mi=MOUSEINPUT(0, 0, 0, 0x0004, 0, 0)),
    )
    return int(user32.SendInput(2, ctypes.byref(arr), ctypes.sizeof(INPUT))) == 2


def _cursor_position(user32) -> AutoPlayPoint | None:
    import ctypes
    from ctypes import wintypes

    point = wintypes.POINT()
    if not user32.GetCursorPos(ctypes.byref(point)):
        return None
    return AutoPlayPoint(int(point.x), int(point.y))




def _cursor_distance(a: AutoPlayPoint | None, b: AutoPlayPoint) -> float:
    if a is None:
        return math.inf
    return math.hypot(float(a.x - b.x), float(a.y - b.y))


def _cursor_within_tolerance(actual: AutoPlayPoint | None, target: AutoPlayPoint, tolerance_px: float = CURSOR_TOLERANCE_PX) -> bool:
    return _cursor_distance(actual, target) <= max(0.0, float(tolerance_px))


def _probe_rejection(user32, target_step: AutoPlayPoint, tolerance_px: float) -> dict:
    """Probe the same waypoint after a hard failure without clicking.

    Classification is based on the physical cursor returned by GetCursorPos, not
    on SetCursorPos's boolean return value.
    """
    import ctypes

    from vision_engine.diagnostics.win32_probe import run_rejection_probe

    def set_cursor(x: int, y: int) -> bool:
        ctypes.set_last_error(0)
        return bool(user32.SetCursorPos(int(x), int(y)))

    def get_cursor() -> tuple[int, int] | None:
        pos = _cursor_position(user32)
        return None if pos is None else (pos.x, pos.y)

    try:
        result = run_rejection_probe(
            target=(target_step.x, target_step.y),
            set_cursor=set_cursor,
            get_cursor=get_cursor,
            get_last_error=lambda: int(ctypes.get_last_error()),
            sleep=time.sleep,
            clock=time.monotonic,
            tolerance_px=max(0, int(round(tolerance_px))),
        )
    except BaseException as exc:
        result = {"classification": "PROBE_ERROR", "error": f"{type(exc).__name__}: {exc}", "attempts": []}
    record_win32("move-recovery-probe", **result)
    return result


def windows_input_preflight() -> tuple[bool, str]:
    if not sys.platform.startswith("win"):
        return False, "Windows input is unavailable on this platform"
    try:
        import ctypes

        user32 = _windows_user32()
        cur = _cursor_position(user32)
        if cur is None:
            return False, "Windows GetCursorPos failed"
        ctypes.set_last_error(0)
        if user32.SetCursorPos(cur.x, cur.y):
            return True, "SetCursorPos"
        if _sendinput_move(user32, cur.x, cur.y):
            return True, "SendInput fallback"
        return False, "Windows blocked cursor control"
    except Exception as exc:
        return False, str(exc)


def windows_move_and_left_click(point: AutoPlayPoint, *, duration_ms: int = 240, tolerance_px: float = CURSOR_TOLERANCE_PX) -> str:
    """Move with V11-style transport; verify only the final destination.

    Intermediate waypoints are transport hints, not correctness checkpoints.  A
    transient cursor clamp/reposition must not kill Auto Play halfway through the
    path.  We therefore keep issuing the remaining waypoints and make the final
    physical cursor position the safety gate before clicking.
    """
    if not sys.platform.startswith("win"):
        raise RuntimeError("Auto Play TEST clicking is supported only on Windows")

    import ctypes

    user32 = _windows_user32()
    start = _cursor_position(user32)
    if start is None:
        raise RuntimeError("Windows GetCursorPos failed")

    duration = max(0, int(duration_ms))
    steps = max(2, int(math.ceil(duration / 16.0))) if duration else 1
    path = motion_points(start, point, steps)
    sleep_s = (duration / 1000.0) / max(1, len(path)) if duration else 0.0
    backend = "SetCursorPos"
    move_started = time.monotonic()
    api_rejections = 0
    record_win32(
        "move-begin",
        start=(start.x, start.y), target=(point.x, point.y),
        duration_ms=duration, steps=len(path), trajectory="v11-linear-smoothstep",
        verification="final-only", path=[(p.x, p.y) for p in path],
    )

    for index, target_step in enumerate(path, start=1):
        ctypes.set_last_error(0)
        moved = bool(user32.SetCursorPos(int(target_step.x), int(target_step.y)))
        last_error = int(ctypes.get_last_error())
        fallback = False
        sendinput_last_error = 0

        # V11 transport semantics: the API result controls whether we try the
        # fallback, but we do not make every intermediate pixel a fatal physical
        # checkpoint.  The next waypoint gets a chance to recover naturally.
        if not moved:
            ctypes.set_last_error(0)
            fallback = _sendinput_move(user32, int(target_step.x), int(target_step.y))
            sendinput_last_error = int(ctypes.get_last_error())
            if fallback:
                backend = "SendInput fallback"
            else:
                api_rejections += 1

        record_win32(
            "move-step" if (moved or fallback) else "move-step-api-rejected",
            step=index, total=len(path), target=(target_step.x, target_step.y),
            set_cursor_ok=moved, sendinput_ok=fallback, last_error=last_error,
            sendinput_last_error=sendinput_last_error,
            fatal=False,
            elapsed_ms=round((time.monotonic() - move_started) * 1000.0, 3),
        )
        if sleep_s:
            time.sleep(sleep_s)

    # Only the final destination is a correctness boundary.  A click must never
    # happen unless the physical cursor is actually on/near the intended cell.
    final_tolerance = max(8.0, float(tolerance_px))
    actual = _cursor_position(user32)
    distance = _cursor_distance(actual, point)
    reached = _cursor_within_tolerance(actual, point, final_tolerance)
    record_win32(
        "move-final-check",
        target=(point.x, point.y),
        actual=None if actual is None else (actual.x, actual.y),
        distance_px=distance, tolerance_px=final_tolerance,
        reached=reached, intermediate_api_rejections=api_rejections,
    )

    if not reached:
        # Keep the detailed probe, but only at the real destination.  It no
        # longer interrupts a valid path because one intermediate waypoint was
        # briefly stale/clamped.
        probe = _probe_rejection(user32, point, final_tolerance)
        actual = _cursor_position(user32)
        distance = _cursor_distance(actual, point)
        reached = _cursor_within_tolerance(actual, point, final_tolerance)
        if reached:
            backend = "delayed final recovery"
            record_win32(
                "move-final-recovered",
                target=(point.x, point.y),
                actual=None if actual is None else (actual.x, actual.y),
                distance_px=distance, tolerance_px=final_tolerance,
                probe=probe.get("classification"),
            )
        else:
            actual_xy = None if actual is None else (actual.x, actual.y)
            boundary = capture_boundary_snapshot(
                {
                    "previous": None,
                    "target": (point.x, point.y),
                    "next": None,
                    "final": (point.x, point.y),
                    "actual": actual_xy,
                },
                set_cursor_ok=False, sendinput_ok=False,
                last_error=0, sendinput_last_error=0,
                position_verified=False,
            )
            record_win32(
                "move-abort", target=(point.x, point.y), actual=actual_xy,
                distance_px=distance, tolerance_px=final_tolerance,
                probe=probe.get("classification"),
                intermediate_api_rejections=api_rejections,
                boundary=boundary,
            )
            raise RuntimeError(
                "Windows cursor movement did not reach the final click target "
                f"(wanted=({point.x},{point.y}), actual={actual_xy}, "
                f"distance={distance:.1f}px, probe={probe.get('classification')})."
            )

    click_ok = _sendinput_left_click(user32)
    record_win32(
        "click", ok=click_ok, target=(point.x, point.y), backend=backend,
        final_distance_px=distance, intermediate_api_rejections=api_rejections,
    )
    if not click_ok:
        raise RuntimeError(
            "Windows blocked the mouse click. Run the Assistant at the same privilege level as MU "
            "(often both as Administrator)."
        )
    return backend

def windows_left_click(point: AutoPlayPoint) -> None:
    windows_move_and_left_click(point, duration_ms=0)
