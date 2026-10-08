"""Process DPI awareness bootstrap (V18.0.27 diagnostics patch).

Must run before the first ``tkinter.Tk()`` and before ``mss`` is imported or
instantiated, so Tk, mss, GetCursorPos and SetCursorPos all share the same
physical-pixel coordinate space. ``mss`` otherwise changes the process DPI
awareness lazily on its first capture, after Tk's main thread already exists.
"""
from __future__ import annotations

import sys
from typing import Any

# DPI_AWARENESS_CONTEXT handles are small negative sentinel values.
DPI_CONTEXT_UNAWARE = -1
DPI_CONTEXT_SYSTEM_AWARE = -2
DPI_CONTEXT_PER_MONITOR_AWARE = -3
DPI_CONTEXT_PER_MONITOR_AWARE_V2 = -4

ERROR_ACCESS_DENIED = 5  # returned when awareness was already fixed (manifest/earlier call)

AWARENESS_LABELS = {0: "unaware", 1: "system", 2: "per-monitor"}

_STARTUP: dict[str, Any] = {"attempted": False}


def startup_state() -> dict[str, Any]:
    """Result of the one-time bootstrap call (safe to embed in diagnostics)."""
    return dict(_STARTUP)


def enable_dpi_awareness() -> dict[str, Any]:
    """Request Per-Monitor-V2 awareness once; never raises."""
    if _STARTUP.get("attempted"):
        return startup_state()
    _STARTUP.update(attempted=True, ok=False, method="none", errors=[])
    if not sys.platform.startswith("win"):
        _STARTUP["method"] = "not-windows"
        return startup_state()

    errors: list[str] = _STARTUP["errors"]
    try:
        import ctypes

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        try:
            fn = user32.SetProcessDpiAwarenessContext
            fn.argtypes = [ctypes.c_void_p]
            fn.restype = ctypes.c_int
            ctypes.set_last_error(0)
            if fn(ctypes.c_void_p(DPI_CONTEXT_PER_MONITOR_AWARE_V2)):
                _STARTUP.update(ok=True, method="SetProcessDpiAwarenessContext(PER_MONITOR_V2)")
                return startup_state()
            err = int(ctypes.get_last_error())
            errors.append(f"SetProcessDpiAwarenessContext:err={err}")
            if err == ERROR_ACCESS_DENIED:
                # Awareness was already fixed by a manifest or earlier call. Do not
                # assume it is PER_MONITOR_V2; record the actual calling-thread context.
                _STARTUP.update(ok=True, method="already-set-before-bootstrap")
                try:
                    user32.GetThreadDpiAwarenessContext.restype = ctypes.c_void_p
                    user32.GetAwarenessFromDpiAwarenessContext.argtypes = [ctypes.c_void_p]
                    user32.GetAwarenessFromDpiAwarenessContext.restype = ctypes.c_int
                    ctx = user32.GetThreadDpiAwarenessContext()
                    awareness = int(user32.GetAwarenessFromDpiAwarenessContext(ctypes.c_void_p(ctx)))
                    _STARTUP["actual_awareness"] = awareness
                    _STARTUP["actual_awareness_label"] = AWARENESS_LABELS.get(awareness, "invalid")
                    try:
                        user32.AreDpiAwarenessContextsEqual.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
                        user32.AreDpiAwarenessContextsEqual.restype = ctypes.c_int
                        _STARTUP["actual_per_monitor_v2"] = bool(
                            user32.AreDpiAwarenessContextsEqual(
                                ctypes.c_void_p(ctx), ctypes.c_void_p(DPI_CONTEXT_PER_MONITOR_AWARE_V2)
                            )
                        )
                    except AttributeError:
                        _STARTUP["actual_per_monitor_v2"] = None
                except BaseException as verify_exc:
                    errors.append(f"verify-existing-context:{type(verify_exc).__name__}:{verify_exc}")
                return startup_state()
        except AttributeError:
            errors.append("SetProcessDpiAwarenessContext:unavailable")

        try:  # Windows 8.1 fallback
            shcore = ctypes.WinDLL("shcore", use_last_error=True)
            shcore.SetProcessDpiAwareness.argtypes = [ctypes.c_int]
            shcore.SetProcessDpiAwareness.restype = ctypes.c_long
            hr = int(shcore.SetProcessDpiAwareness(2))
            if hr == 0:
                _STARTUP.update(ok=True, method="SetProcessDpiAwareness(PER_MONITOR)")
                return startup_state()
            errors.append(f"SetProcessDpiAwareness:hr={hr & 0xFFFFFFFF:#x}")
        except (AttributeError, OSError) as exc:
            errors.append(f"SetProcessDpiAwareness:{type(exc).__name__}")

        try:  # Vista+ system-aware last resort
            user32.SetProcessDPIAware.restype = ctypes.c_int
            if user32.SetProcessDPIAware():
                _STARTUP.update(ok=True, method="SetProcessDPIAware(SYSTEM)")
                return startup_state()
            errors.append("SetProcessDPIAware:false")
        except AttributeError:
            errors.append("SetProcessDPIAware:unavailable")
    except BaseException as exc:  # bootstrap must never stop the app from starting
        errors.append(f"{type(exc).__name__}: {exc}")
    return startup_state()


def thread_dpi_report() -> dict[str, Any]:
    """DPI awareness of the *calling thread* (call it from the Tk thread)."""
    out: dict[str, Any] = {"startup": startup_state()}
    if not sys.platform.startswith("win"):
        out["error"] = "not-windows"
        return out
    try:
        import ctypes

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.GetThreadDpiAwarenessContext.restype = ctypes.c_void_p
        user32.GetAwarenessFromDpiAwarenessContext.argtypes = [ctypes.c_void_p]
        user32.GetAwarenessFromDpiAwarenessContext.restype = ctypes.c_int
        ctx = user32.GetThreadDpiAwarenessContext()
        awareness = int(user32.GetAwarenessFromDpiAwarenessContext(ctypes.c_void_p(ctx)))
        out["thread_awareness"] = awareness
        out["thread_awareness_label"] = AWARENESS_LABELS.get(awareness, "invalid")
        try:
            user32.AreDpiAwarenessContextsEqual.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            user32.AreDpiAwarenessContextsEqual.restype = ctypes.c_int
            out["per_monitor_v2"] = bool(
                user32.AreDpiAwarenessContextsEqual(
                    ctypes.c_void_p(ctx), ctypes.c_void_p(DPI_CONTEXT_PER_MONITOR_AWARE_V2)
                )
            )
        except AttributeError:
            out["per_monitor_v2"] = None
        try:
            user32.GetDpiForSystem.restype = ctypes.c_uint
            out["system_dpi"] = int(user32.GetDpiForSystem())
        except AttributeError:
            out["system_dpi"] = None
    except BaseException as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out
