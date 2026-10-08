from __future__ import annotations

from typing import Any


def _integrity_label(rid: int | None) -> str:
    if rid is None:
        return "unknown"
    if rid < 0x1000:
        return "untrusted"
    if rid < 0x2000:
        return "low"
    if rid < 0x3000:
        return "medium"
    if rid < 0x4000:
        return "high"
    if rid < 0x5000:
        return "system"
    return "protected"


def _integrity_rank(label: str) -> int:
    return {
        "unknown": -1,
        "untrusted": 0,
        "low": 1,
        "medium": 2,
        "high": 3,
        "system": 4,
        "protected": 5,
    }.get(str(label), -1)


def _inside(rect: list[int] | tuple[int, ...] | None, point: tuple[int, int] | None) -> bool | None:
    if not rect or len(rect) != 4 or point is None:
        return None
    x, y = point
    left, top, right, bottom = [int(v) for v in rect]
    return left <= x < right and top <= y < bottom


def _window_key(info: dict[str, Any] | None) -> tuple[int, int]:
    if not isinstance(info, dict):
        return (0, 0)
    return (int(info.get("root_hwnd") or 0), int(info.get("pid") or 0))


def classify_boundary_snapshot(snapshot: dict[str, Any]) -> tuple[str, list[str]]:
    """Classify a captured Windows cursor failure without calling Win32 APIs."""
    details: list[str] = []
    points = snapshot.get("points", {}) if isinstance(snapshot, dict) else {}
    target = points.get("target") if isinstance(points, dict) else None
    previous = points.get("previous") if isinstance(points, dict) else None
    actual = points.get("actual") if isinstance(points, dict) else None
    target_xy = tuple(snapshot.get("target_xy", ())) if snapshot.get("target_xy") else None
    actual_xy = tuple(snapshot.get("actual_xy", ())) if snapshot.get("actual_xy") else None
    clip_rect = snapshot.get("clip_rect")

    process_desktop = str(snapshot.get("process_desktop") or "")
    input_desktop = str(snapshot.get("input_desktop") or "")
    if process_desktop and input_desktop and process_desktop != input_desktop:
        details.append(f"Process desktop={process_desktop!r}, input desktop={input_desktop!r}.")
        return "INPUT_DESKTOP_MISMATCH", details

    target_inside = _inside(clip_rect, target_xy if len(target_xy or ()) == 2 else None)
    actual_inside = _inside(clip_rect, actual_xy if len(actual_xy or ()) == 2 else None)
    if target_inside is False:
        details.append(f"Failed target {target_xy} lies outside ClipCursor rectangle {clip_rect}.")
        if actual_inside is not False:
            details.append(f"Actual cursor {actual_xy} remained inside the clip rectangle.")
        return "CURSOR_CLIPPED", details

    assistant = snapshot.get("assistant", {})
    target_process = target.get("process", {}) if isinstance(target, dict) else {}
    assistant_integrity = str(assistant.get("integrity") or "unknown")
    target_integrity = str(target_process.get("integrity") or "unknown")
    if (
        _integrity_rank(target_integrity) > _integrity_rank(assistant_integrity) >= 0
        and not bool(snapshot.get("sendinput_ok"))
    ):
        details.append(
            f"Target integrity={target_integrity}, Assistant integrity={assistant_integrity}; "
            "SendInput returned 0."
        )
        details.append("This is consistent with a UIPI/integrity-level input boundary.")
        return "UIPI_INTEGRITY_MISMATCH", details

    gui = snapshot.get("gui_resources")
    if (
        isinstance(gui, dict)
        and gui.get("pressure")
        and not bool(snapshot.get("set_cursor_ok"))
        and not bool(snapshot.get("sendinput_ok"))
    ):
        details.append(
            f"Handle pressure in {', '.join(gui.get('pressured_pools', []))}: "
            f"GDI {gui.get('gdi')}/{gui.get('gdi_quota')}, USER {gui.get('user')}/{gui.get('user_quota')}."
        )
        details.append("Both Win32 move methods failed while the process is near its GDI/USER handle quota.")
        return "GUI_RESOURCE_PRESSURE", details

    prev_key = _window_key(previous)
    target_key = _window_key(target)
    if prev_key != (0, 0) and target_key != (0, 0) and prev_key != target_key:
        details.append(f"Previous waypoint root/pid={prev_key}; failed waypoint root/pid={target_key}.")
        details.append("The first SetCursorPos failure coincides with crossing a window/process boundary.")
        return "WINDOW_BOUNDARY_CORRELATED", details

    if snapshot.get("position_verified") is False and (
        bool(snapshot.get("set_cursor_ok")) or bool(snapshot.get("sendinput_ok"))
    ):
        details.append(
            "A Win32 movement API reported success, but GetCursorPos remained outside the physical tolerance."
        )
        details.append("This is consistent with immediate repositioning/interception or a coordinate-context mismatch.")
        return "CURSOR_POSITION_DIVERGED", details

    if not bool(snapshot.get("set_cursor_ok")) and bool(snapshot.get("sendinput_ok")):
        details.append("SetCursorPos failed but SendInput reported success at this waypoint.")
        return "SETCURSORPOS_REJECTED_SENDINPUT_WORKED", details

    if not bool(snapshot.get("set_cursor_ok")) and not bool(snapshot.get("sendinput_ok")):
        if target_key != (0, 0):
            details.append(f"Both Win32 move methods failed over root/pid={target_key}.")
        details.append("No clip, desktop, integrity, or immediate window-boundary explanation was proven.")
        dpi = snapshot.get("dpi")
        if isinstance(dpi, dict) and dpi.get("thread_awareness") == 0:
            details.append("Failing thread is DPI-UNAWARE: cursor coordinates may be virtualized.")
        return "WIN32_INPUT_REJECTED", details

    details.append("Snapshot does not represent a failed move.")
    return "NO_FAILURE", details
