from __future__ import annotations

import os
import sys
import time
from typing import Any

from .win32_boundary import _integrity_label, classify_boundary_snapshot


def capture_boundary_snapshot(
    points: dict[str, tuple[int, int] | None],
    *,
    set_cursor_ok: bool,
    sendinput_ok: bool,
    last_error: int,
    sendinput_last_error: int,
    position_verified: bool | None = None,
) -> dict[str, Any]:
    """Best-effort heavy Win32 snapshot, intended to run only after a move failure."""
    if not sys.platform.startswith("win"):
        return {"platform": sys.platform, "error": "Windows-only snapshot"}
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)

        user32.GetForegroundWindow.restype = wintypes.HWND
        user32.WindowFromPoint.argtypes = [wintypes.POINT]
        user32.WindowFromPoint.restype = wintypes.HWND
        user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
        user32.GetAncestor.restype = wintypes.HWND
        user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        user32.GetWindowTextLengthW.restype = ctypes.c_int
        user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        user32.GetWindowTextW.restype = ctypes.c_int
        user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        user32.GetClassNameW.restype = ctypes.c_int
        user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        user32.GetWindowThreadProcessId.restype = wintypes.DWORD
        user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        user32.GetWindowRect.restype = wintypes.BOOL
        user32.IsWindowVisible.argtypes = [wintypes.HWND]
        user32.IsWindowVisible.restype = wintypes.BOOL
        user32.GetClipCursor.argtypes = [ctypes.POINTER(wintypes.RECT)]
        user32.GetClipCursor.restype = wintypes.BOOL
        user32.GetThreadDesktop.argtypes = [wintypes.DWORD]
        user32.GetThreadDesktop.restype = wintypes.HANDLE
        user32.OpenInputDesktop.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        user32.OpenInputDesktop.restype = wintypes.HANDLE
        user32.CloseDesktop.argtypes = [wintypes.HANDLE]
        user32.CloseDesktop.restype = wintypes.BOOL
        user32.GetProcessWindowStation.restype = wintypes.HANDLE
        user32.GetUserObjectInformationW.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
        user32.GetUserObjectInformationW.restype = wintypes.BOOL
        user32.GetSystemMetrics.argtypes = [ctypes.c_int]
        user32.GetSystemMetrics.restype = ctypes.c_int

        kernel32.GetCurrentThreadId.restype = wintypes.DWORD
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
        kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
        advapi32.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
        advapi32.OpenProcessToken.restype = wintypes.BOOL
        advapi32.GetTokenInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
        advapi32.GetTokenInformation.restype = wintypes.BOOL
        advapi32.GetSidSubAuthorityCount.argtypes = [wintypes.LPVOID]
        advapi32.GetSidSubAuthorityCount.restype = ctypes.POINTER(ctypes.c_ubyte)
        advapi32.GetSidSubAuthority.argtypes = [wintypes.LPVOID, wintypes.DWORD]
        advapi32.GetSidSubAuthority.restype = ctypes.POINTER(wintypes.DWORD)

        class SID_AND_ATTRIBUTES(ctypes.Structure):
            _fields_ = [("Sid", wintypes.LPVOID), ("Attributes", wintypes.DWORD)]

        class TOKEN_MANDATORY_LABEL(ctypes.Structure):
            _fields_ = [("Label", SID_AND_ATTRIBUTES)]

        class CURSORINFO(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.DWORD),
                ("flags", wintypes.DWORD),
                ("hCursor", wintypes.HANDLE),
                ("ptScreenPos", wintypes.POINT),
            ]

        user32.GetCursorInfo.argtypes = [ctypes.POINTER(CURSORINFO)]
        user32.GetCursorInfo.restype = wintypes.BOOL

        def object_name(handle: int | None) -> str | None:
            if not handle:
                return None
            needed = wintypes.DWORD(0)
            user32.GetUserObjectInformationW(handle, 2, None, 0, ctypes.byref(needed))
            if not needed.value:
                return None
            buf = ctypes.create_unicode_buffer(max(2, needed.value // ctypes.sizeof(ctypes.c_wchar) + 1))
            if not user32.GetUserObjectInformationW(handle, 2, buf, needed.value, ctypes.byref(needed)):
                return None
            return buf.value

        def process_info(pid: int) -> dict[str, Any]:
            out: dict[str, Any] = {"pid": int(pid), "path": None, "integrity": "unknown", "integrity_rid": None}
            if pid <= 0:
                return out
            handle = kernel32.OpenProcess(0x1000, False, int(pid))
            if not handle:
                out["open_error"] = int(ctypes.get_last_error())
                return out
            try:
                size = wintypes.DWORD(32768)
                buf = ctypes.create_unicode_buffer(size.value)
                if kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                    out["path"] = buf.value
                token = wintypes.HANDLE()
                if advapi32.OpenProcessToken(handle, 0x0008, ctypes.byref(token)):
                    try:
                        needed = wintypes.DWORD(0)
                        advapi32.GetTokenInformation(token, 25, None, 0, ctypes.byref(needed))
                        if needed.value:
                            raw = ctypes.create_string_buffer(needed.value)
                            if advapi32.GetTokenInformation(token, 25, raw, needed, ctypes.byref(needed)):
                                label = ctypes.cast(raw, ctypes.POINTER(TOKEN_MANDATORY_LABEL)).contents
                                sid = label.Label.Sid
                                count_ptr = advapi32.GetSidSubAuthorityCount(sid)
                                if count_ptr:
                                    count = int(count_ptr.contents.value)
                                    if count > 0:
                                        rid_ptr = advapi32.GetSidSubAuthority(sid, count - 1)
                                        if rid_ptr:
                                            rid = int(rid_ptr.contents.value)
                                            out["integrity_rid"] = rid
                                            out["integrity"] = _integrity_label(rid)
                    finally:
                        kernel32.CloseHandle(token)
            finally:
                kernel32.CloseHandle(handle)
            return out

        def window_info(hwnd: int | None) -> dict[str, Any] | None:
            if not hwnd:
                return None
            root = int(user32.GetAncestor(hwnd, 2) or hwnd)
            pid = wintypes.DWORD(0)
            tid = int(user32.GetWindowThreadProcessId(root, ctypes.byref(pid)))
            n = max(0, int(user32.GetWindowTextLengthW(root)))
            title = ctypes.create_unicode_buffer(max(2, n + 1))
            user32.GetWindowTextW(root, title, len(title))
            cls = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(root, cls, len(cls))
            rect = wintypes.RECT()
            rect_value = None
            if user32.GetWindowRect(root, ctypes.byref(rect)):
                rect_value = [int(rect.left), int(rect.top), int(rect.right), int(rect.bottom)]
            return {
                "hwnd": int(hwnd),
                "root_hwnd": root,
                "title": title.value,
                "class": cls.value,
                "pid": int(pid.value),
                "tid": tid,
                "visible": bool(user32.IsWindowVisible(root)),
                "rect": rect_value,
                "process": process_info(int(pid.value)),
            }

        def point_info(xy: tuple[int, int] | None) -> dict[str, Any] | None:
            if xy is None:
                return None
            x, y = int(xy[0]), int(xy[1])
            info = window_info(user32.WindowFromPoint(wintypes.POINT(x, y)))
            return {"xy": [x, y], **(info or {})}

        clip = wintypes.RECT()
        clip_rect = None
        if user32.GetClipCursor(ctypes.byref(clip)):
            clip_rect = [int(clip.left), int(clip.top), int(clip.right), int(clip.bottom)]

        cursor = CURSORINFO(cbSize=ctypes.sizeof(CURSORINFO))
        cursor_info = None
        if user32.GetCursorInfo(ctypes.byref(cursor)):
            cursor_info = {
                "flags": int(cursor.flags),
                "hcursor": int(cursor.hCursor or 0),
                "xy": [int(cursor.ptScreenPos.x), int(cursor.ptScreenPos.y)],
            }

        process_desktop = object_name(user32.GetThreadDesktop(kernel32.GetCurrentThreadId()))
        input_handle = user32.OpenInputDesktop(0, False, 0x0001)
        input_desktop = None
        if input_handle:
            try:
                input_desktop = object_name(input_handle)
            finally:
                user32.CloseDesktop(input_handle)

        foreground = window_info(user32.GetForegroundWindow())
        point_rows = {name: point_info(xy) for name, xy in points.items()}
        assistant = process_info(os.getpid())
        virtual_screen = [
            int(user32.GetSystemMetrics(76)),
            int(user32.GetSystemMetrics(77)),
            int(user32.GetSystemMetrics(78)),
            int(user32.GetSystemMetrics(79)),
        ]
        target_xy = points.get("target")
        actual_xy = points.get("actual")
        snapshot: dict[str, Any] = {
            "captured_monotonic": time.monotonic(),
            "assistant": assistant,
            "foreground": foreground,
            "points": point_rows,
            "target_xy": list(target_xy) if target_xy else None,
            "actual_xy": list(actual_xy) if actual_xy else None,
            "clip_rect": clip_rect,
            "cursor_info": cursor_info,
            "process_desktop": process_desktop,
            "input_desktop": input_desktop,
            "window_station": object_name(user32.GetProcessWindowStation()),
            "virtual_screen": virtual_screen,
            "set_cursor_ok": bool(set_cursor_ok),
            "sendinput_ok": bool(sendinput_ok),
            "position_verified": position_verified,
            "last_error": int(last_error),
            "sendinput_last_error": int(sendinput_last_error),
        }
        # V18.0.27 diagnostics patch: DPI context of THIS thread, handle pressure
        # and the session's process/handle landscape at the instant of failure.
        from vision_engine.dpi import thread_dpi_report
        from .win32_resources import collect_gui_resources, collect_process_summary

        snapshot["dpi"] = thread_dpi_report()
        snapshot["gui_resources"] = collect_gui_resources()
        snapshot["processes"] = collect_process_summary()
        classification, details = classify_boundary_snapshot(snapshot)
        snapshot["classification"] = classification
        snapshot["analysis"] = details
        return snapshot
    except BaseException as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "platform": sys.platform}
