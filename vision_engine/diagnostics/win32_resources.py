"""GDI/USER handle telemetry and concurrent-process scan (V18.0.27 diagnostics).

Pure helpers (``gui_pressure``, ``summarize_processes``) are unit-testable off
Windows; the ``collect_*`` functions use ctypes and only run on Windows.
"""
from __future__ import annotations

from functools import lru_cache
import os
import sys
from typing import Any, Iterable

GR_GDIOBJECTS = 0
GR_USEROBJECTS = 1
GR_GDIOBJECTS_PEAK = 2
GR_USEROBJECTS_PEAK = 4

DEFAULT_GDI_QUOTA = 10000
DEFAULT_USER_QUOTA = 10000
PRESSURE_WARN_FRACTION = 0.80

PROCESS_NAME_PATTERNS = (
    "python", "pythonw", "py.exe", "wsl", "wslhost", "wslservice", "vmmem",
    "docker", "ubuntu", "main.exe", "mu", "game",
)

MU_PROCESS_PATTERNS = ("main.exe", "mu.exe", "muclient", "muonline", "mugame")


def gui_pressure(
    gdi: int | None,
    user: int | None,
    gdi_quota: int = DEFAULT_GDI_QUOTA,
    user_quota: int = DEFAULT_USER_QUOTA,
    warn_fraction: float = PRESSURE_WARN_FRACTION,
) -> dict[str, Any]:
    """Compare handle counts with the per-process quota."""
    def frac(value: int | None, quota: int) -> float | None:
        if value is None or quota <= 0:
            return None
        return round(float(value) / float(quota), 4)

    gdi_f, user_f = frac(gdi, gdi_quota), frac(user, user_quota)
    hot = [name for name, f in (("GDI", gdi_f), ("USER", user_f)) if f is not None and f >= warn_fraction]
    return {
        "gdi": gdi, "user": user,
        "gdi_quota": int(gdi_quota), "user_quota": int(user_quota),
        "gdi_fraction": gdi_f, "user_fraction": user_f,
        "pressure": bool(hot), "pressured_pools": hot,
    }


def summarize_processes(
    rows: Iterable[dict[str, Any]],
    *,
    own_pid: int,
    session_id: int | None = None,
    patterns: tuple[str, ...] = PROCESS_NAME_PATTERNS,
    top_n: int = 8,
) -> dict[str, Any]:
    """Reduce a full process table to what matters for this investigation.

    rows: {pid, ppid, name, session, gdi, user, handles, path} (counters may be None).
    Reports (a) other Python/WSL/Docker/solver/game processes and (b) the
    heaviest GDI/USER consumers in the same session, because desktop-heap and
    handle exhaustion are session-wide and the culprit may not be this app.
    """
    all_rows = [dict(r) for r in rows]
    same_session = [r for r in all_rows if session_id is None or r.get("session") == session_id]

    def matches(row: dict[str, Any]) -> bool:
        name = str(row.get("name") or "").lower()
        return any(p in name for p in patterns)

    related = [r for r in all_rows if matches(r)]
    other_python = [
        r for r in related
        if int(r.get("pid", -1)) != int(own_pid) and "python" in str(r.get("name") or "").lower()
    ]

    def weight(r: dict[str, Any], key: str) -> int:
        v = r.get(key)
        return int(v) if isinstance(v, int) else -1

    top_gdi = sorted(same_session, key=lambda r: weight(r, "gdi"), reverse=True)[:top_n]
    top_user = sorted(same_session, key=lambda r: weight(r, "user"), reverse=True)[:top_n]
    top_handles = sorted(same_session, key=lambda r: weight(r, "handles"), reverse=True)[:top_n]

    def is_mu_like(row: dict[str, Any]) -> bool:
        name = str(row.get("name") or "").lower()
        path = str(row.get("path") or "").lower()
        return any(p in name or p in path for p in MU_PROCESS_PATTERNS)

    mu_like = [r for r in same_session if is_mu_like(r)]
    def total(key: str) -> int:
        return sum(int(r.get(key) or 0) for r in mu_like if isinstance(r.get(key), int))

    return {
        "own_pid": int(own_pid),
        "session_id": session_id,
        "process_count": len(all_rows),
        "related": related,
        "other_python_pids": [int(r["pid"]) for r in other_python],
        "top_gdi": top_gdi,
        "top_user": top_user,
        "top_handles": top_handles,
        "mu_like_count": len(mu_like),
        "mu_like_totals": {"gdi": total("gdi"), "user": total("user"), "handles": total("handles")},
        "mu_like_processes": mu_like,
    }


# ----------------------------- Windows collectors -----------------------------

def _read_quota(name: str, default: int) -> int:
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Windows",
        ) as key:
            value, _kind = winreg.QueryValueEx(key, name)
            return int(value)
    except BaseException:
        return default


@lru_cache(maxsize=1)
def _quotas() -> tuple[int, int]:
    return (
        _read_quota("GDIProcessHandleQuota", DEFAULT_GDI_QUOTA),
        _read_quota("USERProcessHandleQuota", DEFAULT_USER_QUOTA),
    )


def collect_process_vitals() -> dict[str, Any]:
    """Kernel handle count and memory of the current process (cheap, no admin)."""
    if not sys.platform.startswith("win"):
        return {"error": "not-windows"}
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.GetProcessHandleCount.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.GetProcessHandleCount.restype = wintypes.BOOL
        kernel32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        kernel32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
        proc = kernel32.GetCurrentProcess()
        out: dict[str, Any] = {"pid": os.getpid()}
        count = wintypes.DWORD(0)
        if kernel32.GetProcessHandleCount(proc, ctypes.byref(count)):
            out["handles"] = int(count.value)
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        if kernel32.K32GetProcessMemoryInfo(proc, ctypes.byref(pmc), pmc.cb):
            mb = 1024.0 * 1024.0
            out["working_set_mb"] = round(pmc.WorkingSetSize / mb, 2)
            out["private_mb"] = round(pmc.PrivateUsage / mb, 2)
            out["paged_pool_kb"] = int(pmc.QuotaPagedPoolUsage // 1024)
            out["nonpaged_pool_kb"] = int(pmc.QuotaNonPagedPoolUsage // 1024)
        return out
    except BaseException as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def collect_gui_resources() -> dict[str, Any]:
    """GDI/USER handle counts (and peaks) of the current process."""
    if not sys.platform.startswith("win"):
        return {"error": "not-windows"}
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        user32.GetGuiResources.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        user32.GetGuiResources.restype = wintypes.DWORD
        proc = kernel32.GetCurrentProcess()
        gdi = int(user32.GetGuiResources(proc, GR_GDIOBJECTS))
        usr = int(user32.GetGuiResources(proc, GR_USEROBJECTS))
        gdi_quota, user_quota = _quotas()
        out = gui_pressure(gdi, usr, gdi_quota, user_quota)
        out["gdi_peak"] = int(user32.GetGuiResources(proc, GR_GDIOBJECTS_PEAK))
        out["user_peak"] = int(user32.GetGuiResources(proc, GR_USEROBJECTS_PEAK))
        return out
    except BaseException as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def collect_process_table() -> list[dict[str, Any]]:
    """Toolhelp snapshot of all processes with per-process GDI/USER counts."""
    if not sys.platform.startswith("win"):
        return []
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    user32 = ctypes.WinDLL("user32", use_last_error=True)

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD), ("szExeFile", wintypes.WCHAR * 260),
        ]

    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
    kernel32.Process32FirstW.restype = wintypes.BOOL
    kernel32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
    kernel32.Process32NextW.restype = wintypes.BOOL
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.GetProcessHandleCount.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.GetProcessHandleCount.restype = wintypes.BOOL
    kernel32.ProcessIdToSessionId.argtypes = [wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    kernel32.ProcessIdToSessionId.restype = wintypes.BOOL
    kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    user32.GetGuiResources.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    user32.GetGuiResources.restype = wintypes.DWORD

    TH32CS_SNAPPROCESS = 0x00000002
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    invalid = wintypes.HANDLE(-1).value
    snap = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if snap is None or snap == invalid:
        return []
    rows: list[dict[str, Any]] = []
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
        ok = kernel32.Process32FirstW(snap, ctypes.byref(entry))
        while ok:
            pid = int(entry.th32ProcessID)
            row: dict[str, Any] = {
                "pid": pid, "ppid": int(entry.th32ParentProcessID),
                "name": str(entry.szExeFile), "session": None,
                "gdi": None, "user": None, "handles": None, "path": None,
            }
            session = wintypes.DWORD(0)
            if kernel32.ProcessIdToSessionId(pid, ctypes.byref(session)):
                row["session"] = int(session.value)
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid) if pid else None
            if handle:
                try:
                    row["gdi"] = int(user32.GetGuiResources(handle, GR_GDIOBJECTS))
                    row["user"] = int(user32.GetGuiResources(handle, GR_USEROBJECTS))
                    hcount = wintypes.DWORD(0)
                    if kernel32.GetProcessHandleCount(handle, ctypes.byref(hcount)):
                        row["handles"] = int(hcount.value)
                    size = wintypes.DWORD(32768)
                    buf = ctypes.create_unicode_buffer(size.value)
                    if kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                        row["path"] = buf.value
                finally:
                    kernel32.CloseHandle(handle)
            rows.append(row)
            ok = kernel32.Process32NextW(snap, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snap)
    return rows


def collect_process_summary() -> dict[str, Any]:
    if not sys.platform.startswith("win"):
        return {"error": "not-windows"}
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.ProcessIdToSessionId.argtypes = [wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
        kernel32.ProcessIdToSessionId.restype = wintypes.BOOL
        session = wintypes.DWORD(0)
        sid = int(session.value) if kernel32.ProcessIdToSessionId(os.getpid(), ctypes.byref(session)) else None
        return summarize_processes(collect_process_table(), own_pid=os.getpid(), session_id=sid)
    except BaseException as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
