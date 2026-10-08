from __future__ import annotations

import ast
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "vision_engine"

CRITICAL_DIRS = (PKG / "live", PKG / "input", PKG / "solver", PKG / "diagnostics")
RUNTIME_FILES = {
    "data/pip-install.log",
    "data/wsl-solver.log",
    "data/bingo.sqlite3",
}


def py_modules() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in PKG.rglob("*.py"):
        rel = p.relative_to(PKG).with_suffix("")
        parts = list(rel.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        name = "vision_engine" + ("." + ".".join(parts) if parts else "")
        out[name] = p
    return out


def local_import_edges(modules: dict[str, Path]) -> dict[str, set[str]]:
    edges: dict[str, set[str]] = defaultdict(set)
    for name, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        package_parts = name.split(".")[:-1]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("vision_engine"):
                        edges[name].add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = package_parts[: max(0, len(package_parts) - node.level + 1)]
                    target = ".".join(base + ([node.module] if node.module else []))
                else:
                    target = node.module or ""
                if target.startswith("vision_engine"):
                    edges[name].add(target)
    return edges


def import_cycles(modules: dict[str, Path], edges: dict[str, set[str]]) -> list[list[str]]:
    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    low: dict[str, int] = {}
    cycles: list[list[str]] = []

    def visit(v: str) -> None:
        nonlocal index
        indices[v] = low[v] = index
        index += 1
        stack.append(v)
        on_stack.add(v)
        for w in edges.get(v, ()):
            if w not in modules:
                continue
            if w not in indices:
                visit(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                low[v] = min(low[v], indices[w])
        if low[v] == indices[v]:
            comp: list[str] = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                comp.append(w)
                if w == v:
                    break
            if len(comp) > 1:
                cycles.append(comp)

    for module in modules:
        if module not in indices:
            visit(module)
    return cycles


def line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8", errors="replace").splitlines())


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    modules = py_modules()
    edges = local_import_edges(modules)

    cycles = import_cycles(modules, edges)
    if cycles:
        errors.extend("import cycle: " + " -> ".join(c) for c in cycles)

    gui = (PKG / "gui.py").read_text(encoding="utf-8")
    if "windows_move_and_left_click" in gui:
        errors.append("gui.py bypasses LiveInputExecutor with direct Win32 movement")

    for root in CRITICAL_DIRS:
        for path in root.glob("*.py"):
            count = line_count(path)
            if count > 300:
                errors.append(f"critical module >300 lines: {path.relative_to(ROOT)} ({count})")

    hot_roots = [PKG / "live", PKG / "input"]
    disk_calls = {"write_text", "write_bytes", "write", "dump", "save"}
    for root in hot_roots:
        for path in root.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                fn = node.func
                if isinstance(fn, ast.Name) and fn.id == "open":
                    errors.append(f"disk open in hot path: {path.relative_to(ROOT)}:{node.lineno}")
                elif isinstance(fn, ast.Attribute) and fn.attr in disk_calls:
                    # AdmissionGate.open() is synchronization, not file I/O.
                    if fn.attr == "open" and path.name in {"solver_admission.py", "input_exclusive.py", "solver_bridge.py"}:
                        continue
                    if fn.attr in {"write_text", "write_bytes", "dump", "save"}:
                        errors.append(f"disk-like write in hot path: {path.relative_to(ROOT)}:{node.lineno}")

    for rel in RUNTIME_FILES:
        if (ROOT / rel).exists():
            errors.append(f"runtime state bundled in release: {rel}")
    if (ROOT / ".venv").exists():
        errors.append(".venv bundled in release")

    stale = []
    for rel in ["run_windows.bat", "setup_windows.bat", "setup_wsl_solver.bat"]:
        text = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
        for marker in ["V18.0.11", "V7.0 -", "V7.0.4", "requirements_ok_v7", "V7 will use"]:
            if marker in text:
                stale.append(f"{rel}: {marker}")
    if stale:
        errors.extend("stale launcher branding: " + item for item in stale)

    for path in PKG.glob("*.py"):
        count = line_count(path)
        if count > 300:
            warnings.append(f"legacy module >300 lines: {path.relative_to(ROOT)} ({count})")

    print(f"modules={len(modules)} import_cycles={len(cycles)}")
    print("critical_modules_over_300=0" if not any("critical module" in e for e in errors) else "critical_modules_over_300=FAIL")
    for warning in warnings:
        print("WARN", warning)
    for error in errors:
        print("ERROR", error)
    if errors:
        return 1
    print("V18.0.26 architecture/package audit: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
