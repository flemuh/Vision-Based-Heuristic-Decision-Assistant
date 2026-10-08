from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

EXCLUDE_DIRS = {".venv", ".pytest_cache", "__pycache__", ".git", ".idea", ".vscode"}
EXCLUDE_FILES = {
    "data/pip-install.log", "data/wsl-solver.log", "data/bingo.sqlite3",
}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}


def include(root: Path, path: Path) -> bool:
    rel = path.relative_to(root).as_posix()
    if any(part in EXCLUDE_DIRS for part in path.relative_to(root).parts):
        return False
    if rel in EXCLUDE_FILES or path.suffix in EXCLUDE_SUFFIXES:
        return False
    if rel.startswith(("data/logs/", "data/backups/", "data/game_images/", "data/decision_images/", "data/exports/")):
        return path.name == ".gitkeep"
    return True


def build(root: Path, output: Path, folder_name: str) -> None:
    root = root.resolve()
    output = output.resolve()
    with tempfile.TemporaryDirectory() as td:
        staged = Path(td) / folder_name
        staged.mkdir(parents=True)
        for path in root.rglob("*"):
            if not include(root, path):
                continue
            rel = path.relative_to(root)
            target = staged / rel
            if path.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for path in sorted(staged.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(staged.parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path.cwd())
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--folder-name", required=True)
    args = ap.parse_args()
    build(args.root, args.output, args.folder_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
