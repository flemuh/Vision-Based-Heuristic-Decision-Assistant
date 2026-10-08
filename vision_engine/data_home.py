from __future__ import annotations

import os
import shutil
from pathlib import Path


def default_user_data_dir() -> Path:
    override = os.environ.get("SPEEDLORA_JEWEL_DATA_DIR")
    if override:
        return Path(override).expanduser()
    local = os.environ.get("LOCALAPPDATA")
    if local:
        return Path(local) / "Speedlora" / "JewelBingo"
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / "speedlora" / "jewel-bingo"
    return Path.home() / ".speedlora" / "jewel-bingo"


def _copy_tree_missing(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            try:
                shutil.copy2(path, target)
            except OSError:
                pass


def _version_sibling_candidates(project_root: Path) -> list[Path]:
    parent = project_root.parent
    if not parent.exists():
        return []
    out: list[Path] = []
    for p in parent.iterdir():
        if not p.is_dir() or p.resolve() == project_root.resolve():
            continue
        name = p.name.lower()
        if not name.startswith("speedlora-jewel-bingo-assistant-"):
            continue
        if (p / "data").is_dir():
            out.append(p)
    out.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return out


def prepare_user_data(project_root: Path) -> tuple[Path, list[Path]]:
    """Create the mutable data home without hidden cross-version migration.

    Normal startup only fills files that are missing from the selected data home.
    It never scans sibling ZIP/extraction folders and never merges their SQLite
    databases implicitly.  Portable/profile migration is an explicit user action.

    The legacy sibling importer is retained behind
    SPEEDLORA_JEWEL_ALLOW_LEGACY_MIGRATION=1 solely for controlled recovery/tests.
    """
    project_root = Path(project_root).resolve()
    bundled = project_root / "data"
    user = default_user_data_dir()
    user.mkdir(parents=True, exist_ok=True)

    marker = user / ".persistent_data_v1"
    clean_profile = os.environ.get("SPEEDLORA_JEWEL_CLEAN_PROFILE", "").strip().lower() in {"1", "true", "yes", "on"}
    allow_legacy = os.environ.get("SPEEDLORA_JEWEL_ALLOW_LEGACY_MIGRATION", "").strip().lower() in {"1", "true", "yes", "on"}
    if clean_profile:
        allow_legacy = False
    siblings = _version_sibling_candidates(project_root) if allow_legacy else []

    copied_from: Path | None = None
    if not marker.exists():
        # Historical sibling copying is opt-in only. Normal releases initialize
        # from their bundled defaults and preserve the selected data home.
        if siblings:
            copied_from = siblings[0]
            _copy_tree_missing(copied_from / "data", user)
        _copy_tree_missing(bundled, user)
        marker.write_text("Speedlora Jewel Bingo persistent data home\n", encoding="utf-8")
    else:
        _copy_tree_missing(bundled, user)

    dbs: list[Path] = []
    if allow_legacy:
        for sibling in siblings:
            if copied_from is not None and sibling.resolve() == copied_from.resolve():
                continue
            db = sibling / "data" / "bingo.sqlite3"
            if db.exists():
                dbs.append(db)
        local_db = bundled / "bingo.sqlite3"
        if local_db.exists():
            dbs.append(local_db)
    return user, dbs
