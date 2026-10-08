from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .board import Board

from .versioning import SCHEMA_VERSION


def board_hash(board: Board | Iterable[str | None]) -> str:
    cells = board.cells if isinstance(board, Board) else tuple(board)
    raw = "|".join("MU" if x is None else str(x) for x in cells)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


class BingoDB:
    """SQLite persistence with forward migrations and periodic backups."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.migration_backup_path: Path | None = self._backup_before_migration_if_needed()
        self._init()

    def _existing_schema_version(self) -> int | None:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return None
        try:
            con = sqlite3.connect(self.path, timeout=5)
            try:
                row = con.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
                return int(row[0]) if row else 0
            finally:
                con.close()
        except Exception:
            return 0

    def _backup_before_migration_if_needed(self) -> Path | None:
        """Create a byte-for-byte safety copy before any forward schema change."""
        old = self._existing_schema_version()
        if old is None or old >= SCHEMA_VERSION:
            return None
        backup_dir = self.path.parent / "migration_backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = backup_dir / f"{self.path.stem}_schema{old}_to_{SCHEMA_VERSION}_{stamp}.sqlite3"
        try:
            shutil.copy2(self.path, dest)
            # WAL/SHM are deliberately not copied: migration backup is taken before
            # this process opens the DB and historical releases normally close cleanly.
            return dest
        except OSError:
            return None

    def connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=15)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA foreign_keys=ON")
        con.execute("PRAGMA synchronous=NORMAL")
        return con

    @staticmethod
    def _ensure_column(con: sqlite3.Connection, table: str, column: str, decl: str) -> None:
        cols = {r[1] for r in con.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in cols:
            con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    def _init(self) -> None:
        with self.connect() as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS games (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    finished_at TEXT,
                    source TEXT NOT NULL,
                    session_id TEXT,
                    box_color TEXT,
                    operation_mode TEXT,
                    risk_profile TEXT,
                    board_hash TEXT,
                    board_json TEXT,
                    counts_json TEXT,
                    sequence_json TEXT,
                    selected_mask INTEGER,
                    computed_score_json TEXT,
                    verified_score_json TEXT,
                    one_left_image TEXT,
                    final_image TEXT,
                    notes TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_games_board_hash ON games(board_hash);
                CREATE INDEX IF NOT EXISTS idx_games_created_at ON games(created_at);

                CREATE TABLE IF NOT EXISTS actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    mask_before INTEGER NOT NULL,
                    jewel TEXT NOT NULL,
                    action_pos INTEGER NOT NULL,
                    recommended_pos INTEGER,
                    solver_regret REAL,
                    phase TEXT,
                    strategy TEXT,
                    current_image TEXT,
                    current_conf REAL,
                    current_margin REAL,
                    current_entropy REAL,
                    board_conf REAL,
                    recommendation_json TEXT,
                    FOREIGN KEY(game_id) REFERENCES games(id) ON DELETE CASCADE,
                    UNIQUE(game_id, step)
                );
                CREATE INDEX IF NOT EXISTS idx_actions_game ON actions(game_id, step);

                CREATE TABLE IF NOT EXISTS decision_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    mask_before INTEGER NOT NULL,
                    jewel TEXT NOT NULL,
                    candidate_pos INTEGER NOT NULL,
                    chosen INTEGER NOT NULL DEFAULT 0,
                    rank INTEGER,
                    method TEXT,
                    active_goal TEXT,
                    samples INTEGER DEFAULT 0,
                    stderr_p3 REAL DEFAULT 0,
                    stderr_gt1000 REAL DEFAULT 0,
                    stderr_score REAL DEFAULT 0,
                    solver_regret REAL DEFAULT 0,
                    p3 REAL NOT NULL,
                    p2l1n REAL NOT NULL,
                    p1l2n REAL NOT NULL,
                    p1l1n REAL NOT NULL,
                    p_gt1000 REAL NOT NULL,
                    p_ge999 REAL NOT NULL,
                    expected_score REAL NOT NULL,
                    solver_target REAL NOT NULL,
                    board_json TEXT NOT NULL,
                    image_path TEXT,
                    UNIQUE(game_id, step, candidate_pos),
                    FOREIGN KEY(game_id) REFERENCES games(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_decisions_game ON decision_samples(game_id, step);

                CREATE TABLE IF NOT EXISTS state_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT,
                    created_at TEXT NOT NULL,
                    phase TEXT,
                    kind TEXT NOT NULL,
                    message TEXT,
                    payload_json TEXT,
                    FOREIGN KEY(game_id) REFERENCES games(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_events_game ON state_events(game_id, created_at);

                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS state_checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    state_version INTEGER NOT NULL DEFAULT 0,
                    confirmed_mask INTEGER NOT NULL,
                    provisional_mask INTEGER,
                    phase TEXT,
                    current_jewel TEXT,
                    strategy TEXT,
                    target TEXT,
                    payload_json TEXT,
                    FOREIGN KEY(game_id) REFERENCES games(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_checkpoints_game ON state_checkpoints(game_id, created_at);

                CREATE TABLE IF NOT EXISTS migration_sources (
                    source_path TEXT PRIMARY KEY,
                    source_size INTEGER NOT NULL,
                    source_mtime REAL NOT NULL,
                    merged_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT,
                    created_at TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    duration_ms REAL,
                    cache_hit INTEGER,
                    payload_json TEXT,
                    FOREIGN KEY(game_id) REFERENCES games(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_telemetry_kind_time ON telemetry(kind, created_at);
                """
            )
            # Forward migration from V3 databases.
            for name, decl in (
                ("finished_at", "TEXT"), ("operation_mode", "TEXT"), ("risk_profile", "TEXT"),
                ("status", "TEXT DEFAULT 'playing'"), ("closed_at", "TEXT"),
                ("finalization_source", "TEXT"),
            ):
                self._ensure_column(con, "games", name, decl)
            for name, decl in (
                ("recommended_pos", "INTEGER"), ("solver_regret", "REAL"), ("phase", "TEXT"),
                ("current_margin", "REAL"), ("current_entropy", "REAL"), ("strategy", "TEXT"),
            ):
                self._ensure_column(con, "actions", name, decl)
            for name, decl in (
                ("rank", "INTEGER"), ("active_goal", "TEXT"), ("samples", "INTEGER DEFAULT 0"),
                ("stderr_p3", "REAL DEFAULT 0"), ("stderr_gt1000", "REAL DEFAULT 0"),
                ("stderr_score", "REAL DEFAULT 0"), ("solver_regret", "REAL DEFAULT 0"),
                ("rng_seed", "INTEGER DEFAULT 1337"), ("scenario_set", "TEXT"),
                ("routes_json", "TEXT"), ("primary_stat_tie", "INTEGER DEFAULT 0"),
                ("tie_break_metric", "TEXT"),
            ):
                self._ensure_column(con, "decision_samples", name, decl)
            con.execute("UPDATE games SET status='complete', closed_at=COALESCE(closed_at,finished_at) WHERE finished_at IS NOT NULL")
            con.execute("UPDATE games SET status='playing' WHERE status IS NULL OR status='' ")
            con.execute(
                "INSERT OR REPLACE INTO metadata(key,value) VALUES('schema_version',?)",
                (str(SCHEMA_VERSION),),
            )

    @staticmethod
    def _json(value) -> str | None:
        if value is None:
            return None
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    def ensure_game(
        self,
        game_id: str,
        source: str = "local-auto",
        session_id: str | None = None,
        board: Board | None = None,
        box_color: str | None = None,
        operation_mode: str | None = None,
        risk_profile: str | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as con:
            con.execute(
                """INSERT OR IGNORE INTO games(
                    id,created_at,source,session_id,box_color,operation_mode,risk_profile,board_hash,board_json,status
                ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    game_id, now, source, session_id, box_color, operation_mode, risk_profile,
                    board_hash(board) if board else None,
                    self._json(list(board.cells)) if board else None,
                    "playing",
                ),
            )
            con.execute(
                """UPDATE games SET
                    board_hash=COALESCE(?,board_hash), board_json=COALESCE(?,board_json),
                    box_color=COALESCE(?,box_color), session_id=COALESCE(?,session_id),
                    operation_mode=COALESCE(?,operation_mode), risk_profile=COALESCE(?,risk_profile)
                    WHERE id=?""",
                (
                    board_hash(board) if board else None,
                    self._json(list(board.cells)) if board else None,
                    box_color, session_id, operation_mode, risk_profile, game_id,
                ),
            )

    def log_event(self, game_id: str | None, kind: str, message: str = "", phase: str | None = None, payload: dict | None = None) -> None:
        with self.connect() as con:
            con.execute(
                "INSERT INTO state_events(game_id,created_at,phase,kind,message,payload_json) VALUES(?,?,?,?,?,?)",
                (game_id, datetime.now(timezone.utc).isoformat(), phase, kind, message, self._json(payload)),
            )

    def save_action(
        self,
        game_id: str,
        step: int,
        mask_before: int,
        jewel: str,
        action_pos: int,
        recommendation: dict | None = None,
        current_image: str | None = None,
        current_conf: float | None = None,
        board_conf: float | None = None,
        current_margin: float | None = None,
        current_entropy: float | None = None,
        phase: str | None = None,
        strategy: str | None = None,
        recommended_pos: int | None = None,
        solver_regret: float | None = None,
    ) -> None:
        with self.connect() as con:
            con.execute(
                """INSERT OR REPLACE INTO actions(
                    game_id,step,mask_before,jewel,action_pos,recommended_pos,solver_regret,phase,strategy,
                    current_image,current_conf,current_margin,current_entropy,board_conf,recommendation_json
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    game_id, int(step), int(mask_before), jewel, int(action_pos), recommended_pos,
                    solver_regret, phase, strategy, current_image, current_conf, current_margin, current_entropy,
                    board_conf, self._json(recommendation),
                ),
            )
            con.execute(
                "UPDATE decision_samples SET chosen=CASE WHEN candidate_pos=? THEN 1 ELSE 0 END WHERE game_id=? AND step=?",
                (int(action_pos), game_id, int(step)),
            )

    def save_decision_samples(
        self,
        game_id: str,
        step: int,
        mask_before: int,
        jewel: str,
        board: Board,
        rows: list[dict],
        image_path: str | None = None,
    ) -> None:
        with self.connect() as con:
            for rank, row in enumerate(rows, 1):
                existing = con.execute(
                    "SELECT chosen FROM decision_samples WHERE game_id=? AND step=? AND candidate_pos=?",
                    (game_id, int(step), int(row["position"])),
                ).fetchone()
                chosen = int(existing["chosen"]) if existing is not None else 0
                con.execute(
                    """INSERT OR REPLACE INTO decision_samples(
                        game_id,step,mask_before,jewel,candidate_pos,chosen,rank,method,active_goal,samples,
                        stderr_p3,stderr_gt1000,stderr_score,solver_regret,
                        p3,p2l1n,p1l2n,p1l1n,p_gt1000,p_ge999,expected_score,solver_target,board_json,image_path,
                        rng_seed,scenario_set,routes_json,primary_stat_tie,tie_break_metric
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        game_id, int(step), int(mask_before), jewel, int(row["position"]), chosen,
                        int(row.get("rank", rank)), row.get("method"), row.get("active_goal"), int(row.get("samples", 0)),
                        float(row.get("stderr_p3", 0.0)), float(row.get("stderr_gt1000", 0.0)),
                        float(row.get("stderr_score", 0.0)), float(row.get("solver_regret", 0.0)),
                        float(row["p3"]), float(row["p2l1n"]), float(row["p1l2n"]), float(row["p1l1n"]),
                        float(row["p_gt1000"]), float(row["p_ge999"]), float(row["expected_score"]),
                        float(row["solver_target"]), self._json(list(board.cells)), image_path,
                        int(row.get("rng_seed", 1337)), row.get("scenario_set"), self._json(row.get("routes")),
                        int(bool(row.get("primary_stat_tie", False))), row.get("tie_break_metric"),
                    ),
                )

    def finalize_game(
        self,
        game_id: str,
        board: Board,
        counts: list[int],
        sequence: list[str],
        selected_mask: int,
        computed_score: dict,
        one_left_image: str | None,
        final_image: str | None,
        verified_score: dict | None = None,
    ) -> None:
        self.ensure_game(game_id, board=board)
        with self.connect() as con:
            con.execute(
                """UPDATE games SET finished_at=?,closed_at=?,status='complete',finalization_source=COALESCE(finalization_source,'board-complete'),
                    board_hash=?,board_json=?,counts_json=?,sequence_json=?,selected_mask=?,
                    computed_score_json=?,verified_score_json=COALESCE(?,verified_score_json),one_left_image=?,final_image=?
                    WHERE id=?""",
                (
                    datetime.now(timezone.utc).isoformat(), datetime.now(timezone.utc).isoformat(),
                    board_hash(board), self._json(list(board.cells)),
                    self._json(counts), self._json(sequence), int(selected_mask), self._json(computed_score),
                    self._json(verified_score), one_left_image, final_image, game_id,
                ),
            )

    def finalize_partial_game(
        self, game_id: str, selected_mask: int, verified_score: dict | None = None,
        source: str = "result-screen-without-final-mask", notes: str | None = None,
    ) -> None:
        """Close an episode when the reward screen is certain but final mask was missed.

        This deliberately leaves computed_score/counts unset instead of inventing
        the last placement. Verified OCR, when available, remains authoritative.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as con:
            con.execute(
                """UPDATE games SET finished_at=COALESCE(finished_at,?),closed_at=COALESCE(closed_at,?),
                    status='complete',finalization_source=?,selected_mask=COALESCE(selected_mask,?),
                    verified_score_json=COALESCE(?,verified_score_json),notes=COALESCE(?,notes)
                    WHERE id=?""",
                (now, now, str(source), int(selected_mask), self._json(verified_score), notes, game_id),
            )

    def verify_score(self, game_id: str, score: dict) -> None:
        with self.connect() as con:
            con.execute("UPDATE games SET verified_score_json=? WHERE id=?", (self._json(score), game_id))

    def save_checkpoint(
        self, game_id: str, state_version: int, confirmed_mask: int, provisional_mask: int | None,
        phase: str | None, current_jewel: str | None, strategy: str | None, target: str | None,
        payload: dict | None = None,
    ) -> None:
        self.ensure_game(game_id)
        with self.connect() as con:
            con.execute(
                """INSERT INTO state_checkpoints(
                    game_id,created_at,state_version,confirmed_mask,provisional_mask,phase,current_jewel,strategy,target,payload_json
                ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    game_id, datetime.now(timezone.utc).isoformat(), int(state_version), int(confirmed_mask),
                    None if provisional_mask is None else int(provisional_mask), phase, current_jewel, strategy, target,
                    self._json(payload),
                ),
            )

    def latest_checkpoint(self, game_id: str) -> dict | None:
        with self.connect() as con:
            row = con.execute(
                "SELECT * FROM state_checkpoints WHERE game_id=? ORDER BY id DESC LIMIT 1", (game_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def is_game_complete(self, game_id: str) -> bool:
        with self.connect() as con:
            row = con.execute("SELECT finished_at,status FROM games WHERE id=?", (game_id,)).fetchone()
        return bool(row is not None and (row["finished_at"] is not None or row["status"] == "complete"))

    def mark_finalization_source(self, game_id: str, source: str) -> None:
        with self.connect() as con:
            con.execute(
                "UPDATE games SET finalization_source=?,status=CASE WHEN finished_at IS NULL THEN status ELSE 'complete' END WHERE id=?",
                (str(source), game_id),
            )

    def merge_database(self, source_path: str | Path) -> bool:
        """Merge a previous version database once, preserving historical games.

        New ZIP versions use a stable user-data directory. On first run, sibling
        version databases can be merged without duplicating rows on every start.
        """
        source = Path(source_path)
        if not source.exists() or source.resolve() == self.path.resolve():
            return False
        try:
            st = source.stat()
        except OSError:
            return False
        key = str(source.resolve())
        with self.connect() as con:
            seen = con.execute(
                "SELECT 1 FROM migration_sources WHERE source_path=? AND source_size=? AND source_mtime=?",
                (key, int(st.st_size), float(st.st_mtime)),
            ).fetchone()
            if seen is not None:
                return False

        src = sqlite3.connect(source, timeout=5)
        src.row_factory = sqlite3.Row
        try:
            with self.connect() as dst:
                for table in ("games", "actions", "decision_samples", "state_events", "state_checkpoints", "telemetry"):
                    try:
                        src_cols = [r[1] for r in src.execute(f"PRAGMA table_info({table})").fetchall()]
                        dst_cols = [r[1] for r in dst.execute(f"PRAGMA table_info({table})").fetchall()]
                    except sqlite3.DatabaseError:
                        continue
                    if not src_cols or not dst_cols:
                        continue
                    cols = [c for c in src_cols if c in dst_cols and not (c == "id" and table != "games")]
                    if not cols:
                        continue
                    try:
                        rows = src.execute(f"SELECT {','.join(cols)} FROM {table}").fetchall()
                    except sqlite3.DatabaseError:
                        continue
                    placeholders = ",".join("?" for _ in cols)
                    sql = f"INSERT OR IGNORE INTO {table}({','.join(cols)}) VALUES({placeholders})"
                    for row in rows:
                        try:
                            dst.execute(sql, tuple(row[c] for c in cols))
                        except sqlite3.IntegrityError:
                            pass
                dst.execute(
                    "INSERT OR REPLACE INTO migration_sources(source_path,source_size,source_mtime,merged_at) VALUES(?,?,?,?)",
                    (key, int(st.st_size), float(st.st_mtime), datetime.now(timezone.utc).isoformat()),
                )
            return True
        finally:
            src.close()

    def save_telemetry(
        self, game_id: str | None, kind: str, duration_ms: float | None = None,
        cache_hit: bool | None = None, payload: dict | None = None,
    ) -> None:
        with self.connect() as con:
            con.execute(
                "INSERT INTO telemetry(game_id,created_at,kind,duration_ms,cache_hit,payload_json) VALUES(?,?,?,?,?,?)",
                (
                    game_id, datetime.now(timezone.utc).isoformat(), str(kind),
                    None if duration_ms is None else float(duration_ms),
                    None if cache_hit is None else int(bool(cache_hit)), self._json(payload),
                ),
            )

    def telemetry_summary(self, kind: str = "decision-roundtrip", limit: int = 500) -> dict:
        with self.connect() as con:
            rows = con.execute(
                "SELECT duration_ms,cache_hit FROM telemetry WHERE kind=? AND duration_ms IS NOT NULL ORDER BY id DESC LIMIT ?",
                (str(kind), max(1, int(limit))),
            ).fetchall()
        values = sorted(float(r["duration_ms"]) for r in rows if r["duration_ms"] is not None)
        def pct(q: float):
            if not values:
                return None
            idx = max(0, min(len(values)-1, int(round((len(values)-1)*q))))
            return values[idx]
        cache = [int(r["cache_hit"]) for r in rows if r["cache_hit"] is not None]
        return {
            "count": len(values), "p50": pct(0.50), "p95": pct(0.95), "p99": pct(0.99),
            "cache_hit_rate": (sum(cache)/len(cache)) if cache else None,
        }

    def counterfactual_rows(self) -> list[dict]:
        with self.connect() as con:
            rows = con.execute("SELECT * FROM decision_samples ORDER BY game_id,step,candidate_pos").fetchall()
        return [dict(r) for r in rows]

    def game_rows(self) -> list[dict]:
        with self.connect() as con:
            rows = con.execute("SELECT * FROM games ORDER BY created_at").fetchall()
        return [dict(r) for r in rows]

    def completed_game_count(self) -> int:
        with self.connect() as con:
            return int(con.execute("SELECT COUNT(*) FROM games WHERE finished_at IS NOT NULL").fetchone()[0])

    def repeated_layouts(self, min_count: int = 2) -> list[dict]:
        with self.connect() as con:
            rows = con.execute(
                """SELECT board_hash,COUNT(*) AS games,MIN(board_json) AS board_json
                   FROM games WHERE board_hash IS NOT NULL
                   GROUP BY board_hash HAVING COUNT(*)>=? ORDER BY games DESC""",
                (int(min_count),),
            ).fetchall()
        return [dict(r) for r in rows]

    def backup(self, destination_dir: str | Path, keep: int = 8) -> Path:
        destination_dir = Path(destination_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = destination_dir / f"bingo_{stamp}.sqlite3"
        # sqlite backup API produces a consistent copy even in WAL mode.
        src = self.connect()
        dst = sqlite3.connect(dest)
        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()
        backups = sorted(destination_dir.glob("bingo_*.sqlite3"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in backups[max(1, int(keep)):]:
            try:
                old.unlink()
            except OSError:
                pass
        return dest

    def import_legacy_json(self, history_path: str | Path, episodes_path: str | Path) -> tuple[int, int]:
        """One-way, idempotent-ish migration helper for V1/V2 JSON files."""
        imported_history = 0
        imported_episodes = 0
        hp, ep = Path(history_path), Path(episodes_path)
        if hp.exists():
            try:
                history = json.loads(hp.read_text(encoding="utf-8"))
            except Exception:
                history = []
            for idx, row in enumerate(history):
                gid = str(row.get("episode_id") or f"legacy-history-{idx:04d}")
                board_cells = row.get("board")
                board = None
                if board_cells and len(board_cells) == 25:
                    try:
                        board = Board(tuple(board_cells))
                    except Exception:
                        pass
                self.ensure_game(gid, source=str(row.get("source", "legacy-history")), board=board)
                with self.connect() as con:
                    con.execute(
                        """UPDATE games SET counts_json=COALESCE(counts_json,?),sequence_json=COALESCE(sequence_json,?),
                           selected_mask=COALESCE(selected_mask,?),computed_score_json=COALESCE(computed_score_json,?) WHERE id=?""",
                        (self._json(row.get("counts")), self._json(row.get("sequence")), row.get("selected_mask"), self._json(row.get("computed_score")), gid),
                    )
                imported_history += 1
        if ep.exists():
            try:
                episodes = json.loads(ep.read_text(encoding="utf-8"))
            except Exception:
                episodes = []
            for idx, row in enumerate(episodes):
                gid = str(row.get("id") or f"legacy-episode-{idx:04d}")
                board_cells = row.get("board")
                if not board_cells or len(board_cells) != 25:
                    continue
                try:
                    board = Board(tuple(board_cells))
                except Exception:
                    continue
                self.ensure_game(gid, source=str(row.get("source", "legacy-episode")), board=board)
                for step, action in enumerate(row.get("actions") or []):
                    try:
                        self.save_action(gid, step, int(action["mask_before"]), str(action["jewel"]), int(action["action_pos"]))
                    except Exception:
                        pass
                self.finalize_game(
                    gid, board, list(row.get("counts") or []), list(row.get("sequence") or []),
                    int(row.get("selected_mask") or 0), dict(row.get("computed_score") or {}),
                    row.get("one_left_image"), row.get("final_image"), None,
                )
                imported_episodes += 1
        return imported_history, imported_episodes
