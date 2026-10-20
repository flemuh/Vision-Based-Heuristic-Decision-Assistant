"""Report eligible >=999 completed games; does not touch the model or train anything."""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision_engine.data_home import default_user_data_dir
from vision_engine.persistence import BingoDB
from vision_engine.winning_memory import WinningMemory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=Path, default=default_user_data_dir() / 'bingo.sqlite3')
    args = parser.parse_args()
    db_path = args.db.expanduser()
    if not db_path.is_file():
        print(f'No existing SQLite database at {db_path}. No file was created.')
        return 2
    print(f'Database: {db_path}')
    # Read-only counts first; BingoDB is used only for the existing SQLite schema.
    con = sqlite3.connect(f'file:{db_path.resolve()}?mode=ro', uri=True)
    try:
        completed = con.execute('SELECT COUNT(*) FROM games WHERE finished_at IS NOT NULL').fetchone()[0]
        verified = con.execute('SELECT COUNT(*) FROM games WHERE verified_score_json IS NOT NULL').fetchone()[0]
        actions = con.execute('SELECT COUNT(*) FROM actions').fetchone()[0]
        print(f'Completed games: {completed} | verified scores: {verified} | logged actions: {actions}')
    finally:
        con.close()
    memory = WinningMemory.from_db(BingoDB(db_path))
    print(f'Eligible, replay-consistent games with >=999: {len(memory.examples)}')
    print(f'Exact-match historical decision states: {len(memory._by_state)}')
    print('Other games remain stored. Validated lower-scoring games with possible >1000 alternatives may enter OFFLINE counterfactual training, not live play.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
