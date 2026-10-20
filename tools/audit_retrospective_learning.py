"""Run offline replay audit of completed local games (does not retrain live ML).

Stop game monitoring before running this: it is CPU-intensive on large histories.
Every game is cached by audit version so repeated runs are much cheaper.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision_engine.data_home import default_user_data_dir
from vision_engine.persistence import BingoDB
from vision_engine.retrospective_learning import audit_stored_games


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=Path, default=default_user_data_dir() / 'bingo.sqlite3')
    args = parser.parse_args()
    dbpath=args.db.expanduser()
    if not dbpath.is_file():
        print(f'Database not found: {dbpath}; no file created')
        return 2
    print('Offline audit: original actions/scores are never changed.')
    result=audit_stored_games(BingoDB(dbpath))
    for key, value in result.items():
        print(f'{key}: {value}')
    print('Theoretical opportunities are NOT real victories and are NOT live oracle data.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
