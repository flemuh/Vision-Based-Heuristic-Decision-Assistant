from __future__ import annotations

from _bootstrap import ROOT  # adds repository root to sys.path

import json
from pathlib import Path

from vision_engine.history import HistoryStore
from vision_engine.learning import history_summary
from vision_engine.data_home import default_user_data_dir


ROOT = Path(__file__).resolve().parents[1]
history = HistoryStore(default_user_data_dir() / "history.json").load()
print(json.dumps(history_summary(history), indent=2, ensure_ascii=False))
