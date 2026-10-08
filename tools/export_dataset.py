from __future__ import annotations

from _bootstrap import ROOT
from vision_engine.data_home import default_user_data_dir
from vision_engine.experience import ExperiencePolicy
from vision_engine.persistence import BingoDB
from vision_engine.research_export import export_research_dataset


def main() -> None:
    home = default_user_data_dir()
    db = BingoDB(home / "bingo.sqlite3")
    exp = ExperiencePolicy(home / "bingo.sqlite3", home / "policy_model_v4.joblib", home / "episodes.json")
    out = export_research_dataset(db, home / "exports", exp.info)
    print(f"Exported research dataset to {out}")

if __name__ == "__main__":
    main()
