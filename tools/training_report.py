from __future__ import annotations

from _bootstrap import ROOT  # adds repository root to sys.path

from pathlib import Path

from vision_engine.experience import ExperiencePolicy
from vision_engine.vision import JewelClassifier

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    policy = ExperiencePolicy(ROOT/"data"/"bingo.sqlite3", ROOT/"data"/"policy_model_v4.joblib")
    info = policy.info
    print("Policy model")
    print(f"  bootstrap loaded: {info.bootstrap_loaded}")
    print(f"  solver/counterfactual samples: {info.counterfactual_samples}")
    print(f"  real outcome samples: {info.outcome_samples}")
    print(f"  games: {info.trained_games} | validation games: {info.validation_games}")
    print(f"  counterfactual trusted: {info.counterfactual_trusted}")
    print(f"  outcome residual trusted: {info.outcome_trusted}")
    print(f"  MAE P3={info.mae_p3} >1000={info.mae_gt1000} score={info.mae_score} residual={info.outcome_residual_mae}")
    clf = JewelClassifier(ROOT/"data"/"templates")
    print("\nVision templates")
    for j,n in clf.template_counts().items():
        print(f"  {j}: {n}")


if __name__ == "__main__":
    main()
