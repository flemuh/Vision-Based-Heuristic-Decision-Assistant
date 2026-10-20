from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from .simple_ml import NumpyRidgeRegressor, mean_absolute_error

from .advisor import Utility, utility_learning_target
from .board import Board
from .constants import JEWELS, JEWEL_INDEX, LUCKY_LINES, NORMAL_LINES
from .persistence import BingoDB
from .replay import _terminal_masks_for_counts
from .scoring import score_mask
from .versioning import MODEL_VERSION
from .winning_memory import WinningMemory, validated_episodes
from .retrospective_learning import cached_audit, audit_stored_games


@dataclass
class PolicyModelInfo:
    trained_samples: int = 0
    trained_games: int = 0
    counterfactual_samples: int = 0
    outcome_samples: int = 0
    validation_games: int = 0
    outcome_validation_games: int = 0
    mae_p3: float | None = None
    mae_gt1000: float | None = None
    mae_score: float | None = None
    outcome_residual_mae: float | None = None
    outcome_baseline_mae: float | None = None
    outcome_improvement: float | None = None
    counterfactual_trusted: bool = False
    outcome_trusted: bool = False
    hindsight_samples: int = 0
    hindsight_games: int = 0
    hindsight_validation_games: int = 0
    hindsight_mae_p3: float | None = None
    hindsight_mae_gt1000: float | None = None
    hindsight_mae_score: float | None = None
    hindsight_trusted: bool = False
    bootstrap_loaded: bool = False
    counterfactual_recovery_games: int = 0
    retrospectively_audited_games: int = 0


def _line_progress(mask: int, cells: Iterable[int]) -> float:
    cells = tuple(cells)
    return sum(1 for i in cells if (mask >> i) & 1) / float(len(cells))


def action_features(board: Board, mask: int, jewel: str, pos: int) -> np.ndarray:
    after = mask | (1 << pos)
    feats: list[float] = []
    feats.append(mask.bit_count() / 14.0)
    feats.extend(1.0 if jewel == j else 0.0 for j in JEWELS)
    counts = board.jewel_counts_in_mask(mask)
    feats.extend(v / 4.0 for v in counts)
    feats.extend(1.0 if i == pos else 0.0 for i in range(25))
    feats.extend(0.0 if i == 12 else float((mask >> i) & 1) for i in range(25))
    for i, cell in enumerate(board.cells):
        if i == 12:
            feats.extend([0.0] * 6)
        else:
            idx = JEWEL_INDEX[cell]  # type: ignore[index]
            feats.extend(1.0 if k == idx else 0.0 for k in range(6))
    feats.extend(_line_progress(after, cells) for cells in LUCKY_LINES.values())
    feats.extend(_line_progress(after, cells) for cells in NORMAL_LINES.values())
    return np.asarray(feats, dtype=np.float32)


def _outcome_utility(score: dict) -> Utility:
    lucky = int(score.get("lucky") or 0)
    normal = int(score.get("normal") or 0)
    total = float(score.get("total") or 0)
    return Utility(
        float(lucky >= 3),
        float(lucky >= 2 and normal >= 1),
        float(lucky >= 1 and normal >= 2),
        float(lucky >= 1 and normal >= 1),
        float(total > 1000),
        float(total >= 999),
        total,
    )


def _stable_holdout(game_id: str, fraction: float) -> bool:
    """Deterministic group split: every state from a game stays on one side."""
    if fraction <= 0:
        return False
    h = int.from_bytes(hashlib.sha1(game_id.encode()).digest()[:4], "big") / 2**32
    return h < fraction


class ExperiencePolicy:
    """Local ML that learns *action values*, not just final game labels.

    Counterfactual model: multi-output imitation of the mathematical solver for
    every candidate action. It predicts the seven Utility components directly.

    Replay-validated completed >=999 scores train factual ML. Replay-valid
    lower scores with a provable >1000 alternative train offline hindsight. Live actions NEVER use the generalized model; only exact
    board + observed jewel/position prefix matches from WinningMemory may
    break mathematically indistinguishable solver decisions.
    """

    UTILITY_COLUMNS = ("p3", "p2l1n", "p1l2n", "p1l1n", "p_gt1000", "p_ge999", "expected_score")

    def __init__(self, db_path: str | Path, model_path: str | Path, legacy_episodes_path: str | Path | None = None):
        self.db = BingoDB(db_path)
        self.model_path = Path(model_path)
        self.legacy_episodes_path = Path(legacy_episodes_path) if legacy_episodes_path else None
        self.counterfactual_model = None
        self.outcome_residual_model = None
        self.hindsight_model = None
        self.bootstrap_model = None
        self.bootstrap_model_path = self.model_path.parent / "bootstrap_solver_model.joblib"
        self.info = PolicyModelInfo()
        self.winning_memory = WinningMemory()
        self.reload()

    def reload(self) -> None:
        self.winning_memory = WinningMemory.from_db(self.db)
        self.bootstrap_model = None
        if self.bootstrap_model_path.exists():
            try:
                bp = joblib.load(self.bootstrap_model_path)
                self.bootstrap_model = bp.get("model") if isinstance(bp, dict) else bp
            except Exception:
                self.bootstrap_model = None
        if not self.model_path.exists():
            self.counterfactual_model = None
            self.outcome_residual_model = None
            self.hindsight_model = None
            self.info = PolicyModelInfo(bootstrap_loaded=self.bootstrap_model is not None)
            return
        try:
            payload = joblib.load(self.model_path)
            if payload.get("training_policy") != "validated-ge999-and-counterfactual-v2":
                # An old model was trained on losses/partial games: quarantine it.
                # Do not delete the user's model file or SQLite history.
                self.counterfactual_model = None
                self.outcome_residual_model = None
                self.hindsight_model = None
                self.info = PolicyModelInfo(bootstrap_loaded=self.bootstrap_model is not None)
                return
            self.counterfactual_model = payload.get("counterfactual_model")
            self.outcome_residual_model = payload.get("outcome_residual_model")
            self.hindsight_model = payload.get("hindsight_model")
            raw = payload.get("info") or {}
            allowed = set(PolicyModelInfo.__dataclass_fields__)
            self.info = PolicyModelInfo(**{k: v for k, v in raw.items() if k in allowed})
            self.info.bootstrap_loaded = self.bootstrap_model is not None
        except Exception:
            self.counterfactual_model = None
            self.outcome_residual_model = None
            self.hindsight_model = None
            self.info = PolicyModelInfo(bootstrap_loaded=self.bootstrap_model is not None)

    @staticmethod
    def _board_from_json(raw: str | list | tuple) -> Board | None:
        try:
            cells = json.loads(raw) if isinstance(raw, str) else list(raw)
            return Board(tuple(cells))
        except Exception:
            return None

    def train(
        self,
        min_counterfactual: int = 80,
        min_outcome_games: int = 20,
        min_outcome_samples: int = 180,
        min_hindsight_games: int = 3,
        min_hindsight_samples: int = 30,
        holdout_fraction: float = 0.25,
        min_validation_games: int = 8,
    ) -> PolicyModelInfo:
        # Factual >=999 episodes can enter the exact live history. Completed
        # replay-valid losses with a >1000 feasible alternative enter offline
        # retrospective training only; they NEVER masquerade as real wins.
        self.winning_memory = WinningMemory.from_db(self.db)
        winning_ids = self.winning_memory.winning_game_ids
        winning_examples = {x.game_id: x for x in self.winning_memory.examples}
        # Validate every saved candidate's state against the actual winning
        # episode; final totals alone do not make unrelated snapshots eligible.
        cf_rows = []
        for row in self.db.counterfactual_rows():
            example = winning_examples.get(str(row.get("game_id")))
            if example is None:
                continue
            try:
                step = int(row["step"])
                expected_mask = sum(1 << p for p in example.actions[:step])
                if (not 0 <= step < 14
                    or int(row["mask_before"]) != expected_mask
                    or str(row["jewel"]) != example.sequence[step]
                    or tuple(json.loads(row["board_json"])) != example.board):
                    continue
            except (KeyError, TypeError, ValueError, IndexError, json.JSONDecodeError):
                continue
            cf_rows.append(row)
        x_train: list[np.ndarray] = []
        y_train: list[list[float]] = []
        x_val: list[np.ndarray] = []
        y_val: list[list[float]] = []
        cf_games = set()
        val_games = set()

        for row in cf_rows:
            board = self._board_from_json(row.get("board_json"))
            if board is None:
                continue
            try:
                mask = int(row["mask_before"])
                jewel = str(row["jewel"])
                pos = int(row["candidate_pos"])
                if jewel not in JEWELS or board.cells[pos] != jewel:
                    continue
                x = action_features(board, mask, jewel, pos)
                y = [float(row[c]) for c in self.UTILITY_COLUMNS]
                gid = str(row["game_id"])
                cf_games.add(gid)
                if _stable_holdout(gid, holdout_fraction):
                    x_val.append(x); y_val.append(y); val_games.add(gid)
                else:
                    x_train.append(x); y_train.append(y)
            except Exception:
                continue

        cf_model = None
        mae_p3 = mae_gt = mae_score = None
        cf_trusted = False
        if len(x_train) >= min_counterfactual:
            cf_model = NumpyRidgeRegressor(alpha=4.0)
            cf_model.fit(np.stack(x_train), np.asarray(y_train, dtype=np.float64))
            if x_val:
                pred = np.asarray(cf_model.predict(np.stack(x_val)), dtype=np.float64)
                true = np.asarray(y_val, dtype=np.float64)
                mae_p3 = float(mean_absolute_error(true[:, 0], pred[:, 0]))
                mae_gt = float(mean_absolute_error(true[:, 4], pred[:, 4]))
                mae_score = float(mean_absolute_error(true[:, 6], pred[:, 6]))
                cf_trusted = len(val_games) >= int(min_validation_games) and mae_p3 <= 0.14 and mae_gt <= 0.16 and mae_score <= 125
            else:
                # Solver labels are deterministic; with no holdout we can train,
                # but mark it unvalidated so the GUI reports that honestly.
                cf_trusted = False
            # Refit on all solver-labeled examples after measuring holdout error.
            if x_val:
                all_x = x_train + x_val
                all_y = y_train + y_val
                cf_model.fit(np.stack(all_x), np.asarray(all_y, dtype=np.float64))

        # Legacy real-outcome residuals were not causally assigned to actions.
        # Retain raw SQLite for future research, but omit this model, its MAE
        # and its expensive feature extraction entirely in this release.
        x_out_train: list[np.ndarray] = []
        x_out_val: list[np.ndarray] = []
        out_games: set[str] = set()
        out_val_games: set[str] = set()
        outcome_model = None
        residual_mae = baseline_mae = outcome_improvement = None
        outcome_trusted = False

        # Validated end-of-game hindsight. The complete future draw is allowed
        # ONLY in these cached offline targets. This includes alternate wins
        # >1000 from real losses; these are research labels, NOT live memory.
        # Observable games need no saved solver candidate rows to contribute.
        h_x_train: list[np.ndarray] = []
        h_y_train: list[list[float]] = []
        h_x_val: list[np.ndarray] = []
        h_y_val: list[list[float]] = []
        h_games: set[str] = set()
        h_val_games: set[str] = set()
        retrospective_recoverable = 0
        audited_games = 0
        for episode in validated_episodes(self.db):
            audit = cached_audit(self.db, episode)
            if not audit or not audit.get("valid"):
                continue
            audited_games += 1
            can_recover = episode.total < 999 and bool(audit["could_exceed_1000"])
            if can_recover:
                retrospective_recoverable += 1
            if episode.total < 999 and not can_recover:
                continue
            board = Board(episode.board)
            is_val_game = _stable_holdout(episode.game_id, holdout_fraction)
            count = 0
            for step in audit["steps"]:
                mask = int(step["mask_before"])
                jewel = step["jewel"]
                for candidate in step["candidates"]:
                    pos = int(candidate["position"])
                    labels = [
                        float(candidate["possible_3lucky"]),
                        float(candidate["possible_2l1n"]),
                        float(candidate["possible_1l2n"]),
                        float(candidate["possible_1l1n"]),
                        float(candidate["possible_gt1000"]),
                        float(candidate["possible_ge999"]),
                        float(candidate["best_score"]),
                    ]
                    features = action_features(board, mask, jewel, pos)
                    if is_val_game:
                        h_x_val.append(features); h_y_val.append(labels); h_val_games.add(episode.game_id)
                    else:
                        h_x_train.append(features); h_y_train.append(labels)
                    count += 1
            if count:
                h_games.add(episode.game_id)

        hindsight_model = None
        h_mae_p3 = h_mae_gt = h_mae_score = None
        h_trusted = False
        h_total_samples = len(h_x_train) + len(h_x_val)
        if len(h_games) >= int(min_hindsight_games) and h_total_samples >= int(min_hindsight_samples):
            fit_x = h_x_train or h_x_val
            fit_y = h_y_train or h_y_val
            hindsight_model = NumpyRidgeRegressor(alpha=12.0)
            hindsight_model.fit(np.stack(fit_x), np.asarray(fit_y, dtype=np.float64))
            if h_x_train and h_x_val:
                pred = np.asarray(hindsight_model.predict(np.stack(h_x_val)), dtype=np.float64)
                true = np.asarray(h_y_val, dtype=np.float64)
                h_mae_p3 = float(mean_absolute_error(true[:, 0], np.clip(pred[:, 0], 0.0, 1.0)))
                h_mae_gt = float(mean_absolute_error(true[:, 4], np.clip(pred[:, 4], 0.0, 1.0)))
                h_mae_score = float(mean_absolute_error(true[:, 6], pred[:, 6]))
                h_trusted = (
                    len(h_val_games) >= int(min_validation_games)
                    and h_mae_p3 <= 0.34
                    and h_mae_gt <= 0.34
                    and h_mae_score <= 150.0
                )
                hindsight_model.fit(
                    np.stack(h_x_train + h_x_val),
                    np.asarray(h_y_train + h_y_val, dtype=np.float64),
                )

        self.counterfactual_model = cf_model
        self.outcome_residual_model = outcome_model
        self.hindsight_model = hindsight_model
        self.info = PolicyModelInfo(
            trained_samples=len(x_train) + len(x_val) + len(x_out_train) + len(x_out_val) + h_total_samples,
            trained_games=len(cf_games | out_games | h_games),
            counterfactual_samples=len(x_train) + len(x_val),
            outcome_samples=len(x_out_train) + len(x_out_val),
            validation_games=len(val_games),
            outcome_validation_games=len(out_val_games),
            mae_p3=mae_p3,
            mae_gt1000=mae_gt,
            mae_score=mae_score,
            outcome_residual_mae=residual_mae,
            outcome_baseline_mae=baseline_mae,
            outcome_improvement=outcome_improvement,
            counterfactual_trusted=cf_trusted,
            outcome_trusted=outcome_trusted,
            hindsight_samples=h_total_samples,
            hindsight_games=len(h_games),
            hindsight_validation_games=len(h_val_games),
            hindsight_mae_p3=h_mae_p3,
            hindsight_mae_gt1000=h_mae_gt,
            hindsight_mae_score=h_mae_score,
            hindsight_trusted=h_trusted,
            bootstrap_loaded=self.bootstrap_model is not None,
            counterfactual_recovery_games=retrospective_recoverable,
            retrospectively_audited_games=audited_games,
        )
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "counterfactual_model": cf_model,
            "outcome_residual_model": outcome_model,
            "hindsight_model": hindsight_model,
            "info": asdict(self.info),
            "utility_columns": self.UTILITY_COLUMNS,
            "model_version": MODEL_VERSION,
            "training_policy": "validated-ge999-and-counterfactual-v2",
        }
        tmp_path = self.model_path.with_suffix(self.model_path.suffix + ".tmp")
        joblib.dump(payload, tmp_path)
        tmp_path.replace(self.model_path)
        return self.info

    def predict(
        self, board: Board, mask: int, jewel: str, positions: Iterable[int],
        *, observed_jewels: Iterable[str] | None = None,
        observed_positions: Iterable[int] | None = None,
    ) -> dict[int, float]:
        """Exact >=999 observed-history tie-break only. Offline alternatives excluded."""
        return self.winning_memory.matching_scores(
            board, mask, jewel, positions, observed_jewels, observed_positions,
        )

