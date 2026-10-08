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

    Outcome model: learns only the residual between a chosen action's solver
    target and the verified/computed real outcome. It is deliberately gated by
    game-level validation and receives a small weight, so a few lucky games
    cannot override the mathematical policy.
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
        self.reload()

    def reload(self) -> None:
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
        cf_rows = self.db.counterfactual_rows()
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

        # Real-outcome residuals on chosen actions. This tackles credit assignment
        # better than giving every move the same raw final-score target: each move
        # learns how reality differed from the solver's own prediction at that state.
        x_out_train: list[np.ndarray] = []
        y_out_train: list[float] = []
        x_out_val: list[np.ndarray] = []
        y_out_val: list[float] = []
        out_games = set(); out_val_games = set()
        with self.db.connect() as con:
            rows = con.execute(
                """SELECT a.game_id,a.mask_before,a.jewel,a.action_pos,g.board_json,
                          COALESCE(g.verified_score_json,g.computed_score_json) AS score_json,
                          d.solver_target
                   FROM actions a JOIN games g ON g.id=a.game_id
                   LEFT JOIN decision_samples d ON d.game_id=a.game_id AND d.step=a.step AND d.candidate_pos=a.action_pos
                   WHERE g.board_json IS NOT NULL
                     AND COALESCE(g.verified_score_json,g.computed_score_json) IS NOT NULL"""
            ).fetchall()
        for row in rows:
            board = self._board_from_json(row["board_json"])
            if board is None or row["solver_target"] is None:
                continue
            try:
                score = json.loads(row["score_json"])
                pos = int(row["action_pos"]); jewel = str(row["jewel"]); mask = int(row["mask_before"])
                if board.cells[pos] != jewel:
                    continue
                actual_target = utility_learning_target(_outcome_utility(score))
                residual = float(actual_target - float(row["solver_target"]))
                x = action_features(board, mask, jewel, pos)
                gid = str(row["game_id"]); out_games.add(gid)
                if _stable_holdout(gid, holdout_fraction):
                    x_out_val.append(x); y_out_val.append(residual); out_val_games.add(gid)
                else:
                    x_out_train.append(x); y_out_train.append(residual)
            except Exception:
                continue

        outcome_model = None
        residual_mae = None
        baseline_mae = None
        outcome_improvement = None
        outcome_trusted = False
        if len(out_games) >= min_outcome_games and len(x_out_train) >= min_outcome_samples:
            outcome_model = NumpyRidgeRegressor(alpha=8.0)
            outcome_model.fit(np.stack(x_out_train), np.asarray(y_out_train, dtype=np.float64))
            if x_out_val:
                pred = outcome_model.predict(np.stack(x_out_val))
                residual_mae = float(mean_absolute_error(y_out_val, pred))
                baseline_mae = float(mean_absolute_error(y_out_val, np.zeros(len(y_out_val))))
                if baseline_mae > 0:
                    outcome_improvement = 1.0 - (residual_mae / baseline_mae)
                outcome_trusted = (
                    len(out_val_games) >= int(min_validation_games)
                    and baseline_mae > 0
                    and residual_mae < baseline_mae * 0.92
                )
            if x_out_val:
                outcome_model.fit(
                    np.stack(x_out_train + x_out_val),
                    np.asarray(y_out_train + y_out_val, dtype=np.float64),
                )

        # Retrospective candidate reward. Unlike the legacy outcome model, this
        # labels *every saved candidate*, including actions that were not chosen.
        # For a completed game we know the 14 jewels that actually arrived. For
        # each historical state/candidate we ask whether at least one legal final
        # allocation with that same draw composition still reaches each target.
        # The future sequence is used ONLY to create offline labels; it is never
        # part of action_features(), so live inference cannot peek at future jewels.
        h_x_train: list[np.ndarray] = []
        h_y_train: list[list[float]] = []
        h_x_val: list[np.ndarray] = []
        h_y_val: list[list[float]] = []
        h_games: set[str] = set()
        h_val_games: set[str] = set()
        game_map = {str(g.get("id")): g for g in self.db.game_rows()}
        rows_by_game: dict[str, list[dict]] = defaultdict(list)
        for row in cf_rows:
            rows_by_game[str(row.get("game_id"))].append(row)

        for gid, game_rows in rows_by_game.items():
            game = game_map.get(gid)
            if not game or not game.get("finished_at") or not game.get("board_json") or not game.get("sequence_json"):
                continue
            board = self._board_from_json(game.get("board_json"))
            if board is None:
                continue
            try:
                sequence = json.loads(game.get("sequence_json") or "[]")
            except Exception:
                continue
            sequence = [str(j) for j in sequence if str(j) in JEWELS]
            if len(sequence) != 14:
                continue
            counts_counter = Counter(sequence)
            counts = tuple(int(counts_counter.get(j, 0)) for j in JEWELS)
            if any(c > 4 for c in counts):
                continue
            terminal_masks = _terminal_masks_for_counts(board.cells, counts)
            if not terminal_masks:
                continue
            terminal = [(m, score_mask(m)) for m in terminal_masks]
            step_rows: dict[int, list[dict]] = defaultdict(list)
            for row in game_rows:
                try:
                    step_rows[int(row["step"])].append(row)
                except Exception:
                    pass

            game_examples = 0
            is_val_game = _stable_holdout(gid, holdout_fraction)
            for _step, candidates in sorted(step_rows.items()):
                if not candidates:
                    continue
                try:
                    prefix = int(candidates[0]["mask_before"])
                except Exception:
                    continue
                feasible_from_prefix = [(m, sc) for m, sc in terminal if (m & prefix) == prefix]
                if not feasible_from_prefix:
                    continue
                for row in candidates:
                    try:
                        jewel = str(row["jewel"]); pos = int(row["candidate_pos"]); mask = int(row["mask_before"])
                    except Exception:
                        continue
                    if jewel not in JEWELS or mask != prefix or board.cells[pos] != jewel or ((prefix >> pos) & 1):
                        continue
                    matching = [sc for m, sc in feasible_from_prefix if (m >> pos) & 1]
                    if not matching:
                        continue
                    labels = [
                        float(any(sc.lucky >= 3 for sc in matching)),
                        float(any(sc.lucky >= 2 and sc.normal >= 1 for sc in matching)),
                        float(any(sc.lucky >= 1 and sc.normal >= 2 for sc in matching)),
                        float(any(sc.lucky >= 1 and sc.normal >= 1 for sc in matching)),
                        float(any(sc.total > 1000 for sc in matching)),
                        float(any(sc.total >= 999 for sc in matching)),
                        float(max(sc.total for sc in matching)),
                    ]
                    x = action_features(board, mask, jewel, pos)
                    if is_val_game:
                        h_x_val.append(x); h_y_val.append(labels); h_val_games.add(gid)
                    else:
                        h_x_train.append(x); h_y_train.append(labels)
                    game_examples += 1
            if game_examples:
                h_games.add(gid)

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
        )
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "counterfactual_model": cf_model,
            "outcome_residual_model": outcome_model,
            "hindsight_model": hindsight_model,
            "info": asdict(self.info),
            "utility_columns": self.UTILITY_COLUMNS,
            "model_version": MODEL_VERSION,
        }
        tmp_path = self.model_path.with_suffix(self.model_path.suffix + ".tmp")
        joblib.dump(payload, tmp_path)
        tmp_path.replace(self.model_path)
        return self.info

    def predict(self, board: Board, mask: int, jewel: str, positions: Iterable[int]) -> dict[int, float]:
        positions = list(positions)
        if not positions:
            return {}
        x = np.stack([action_features(board, mask, jewel, p) for p in positions])

        # Base learned score imitates the mathematical solver. If no base model
        # exists yet, hindsight can still contribute a conservative tie-break.
        model = self.counterfactual_model or self.bootstrap_model
        if model is None:
            scores = np.zeros(len(positions), dtype=np.float64)
        else:
            raw = np.asarray(model.predict(x), dtype=np.float64)
            # If both models exist but local solver imitation is still unvalidated,
            # softly blend the generic bootstrap. Once local validation passes, the
            # board/user-specific model becomes authoritative for ML tie-breaking.
            if self.counterfactual_model is not None and self.bootstrap_model is not None and not self.info.counterfactual_trusted:
                boot = np.asarray(self.bootstrap_model.predict(x), dtype=np.float64)
                if boot.shape == raw.shape:
                    raw = 0.70 * raw + 0.30 * boot
            if raw.ndim == 1:
                raw = raw.reshape(-1, 1)
            base_scores = []
            for row in raw:
                if len(row) >= 7:
                    u = Utility(*[float(v) for v in row[:7]])
                    base_scores.append(utility_learning_target(u))
                else:
                    base_scores.append(float(row[0]))
            scores = np.asarray(base_scores, dtype=np.float64)

        # Real-outcome learning is a small residual correction and only activates
        # if it beat a zero-residual baseline on held-out games.
        if self.outcome_residual_model is not None and self.info.outcome_trusted:
            residual = np.asarray(self.outcome_residual_model.predict(x), dtype=np.float64)
            cap = np.quantile(np.abs(residual), 0.90) if residual.size > 1 else abs(residual[0])
            cap = max(100.0, min(float(cap), 1800.0))
            scores = scores + 0.15 * np.clip(residual, -cap, cap)

        # V18.0.6 closes the hindsight-learning loop. A candidate that preserved
        # a winning allocation in real historical draws receives positive reward;
        # one that destroyed it receives negative relative credit. The model sees
        # only live-state features at inference time. Because Advisor consults
        # learned scores only for true mathematical/statistical ties, this cannot
        # override a clearly superior solver action.
        if self.hindsight_model is not None:
            hp = np.asarray(self.hindsight_model.predict(x), dtype=np.float64)
            if hp.ndim == 1:
                hp = hp.reshape(-1, 1)
            if hp.shape[1] >= 7:
                hindsight_scores = []
                for row in hp:
                    probs = [float(np.clip(v, 0.0, 1.0)) for v in row[:6]]
                    best_score = float(np.clip(row[6], 0.0, 2000.0))
                    hindsight_scores.append(utility_learning_target(Utility(*probs, best_score)))
                hs = np.asarray(hindsight_scores, dtype=np.float64)
                # Centering makes this pure relative credit among the legal moves.
                # Early/unvalidated history gets a small influence; validated
                # history can contribute more, still only inside solver ties.
                hs = hs - float(np.mean(hs))
                if self.info.hindsight_trusted:
                    weight = 0.22
                else:
                    weight = min(0.10, 0.025 + 0.005 * max(0, self.info.hindsight_games))
                scores = scores + weight * hs

        return {p: float(v) for p, v in zip(positions, scores)}

