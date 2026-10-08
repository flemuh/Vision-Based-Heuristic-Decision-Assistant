from __future__ import annotations

from dataclasses import dataclass
import numpy as np


class NumpyKNNClassifier:
    """Small dependency-free KNN classifier using NumPy only.

    It implements just the API used by the vision layer: fit(), predict_proba()
    and classes_. This intentionally avoids scikit-learn compiled extensions so
    the assistant can run on Windows machines with restrictive Application
    Control policies.
    """

    def __init__(self, n_neighbors: int = 3, weights: str = "distance"):
        self.n_neighbors = max(1, int(n_neighbors))
        self.weights = weights
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self.classes_: np.ndarray = np.asarray([], dtype=object)

    def fit(self, x: np.ndarray, y) -> "NumpyKNNClassifier":
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(list(y), dtype=object)
        if x.ndim != 2 or len(x) != len(y) or len(x) == 0:
            raise ValueError("Invalid KNN training data")
        self._x = x
        self._y = y
        self.classes_ = np.asarray(sorted({str(v) for v in y}), dtype=object)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self._x is None or self._y is None or self.classes_.size == 0:
            raise RuntimeError("KNN model is not fitted")
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        out = np.zeros((len(x), len(self.classes_)), dtype=np.float64)
        class_to_idx = {str(c): i for i, c in enumerate(self.classes_)}
        k = min(self.n_neighbors, len(self._x))
        for row_idx, row in enumerate(x):
            d2 = np.sum((self._x - row) ** 2, axis=1, dtype=np.float64)
            nn = np.argpartition(d2, k - 1)[:k] if k < len(d2) else np.arange(len(d2))
            if self.weights == "distance":
                # Exact template match should dominate without numerical blow-up.
                distances = np.sqrt(np.maximum(d2[nn], 0.0))
                exact = distances <= 1e-12
                if np.any(exact):
                    w = exact.astype(np.float64)
                else:
                    w = 1.0 / np.maximum(distances, 1e-9)
            else:
                w = np.ones(len(nn), dtype=np.float64)
            for idx, weight in zip(nn, w):
                out[row_idx, class_to_idx[str(self._y[idx])]] += float(weight)
            s = out[row_idx].sum()
            if s <= 0:
                out[row_idx] = 1.0 / len(self.classes_)
            else:
                out[row_idx] /= s
        return out


@dataclass
class NumpyRidgeRegressor:
    """Compact multi-output ridge regressor implemented only with NumPy.

    It is intentionally conservative: the mathematical solver remains the
    authority and this model is used only for approximation/tie-breaking.
    """

    alpha: float = 4.0
    x_mean_: np.ndarray | None = None
    x_scale_: np.ndarray | None = None
    y_mean_: np.ndarray | None = None
    y_scale_: np.ndarray | None = None
    coef_: np.ndarray | None = None
    single_output_: bool = False

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NumpyRidgeRegressor":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.ndim != 2 or x.shape[0] == 0:
            raise ValueError("Invalid regression training data")
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            self.single_output_ = True
        else:
            self.single_output_ = False
        if y.shape[0] != x.shape[0]:
            raise ValueError("X/y row mismatch")

        self.x_mean_ = x.mean(axis=0)
        self.x_scale_ = x.std(axis=0)
        self.x_scale_[self.x_scale_ < 1e-8] = 1.0
        self.y_mean_ = y.mean(axis=0)
        self.y_scale_ = y.std(axis=0)
        self.y_scale_[self.y_scale_ < 1e-8] = 1.0

        xs = (x - self.x_mean_) / self.x_scale_
        ys = (y - self.y_mean_) / self.y_scale_
        # Add an explicit bias feature after centering for numerical robustness.
        xa = np.concatenate([xs, np.ones((len(xs), 1), dtype=np.float64)], axis=1)
        d = xa.shape[1]
        ridge = np.eye(d, dtype=np.float64) * float(self.alpha)
        ridge[-1, -1] = 1e-9
        if xa.shape[0] >= d:
            self.coef_ = np.linalg.solve(xa.T @ xa + ridge, xa.T @ ys)
        else:
            # Dual form is much cheaper when there are fewer samples than features.
            dual = np.linalg.solve(xa @ xa.T + np.eye(len(xa)) * float(self.alpha), ys)
            self.coef_ = xa.T @ dual
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if any(v is None for v in (self.x_mean_, self.x_scale_, self.y_mean_, self.y_scale_, self.coef_)):
            raise RuntimeError("Regressor is not fitted")
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        xs = (x - self.x_mean_) / self.x_scale_
        xa = np.concatenate([xs, np.ones((len(xs), 1), dtype=np.float64)], axis=1)
        pred = xa @ self.coef_
        pred = pred * self.y_scale_ + self.y_mean_
        if self.single_output_:
            return pred[:, 0]
        return pred


def mean_absolute_error(y_true, y_pred) -> float:
    a = np.asarray(y_true, dtype=np.float64)
    b = np.asarray(y_pred, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("Shape mismatch")
    return float(np.mean(np.abs(a - b)))
