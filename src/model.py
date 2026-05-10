from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class CapitalModelResult:
    coef: np.ndarray
    intercept: float
    r2: float
    model_type: str


class CapitalAccumulationModel:
    """Finite-dimensional approximation with optional regularization.

    ΔK_t = α + β^T x_t + 1/2 x_t^T H x_t + ε_t
    """

    def __init__(
        self,
        feature_names: Sequence[str],
        model_type: Literal["ols", "ridge", "lasso"] = "ridge",
        alpha: float = 1.0,
    ):
        self.feature_names = list(feature_names)
        self.model_type = model_type
        self.alpha = alpha

        if model_type == "ols":
            reg = LinearRegression()
        elif model_type == "ridge":
            reg = Ridge(alpha=alpha)
        elif model_type == "lasso":
            reg = Lasso(alpha=alpha, max_iter=10000)
        else:
            raise ValueError("model_type must be one of: ols, ridge, lasso")

        self.reg = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", reg),
        ])
        self._fitted = False

    def _design_matrix(self, x: np.ndarray) -> np.ndarray:
        n, d = x.shape
        cols = [x]
        lag_x = np.roll(x, shift=1, axis=0)
        lag_x[0, :] = x[0, :]
        cols.append(lag_x)

        quad_terms = []
        for i in range(d):
            for j in range(i, d):
                quad_terms.append((x[:, i] * x[:, j]).reshape(n, 1))
        if quad_terms:
            cols.append(np.hstack(quad_terms))
        return np.hstack(cols)

    def fit(self, k_t: np.ndarray, x_t: np.ndarray, k_tp1: np.ndarray) -> CapitalModelResult:
        y = k_tp1 - k_t
        X = self._design_matrix(x_t)
        self.reg.fit(X, y)
        self._fitted = True

        pred = self.reg.predict(X)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        reg = self.reg.named_steps["reg"]
        return CapitalModelResult(
            coef=np.asarray(reg.coef_).copy(),
            intercept=float(reg.intercept_),
            r2=float(r2),
            model_type=self.model_type,
        )

    def predict_next(self, k_t: np.ndarray, x_t: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model is not fitted.")
        X = self._design_matrix(x_t)
        delta = self.reg.predict(X)
        return k_t + delta

    def marginal_effect(self, x_point: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        if x_point.ndim != 1:
            raise ValueError("x_point must be 1D")
        base = self.reg.predict(self._design_matrix(x_point.reshape(1, -1)))[0]
        grad = np.zeros_like(x_point, dtype=float)
        for i in range(len(x_point)):
            x2 = x_point.copy()
            x2[i] += eps
            bumped = self.reg.predict(self._design_matrix(x2.reshape(1, -1)))[0]
            grad[i] = (bumped - base) / eps
        return grad
