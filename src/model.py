from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.linear_model import LinearRegression


@dataclass
class CapitalModelResult:
    coef: np.ndarray
    intercept: float
    r2: float


class CapitalAccumulationModel:
    """Finite-dimensional approximation of
    K_{t+1} = K_t + alpha + beta^T x_t + 1/2 x_t^T H x_t + eps_t.
    """

    def __init__(self, feature_names: Sequence[str]):
        self.feature_names = list(feature_names)
        self.reg = LinearRegression()
        self._fitted = False

    def _design_matrix(self, x: np.ndarray) -> np.ndarray:
        n, d = x.shape
        cols = [x]
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
        return CapitalModelResult(
            coef=self.reg.coef_.copy(),
            intercept=float(self.reg.intercept_),
            r2=float(self.reg.score(X, y)),
        )

    def predict_next(self, k_t: np.ndarray, x_t: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model is not fitted.")
        X = self._design_matrix(x_t)
        delta = self.reg.predict(X)
        return k_t + delta

    def marginal_effect(self, x_point: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Numerical gradient of ΔK wrt x at a point."""
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
