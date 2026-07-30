"""Power transformations implemented from first principles."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


class YeoJohnsonTransformer:
    """Estimate and apply a Yeo-Johnson transformation column by column.

    Parameters
    ----------
    bounds:
        Search interval used for maximum-likelihood estimation of lambda.
    """

    def __init__(self, bounds=(-2.0, 2.0)):
        if len(bounds) != 2 or bounds[0] >= bounds[1]:
            raise ValueError("bounds must contain two increasing values.")
        self.bounds = tuple(float(value) for value in bounds)
        self.lambdas_ = None
        self.n_features_in_ = None

    @staticmethod
    def _as_2d(X):
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError("X must be a 1D or 2D array-like object.")
        if not np.isfinite(arr).all():
            raise ValueError("X must contain only finite values.")
        return arr

    @staticmethod
    def _transform_column(x, lam):
        z = np.empty_like(x, dtype=float)
        positive = x >= 0

        if abs(lam) < 1e-8:
            z[positive] = np.log1p(x[positive])
        else:
            z[positive] = ((x[positive] + 1.0) ** lam - 1.0) / lam

        if abs(lam - 2.0) < 1e-8:
            z[~positive] = -np.log1p(-x[~positive])
        else:
            z[~positive] = -(
                ((-x[~positive] + 1.0) ** (2.0 - lam) - 1.0)
                / (2.0 - lam)
            )

        return z

    @staticmethod
    def _log_jacobian(x, lam):
        positive = x >= 0
        return (
            (lam - 1.0) * np.log1p(x[positive]).sum()
            + (1.0 - lam) * np.log1p(-x[~positive]).sum()
        )

    def fit(self, X, y=None):
        X_arr = self._as_2d(X)
        self.n_features_in_ = X_arr.shape[1]
        lambdas = []

        for col_idx in range(self.n_features_in_):
            x = X_arr[:, col_idx]

            def objective(lam):
                z = self._transform_column(x, lam)
                variance = np.var(z)
                if variance <= 0 or not np.isfinite(variance):
                    return np.inf
                return (
                    0.5 * len(x) * np.log(variance)
                    - self._log_jacobian(x, lam)
                )

            result = minimize_scalar(
                objective,
                bounds=self.bounds,
                method="bounded",
            )
            if not result.success:
                raise RuntimeError("Unable to estimate a Yeo-Johnson parameter.")
            lambdas.append(result.x)

        self.lambdas_ = np.asarray(lambdas)
        return self

    def transform(self, X):
        if self.lambdas_ is None:
            raise RuntimeError(
                "YeoJohnsonTransformer must be fitted before transform."
            )

        X_arr = self._as_2d(X)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of columns than during fit.")

        transformed = np.empty_like(X_arr, dtype=float)
        for col_idx, lam in enumerate(self.lambdas_):
            transformed[:, col_idx] = self._transform_column(
                X_arr[:, col_idx], lam
            )
        return transformed

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
