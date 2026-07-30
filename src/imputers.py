"""NumPy-based imputation utilities with a fit/transform interface."""

from __future__ import annotations

import numpy as np


def _as_2d_float_array(X):
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("X must be a 1D or 2D array-like object.")
    return arr


class SimpleImputer:
    """Replace missing numeric values with a fitted mean or median."""

    def __init__(self, strategy="mean"):
        if strategy not in {"mean", "median"}:
            raise ValueError("strategy must be either 'mean' or 'median'.")
        self.strategy = strategy
        self.statistics_ = None
        self.n_features_in_ = None

    def fit(self, X, y=None):
        X_arr = _as_2d_float_array(X)
        self.n_features_in_ = X_arr.shape[1]

        if np.any(np.all(np.isnan(X_arr), axis=0)):
            raise ValueError("Each column must contain at least one observed value.")

        if self.strategy == "mean":
            statistics = np.nanmean(X_arr, axis=0)
        else:
            statistics = np.nanmedian(X_arr, axis=0)

        self.statistics_ = statistics
        return self

    def transform(self, X):
        if self.statistics_ is None:
            raise RuntimeError("SimpleImputer must be fitted before transform.")

        X_arr = _as_2d_float_array(X).copy()
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of columns than during fit.")

        missing_rows, missing_cols = np.where(np.isnan(X_arr))
        X_arr[missing_rows, missing_cols] = self.statistics_[missing_cols]
        return X_arr

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class KNNImputer:
    """Simplified KNN imputer with Euclidean distance on shared values."""

    def __init__(self, k=5):
        if k <= 0:
            raise ValueError("k must be positive.")
        self.k = int(k)
        self.X_fit_ = None
        self.n_features_in_ = None
        self.column_means_ = None

    def fit(self, X, y=None):
        X_arr = _as_2d_float_array(X)
        self.n_features_in_ = X_arr.shape[1]
        self.X_fit_ = X_arr.copy()

        observed_values = self.X_fit_[~np.isnan(self.X_fit_)]
        overall_mean = float(np.mean(observed_values)) if observed_values.size else 0.0
        column_means = np.array([
            np.mean(column[~np.isnan(column)]) if np.any(~np.isnan(column)) else overall_mean
            for column in self.X_fit_.T
        ])
        self.column_means_ = column_means
        return self

    def transform(self, X):
        if self.X_fit_ is None:
            raise RuntimeError("KNNImputer must be fitted before transform.")

        X_arr = _as_2d_float_array(X).copy()
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of columns than during fit.")

        original = X_arr.copy()
        missing_rows, missing_cols = np.where(np.isnan(original))

        for row_idx, col_idx in zip(missing_rows, missing_cols):
            row = original[row_idx]
            candidates = self.X_fit_[~np.isnan(self.X_fit_[:, col_idx])]

            if len(candidates) == 0:
                X_arr[row_idx, col_idx] = self.column_means_[col_idx]
                continue

            shared = (~np.isnan(row)) & (~np.isnan(candidates))
            valid = shared.sum(axis=1) > 0
            if not np.any(valid):
                X_arr[row_idx, col_idx] = self.column_means_[col_idx]
                continue

            candidate_rows = candidates[valid]
            shared = shared[valid]
            differences = np.where(shared, candidate_rows - row, 0.0)
            distances = np.sqrt(np.sum(differences**2, axis=1))

            n_neighbors = min(self.k, len(candidate_rows))
            nearest = np.argsort(distances, kind="stable")[:n_neighbors]
            X_arr[row_idx, col_idx] = np.mean(candidate_rows[nearest, col_idx])

        return X_arr

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
