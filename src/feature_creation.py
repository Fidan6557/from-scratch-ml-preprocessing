"""Feature-generation utilities implemented with NumPy."""

from __future__ import annotations

from itertools import combinations, combinations_with_replacement

import numpy as np


def _as_2d_float_array(X):
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("X must be a 1D or 2D array-like object.")
    return arr


class PolynomialFeatures:
    """Generate polynomial and interaction terms up to a chosen degree."""

    def __init__(self, degree=2, include_bias=True, interaction_only=False):
        if degree < 1:
            raise ValueError("degree must be at least 1.")
        self.degree = int(degree)
        self.include_bias = bool(include_bias)
        self.interaction_only = bool(interaction_only)
        self.powers_ = None
        self.n_features_in_ = None

    def fit(self, X, y=None):
        X_arr = _as_2d_float_array(X)
        self.n_features_in_ = X_arr.shape[1]

        powers = []
        if self.include_bias:
            powers.append((0,) * self.n_features_in_)

        combination_fn = combinations if self.interaction_only else combinations_with_replacement
        for current_degree in range(1, self.degree + 1):
            for combination in combination_fn(range(self.n_features_in_), current_degree):
                power = [0] * self.n_features_in_
                for feature_idx in combination:
                    power[feature_idx] += 1
                powers.append(tuple(power))

        self.powers_ = powers
        return self

    def transform(self, X):
        if self.powers_ is None:
            raise RuntimeError("PolynomialFeatures must be fitted before transform.")

        X_arr = _as_2d_float_array(X)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of columns than during fit.")

        transformed = np.ones((X_arr.shape[0], len(self.powers_)), dtype=float)
        for output_idx, power in enumerate(self.powers_):
            for feature_idx, exponent in enumerate(power):
                if exponent:
                    transformed[:, output_idx] *= X_arr[:, feature_idx] ** exponent
        return transformed

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class KBinsDiscretizer:
    """Create uniform or quantile bins and return dense one-hot columns."""

    def __init__(self, n_bins=10, strategy="uniform", encode="onehot-dense"):
        if n_bins < 2:
            raise ValueError("n_bins must be at least 2.")
        if strategy not in {"uniform", "quantile"}:
            raise ValueError("strategy must be 'uniform' or 'quantile'.")
        if encode != "onehot-dense":
            raise ValueError("Only encode='onehot-dense' is supported.")
        self.n_bins = int(n_bins)
        self.strategy = strategy
        self.encode = encode
        self.bin_edges_ = None
        self.n_features_in_ = None

    def fit(self, X, y=None):
        X_arr = _as_2d_float_array(X)
        self.n_features_in_ = X_arr.shape[1]

        if self.strategy == "uniform":
            mins = np.min(X_arr, axis=0)
            maxs = np.max(X_arr, axis=0)
            self.bin_edges_ = [
                np.linspace(mins[i], maxs[i], self.n_bins + 1)
                for i in range(self.n_features_in_)
            ]
        else:
            percentiles = np.linspace(0, 100, self.n_bins + 1)
            self.bin_edges_ = [
                np.percentile(X_arr[:, i], percentiles)
                for i in range(self.n_features_in_)
            ]

        return self

    def transform(self, X):
        if self.bin_edges_ is None:
            raise RuntimeError("KBinsDiscretizer must be fitted before transform.")

        X_arr = _as_2d_float_array(X)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of columns than during fit.")

        n_samples = X_arr.shape[0]
        transformed = np.zeros((n_samples, self.n_features_in_ * self.n_bins), dtype=float)
        row_indices = np.arange(n_samples)

        for feature_idx, edges in enumerate(self.bin_edges_):
            bin_indices = np.searchsorted(edges[1:-1], X_arr[:, feature_idx], side="right")
            bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
            offset = feature_idx * self.n_bins
            transformed[row_indices, offset + bin_indices] = 1.0

        return transformed

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
