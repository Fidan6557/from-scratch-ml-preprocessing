"""Categorical encoders implemented with NumPy and pandas-compatible inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd


class OneHotEncoder:
    """Dense one-hot encoder with an optional dropped reference level."""

    def __init__(self, drop="first"):
        if drop not in {"first", None}:
            raise ValueError("drop must be 'first' or None.")
        self.drop = drop
        self.categories_ = None
        self.columns_ = None
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        X_df = self._to_dataframe(X)
        self.columns_ = list(X_df.columns)
        self.categories_ = {}
        feature_names = []

        for column in self.columns_:
            categories = list(pd.unique(X_df[column]))
            self.categories_[column] = categories
            kept = categories[1:] if self.drop == "first" else categories
            feature_names.extend(f"{column}_{category}" for category in kept)

        self.feature_names_out_ = feature_names
        return self

    def transform(self, X):
        if self.categories_ is None:
            raise RuntimeError("OneHotEncoder must be fitted before transform.")

        X_df = self._to_dataframe(X)
        if list(X_df.columns) != self.columns_:
            if X_df.shape[1] != len(self.columns_):
                raise ValueError("X has a different number of columns than during fit.")
            X_df.columns = self.columns_

        encoded = np.zeros((len(X_df), len(self.feature_names_out_)), dtype=float)
        output_col = 0

        for column in self.columns_:
            categories = self.categories_[column]
            kept = categories[1:] if self.drop == "first" else categories
            if not kept:
                continue

            codes = pd.Categorical(X_df[column], categories=kept).codes
            known = codes >= 0
            encoded[np.flatnonzero(known), output_col + codes[known]] = 1.0
            output_col += len(kept)

        return encoded

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self):
        if self.feature_names_out_ is None:
            raise RuntimeError("Encoder has not been fitted.")
        return list(self.feature_names_out_)

    @staticmethod
    def _to_dataframe(X):
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if isinstance(X, pd.Series):
            return X.to_frame()
        arr = np.asarray(X, dtype=object)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError("X must be a 1D or 2D array-like object.")
        return pd.DataFrame(arr, columns=[f"x{i}" for i in range(arr.shape[1])])


class TargetEncoder:
    """Mean target encoder with smoothing and out-of-fold training support."""

    def __init__(self, smoothing=1.0):
        if smoothing < 0:
            raise ValueError("smoothing must be non-negative.")
        self.smoothing = float(smoothing)
        self.global_mean_ = None
        self.category_stats_ = None
        self.columns_ = None

    def fit(self, X, y):
        X_df = OneHotEncoder._to_dataframe(X)
        y_arr = np.asarray(y, dtype=float)
        if len(X_df) != len(y_arr):
            raise ValueError("X and y must have the same number of rows.")
        if len(y_arr) == 0:
            raise ValueError("X and y must contain at least one row.")

        self.global_mean_ = float(np.mean(y_arr))
        self.columns_ = list(X_df.columns)
        target = pd.Series(y_arr, index=X_df.index, name="target")
        self.category_stats_ = {}

        for column in self.columns_:
            stats = target.groupby(X_df[column], dropna=False).agg(["count", "mean"])
            stats["encoding"] = (
                stats["count"] * stats["mean"] + self.smoothing * self.global_mean_
            ) / (stats["count"] + self.smoothing)
            self.category_stats_[column] = stats

        return self

    def transform(self, X):
        if self.category_stats_ is None:
            raise RuntimeError("TargetEncoder must be fitted before transform.")

        X_df = OneHotEncoder._to_dataframe(X)
        if X_df.shape[1] != len(self.columns_):
            raise ValueError("X has a different number of columns than during fit.")
        X_df.columns = self.columns_

        encoded = np.empty((len(X_df), len(self.columns_)), dtype=float)
        for i, column in enumerate(self.columns_):
            mapping = self.category_stats_[column]["encoding"]
            encoded[:, i] = (
                X_df[column].map(mapping).fillna(self.global_mean_).to_numpy(dtype=float)
            )
        return encoded

    def fit_transform_cv(self, X, y, cv=5, random_state=42):
        """Return out-of-fold encodings, then fit on all training rows."""
        X_df = OneHotEncoder._to_dataframe(X)
        y_arr = np.asarray(y, dtype=float)
        if len(X_df) != len(y_arr):
            raise ValueError("X and y must have the same number of rows.")
        if len(X_df) < 2:
            raise ValueError("At least two rows are required for CV target encoding.")

        cv = int(cv)
        if cv < 2:
            raise ValueError("cv must be at least 2.")
        cv = min(cv, len(X_df))

        rng = np.random.default_rng(random_state)
        folds = np.array_split(rng.permutation(len(X_df)), cv)
        encoded = np.empty((len(X_df), X_df.shape[1]), dtype=float)

        for holdout_idx in folds:
            train_mask = np.ones(len(X_df), dtype=bool)
            train_mask[holdout_idx] = False
            fold_encoder = TargetEncoder(smoothing=self.smoothing)
            fold_encoder.fit(X_df.iloc[train_mask], y_arr[train_mask])
            encoded[holdout_idx] = fold_encoder.transform(X_df.iloc[holdout_idx])

        self.fit(X_df, y_arr)
        return encoded
