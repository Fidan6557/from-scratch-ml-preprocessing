"""Scaling utilities implemented with NumPy."""

from __future__ import annotations

import numpy as np


def _as_2d_float_array(X):
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("X must be a 1D or 2D array-like object.")
    return arr


def _check_feature_count(X_arr, expected):
    if X_arr.shape[1] != expected:
        raise ValueError("X has a different number of columns than during fit.")


class StandardScaler:
    """Center each column and divide by its standard deviation."""

    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.n_features_in_ = None

    def fit(self, X, y=None):
        X_arr = _as_2d_float_array(X)
        self.mean_ = np.mean(X_arr, axis=0)
        self.scale_ = np.std(X_arr, axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        self.n_features_in_ = X_arr.shape[1]
        return self

    def transform(self, X):
        if self.mean_ is None:
            raise RuntimeError("StandardScaler must be fitted before transform.")
        X_arr = _as_2d_float_array(X)
        _check_feature_count(X_arr, self.n_features_in_)
        return (X_arr - self.mean_) / self.scale_

    def inverse_transform(self, X):
        if self.mean_ is None:
            raise RuntimeError("StandardScaler must be fitted before inverse_transform.")
        X_arr = _as_2d_float_array(X)
        _check_feature_count(X_arr, self.n_features_in_)
        return X_arr * self.scale_ + self.mean_

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class MinMaxScaler:
    """Map each fitted column to the interval [0, 1]."""

    def __init__(self):
        self.data_min_ = None
        self.data_max_ = None
        self.scale_ = None
        self.n_features_in_ = None

    def fit(self, X, y=None):
        X_arr = _as_2d_float_array(X)
        self.data_min_ = np.min(X_arr, axis=0)
        self.data_max_ = np.max(X_arr, axis=0)
        self.scale_ = self.data_max_ - self.data_min_
        self.scale_[self.scale_ == 0] = 1.0
        self.n_features_in_ = X_arr.shape[1]
        return self

    def transform(self, X):
        if self.data_min_ is None:
            raise RuntimeError("MinMaxScaler must be fitted before transform.")
        X_arr = _as_2d_float_array(X)
        _check_feature_count(X_arr, self.n_features_in_)
        return (X_arr - self.data_min_) / self.scale_

    def inverse_transform(self, X):
        if self.data_min_ is None:
            raise RuntimeError("MinMaxScaler must be fitted before inverse_transform.")
        X_arr = _as_2d_float_array(X)
        _check_feature_count(X_arr, self.n_features_in_)
        return X_arr * self.scale_ + self.data_min_

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class RobustScaler:
    """Center by the median and scale by the interquartile range."""

    def __init__(self):
        self.center_ = None
        self.scale_ = None
        self.n_features_in_ = None

    def fit(self, X, y=None):
        X_arr = _as_2d_float_array(X)
        self.center_ = np.median(X_arr, axis=0)
        q25, q75 = np.percentile(X_arr, [25, 75], axis=0)
        self.scale_ = q75 - q25
        self.scale_[self.scale_ == 0] = 1.0
        self.n_features_in_ = X_arr.shape[1]
        return self

    def transform(self, X):
        if self.center_ is None:
            raise RuntimeError("RobustScaler must be fitted before transform.")
        X_arr = _as_2d_float_array(X)
        _check_feature_count(X_arr, self.n_features_in_)
        return (X_arr - self.center_) / self.scale_

    def inverse_transform(self, X):
        if self.center_ is None:
            raise RuntimeError("RobustScaler must be fitted before inverse_transform.")
        X_arr = _as_2d_float_array(X)
        _check_feature_count(X_arr, self.n_features_in_)
        return X_arr * self.scale_ + self.center_

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
