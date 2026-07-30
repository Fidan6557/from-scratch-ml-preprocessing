"""Tests for the custom preprocessing classes."""

import numpy as np
import pytest

from ml_preprocessing.encoders import OneHotEncoder, TargetEncoder
from ml_preprocessing.feature_creation import KBinsDiscretizer, PolynomialFeatures
from ml_preprocessing.imputers import KNNImputer, SimpleImputer
from ml_preprocessing.scalers import MinMaxScaler, RobustScaler, StandardScaler


def test_simple_imputer_mean_uses_training_statistics():
    X_train = np.array([[1.0, np.nan], [3.0, 10.0], [5.0, 14.0]])
    X_test = np.array([[np.nan, 12.0]])

    transformed = SimpleImputer(strategy="mean").fit(X_train).transform(X_test)

    assert np.allclose(transformed, np.array([[3.0, 12.0]]))


def test_simple_imputer_median():
    X = np.array([[1.0], [100.0], [np.nan], [3.0]])
    transformed = SimpleImputer(strategy="median").fit_transform(X)
    assert np.isclose(transformed[2, 0], 3.0)


def test_simple_imputer_rejects_all_missing_column():
    with pytest.raises(ValueError):
        SimpleImputer().fit(np.array([[np.nan], [np.nan]]))


def test_knn_imputer_uses_nearest_observed_neighbours():
    X_train = np.array(
        [[1.0, 1.0], [2.0, 2.0], [10.0, 10.0], [11.0, 11.0]]
    )
    X_test = np.array([[np.nan, 1.4], [10.6, np.nan]])

    transformed = KNNImputer(k=2).fit(X_train).transform(X_test)

    assert np.allclose(transformed, np.array([[1.5, 1.4], [10.6, 10.5]]))


def test_knn_imputer_falls_back_to_training_mean_without_overlap():
    X_train = np.array([[1.0, np.nan], [3.0, np.nan]])
    X_test = np.array([[np.nan, np.nan]])

    transformed = KNNImputer(k=2).fit(X_train).transform(X_test)

    assert np.isclose(transformed[0, 0], 2.0)
    assert np.isfinite(transformed).all()


def test_standard_scaler_round_trip():
    X = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)

    assert np.allclose(Z.mean(axis=0), 0.0)
    assert np.allclose(scaler.inverse_transform(Z), X)


def test_minmax_scaler_bounds_training_data():
    X = np.array([[1.0], [3.0], [5.0]])
    Z = MinMaxScaler().fit_transform(X)
    assert np.allclose(Z.ravel(), np.array([0.0, 0.5, 1.0]))


def test_robust_scaler_uses_median_and_round_trips():
    X = np.array([[1.0, 10.0], [2.0, 20.0], [100.0, 30.0]])
    scaler = RobustScaler().fit(X)
    Z = scaler.transform(X)

    assert np.allclose(np.median(Z, axis=0), 0.0)
    assert np.allclose(scaler.inverse_transform(Z), X)


def test_scaler_checks_feature_count():
    scaler = StandardScaler().fit(np.array([[1.0, 2.0], [3.0, 4.0]]))
    with pytest.raises(ValueError):
        scaler.transform(np.array([[1.0]]))


def test_polynomial_features_degree_2_two_columns():
    X = np.array([[3.0, 5.0]])
    out = PolynomialFeatures(degree=2, include_bias=True).fit_transform(X)
    assert np.allclose(out, np.array([[1.0, 3.0, 5.0, 9.0, 15.0, 25.0]]))


def test_kbins_quantile_shape_and_one_hot_rows():
    X = np.arange(10, dtype=float).reshape(-1, 1)
    out = KBinsDiscretizer(n_bins=5, strategy="quantile").fit_transform(X)

    assert out.shape == (10, 5)
    assert np.allclose(out.sum(axis=1), 1.0)


def test_one_hot_encoder_drops_first_and_ignores_unknowns():
    X_train = np.array([["red"], ["blue"], ["green"]])
    X_test = np.array([["blue"], ["yellow"]])

    encoder = OneHotEncoder(drop="first").fit(X_train)
    out = encoder.transform(X_test)

    assert encoder.get_feature_names_out() == ["x0_blue", "x0_green"]
    assert np.allclose(out, np.array([[1.0, 0.0], [0.0, 0.0]]))


def test_target_encoder_cv_is_finite_and_handles_unknown_category():
    X = np.array([["A"], ["A"], ["B"], ["B"], ["C"], ["C"]])
    y = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.0])

    encoder = TargetEncoder(smoothing=1.0)
    encoded = encoder.fit_transform_cv(X, y, cv=3, random_state=42)
    transformed = encoder.transform(np.array([["A"], ["missing"]]))

    assert encoded.shape == (6, 1)
    assert transformed.shape == (2, 1)
    assert np.isfinite(encoded).all()
    assert np.isclose(transformed[1, 0], y.mean())


def test_yeo_johnson_transformer_produces_finite_output():
    from ml_preprocessing.power_transformers import YeoJohnsonTransformer

    X = np.array([[-4.0], [-1.0], [0.0], [1.0], [4.0], [10.0]])
    transformer = YeoJohnsonTransformer().fit(X)
    transformed = transformer.transform(X)

    assert transformed.shape == X.shape
    assert np.isfinite(transformed).all()
    assert transformer.lambdas_.shape == (1,)


def test_yeo_johnson_checks_feature_count():
    from ml_preprocessing.power_transformers import YeoJohnsonTransformer

    transformer = YeoJohnsonTransformer().fit(np.array([[1.0], [2.0], [3.0]]))
    with pytest.raises(ValueError):
        transformer.transform(np.array([[1.0, 2.0]]))
