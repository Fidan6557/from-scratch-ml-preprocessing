"""Minimal usage examples for the preprocessing components."""

import numpy as np

from ml_preprocessing import OneHotEncoder, SimpleImputer, StandardScaler


numeric = np.array([[1.0, np.nan], [3.0, 10.0], [5.0, 14.0]])
imputed = SimpleImputer(strategy="mean").fit_transform(numeric)
scaled = StandardScaler().fit_transform(imputed)

categories = np.array([["red"], ["blue"], ["red"]], dtype=object)
encoded = OneHotEncoder(drop="first").fit_transform(categories)

print("Scaled numeric features:\n", scaled)
print("Encoded categories:\n", encoded)
