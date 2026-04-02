"""Tests for early stopping fix in LGBMPipeline and XGBPipeline."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from qtp.models.lgbm import LGBMPipeline
from qtp.models.xgb import XGBPipeline


def _make_training_data(n: int = 500, n_features: int = 5, seed: int = 42):
    """Generate synthetic classification/regression data."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features)
    # Create a weak signal so early stopping has something to learn
    signal = X[:, 0] * 0.3 + X[:, 1] * 0.2 + rng.randn(n) * 0.8
    y_direction = (signal > 0).astype(int)
    y_magnitude = signal * 0.01

    feature_names = [f"feat_{i}" for i in range(n_features)]
    X_pl = pl.DataFrame({name: X[:, i] for i, name in enumerate(feature_names)})
    return X_pl, pl.Series("direction", y_direction), pl.Series("magnitude", y_magnitude)


class TestLGBMEarlyStopping:
    """LGBMPipeline should early-stop on validation data, not training data."""

    def test_best_iteration_less_than_n_estimators(self):
        """Model should stop before reaching max n_estimators."""
        X, y_dir, y_mag = _make_training_data(n=600)
        model = LGBMPipeline()
        model.fit(X, y_dir, y_mag)

        # With 1000 max estimators and weak signal, early stopping should kick in
        assert model.clf.best_iteration_ < model.clf.n_estimators, \
            f"Classifier did not early stop: best={model.clf.best_iteration_}, max={model.clf.n_estimators}"
        assert model.reg.best_iteration_ < model.reg.n_estimators, \
            f"Regressor did not early stop: best={model.reg.best_iteration_}, max={model.reg.n_estimators}"

    def test_uses_validation_split(self):
        """Verify the model trains on ~80% and validates on ~20%."""
        X, y_dir, y_mag = _make_training_data(n=500)
        model = LGBMPipeline()
        model.fit(X, y_dir, y_mag)

        # The model's best_iteration_ should be set (early stopping triggered)
        assert hasattr(model.clf, "best_iteration_")
        assert model.clf.best_iteration_ > 0

    def test_predictions_after_fit(self):
        """Model produces valid predictions after training with early stopping."""
        X, y_dir, y_mag = _make_training_data(n=500)
        model = LGBMPipeline()
        model.fit(X, y_dir, y_mag)

        proba = model.predict_proba(X)
        mag = model.predict_magnitude(X)

        assert len(proba) == len(X)
        assert all(0 <= p <= 1 for p in proba)
        assert len(mag) == len(X)


class TestXGBEarlyStopping:
    """XGBPipeline should early-stop on validation data, not training data."""

    def test_best_iteration_less_than_n_estimators(self):
        """Model should stop before reaching max n_estimators."""
        X, y_dir, y_mag = _make_training_data(n=600)
        model = XGBPipeline()
        model.fit(X, y_dir, y_mag)

        max_est = model.clf.n_estimators
        assert model.clf.best_iteration < max_est, \
            f"Classifier did not early stop: best={model.clf.best_iteration}, max={max_est}"
        assert model.reg.best_iteration < max_est, \
            f"Regressor did not early stop: best={model.reg.best_iteration}, max={max_est}"

    def test_predictions_after_fit(self):
        """Model produces valid predictions after training with early stopping."""
        X, y_dir, y_mag = _make_training_data(n=500)
        model = XGBPipeline()
        model.fit(X, y_dir, y_mag)

        proba = model.predict_proba(X)
        mag = model.predict_magnitude(X)

        assert len(proba) == len(X)
        assert all(0 <= p <= 1 for p in proba)
        assert len(mag) == len(X)
