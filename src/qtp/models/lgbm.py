"""Dual-head LightGBM: classifier (direction) + regressor (magnitude).

Includes optional isotonic calibration to fix probability estimates.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
import structlog
from sklearn.calibration import CalibratedClassifierCV

from qtp.models.base import ModelWrapper

logger = structlog.get_logger()


class LGBMPipeline(ModelWrapper):
    def __init__(
        self,
        clf_params: dict | None = None,
        reg_params: dict | None = None,
        calibrate: bool = True,
    ):
        self.clf = lgb.LGBMClassifier(**(clf_params or self._default_clf_params()))
        self.reg = lgb.LGBMRegressor(**(reg_params or self._default_reg_params()))
        self.calibrator: CalibratedClassifierCV | None = None
        self.calibrate = calibrate
        self.feature_names: list[str] = []
        self.version: str = ""

    @staticmethod
    def _default_clf_params() -> dict:
        return {
            "n_estimators": 1000,
            "learning_rate": 0.01,
            "max_depth": 6,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

    @staticmethod
    def _default_reg_params() -> dict:
        return {
            "n_estimators": 1000,
            "learning_rate": 0.01,
            "max_depth": 6,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
            "objective": "regression",
        }

    def fit(
        self,
        X: pl.DataFrame,
        y_direction: pl.Series,
        y_magnitude: pl.Series,
        sample_weight: pl.Series | None = None,
    ) -> None:
        X_pd = X.to_pandas()
        y_dir_pd = y_direction.to_pandas()
        y_mag_pd = y_magnitude.to_pandas()
        w_pd = sample_weight.to_pandas() if sample_weight is not None else None

        self.feature_names = list(X.columns)

        # Split 80/20 for early stopping validation (time-series: use last 20%)
        split_idx = int(len(X_pd) * 0.8)
        X_train, X_val = X_pd.iloc[:split_idx], X_pd.iloc[split_idx:]
        y_dir_train, y_dir_val = y_dir_pd.iloc[:split_idx], y_dir_pd.iloc[split_idx:]
        y_mag_train, y_mag_val = y_mag_pd.iloc[:split_idx], y_mag_pd.iloc[split_idx:]
        w_train = w_pd.iloc[:split_idx] if w_pd is not None else None

        logger.info(
            "training_classifier",
            n_samples=len(X_train),
            n_val_samples=len(X_val),
            n_features=len(self.feature_names),
        )
        self.clf.fit(
            X_train,
            y_dir_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_dir_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        logger.info("clf_early_stopped", best_iteration=self.clf.best_iteration_)

        logger.info("training_regressor", n_samples=len(X_train))
        self.reg.fit(
            X_train,
            y_mag_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_mag_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        logger.info("reg_early_stopped", best_iteration=self.reg.best_iteration_)

        # Isotonic calibration on validation set
        if self.calibrate and len(X_val) >= 50:
            raw_proba = self.clf.predict_proba(X_val)[:, 1]
            from sklearn.isotonic import IsotonicRegression

            self.calibrator = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
            self.calibrator.fit(raw_proba, y_dir_val.values)
            cal_proba = self.calibrator.predict(raw_proba)
            logger.info(
                "calibration_applied",
                raw_mean=round(float(np.mean(raw_proba)), 3),
                cal_mean=round(float(np.mean(cal_proba)), 3),
                val_accuracy=round(float(np.mean((raw_proba >= 0.5) == y_dir_val.values)), 3),
            )

        self.version = f"lgbm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info("training_complete", version=self.version)

    def predict_proba(self, X: pl.DataFrame) -> list[float]:
        X_pd = X.to_pandas()
        proba = self.clf.predict_proba(X_pd)[:, 1]
        if self.calibrator is not None:
            proba = self.calibrator.predict(proba)
        return proba.tolist()

    def predict_magnitude(self, X: pl.DataFrame) -> list[float]:
        X_pd = X.to_pandas()
        mag = self.reg.predict(X_pd)
        return mag.tolist()

    def get_params(self) -> dict:
        return {
            "clf": self.clf.get_params(),
            "reg": self.reg.get_params(),
        }

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.clf, path / "clf.joblib")
        joblib.dump(self.reg, path / "reg.joblib")
        if self.calibrator is not None:
            joblib.dump(self.calibrator, path / "calibrator.joblib")
        metadata = {
            "version": self.version,
            "feature_names": self.feature_names,
            "clf_params": {k: str(v) for k, v in self.clf.get_params().items()},
            "reg_params": {k: str(v) for k, v in self.reg.get_params().items()},
            "created_at": datetime.now().isoformat(),
        }
        (path / "metadata.json").write_text(json.dumps(metadata, indent=2))
        logger.info("model_saved", path=str(path))

    @classmethod
    def load(cls, path: Path) -> LGBMPipeline:
        instance = cls()
        instance.clf = joblib.load(path / "clf.joblib")
        instance.reg = joblib.load(path / "reg.joblib")
        cal_path = path / "calibrator.joblib"
        if cal_path.exists():
            instance.calibrator = joblib.load(cal_path)
        metadata = json.loads((path / "metadata.json").read_text())
        instance.version = metadata["version"]
        instance.feature_names = metadata["feature_names"]
        logger.info(
            "model_loaded", version=instance.version, calibrated=instance.calibrator is not None
        )
        return instance
