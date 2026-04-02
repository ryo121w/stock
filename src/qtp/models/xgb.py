"""XGBoost dual-head pipeline (for ensemble diversity)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import polars as pl
import structlog
import xgboost as xgb

from qtp.models.base import ModelWrapper

logger = structlog.get_logger()


class XGBPipeline(ModelWrapper):
    def __init__(self, clf_params: dict | None = None, reg_params: dict | None = None):
        self.clf = xgb.XGBClassifier(**(clf_params or self._default_clf_params()))
        self.reg = xgb.XGBRegressor(**(reg_params or self._default_reg_params()))
        self.feature_names: list[str] = []
        self.version: str = ""

    @staticmethod
    def _default_clf_params() -> dict:
        return {
            "n_estimators": 800,
            "learning_rate": 0.01,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "logloss",
            "verbosity": 0,
            "early_stopping_rounds": 50,
        }

    @staticmethod
    def _default_reg_params() -> dict:
        return {
            "n_estimators": 800,
            "learning_rate": 0.01,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "rmse",
            "verbosity": 0,
            "early_stopping_rounds": 50,
        }

    def fit(self, X: pl.DataFrame, y_direction: pl.Series, y_magnitude: pl.Series) -> None:
        X_pd = X.to_pandas()
        y_dir_pd = y_direction.to_pandas()
        y_mag_pd = y_magnitude.to_pandas()
        self.feature_names = list(X.columns)

        # Split 80/20 for early stopping validation (time-series: use last 20%)
        split_idx = int(len(X_pd) * 0.8)
        X_train, X_val = X_pd.iloc[:split_idx], X_pd.iloc[split_idx:]
        y_dir_train, y_dir_val = y_dir_pd.iloc[:split_idx], y_dir_pd.iloc[split_idx:]
        y_mag_train, y_mag_val = y_mag_pd.iloc[:split_idx], y_mag_pd.iloc[split_idx:]

        logger.info("xgb_training_classifier", n_samples=len(X_train), n_val_samples=len(X_val))
        self.clf.fit(
            X_train, y_dir_train,
            eval_set=[(X_val, y_dir_val)],
            verbose=False,
        )
        logger.info("xgb_clf_early_stopped", best_iteration=self.clf.best_iteration)

        logger.info("xgb_training_regressor", n_samples=len(X_train))
        self.reg.fit(
            X_train, y_mag_train,
            eval_set=[(X_val, y_mag_val)],
            verbose=False,
        )
        logger.info("xgb_reg_early_stopped", best_iteration=self.reg.best_iteration)
        self.version = f"xgb_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def predict_proba(self, X: pl.DataFrame) -> list[float]:
        return self.clf.predict_proba(X.to_pandas())[:, 1].tolist()

    def predict_magnitude(self, X: pl.DataFrame) -> list[float]:
        return self.reg.predict(X.to_pandas()).tolist()

    def get_params(self) -> dict:
        return {"clf": self.clf.get_params(), "reg": self.reg.get_params()}

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.clf, path / "clf.joblib")
        joblib.dump(self.reg, path / "reg.joblib")
        metadata = {
            "version": self.version,
            "feature_names": self.feature_names,
            "created_at": datetime.now().isoformat(),
        }
        (path / "metadata.json").write_text(json.dumps(metadata, indent=2))

    @classmethod
    def load(cls, path: Path) -> XGBPipeline:
        instance = cls()
        instance.clf = joblib.load(path / "clf.joblib")
        instance.reg = joblib.load(path / "reg.joblib")
        metadata = json.loads((path / "metadata.json").read_text())
        instance.version = metadata["version"]
        instance.feature_names = metadata["feature_names"]
        return instance
