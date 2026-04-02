"""Stacking ensemble: Level-1 (LGBM + XGB + RF) → Level-2 meta-learner.

Uses time-series aware 3-fold OOF predictions to train the meta-learner,
avoiding any future data leakage.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import structlog
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from qtp.models.base import ModelWrapper
from qtp.models.lgbm import LGBMPipeline
from qtp.models.xgb import XGBPipeline

logger = structlog.get_logger()

# Default best_params.json location (relative to project root)
_DEFAULT_PARAMS_PATH = Path(__file__).resolve().parents[3] / "configs" / "best_params.json"


def _load_best_params(path: Path | None = None) -> dict | None:
    """Load tuned hyperparameters from best_params.json if it exists."""
    p = path or _DEFAULT_PARAMS_PATH
    if p.exists():
        return json.loads(p.read_text())
    return None


class StackingEnsemble(ModelWrapper):
    """Level-1 models + Level-2 meta-learner (stacking).

    Level-1: LGBMPipeline, XGBPipeline, RandomForest
    Level-2: LogisticRegression (classification), Ridge (regression)

    During fit(), 3-fold time-series split generates out-of-fold (OOF)
    predictions from Level-1 models.  Level-2 is trained on those OOF
    predictions.  This prevents the meta-learner from seeing in-sample
    predictions and avoids overfitting.
    """

    def __init__(
        self,
        n_oof_folds: int = 3,
        best_params_path: Path | None = None,
        rf_clf_params: dict | None = None,
        rf_reg_params: dict | None = None,
    ):
        self.n_oof_folds = n_oof_folds
        self.version: str = ""

        # Load tuned params for LGBM / XGB
        bp = _load_best_params(best_params_path)
        lgbm_clf_params = bp["lgbm_clf_params"] if bp else None
        xgb_clf_params = bp["xgb_clf_params"] if bp else None

        # Level-1 model factories (re-created per fold)
        self._lgbm_clf_params = lgbm_clf_params
        self._xgb_clf_params = xgb_clf_params
        self._rf_clf_params = rf_clf_params or {
            "n_estimators": 300,
            "max_depth": 8,
            "min_samples_leaf": 20,
            "max_features": "sqrt",
            "random_state": 42,
            "n_jobs": -1,
        }
        self._rf_reg_params = rf_reg_params or {
            "n_estimators": 300,
            "max_depth": 8,
            "min_samples_leaf": 20,
            "max_features": "sqrt",
            "random_state": 42,
            "n_jobs": -1,
        }

        # Final Level-1 models (trained on full training data after OOF)
        self._lgbm: LGBMPipeline | None = None
        self._xgb: XGBPipeline | None = None
        self._rf_clf: RandomForestClassifier | None = None
        self._rf_reg: RandomForestRegressor | None = None

        # Level-2 meta-learners
        self._meta_clf: LogisticRegression | None = None
        self._meta_reg: Ridge | None = None

        self.feature_names: list[str] = []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_lgbm(self) -> LGBMPipeline:
        return LGBMPipeline(clf_params=self._lgbm_clf_params)

    def _make_xgb(self) -> XGBPipeline:
        return XGBPipeline(clf_params=self._xgb_clf_params)

    def _make_rf_clf(self) -> RandomForestClassifier:
        return RandomForestClassifier(**self._rf_clf_params)

    def _make_rf_reg(self) -> RandomForestRegressor:
        return RandomForestRegressor(**self._rf_reg_params)

    @staticmethod
    def _ts_split_indices(n: int, n_folds: int):
        """Generate time-series aware train/val splits.

        Each fold uses an expanding window: fold k trains on the first
        (k+1)/n_folds fraction of data and validates on the next slice.
        """
        fold_size = n // (n_folds + 1)
        for k in range(n_folds):
            train_end = fold_size * (k + 1)
            val_start = train_end
            val_end = min(train_end + fold_size, n)
            if val_end <= val_start:
                continue
            train_idx = np.arange(0, train_end)
            val_idx = np.arange(val_start, val_end)
            yield train_idx, val_idx

    # ------------------------------------------------------------------
    # ModelWrapper interface
    # ------------------------------------------------------------------

    def fit(self, X: pl.DataFrame, y_direction: pl.Series, y_magnitude: pl.Series) -> None:
        self.feature_names = list(X.columns)
        n = X.height

        X_pd = X.to_pandas()
        y_dir_np = y_direction.to_numpy().astype(int)
        y_mag_np = y_magnitude.to_numpy().astype(float)

        # --- Phase 1: Generate OOF predictions via time-series folds ---
        oof_proba_lgbm = np.full(n, np.nan)
        oof_proba_xgb = np.full(n, np.nan)
        oof_proba_rf = np.full(n, np.nan)
        oof_mag_lgbm = np.full(n, np.nan)
        oof_mag_xgb = np.full(n, np.nan)
        oof_mag_rf = np.full(n, np.nan)

        for fold_i, (tr_idx, va_idx) in enumerate(self._ts_split_indices(n, self.n_oof_folds)):
            logger.info(
                "stacking_oof_fold",
                fold=fold_i,
                train=len(tr_idx),
                val=len(va_idx),
            )

            X_tr = X[tr_idx]
            X_va = X[va_idx]
            y_dir_tr = pl.Series(y_dir_np[tr_idx])
            y_mag_tr = pl.Series(y_mag_np[tr_idx])

            # LGBM
            lgbm = self._make_lgbm()
            lgbm.fit(X_tr, y_dir_tr, y_mag_tr)
            oof_proba_lgbm[va_idx] = np.array(lgbm.predict_proba(X_va))
            oof_mag_lgbm[va_idx] = np.array(lgbm.predict_magnitude(X_va))

            # XGB
            xgb_m = self._make_xgb()
            xgb_m.fit(X_tr, y_dir_tr, y_mag_tr)
            oof_proba_xgb[va_idx] = np.array(xgb_m.predict_proba(X_va))
            oof_mag_xgb[va_idx] = np.array(xgb_m.predict_magnitude(X_va))

            # RandomForest (sklearn, uses pandas)
            X_tr_pd = X_tr.to_pandas()
            X_va_pd = X_va.to_pandas()
            y_dir_tr_np = y_dir_np[tr_idx]
            y_mag_tr_np = y_mag_np[tr_idx]

            rf_clf = self._make_rf_clf()
            rf_clf.fit(X_tr_pd, y_dir_tr_np)
            oof_proba_rf[va_idx] = rf_clf.predict_proba(X_va_pd)[:, 1]

            rf_reg = self._make_rf_reg()
            rf_reg.fit(X_tr_pd, y_mag_tr_np)
            oof_mag_rf[va_idx] = rf_reg.predict(X_va_pd)

        # --- Phase 2: Train Level-2 meta-learner on OOF predictions ---
        # Only use rows where we have OOF predictions (non-NaN)
        valid_mask = ~(np.isnan(oof_proba_lgbm) | np.isnan(oof_proba_xgb) | np.isnan(oof_proba_rf))
        n_valid = valid_mask.sum()
        logger.info("stacking_meta_training", n_valid_oof=int(n_valid), n_total=n)

        meta_X_clf = np.column_stack(
            [oof_proba_lgbm[valid_mask], oof_proba_xgb[valid_mask], oof_proba_rf[valid_mask]]
        )
        meta_y_dir = y_dir_np[valid_mask]

        self._meta_clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        self._meta_clf.fit(meta_X_clf, meta_y_dir)

        meta_X_reg = np.column_stack(
            [oof_mag_lgbm[valid_mask], oof_mag_xgb[valid_mask], oof_mag_rf[valid_mask]]
        )
        meta_y_mag = y_mag_np[valid_mask]

        self._meta_reg = Ridge(alpha=1.0)
        self._meta_reg.fit(meta_X_reg, meta_y_mag)

        logger.info(
            "stacking_meta_weights",
            clf_coef=self._meta_clf.coef_[0].tolist(),
            reg_coef=self._meta_reg.coef_.tolist(),
        )

        # --- Phase 3: Re-train Level-1 models on FULL training data ---
        logger.info("stacking_full_retrain", n_samples=n)
        self._lgbm = self._make_lgbm()
        self._lgbm.fit(X, y_direction, y_magnitude)

        self._xgb = self._make_xgb()
        self._xgb.fit(X, y_direction, y_magnitude)

        self._rf_clf = self._make_rf_clf()
        self._rf_clf.fit(X_pd, y_dir_np)

        self._rf_reg = self._make_rf_reg()
        self._rf_reg.fit(X_pd, y_mag_np)

        self.version = f"stacking_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info("stacking_fit_complete", version=self.version)

    def predict_proba(self, X: pl.DataFrame) -> list[float]:
        X_pd = X.to_pandas()
        p_lgbm = np.array(self._lgbm.predict_proba(X))
        p_xgb = np.array(self._xgb.predict_proba(X))
        p_rf = self._rf_clf.predict_proba(X_pd)[:, 1]

        meta_X = np.column_stack([p_lgbm, p_xgb, p_rf])
        return self._meta_clf.predict_proba(meta_X)[:, 1].tolist()

    def predict_magnitude(self, X: pl.DataFrame) -> list[float]:
        X_pd = X.to_pandas()
        m_lgbm = np.array(self._lgbm.predict_magnitude(X))
        m_xgb = np.array(self._xgb.predict_magnitude(X))
        m_rf = self._rf_reg.predict(X_pd)

        meta_X = np.column_stack([m_lgbm, m_xgb, m_rf])
        return self._meta_reg.predict(meta_X).tolist()

    def get_params(self) -> dict:
        return {
            "n_oof_folds": self.n_oof_folds,
            "lgbm_clf_params": self._lgbm_clf_params,
            "xgb_clf_params": self._xgb_clf_params,
            "rf_clf_params": self._rf_clf_params,
            "rf_reg_params": self._rf_reg_params,
            "meta_clf_coef": (self._meta_clf.coef_[0].tolist() if self._meta_clf else None),
            "meta_reg_coef": (self._meta_reg.coef_.tolist() if self._meta_reg else None),
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save Level-1 models
        self._lgbm.save(path / "lgbm")
        self._xgb.save(path / "xgb")
        joblib.dump(self._rf_clf, path / "rf_clf.joblib")
        joblib.dump(self._rf_reg, path / "rf_reg.joblib")

        # Save Level-2 meta-learners
        joblib.dump(self._meta_clf, path / "meta_clf.joblib")
        joblib.dump(self._meta_reg, path / "meta_reg.joblib")

        # Metadata
        metadata = {
            "version": self.version,
            "feature_names": self.feature_names,
            "n_oof_folds": self.n_oof_folds,
            "meta_clf_coef": self._meta_clf.coef_[0].tolist(),
            "meta_reg_coef": self._meta_reg.coef_.tolist(),
            "created_at": datetime.now().isoformat(),
        }
        (path / "metadata.json").write_text(json.dumps(metadata, indent=2))
        logger.info("stacking_saved", path=str(path))

    @classmethod
    def load(cls, path: Path) -> StackingEnsemble:
        path = Path(path)
        metadata = json.loads((path / "metadata.json").read_text())

        instance = cls(n_oof_folds=metadata["n_oof_folds"])
        instance.version = metadata["version"]
        instance.feature_names = metadata["feature_names"]

        # Load Level-1
        instance._lgbm = LGBMPipeline.load(path / "lgbm")
        instance._xgb = XGBPipeline.load(path / "xgb")
        instance._rf_clf = joblib.load(path / "rf_clf.joblib")
        instance._rf_reg = joblib.load(path / "rf_reg.joblib")

        # Load Level-2
        instance._meta_clf = joblib.load(path / "meta_clf.joblib")
        instance._meta_reg = joblib.load(path / "meta_reg.joblib")

        logger.info("stacking_loaded", version=instance.version)
        return instance
