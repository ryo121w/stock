"""Post-hoc probability calibration for overconfident models.

Fixes the overconfidence problem where model outputs 65%+ confidence
but actual hit rate is much lower. Uses isotonic regression (default)
or Platt scaling to map raw probabilities to calibrated ones.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import structlog

logger = structlog.get_logger()


class ProbabilityCalibrator:
    """Post-hoc probability calibration.

    Trained on out-of-sample (validation) predictions, then applied
    to test predictions to produce well-calibrated probabilities.

    Parameters
    ----------
    method : str
        "isotonic" for IsotonicRegression (non-parametric, default)
        "sigmoid" for Platt Scaling (parametric, logistic regression)
    """

    def __init__(self, method: str = "isotonic"):
        if method not in ("isotonic", "sigmoid"):
            raise ValueError(f"method must be 'isotonic' or 'sigmoid', got '{method}'")
        self.method = method
        self.calibrator = None
        self.fitted = False
        self.n_train_samples: int = 0

    def fit(self, raw_proba: np.ndarray, actual_labels: np.ndarray) -> None:
        """Fit calibration model on OOS predicted probabilities and actual labels.

        Parameters
        ----------
        raw_proba : np.ndarray
            Raw model output probabilities, shape (n_samples,)
        actual_labels : np.ndarray
            Actual binary labels (0 or 1), shape (n_samples,)
        """
        raw_proba = np.asarray(raw_proba, dtype=np.float64)
        actual_labels = np.asarray(actual_labels, dtype=np.float64)

        if len(raw_proba) != len(actual_labels):
            raise ValueError(
                f"Length mismatch: raw_proba={len(raw_proba)}, actual_labels={len(actual_labels)}"
            )

        if len(raw_proba) < 10:
            logger.warning("calibration_small_sample", n=len(raw_proba))

        self.n_train_samples = len(raw_proba)

        if self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            self.calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            self.calibrator.fit(raw_proba, actual_labels)
        else:
            from sklearn.linear_model import LogisticRegression

            self.calibrator = LogisticRegression(C=1.0, solver="lbfgs")
            self.calibrator.fit(raw_proba.reshape(-1, 1), actual_labels)

        self.fitted = True
        logger.info(
            "calibrator_fitted",
            method=self.method,
            n_samples=self.n_train_samples,
        )

    def transform(self, raw_proba: np.ndarray) -> np.ndarray:
        """Transform raw probabilities to calibrated probabilities.

        Parameters
        ----------
        raw_proba : np.ndarray
            Raw model output probabilities, shape (n_samples,)

        Returns
        -------
        np.ndarray
            Calibrated probabilities, shape (n_samples,)
        """
        if not self.fitted:
            raise RuntimeError("Calibrator has not been fitted. Call fit() first.")

        raw_proba = np.asarray(raw_proba, dtype=np.float64)

        if self.method == "isotonic":
            return self.calibrator.transform(raw_proba)
        else:
            return self.calibrator.predict_proba(raw_proba.reshape(-1, 1))[:, 1]

    def save(self, path: Path) -> None:
        """Save calibrator to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.calibrator, path / "calibrator.joblib")
        metadata = {
            "method": self.method,
            "n_train_samples": self.n_train_samples,
            "fitted": self.fitted,
            "created_at": datetime.now().isoformat(),
        }
        (path / "calibrator_metadata.json").write_text(json.dumps(metadata, indent=2))
        logger.info("calibrator_saved", path=str(path))

    @classmethod
    def load(cls, path: Path) -> ProbabilityCalibrator:
        """Load calibrator from disk."""
        path = Path(path)
        metadata = json.loads((path / "calibrator_metadata.json").read_text())
        instance = cls(method=metadata["method"])
        instance.calibrator = joblib.load(path / "calibrator.joblib")
        instance.fitted = metadata["fitted"]
        instance.n_train_samples = metadata["n_train_samples"]
        logger.info("calibrator_loaded", method=instance.method)
        return instance
