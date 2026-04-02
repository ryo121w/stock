"""Weighted ensemble of multiple models."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import structlog

from qtp.models.base import ModelWrapper

logger = structlog.get_logger()


class WeightedEnsemble(ModelWrapper):
    """Weighted average of multiple ModelWrapper predictions."""

    def __init__(self, models: list[tuple[ModelWrapper, float]]):
        total_weight = sum(w for _, w in models)
        self.models = [(m, w / total_weight) for m, w in models]
        self.version = "ensemble_" + "+".join(
            f"{m.version}({w:.2f})" for m, w in self.models
        )

    def fit(self, X: pl.DataFrame, y_direction: pl.Series, y_magnitude: pl.Series) -> None:
        for model, _ in self.models:
            model.fit(X, y_direction, y_magnitude)

    def predict_proba(self, X: pl.DataFrame) -> list[float]:
        weighted = [0.0] * X.height
        for model, weight in self.models:
            probas = model.predict_proba(X)
            for i, p in enumerate(probas):
                weighted[i] += p * weight
        return weighted

    def predict_magnitude(self, X: pl.DataFrame) -> list[float]:
        weighted = [0.0] * X.height
        for model, weight in self.models:
            mags = model.predict_magnitude(X)
            for i, m in enumerate(mags):
                weighted[i] += m * weight
        return weighted

    def get_params(self) -> dict:
        return {f"model_{i}": {"weight": w, "params": m.get_params()}
                for i, (m, w) in enumerate(self.models)}

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        for i, (model, weight) in enumerate(self.models):
            model.save(path / f"model_{i}")

    @classmethod
    def load(cls, path: Path) -> WeightedEnsemble:
        raise NotImplementedError("Load individual models and reconstruct ensemble")
