"""Abstract model wrapper with prediction result."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import polars as pl


@dataclass
class PredictionResult:
    ticker: str
    prediction_date: date       # Date the prediction is FOR (next trading day)
    direction: int              # 1=up, 0=down
    direction_proba: float      # Confidence [0.0, 1.0]
    magnitude: float            # Expected return magnitude
    model_version: str
    features_used: list[str] = field(default_factory=list)


class ModelWrapper(abc.ABC):
    version: str = ""

    @abc.abstractmethod
    def fit(self, X: pl.DataFrame, y_direction: pl.Series, y_magnitude: pl.Series) -> None:
        ...

    @abc.abstractmethod
    def predict_proba(self, X: pl.DataFrame) -> list[float]:
        """Return probability of direction=1 for each row."""
        ...

    @abc.abstractmethod
    def predict_magnitude(self, X: pl.DataFrame) -> list[float]:
        """Return predicted magnitude for each row."""
        ...

    @abc.abstractmethod
    def get_params(self) -> dict:
        ...

    @abc.abstractmethod
    def save(self, path: Path) -> None:
        ...

    @classmethod
    @abc.abstractmethod
    def load(cls, path: Path) -> ModelWrapper:
        ...
