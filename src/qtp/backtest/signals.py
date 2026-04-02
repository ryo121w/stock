"""Signal generation and position sizing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from qtp.models.base import PredictionResult


@dataclass
class Signal:
    date: date
    ticker: str
    direction: int          # 1=long, 0=no position
    confidence: float       # [0, 1]
    expected_magnitude: float


class SignalGenerator:
    """Convert model predictions to trading signals."""

    def __init__(self, confidence_threshold: float = 0.55, magnitude_threshold: float = 0.002):
        self.confidence_threshold = confidence_threshold
        self.magnitude_threshold = magnitude_threshold

    def generate(self, predictions: list[PredictionResult]) -> list[Signal]:
        signals = []
        for pred in predictions:
            if (pred.direction_proba >= self.confidence_threshold
                    and pred.direction == 1
                    and pred.magnitude >= self.magnitude_threshold):
                signals.append(Signal(
                    date=pred.prediction_date,
                    ticker=pred.ticker,
                    direction=1,
                    confidence=pred.direction_proba,
                    expected_magnitude=pred.magnitude,
                ))
        return signals


class PositionSizer:
    """Kelly-inspired position sizing scaled by confidence."""

    def __init__(self, max_position_pct: float = 0.05, total_capital: float = 10_000_000):
        self.max_position_pct = max_position_pct
        self.total_capital = total_capital

    def size(self, signal: Signal) -> float:
        """Return dollar amount for this signal."""
        scale = (signal.confidence - 0.5) / 0.5
        scale = min(max(scale, 0.0), 1.0)
        return self.total_capital * self.max_position_pct * scale
