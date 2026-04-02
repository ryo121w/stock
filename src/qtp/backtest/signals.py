"""Signal generation and position sizing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from qtp.models.base import PredictionResult


@dataclass
class Signal:
    date: date
    ticker: str
    direction: int  # 1=long, 0=no position
    confidence: float  # [0, 1]
    expected_magnitude: float


class SignalGenerator:
    """Convert model predictions to trading signals."""

    def __init__(self, confidence_threshold: float = 0.55, magnitude_threshold: float = 0.002):
        self.confidence_threshold = confidence_threshold
        self.magnitude_threshold = magnitude_threshold

    def generate(self, predictions: list[PredictionResult]) -> list[Signal]:
        signals = []
        for pred in predictions:
            if (
                pred.direction_proba >= self.confidence_threshold
                and pred.direction == 1
                and pred.magnitude >= self.magnitude_threshold
            ):
                signals.append(
                    Signal(
                        date=pred.prediction_date,
                        ticker=pred.ticker,
                        direction=1,
                        confidence=pred.direction_proba,
                        expected_magnitude=pred.magnitude,
                    )
                )
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


class AdvancedPositionSizer:
    """Volatility-targeted position sizing with Half-Kelly and portfolio limits."""

    def __init__(
        self,
        max_position_pct: float = 0.05,
        max_portfolio_pct: float = 0.30,
        target_volatility: float = 0.15,
        kelly_fraction: float = 0.5,
    ):
        self.max_position_pct = max_position_pct
        self.max_portfolio_pct = max_portfolio_pct
        self.target_volatility = target_volatility
        self.kelly_fraction = kelly_fraction

    def size(
        self,
        confidence: float,
        ticker_volatility: float,
        avg_win: float = 0.02,
        avg_loss: float = 0.015,
        current_exposure: float = 0.0,
    ) -> float:
        """Calculate position size as fraction of portfolio.

        Args:
            confidence: Model prediction confidence [0.5, 1.0]
            ticker_volatility: Annualized volatility of the ticker
            avg_win: Historical average winning trade return
            avg_loss: Historical average losing trade return (positive number)
            current_exposure: Current total portfolio exposure [0.0, 1.0]

        Returns:
            Position size as fraction of portfolio [0.0, max_position_pct]
        """
        # 1. Half-Kelly
        win_rate = confidence
        edge = win_rate * avg_win - (1 - win_rate) * avg_loss
        if edge <= 0:
            return 0.0
        kelly = (edge / avg_win) * self.kelly_fraction

        # 2. Volatility targeting
        if ticker_volatility > 0:
            vol_scale = self.target_volatility / ticker_volatility
        else:
            vol_scale = 1.0

        # 3. Combine and cap
        size = min(kelly, vol_scale * self.max_position_pct, self.max_position_pct)

        # 4. Portfolio exposure cap
        remaining = max(0, self.max_portfolio_pct - current_exposure)
        size = min(size, remaining)

        return max(0.0, size)
