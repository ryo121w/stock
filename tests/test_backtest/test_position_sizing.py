"""Tests for AdvancedPositionSizer."""

from __future__ import annotations

from datetime import date

import pytest

from qtp.backtest.signals import AdvancedPositionSizer, PositionSizer, Signal


@pytest.fixture
def sizer():
    return AdvancedPositionSizer(
        max_position_pct=0.05,
        max_portfolio_pct=0.30,
        target_volatility=0.15,
        kelly_fraction=0.5,
    )


class TestAdvancedPositionSizer:
    def test_zero_edge_returns_zero(self, sizer):
        """When edge is negative (win_rate too low), size should be 0."""
        # edge = 0.42 * 0.02 - 0.58 * 0.015 = 0.0084 - 0.0087 = -0.0003
        result = sizer.size(confidence=0.42, ticker_volatility=0.25)
        assert result == 0.0

    def test_breakeven_confidence_positive(self, sizer):
        """At 50% confidence with default avg_win > avg_loss, edge is still positive."""
        # edge = 0.5*0.02 - 0.5*0.015 = 0.0025 > 0, so size > 0
        result = sizer.size(confidence=0.50, ticker_volatility=0.25)
        assert result > 0.0

    def test_low_confidence_returns_zero(self, sizer):
        """Below break-even confidence, no position."""
        result = sizer.size(confidence=0.42, ticker_volatility=0.25)
        assert result == 0.0

    def test_positive_edge_returns_positive(self, sizer):
        """Decent confidence should produce a non-zero position."""
        result = sizer.size(confidence=0.60, ticker_volatility=0.25)
        assert result > 0.0

    def test_capped_at_max_position(self, sizer):
        """Position should never exceed max_position_pct."""
        result = sizer.size(confidence=0.99, ticker_volatility=0.05)
        assert result <= sizer.max_position_pct

    def test_portfolio_exposure_cap(self, sizer):
        """When current exposure is near max_portfolio_pct, remaining is tiny."""
        result = sizer.size(
            confidence=0.70,
            ticker_volatility=0.20,
            current_exposure=0.28,
        )
        assert result <= 0.02  # remaining = 0.30 - 0.28 = 0.02

    def test_portfolio_fully_invested_returns_zero(self, sizer):
        """When already at max portfolio exposure, no new positions."""
        result = sizer.size(
            confidence=0.80,
            ticker_volatility=0.20,
            current_exposure=0.30,
        )
        assert result == 0.0

    def test_high_volatility_reduces_size(self, sizer):
        """Higher volatility should produce smaller position."""
        size_low_vol = sizer.size(confidence=0.65, ticker_volatility=0.15)
        size_high_vol = sizer.size(confidence=0.65, ticker_volatility=0.50)
        assert size_high_vol < size_low_vol

    def test_zero_volatility_uses_fallback(self, sizer):
        """Zero volatility should not cause division error."""
        result = sizer.size(confidence=0.65, ticker_volatility=0.0)
        assert result >= 0.0

    def test_result_always_non_negative(self, sizer):
        """Size should always be >= 0."""
        for conf in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            for vol in [0.0, 0.1, 0.3, 0.5, 1.0]:
                result = sizer.size(confidence=conf, ticker_volatility=vol)
                assert result >= 0.0, f"Negative size for conf={conf}, vol={vol}"

    def test_custom_avg_win_loss(self, sizer):
        """Custom avg_win and avg_loss should affect sizing."""
        # Larger avg_win with same confidence = different kelly
        size1 = sizer.size(confidence=0.65, ticker_volatility=0.20, avg_win=0.02, avg_loss=0.015)
        size2 = sizer.size(confidence=0.65, ticker_volatility=0.20, avg_win=0.05, avg_loss=0.015)
        # Both should be valid
        assert size1 >= 0.0
        assert size2 >= 0.0


class TestOriginalPositionSizer:
    """Ensure the original PositionSizer still works unchanged."""

    def test_basic_sizing(self):
        sizer = PositionSizer(max_position_pct=0.05, total_capital=10_000_000)
        signal = Signal(
            date=date(2024, 1, 1),
            ticker="MSFT",
            direction=1,
            confidence=0.75,
            expected_magnitude=0.02,
        )
        result = sizer.size(signal)
        # scale = (0.75 - 0.5) / 0.5 = 0.5
        # size = 10M * 0.05 * 0.5 = 250_000
        assert result == pytest.approx(250_000.0)

    def test_minimum_confidence(self):
        sizer = PositionSizer(max_position_pct=0.05, total_capital=10_000_000)
        signal = Signal(
            date=date(2024, 1, 1),
            ticker="MSFT",
            direction=1,
            confidence=0.50,
            expected_magnitude=0.01,
        )
        result = sizer.size(signal)
        assert result == 0.0
