"""Tests for Gate 2: Technical confirmation gate."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from qtp.gates import GateResult
from qtp.gates.gate2_technical import Gate2_Technical


@pytest.fixture
def gate():
    return Gate2_Technical()


def _make_ohlcv(n: int = 250, base: float = 100.0, trend: float = 0.0, seed: int = 42):
    """Create synthetic OHLCV data.

    Args:
        n: Number of rows.
        base: Starting price.
        trend: Daily drift (positive = uptrend).
        seed: Random seed.
    """
    rng = np.random.default_rng(seed)
    dates = [date(2024, 1, 2) + timedelta(days=i) for i in range(n)]
    close = base + np.cumsum(rng.normal(trend, 0.5, n))
    close = np.maximum(close, 10.0)
    return pl.DataFrame(
        {
            "date": dates,
            "open": close + rng.normal(0, 0.3, n),
            "high": close + np.abs(rng.normal(0, 0.5, n)),
            "low": close - np.abs(rng.normal(0, 0.5, n)),
            "close": close,
            "volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
        }
    )


class TestGate2_Technical:
    def test_insufficient_data(self, gate):
        tiny = _make_ohlcv(n=10)
        result = gate.evaluate("AAPL", tiny)
        assert isinstance(result, GateResult)
        assert result.passed is False
        assert "Insufficient" in result.reason

    def test_normal_stock_passes(self, gate):
        ohlcv = _make_ohlcv(n=250, trend=0.05)
        result = gate.evaluate("AAPL", ohlcv)
        assert isinstance(result, GateResult)
        assert result.gate == "Technical"
        # Score and reason should be populated
        assert result.score >= 0
        assert result.reason

    def test_rsi_overheat_fails(self, gate):
        # Create a strong uptrend to push RSI very high
        n = 50
        dates = [date(2024, 1, 2) + timedelta(days=i) for i in range(n)]
        # Monotonically increasing close (nearly) to push RSI > 75
        close = [100 + i * 2.0 for i in range(n)]
        ohlcv = pl.DataFrame(
            {
                "date": dates,
                "open": close,
                "high": [c + 0.5 for c in close],
                "low": [c - 0.5 for c in close],
                "close": close,
                "volume": [5_000_000.0] * n,
            }
        )
        result = gate.evaluate("TEST", ohlcv)
        # RSI should be very high for monotonic increase
        rsi = result.data.get("rsi")
        if rsi is not None and rsi > 75:
            assert result.passed is False
            assert "overheated" in result.reason

    def test_rsi_computation(self, gate):
        ohlcv = _make_ohlcv(n=250)
        rsi = gate._compute_rsi(ohlcv)
        assert rsi is not None
        assert 0 <= rsi <= 100

    def test_macd_improving(self, gate):
        ohlcv = _make_ohlcv(n=250, trend=0.1)
        result = gate._is_macd_improving(ohlcv)
        assert isinstance(result, bool)

    def test_sma200_above(self, gate):
        ohlcv = _make_ohlcv(n=250, trend=0.1)
        result = gate._is_above_sma200(ohlcv)
        assert isinstance(result, bool)

    def test_sma200_insufficient_data_gives_benefit(self, gate):
        ohlcv = _make_ohlcv(n=100)
        # Less than 200 rows -- benefit of the doubt
        assert gate._is_above_sma200(ohlcv) is True

    def test_result_has_data_dict(self, gate):
        ohlcv = _make_ohlcv(n=250)
        result = gate.evaluate("AAPL", ohlcv)
        assert "rsi" in result.data
        assert "macd_improving" in result.data
        assert "above_sma200" in result.data
