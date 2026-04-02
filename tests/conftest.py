"""Shared test fixtures."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest


@pytest.fixture
def sample_ohlcv() -> pl.DataFrame:
    """Generate 100 days of synthetic OHLCV data."""
    import numpy as np

    np.random.seed(42)
    n = 250
    dates = pl.date_range(date(2024, 1, 2), date(2024, 12, 31), eager=True)[:n]
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    close = np.maximum(close, 10)  # Ensure positive

    return pl.DataFrame({
        "date": dates,
        "open": close + np.random.randn(n) * 0.3,
        "high": close + abs(np.random.randn(n) * 0.5),
        "low": close - abs(np.random.randn(n) * 0.5),
        "close": close,
        "volume": np.random.randint(1_000_000, 10_000_000, n).astype(float),
    })
