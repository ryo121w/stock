"""Tests for cross-sectional features."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from qtp.features.cross_sectional import CROSS_SECTIONAL_FEATURES, compute_cross_sectional_features


@pytest.fixture
def multi_ticker_dataset() -> pl.DataFrame:
    """Synthetic multi-ticker dataset with known values."""
    # 3 tickers, 3 dates each
    return pl.DataFrame(
        {
            "date": [
                date(2024, 1, 1),
                date(2024, 1, 2),
                date(2024, 1, 3),
                date(2024, 1, 1),
                date(2024, 1, 2),
                date(2024, 1, 3),
                date(2024, 1, 1),
                date(2024, 1, 2),
                date(2024, 1, 3),
            ],
            "ticker": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "ret_21d": [0.10, 0.05, -0.02, 0.20, 0.03, 0.01, -0.05, 0.10, 0.04],
            "volume_ratio_20d": [1.2, 0.8, 1.0, 2.0, 1.5, 0.9, 0.5, 0.7, 1.1],
            "realized_vol_21d": [0.15, 0.20, 0.25, 0.30, 0.10, 0.18, 0.22, 0.15, 0.12],
        }
    )


def test_cross_sectional_features_exist(multi_ticker_dataset):
    """All expected cross-sectional columns are added."""
    result = compute_cross_sectional_features(multi_ticker_dataset)
    for col in CROSS_SECTIONAL_FEATURES:
        assert col in result.columns, f"Missing column: {col}"


def test_relative_strength_sums_to_zero(multi_ticker_dataset):
    """Relative strength should sum to ~0 within each date (demeaned)."""
    result = compute_cross_sectional_features(multi_ticker_dataset)
    for d in result["date"].unique().to_list():
        daily = result.filter(pl.col("date") == d)
        assert abs(daily["relative_strength_21d"].sum()) < 1e-10


def test_momentum_rank_range(multi_ticker_dataset):
    """Momentum rank should be in (0, 1]."""
    result = compute_cross_sectional_features(multi_ticker_dataset)
    ranks = result["momentum_rank"]
    assert ranks.min() > 0.0
    assert ranks.max() <= 1.0


def test_volatility_rank_range(multi_ticker_dataset):
    """Volatility rank should be in (0, 1]."""
    result = compute_cross_sectional_features(multi_ticker_dataset)
    ranks = result["volatility_rank"]
    assert ranks.min() > 0.0
    assert ranks.max() <= 1.0


def test_dispersion_non_negative(multi_ticker_dataset):
    """Cross-sectional dispersion (std) should be >= 0."""
    result = compute_cross_sectional_features(multi_ticker_dataset)
    assert (result["cross_sectional_dispersion"] >= 0).all()


def test_no_nulls_with_multiple_tickers(multi_ticker_dataset):
    """No nulls should remain when we have multiple tickers per date."""
    result = compute_cross_sectional_features(multi_ticker_dataset)
    for col in CROSS_SECTIONAL_FEATURES:
        assert result[col].null_count() == 0, f"Nulls in {col}"


def test_missing_columns_graceful():
    """Function should not fail if input columns are missing."""
    df = pl.DataFrame(
        {
            "date": [date(2024, 1, 1), date(2024, 1, 1)],
            "ticker": ["A", "B"],
            "some_other_col": [1.0, 2.0],
        }
    )
    result = compute_cross_sectional_features(df)
    # Should return the original columns without cross-sectional features
    assert "relative_strength_21d" not in result.columns


def test_single_ticker_skipped():
    """With only 1 ticker, cross-sectional features should be skipped."""
    df = pl.DataFrame(
        {
            "date": [date(2024, 1, 1), date(2024, 1, 2)],
            "ticker": ["A", "A"],
            "ret_21d": [0.05, 0.03],
        }
    )
    result = compute_cross_sectional_features(df)
    assert "relative_strength_21d" not in result.columns


def test_missing_date_or_ticker():
    """Without date or ticker columns, should return dataset unchanged."""
    df = pl.DataFrame(
        {
            "ret_21d": [0.05, 0.03],
        }
    )
    result = compute_cross_sectional_features(df)
    assert result.columns == ["ret_21d"]
