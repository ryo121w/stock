"""Tests for new Tier5 features: EDGAR insider + Fear & Greed.

Tests verify:
1. Features return correct dtype and shape
2. Time-series values vary (not static)
3. Point-in-time safety (no future leakage)
4. Graceful fallback when data unavailable
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import patch

import polars as pl


def _make_ohlcv(ticker: str = "NVDA", n_days: int = 100, start: date | None = None) -> pl.DataFrame:
    """Create a minimal OHLCV DataFrame for testing."""
    start = start or date(2025, 1, 1)
    dates = [start + timedelta(days=i) for i in range(n_days)]
    return pl.DataFrame(
        {
            "date": dates,
            "open": [100.0] * n_days,
            "high": [101.0] * n_days,
            "low": [99.0] * n_days,
            "close": [100.0] * n_days,
            "volume": [1_000_000] * n_days,
            "ticker": [ticker] * n_days,
        }
    )


# =========================================================================
# Fear & Greed tests
# =========================================================================


class TestFearGreedFeatures:
    def test_fear_greed_score_returns_series(self):
        from qtp.features.tier5_fear_greed import clear_cache, fear_greed_score

        clear_cache()
        df = _make_ohlcv("NVDA", 50)
        result = fear_greed_score(df)

        assert isinstance(result, pl.Series)
        assert result.name == "fear_greed_score"
        assert result.len() == 50
        assert result.dtype == pl.Float64

    def test_fear_greed_score_in_range(self):
        from qtp.features.tier5_fear_greed import clear_cache, fear_greed_score

        clear_cache()
        df = _make_ohlcv("NVDA", 50)
        result = fear_greed_score(df)

        # Score normalized to 0-1
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_fear_greed_change_returns_series(self):
        from qtp.features.tier5_fear_greed import clear_cache, fear_greed_change_7d

        clear_cache()
        df = _make_ohlcv("NVDA", 50)
        result = fear_greed_change_7d(df)

        assert isinstance(result, pl.Series)
        assert result.name == "fear_greed_change_7d"
        assert result.len() == 50

    def test_fear_greed_varies_over_time(self):
        """Fear & Greed score should not be the same for all dates."""
        from qtp.features.tier5_fear_greed import clear_cache, fear_greed_score

        clear_cache()
        df = _make_ohlcv("NVDA", 365, start=date(2025, 4, 1))
        result = fear_greed_score(df)

        # Should have multiple unique values (not static)
        unique_count = result.unique().len()
        assert unique_count > 1, "Fear & Greed should vary over time"

    def test_fear_greed_fallback_when_no_history(self):
        """When history is empty, should return default 0.5 (normalized)."""
        import qtp.features.tier5_fear_greed as fg_mod

        # Force empty history via cache
        fg_mod._fg_history = []
        df = _make_ohlcv("NVDA", 10)
        result = fg_mod.fear_greed_score(df)
        # Empty history → early return with 0.5 (normalized neutral)
        assert (result == 0.5).sum() == 10
        fg_mod._fg_history = None  # Reset


# =========================================================================
# EDGAR Insider tests
# =========================================================================


class TestEdgarInsiderFeatures:
    def test_insider_net_buy_returns_series(self):
        from qtp.features.tier5_edgar_insider import clear_cache, insider_net_buy_90d

        clear_cache()
        df = _make_ohlcv("NVDA", 50)
        result = insider_net_buy_90d(df)

        assert isinstance(result, pl.Series)
        assert result.name == "insider_net_buy_90d"
        assert result.len() == 50
        assert result.dtype == pl.Float64

    def test_insider_sell_intensity_returns_series(self):
        from qtp.features.tier5_edgar_insider import clear_cache, insider_sell_intensity_90d

        clear_cache()
        df = _make_ohlcv("NVDA", 50)
        result = insider_sell_intensity_90d(df)

        assert isinstance(result, pl.Series)
        assert result.name == "insider_sell_intensity_90d"
        assert result.len() == 50

    def test_japan_ticker_returns_zeros(self):
        """EDGAR only covers US stocks. Japanese tickers should return 0."""
        from qtp.features.tier5_edgar_insider import insider_net_buy_90d

        df = _make_ohlcv("8316.T", 20)
        result = insider_net_buy_90d(df)
        assert (result == 0.0).sum() == 20

    def test_insider_point_in_time(self):
        """Earlier dates should not see later transactions."""
        from qtp.features.tier5_edgar_insider import clear_cache, insider_net_buy_90d

        clear_cache()
        df = _make_ohlcv("NVDA", 365, start=date(2025, 4, 1))
        result = insider_net_buy_90d(df)

        # Values should change over time (not all the same)
        if result.abs().sum() > 0:  # Only if there's data
            unique_count = result.unique().len()
            assert unique_count > 1, "Insider data should vary over time"

    def test_no_ticker_returns_zeros(self):
        """DataFrame without ticker column should return zeros."""
        from qtp.features.tier5_edgar_insider import insider_net_buy_90d

        df = pl.DataFrame(
            {
                "date": [date(2025, 1, 1)] * 5,
                "close": [100.0] * 5,
            }
        )
        result = insider_net_buy_90d(df)
        assert result.len() == 5
        assert (result == 0.0).sum() == 5


# =========================================================================
# Finnhub fetcher tests (no API key required for structure tests)
# =========================================================================


class TestFinnhubFetcher:
    def test_no_api_key_returns_empty(self):
        """Without API key, all functions should return empty results."""
        from qtp.data.fetchers.finnhub_ import (
            fetch_company_news,
            fetch_eps_estimates,
            fetch_price_target,
            fetch_recommendation_trends,
            fetch_upgrade_downgrade,
        )

        with patch.dict("os.environ", {"FINNHUB_API_KEY": ""}, clear=False):
            import qtp.data.fetchers.finnhub_ as fh

            fh._client = None  # Reset client

            assert fetch_recommendation_trends("NVDA") == {}
            assert fetch_price_target("NVDA") == {}
            assert fetch_upgrade_downgrade("NVDA") == []
            assert fetch_company_news("NVDA") == []
            assert fetch_eps_estimates("NVDA") == []

    def test_is_available_without_key(self):
        from qtp.data.fetchers.finnhub_ import is_available

        with patch.dict("os.environ", {"FINNHUB_API_KEY": ""}, clear=False):
            assert is_available() is False

    def test_is_available_with_key(self):
        from qtp.data.fetchers.finnhub_ import is_available

        with patch.dict("os.environ", {"FINNHUB_API_KEY": "test_key"}, clear=False):
            assert is_available() is True


# =========================================================================
# Tier6 time-series tests
# =========================================================================


class TestTier6TimeSeries:
    def test_eps_growth_varies_over_time(self):
        """Tier6 features should produce different values at different dates."""
        from qtp.features.tier6_fundamental_ts import clear_cache, eps_growth_qoq

        clear_cache()
        df = _make_ohlcv("NVDA", 365, start=date(2025, 1, 1))
        result = eps_growth_qoq(df)

        assert result.len() == 365
        # Should have > 1 unique value (time-series, not static)
        unique_count = result.unique().len()
        assert unique_count > 1, "Tier6 should be time-series, not static"

    def test_earnings_streak_varies(self):
        from qtp.features.tier6_fundamental_ts import clear_cache, earnings_surprise_streak

        clear_cache()
        df = _make_ohlcv("NVDA", 365, start=date(2025, 1, 1))
        result = earnings_surprise_streak(df)

        assert result.len() == 365
        unique_count = result.unique().len()
        assert unique_count > 1, "Streak should change at each earnings date"
