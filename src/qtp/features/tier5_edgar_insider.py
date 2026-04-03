"""Tier 5: SEC EDGAR insider transaction features.

Unlike the old Tier5 (MCP snapshot), these features use FULL HISTORICAL DATA
from SEC EDGAR Form 4 filings. Each row gets a rolling window calculation
based on transactions available up to that date (point-in-time safe).

Data source: EdgarTools → SEC EDGAR (free, no rate limit, decades of history)
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import structlog

from qtp.features.registry import FeatureRegistry, FeatureTier

reg = FeatureRegistry.instance()
logger = structlog.get_logger()

# Cache per ticker per session
_txn_cache: dict[str, list[dict]] = {}


def _get_transactions(ticker: str) -> list[dict]:
    """Get insider transactions for ticker. Cached per session."""
    if ticker in _txn_cache:
        return _txn_cache[ticker]

    try:
        from qtp.data.fetchers.edgar_ import fetch_insider_transactions

        txns = fetch_insider_transactions(ticker, months=36, max_filings=100)
        _txn_cache[ticker] = txns
        return txns
    except Exception as e:
        logger.warning("edgar_feature_load_failed", ticker=ticker, error=str(e))
        _txn_cache[ticker] = []
        return []


def _infer_ticker(df: pl.DataFrame) -> str | None:
    if "ticker" in df.columns:
        tickers = df["ticker"].unique().to_list()
        if len(tickers) == 1:
            return tickers[0]
    return None


def _parse_date(d) -> date | None:
    """Parse various date formats to date object."""
    if isinstance(d, date):
        return d
    if hasattr(d, "date"):
        return d.date()
    try:
        return date.fromisoformat(str(d)[:10])
    except (ValueError, TypeError):
        return None


# =============================================================================
# Feature: insider_net_buy_90d
# =============================================================================


@reg.register(
    "insider_net_buy_90d",
    FeatureTier.TIER5_ALTERNATIVE,
    lookback_days=1,
    description="Net insider buys minus sells in rolling 90-day window (SEC Form 4)",
)
def insider_net_buy_90d(df: pl.DataFrame) -> pl.Series:
    """For each date, count (buys - sells) in the preceding 90 days."""
    n = df.height
    ticker = _infer_ticker(df)
    if not ticker or ticker.endswith(".T"):  # EDGAR = US stocks only
        return pl.Series("insider_net_buy_90d", [0.0] * n, dtype=pl.Float64)

    txns = _get_transactions(ticker)
    if not txns:
        return pl.Series("insider_net_buy_90d", [0.0] * n, dtype=pl.Float64)

    # Pre-parse transaction dates
    parsed_txns = []
    for t in txns:
        d = _parse_date(t.get("date"))
        if d:
            parsed_txns.append((d, t["type"]))

    dates = df["date"].to_list()
    result = []
    for row_date in dates:
        d = _parse_date(row_date)
        if d is None:
            result.append(0.0)
            continue
        window_start = d - timedelta(days=90)
        buys = sum(1 for td, tt in parsed_txns if window_start <= td <= d and tt == "BUY")
        sells = sum(1 for td, tt in parsed_txns if window_start <= td <= d and tt == "SELL")
        result.append(float(buys - sells))

    return pl.Series("insider_net_buy_90d", result, dtype=pl.Float64)


# =============================================================================
# Feature: insider_sell_intensity_90d
# =============================================================================


@reg.register(
    "insider_sell_intensity_90d",
    FeatureTier.TIER5_ALTERNATIVE,
    lookback_days=1,
    description="Total insider sell value ($) in rolling 90-day window, log-scaled",
)
def insider_sell_intensity_90d(df: pl.DataFrame) -> pl.Series:
    """For each date, sum sell value in preceding 90 days (log10 scaled)."""
    import numpy as np

    n = df.height
    ticker = _infer_ticker(df)
    if not ticker or ticker.endswith(".T"):
        return pl.Series("insider_sell_intensity_90d", [0.0] * n, dtype=pl.Float64)

    txns = _get_transactions(ticker)
    if not txns:
        return pl.Series("insider_sell_intensity_90d", [0.0] * n, dtype=pl.Float64)

    parsed_txns = []
    for t in txns:
        d = _parse_date(t.get("date"))
        if d and t["type"] == "SELL":
            parsed_txns.append((d, abs(t.get("value", 0))))

    dates = df["date"].to_list()
    result = []
    for row_date in dates:
        d = _parse_date(row_date)
        if d is None:
            result.append(0.0)
            continue
        window_start = d - timedelta(days=90)
        total_sell = sum(val for td, val in parsed_txns if window_start <= td <= d)
        # Log scale: $0 → 0, $1M → 6, $10M → 7, $100M → 8
        result.append(np.log10(total_sell + 1) if total_sell > 0 else 0.0)

    return pl.Series("insider_sell_intensity_90d", result, dtype=pl.Float64)


def clear_cache() -> None:
    _txn_cache.clear()
