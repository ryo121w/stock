"""Tier 6: Fundamental time-series features from quarterly financial data.

Unlike Tier3 (price-derived proxies) and Tier5 (MCP snapshots),
Tier6 uses actual quarterly financial statement data from yfinance
to compute growth trajectories, earnings momentum, and surprise patterns.

Data source: yfinance Ticker.quarterly_income_stmt / quarterly_balance_sheet / earnings_dates

These features are STATIC per ticker (same value for all dates in the OHLCV).
The model uses them as cross-sectional discriminators across tickers.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import structlog
import yfinance as yf

from qtp.features.registry import FeatureRegistry, FeatureTier

reg = FeatureRegistry.instance()
logger = structlog.get_logger()

# Cache quarterly data per ticker per session
_quarterly_cache: dict[str, dict] = {}


def _load_quarterly(ticker: str) -> dict:
    """Load quarterly financial data from yfinance. Cached per session."""
    if ticker in _quarterly_cache:
        return _quarterly_cache[ticker]

    try:
        tk = yf.Ticker(ticker)
        data: dict = {
            "income": tk.quarterly_income_stmt,
            "balance": tk.quarterly_balance_sheet,
            "earnings_dates": None,
        }
        try:
            data["earnings_dates"] = tk.earnings_dates
        except Exception:
            pass
        _quarterly_cache[ticker] = data
        return data
    except Exception as e:
        logger.warning("tier6_load_failed", ticker=ticker, error=str(e))
        _quarterly_cache[ticker] = {}
        return {}


def _infer_ticker(df: pl.DataFrame) -> str | None:
    if "ticker" in df.columns:
        tickers = df["ticker"].unique().to_list()
        if len(tickers) == 1:
            return tickers[0]
    return None


def _static_series(name: str, n: int, value: float) -> pl.Series:
    """Return a series filled with a static value."""
    return pl.Series(name, [value] * n, dtype=pl.Float64)


def _null_series(name: str, n: int) -> pl.Series:
    return pl.Series(name, [0.0] * n, dtype=pl.Float64)


def _safe_get_row(income_df, row_name: str) -> list[float] | None:
    """Extract a row from yfinance quarterly statement by name.

    yfinance DataFrames have metric names as index and quarter-end dates as columns.
    Columns are ordered newest-first (left = most recent quarter).
    Returns list of float values (newest first), or None if row not found.
    """
    if income_df is None or income_df.empty:
        return None
    # Try exact match first, then partial match
    for name in [row_name]:
        if name in income_df.index:
            vals = income_df.loc[name].tolist()
            return [float(v) if v is not None and not np.isnan(float(v)) else None for v in vals]
    return None


# =============================================================================
# Feature: eps_growth_qoq
# =============================================================================


@reg.register(
    "eps_growth_qoq",
    FeatureTier.TIER6_FUNDAMENTAL_TS,
    lookback_days=1,
    description="Quarter-over-quarter EPS growth rate (latest vs previous quarter)",
)
def eps_growth_qoq(df: pl.DataFrame) -> pl.Series:
    n = df.height
    ticker = _infer_ticker(df)
    if not ticker:
        return _null_series("eps_growth_qoq", n)

    data = _load_quarterly(ticker)
    income = data.get("income")
    if income is None or (hasattr(income, "empty") and income.empty):
        return _null_series("eps_growth_qoq", n)

    eps_vals = _safe_get_row(income, "Basic EPS")
    if eps_vals is None:
        # Try alternative row names
        eps_vals = _safe_get_row(income, "Diluted EPS")
    if eps_vals is None or len(eps_vals) < 2:
        return _null_series("eps_growth_qoq", n)

    # eps_vals[0] = most recent, eps_vals[1] = previous quarter
    latest, prev = eps_vals[0], eps_vals[1]
    if latest is None or prev is None or prev == 0:
        return _null_series("eps_growth_qoq", n)

    growth = (latest - prev) / abs(prev)
    return _static_series("eps_growth_qoq", n, growth)


# =============================================================================
# Feature: revenue_growth_qoq
# =============================================================================


@reg.register(
    "revenue_growth_qoq",
    FeatureTier.TIER6_FUNDAMENTAL_TS,
    lookback_days=1,
    description="Quarter-over-quarter revenue growth rate",
)
def revenue_growth_qoq(df: pl.DataFrame) -> pl.Series:
    n = df.height
    ticker = _infer_ticker(df)
    if not ticker:
        return _null_series("revenue_growth_qoq", n)

    data = _load_quarterly(ticker)
    income = data.get("income")
    if income is None or (hasattr(income, "empty") and income.empty):
        return _null_series("revenue_growth_qoq", n)

    rev_vals = _safe_get_row(income, "Total Revenue")
    if rev_vals is None:
        rev_vals = _safe_get_row(income, "Operating Revenue")
    if rev_vals is None or len(rev_vals) < 2:
        return _null_series("revenue_growth_qoq", n)

    latest, prev = rev_vals[0], rev_vals[1]
    if latest is None or prev is None or prev == 0:
        return _null_series("revenue_growth_qoq", n)

    growth = (latest - prev) / abs(prev)
    return _static_series("revenue_growth_qoq", n, growth)


# =============================================================================
# Feature: earnings_surprise_avg
# =============================================================================


@reg.register(
    "earnings_surprise_avg",
    FeatureTier.TIER6_FUNDAMENTAL_TS,
    lookback_days=1,
    description="Average earnings surprise (%) over last 4 reported quarters",
)
def earnings_surprise_avg(df: pl.DataFrame) -> pl.Series:
    n = df.height
    ticker = _infer_ticker(df)
    if not ticker:
        return _null_series("earnings_surprise_avg", n)

    data = _load_quarterly(ticker)
    ed = data.get("earnings_dates")
    if ed is None or (hasattr(ed, "empty") and ed.empty):
        return _null_series("earnings_surprise_avg", n)

    try:
        # earnings_dates has columns: 'EPS Estimate', 'Reported EPS', 'Surprise(%)'
        # Rows are indexed by date, both past and future. Filter for reported quarters.
        if "Surprise(%)" in ed.columns:
            surprises = ed["Surprise(%)"].dropna()
            if len(surprises) == 0:
                return _null_series("earnings_surprise_avg", n)
            # Take last 4 reported quarters
            recent = surprises.head(4) if len(surprises) >= 4 else surprises
            avg_surprise = float(recent.mean())
            # Normalize: yfinance Surprise(%) is already in percentage form
            # Convert to fraction for consistency (e.g., 5% -> 0.05)
            avg_surprise_frac = avg_surprise / 100.0
            return _static_series("earnings_surprise_avg", n, avg_surprise_frac)
        return _null_series("earnings_surprise_avg", n)
    except Exception:
        return _null_series("earnings_surprise_avg", n)


# =============================================================================
# Feature: earnings_surprise_streak
# =============================================================================


@reg.register(
    "earnings_surprise_streak",
    FeatureTier.TIER6_FUNDAMENTAL_TS,
    lookback_days=1,
    description="Consecutive earnings beats (+) or misses (-) streak count",
)
def earnings_surprise_streak(df: pl.DataFrame) -> pl.Series:
    n = df.height
    ticker = _infer_ticker(df)
    if not ticker:
        return _null_series("earnings_surprise_streak", n)

    data = _load_quarterly(ticker)
    ed = data.get("earnings_dates")
    if ed is None or (hasattr(ed, "empty") and ed.empty):
        return _null_series("earnings_surprise_streak", n)

    try:
        if "Surprise(%)" not in ed.columns:
            return _null_series("earnings_surprise_streak", n)

        surprises = ed["Surprise(%)"].dropna()
        if len(surprises) == 0:
            return _null_series("earnings_surprise_streak", n)

        # Count consecutive beats (positive surprise) from most recent
        streak = 0
        for val in surprises:
            if val > 0:
                streak += 1
            elif val < 0:
                if streak == 0:
                    # Started with misses — count negative streak
                    streak -= 1
                else:
                    break
            else:
                break  # Exact meet, break streak

        return _static_series("earnings_surprise_streak", n, float(streak))
    except Exception:
        return _null_series("earnings_surprise_streak", n)


# =============================================================================
# Feature: operating_margin_trend
# =============================================================================


@reg.register(
    "operating_margin_trend",
    FeatureTier.TIER6_FUNDAMENTAL_TS,
    lookback_days=1,
    description="Change in operating margin (latest quarter minus oldest available quarter)",
)
def operating_margin_trend(df: pl.DataFrame) -> pl.Series:
    n = df.height
    ticker = _infer_ticker(df)
    if not ticker:
        return _null_series("operating_margin_trend", n)

    data = _load_quarterly(ticker)
    income = data.get("income")
    if income is None or (hasattr(income, "empty") and income.empty):
        return _null_series("operating_margin_trend", n)

    op_income = _safe_get_row(income, "Operating Income")
    revenue = _safe_get_row(income, "Total Revenue")
    if revenue is None:
        revenue = _safe_get_row(income, "Operating Revenue")

    if op_income is None or revenue is None:
        return _null_series("operating_margin_trend", n)

    # Compute operating margin for available quarters
    margins = []
    for oi, rev in zip(op_income, revenue):
        if oi is not None and rev is not None and rev != 0:
            margins.append(oi / rev)
        else:
            margins.append(None)

    # Need at least 2 data points
    valid_margins = [(i, m) for i, m in enumerate(margins) if m is not None]
    if len(valid_margins) < 2:
        return _null_series("operating_margin_trend", n)

    # Latest margin minus oldest margin (newest is index 0)
    latest_margin = valid_margins[0][1]
    oldest_margin = valid_margins[-1][1]
    trend = latest_margin - oldest_margin

    return _static_series("operating_margin_trend", n, trend)


# =============================================================================
# Feature: net_income_acceleration
# =============================================================================


@reg.register(
    "net_income_acceleration",
    FeatureTier.TIER6_FUNDAMENTAL_TS,
    lookback_days=1,
    description="Net income growth acceleration (recent growth rate minus older growth rate)",
)
def net_income_acceleration(df: pl.DataFrame) -> pl.Series:
    n = df.height
    ticker = _infer_ticker(df)
    if not ticker:
        return _null_series("net_income_acceleration", n)

    data = _load_quarterly(ticker)
    income = data.get("income")
    if income is None or (hasattr(income, "empty") and income.empty):
        return _null_series("net_income_acceleration", n)

    ni_vals = _safe_get_row(income, "Net Income")
    if ni_vals is None or len(ni_vals) < 4:
        return _null_series("net_income_acceleration", n)

    # ni_vals: [Q0, Q1, Q2, Q3] where Q0 = most recent
    # Recent growth: Q0 vs Q1
    # Older growth: Q2 vs Q3
    q0, q1, q2, q3 = ni_vals[0], ni_vals[1], ni_vals[2], ni_vals[3]

    if any(v is None for v in [q0, q1, q2, q3]):
        return _null_series("net_income_acceleration", n)

    if q1 == 0 or q3 == 0:
        return _null_series("net_income_acceleration", n)

    recent_growth = (q0 - q1) / abs(q1)
    older_growth = (q2 - q3) / abs(q3)
    acceleration = recent_growth - older_growth

    return _static_series("net_income_acceleration", n, acceleration)


def clear_cache() -> None:
    """Clear the quarterly data cache (useful for testing or session reset)."""
    _quarterly_cache.clear()
