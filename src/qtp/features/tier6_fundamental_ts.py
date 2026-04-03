"""Tier 6: Fundamental time-series features from quarterly financial data.

Unlike Tier3 (price-derived proxies) and Tier5 (MCP snapshots),
Tier6 uses actual quarterly financial statement data from yfinance
to compute growth trajectories, earnings momentum, and surprise patterns.

Data source: yfinance Ticker.quarterly_income_stmt / quarterly_balance_sheet / earnings_dates

These features are TIME-SERIES: each row gets the value corresponding to the
most recent earnings report available on that date. This means the model can
learn from temporal changes (e.g., NVDA going from 3 consecutive beats to 13).

Anti-leakage: We use earnings report dates (not quarter-end dates) to determine
when data becomes publicly available. A row on 2025-01-15 only sees quarterly
data from reports filed before that date.
"""

from __future__ import annotations

from datetime import timedelta

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
    if row_name in income_df.index:
        vals = income_df.loc[row_name].tolist()
        return [float(v) if v is not None and not np.isnan(float(v)) else None for v in vals]
    return None


def _get_report_dates(data: dict) -> list:
    """Get earnings report dates from yfinance earnings_dates.

    Returns list of (report_date, quarter_index) sorted oldest-first,
    where quarter_index 0 = most recent quarter in the income statement.

    Falls back to quarter-end + 45 days if earnings_dates unavailable.
    """
    import pandas as pd

    income = data.get("income")
    if income is None or income.empty:
        return []

    # Quarter-end dates from income statement columns (newest first)
    quarter_ends = list(income.columns)

    ed = data.get("earnings_dates")
    if ed is not None and not ed.empty:
        # earnings_dates index = report dates, has columns like 'Reported EPS'
        # Filter to only past reports (those with actual reported EPS)
        try:
            reported = ed[ed["Reported EPS"].notna()]
            report_dates_index = reported.index
            # Convert to date objects
            report_dates_list = []
            for rd in report_dates_index:
                if isinstance(rd, pd.Timestamp):
                    report_dates_list.append(rd.date())
                else:
                    report_dates_list.append(pd.Timestamp(rd).date())

            # Match report dates to quarter-end dates by proximity
            # Each quarter-end should have a report date ~30-60 days later
            result = []
            for qi, qe in enumerate(quarter_ends):
                qe_date = pd.Timestamp(qe).date()
                # Find the closest report date that is after the quarter end
                best_rd = None
                best_delta = timedelta(days=999)
                for rd in report_dates_list:
                    delta = rd - qe_date
                    if timedelta(days=0) <= delta < best_delta:
                        best_delta = delta
                        best_rd = rd
                if best_rd is not None and best_delta <= timedelta(days=120):
                    result.append((best_rd, qi))
                else:
                    # Fallback: quarter-end + 45 days
                    result.append((qe_date + timedelta(days=45), qi))

            # Sort oldest-first
            result.sort(key=lambda x: x[0])
            return result
        except Exception:
            pass

    # Fallback: quarter-end + 45 days (conservative estimate)
    result = []
    for qi, qe in enumerate(quarter_ends):
        qe_date = pd.Timestamp(qe).date()
        result.append((qe_date + timedelta(days=45), qi))
    result.sort(key=lambda x: x[0])
    return result


def _map_to_timeseries(
    df: pl.DataFrame,
    values_by_quarter: list[float | None],
    report_dates: list[tuple],
    name: str,
) -> pl.Series:
    """Map quarterly values to each OHLCV row based on report dates.

    For each date in df, find the most recent report_date <= that date,
    and use the corresponding value.

    Args:
        df: OHLCV DataFrame with 'date' column
        values_by_quarter: List indexed by quarter_index (0=newest)
        report_dates: List of (report_date, quarter_index) sorted oldest-first
        name: Feature name for the Series
    """
    n = df.height
    if not report_dates or not values_by_quarter:
        return _null_series(name, n)

    dates = df["date"].to_list()
    result = []

    for d in dates:
        # Convert to date if needed
        if hasattr(d, "date"):
            d = d.date()

        # Find most recent report_date <= d (binary search would be faster but
        # N is small enough)
        best_val = 0.0  # default when no report is available yet
        for report_date, qi in report_dates:
            if report_date <= d:
                if qi < len(values_by_quarter) and values_by_quarter[qi] is not None:
                    best_val = values_by_quarter[qi]
            else:
                break  # report_dates is sorted, so no more matches

        result.append(best_val)

    return pl.Series(name, result, dtype=pl.Float64)


def _compute_growth_series(
    vals_by_quarter: list[float | None],
) -> list[float | None]:
    """Compute QoQ growth for each quarter vs previous quarter.

    Returns list with same indexing as vals_by_quarter.
    Index 0 = growth of newest quarter vs next-newest.
    """
    n = len(vals_by_quarter)
    growth = [None] * n
    for i in range(n - 1):
        curr = vals_by_quarter[i]
        prev = vals_by_quarter[i + 1]
        if curr is not None and prev is not None and prev != 0:
            growth[i] = (curr - prev) / abs(prev)
    return growth


# =============================================================================
# Feature: eps_growth_qoq
# =============================================================================


@reg.register(
    "eps_growth_qoq",
    FeatureTier.TIER6_FUNDAMENTAL_TS,
    lookback_days=1,
    description="Quarter-over-quarter EPS growth rate, time-series aligned to report dates",
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
        eps_vals = _safe_get_row(income, "Diluted EPS")
    if eps_vals is None or len(eps_vals) < 2:
        return _null_series("eps_growth_qoq", n)

    growth = _compute_growth_series(eps_vals)
    report_dates = _get_report_dates(data)

    return _map_to_timeseries(df, growth, report_dates, "eps_growth_qoq")


# =============================================================================
# Feature: revenue_growth_qoq
# =============================================================================


@reg.register(
    "revenue_growth_qoq",
    FeatureTier.TIER6_FUNDAMENTAL_TS,
    lookback_days=1,
    description="Quarter-over-quarter revenue growth rate, time-series aligned",
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

    growth = _compute_growth_series(rev_vals)
    report_dates = _get_report_dates(data)

    return _map_to_timeseries(df, growth, report_dates, "revenue_growth_qoq")


# =============================================================================
# Feature: earnings_surprise_avg
# =============================================================================


@reg.register(
    "earnings_surprise_avg",
    FeatureTier.TIER6_FUNDAMENTAL_TS,
    lookback_days=1,
    description="Rolling average earnings surprise (%) as of each report date",
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

    import pandas as pd

    try:
        if "Surprise(%)" not in ed.columns:
            return _null_series("earnings_surprise_avg", n)

        # Get reported earnings with surprise data
        reported = ed[ed["Reported EPS"].notna()].copy()
        if len(reported) == 0:
            return _null_series("earnings_surprise_avg", n)

        # Build time-series: for each report date, compute rolling 4Q average surprise
        report_entries = []  # (report_date, rolling_avg_surprise)
        surprises = reported["Surprise(%)"].tolist()
        report_idx = reported.index

        # reported is newest-first, reverse for chronological order
        surprises_chrono = list(reversed(surprises))
        dates_chrono = list(reversed(report_idx))

        for i in range(len(surprises_chrono)):
            # Rolling window of up to 4 past quarters
            window = [s for s in surprises_chrono[max(0, i - 3) : i + 1] if not np.isnan(s)]
            if window:
                avg = sum(window) / len(window) / 100.0  # Convert % to fraction
                rd = dates_chrono[i]
                if isinstance(rd, pd.Timestamp):
                    rd = rd.date()
                report_entries.append((rd, avg))

        if not report_entries:
            return _null_series("earnings_surprise_avg", n)

        # Map to OHLCV dates
        dates = df["date"].to_list()
        result = []
        for d in dates:
            if hasattr(d, "date"):
                d = d.date()
            best_val = 0.0
            for rd, val in report_entries:
                if rd <= d:
                    best_val = val
                else:
                    break
            result.append(best_val)

        return pl.Series("earnings_surprise_avg", result, dtype=pl.Float64)
    except Exception:
        return _null_series("earnings_surprise_avg", n)


# =============================================================================
# Feature: earnings_surprise_streak
# =============================================================================


@reg.register(
    "earnings_surprise_streak",
    FeatureTier.TIER6_FUNDAMENTAL_TS,
    lookback_days=1,
    description="Consecutive beats (+) or misses (-) as of each report date",
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

    import pandas as pd

    try:
        if "Surprise(%)" not in ed.columns:
            return _null_series("earnings_surprise_streak", n)

        reported = ed[ed["Reported EPS"].notna()].copy()
        if len(reported) == 0:
            return _null_series("earnings_surprise_streak", n)

        surprises = reported["Surprise(%)"].tolist()
        report_idx = reported.index

        # Reverse to chronological order (oldest first)
        surprises_chrono = list(reversed(surprises))
        dates_chrono = list(reversed(report_idx))

        # Build streak at each report date
        report_entries = []  # (report_date, streak_count)
        streak = 0
        for i, val in enumerate(surprises_chrono):
            if np.isnan(val):
                # Keep previous streak
                pass
            elif val > 0:
                streak = streak + 1 if streak > 0 else 1
            elif val < 0:
                streak = streak - 1 if streak < 0 else -1
            else:
                streak = 0  # Exact meet resets streak

            rd = dates_chrono[i]
            if isinstance(rd, pd.Timestamp):
                rd = rd.date()
            report_entries.append((rd, float(streak)))

        if not report_entries:
            return _null_series("earnings_surprise_streak", n)

        # Map to OHLCV dates
        dates = df["date"].to_list()
        result = []
        for d in dates:
            if hasattr(d, "date"):
                d = d.date()
            best_val = 0.0
            for rd, val in report_entries:
                if rd <= d:
                    best_val = val
                else:
                    break
            result.append(best_val)

        return pl.Series("earnings_surprise_streak", result, dtype=pl.Float64)
    except Exception:
        return _null_series("earnings_surprise_streak", n)


# =============================================================================
# Feature: operating_margin_trend
# =============================================================================


@reg.register(
    "operating_margin_trend",
    FeatureTier.TIER6_FUNDAMENTAL_TS,
    lookback_days=1,
    description="Change in operating margin vs 4Q ago, time-series aligned",
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

    # Compute margins per quarter
    margins = []
    for oi, rev in zip(op_income, revenue):
        if oi is not None and rev is not None and rev != 0:
            margins.append(oi / rev)
        else:
            margins.append(None)

    # For each quarter, compute margin change vs oldest available
    # (simplification: trend = latest margin - margin at index min(i+3, len-1))
    trend_by_q = [None] * len(margins)
    for i in range(len(margins)):
        if margins[i] is None:
            continue
        # Compare to the quarter 3 positions older (or oldest available)
        compare_idx = min(i + 3, len(margins) - 1)
        if margins[compare_idx] is not None:
            trend_by_q[i] = margins[i] - margins[compare_idx]

    report_dates = _get_report_dates(data)
    return _map_to_timeseries(df, trend_by_q, report_dates, "operating_margin_trend")


# =============================================================================
# Feature: net_income_acceleration
# =============================================================================


@reg.register(
    "net_income_acceleration",
    FeatureTier.TIER6_FUNDAMENTAL_TS,
    lookback_days=1,
    description="Net income growth acceleration (recent vs older growth rate), time-series",
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

    # For each quarter i (where i+3 exists):
    # recent_growth = (Q[i] - Q[i+1]) / abs(Q[i+1])
    # older_growth  = (Q[i+2] - Q[i+3]) / abs(Q[i+3])
    # acceleration  = recent_growth - older_growth
    accel_by_q = [None] * len(ni_vals)
    for i in range(len(ni_vals) - 3):
        q0, q1, q2, q3 = ni_vals[i], ni_vals[i + 1], ni_vals[i + 2], ni_vals[i + 3]
        if any(v is None for v in [q0, q1, q2, q3]):
            continue
        if q1 == 0 or q3 == 0:
            continue
        recent = (q0 - q1) / abs(q1)
        older = (q2 - q3) / abs(q3)
        accel_by_q[i] = recent - older

    report_dates = _get_report_dates(data)
    return _map_to_timeseries(df, accel_by_q, report_dates, "net_income_acceleration")


def clear_cache() -> None:
    """Clear the quarterly data cache (useful for testing or session reset)."""
    _quarterly_cache.clear()
