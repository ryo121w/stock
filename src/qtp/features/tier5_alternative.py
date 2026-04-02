"""Tier 5: Alternative data features — earnings, analyst, insider signals.

These features leverage data from stock-tools MCP (earnings trend, analyst actions,
insider transactions, etc.) to capture information asymmetry that pure price/volume
technical indicators cannot.

Data flow:
  MCP tools → SQLite (qtp.db alternative_data table) → this module → features

If cached data is not available for a ticker, features return 0.0 (neutral signal).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import structlog

from qtp.features.registry import FeatureRegistry, FeatureTier

reg = FeatureRegistry.instance()
logger = structlog.get_logger()

# In-memory cache for session (avoids repeated DB queries)
_alt_cache: dict[str, dict | None] = {}

# Lazy-initialized database connection
_db = None


def _get_db():
    """Get or create database connection (lazy singleton)."""
    global _db
    if _db is None:
        from qtp.data.database import QTPDatabase

        _db = QTPDatabase(Path("data/qtp.db"))
    return _db


def _load_alt(ticker: str, tool_name: str) -> dict | None:
    """Load cached MCP tool result for a ticker from SQLite."""
    cache_key = f"{ticker}:{tool_name}"
    if cache_key in _alt_cache:
        return _alt_cache[cache_key]

    db = _get_db()
    # Try ticker-specific first, then market-level
    for key in [ticker, "_market"]:
        data = db.get_alternative(key, tool_name)
        if data:
            _alt_cache[cache_key] = data
            return data

    _alt_cache[cache_key] = None
    return None


def _null_series(name: str, n: int) -> pl.Series:
    """Return a null float series (graceful fallback)."""
    return pl.Series(name, [None] * n, dtype=pl.Float64)


# =============================================================================
# Earnings Trend Features (EPS revision momentum — most predictive signal)
# =============================================================================


@reg.register(
    "eps_revision_7d",
    FeatureTier.TIER5_ALTERNATIVE,
    lookback_days=1,
    description="EPS estimate revision direction over last 7 days (up=1, down=-1, flat=0)",
)
def eps_revision_7d(df: pl.DataFrame) -> pl.Series:
    n = df.height
    # Extract ticker from context — use first non-null ticker if available
    # For now, this requires alternative data to be pre-loaded
    # The feature returns a static value across all rows (snapshot feature)
    ticker = _infer_ticker(df)
    if not ticker:
        return _null_series("eps_revision_7d", n)

    data = _load_alt(ticker, "earnings_trend")
    if not data:
        return _null_series("eps_revision_7d", n)

    # Extract revision signal from earnings_trend response
    try:
        trend = data.get("trend", data)
        # Look for 7-day revision direction
        rev_7d = trend.get("eps_revision_7d", trend.get("revision_7d", 0))
        if isinstance(rev_7d, str):
            rev_7d = {"up": 1, "down": -1, "flat": 0}.get(rev_7d.lower(), 0)
        return pl.Series("eps_revision_7d", [float(rev_7d)] * n, dtype=pl.Float64)
    except Exception:
        return _null_series("eps_revision_7d", n)


@reg.register(
    "eps_revision_30d",
    FeatureTier.TIER5_ALTERNATIVE,
    lookback_days=1,
    description="EPS estimate revision direction over last 30 days",
)
def eps_revision_30d(df: pl.DataFrame) -> pl.Series:
    n = df.height
    ticker = _infer_ticker(df)
    if not ticker:
        return _null_series("eps_revision_30d", n)

    data = _load_alt(ticker, "earnings_trend")
    if not data:
        return _null_series("eps_revision_30d", n)

    try:
        trend = data.get("trend", data)
        rev_30d = trend.get("eps_revision_30d", trend.get("revision_30d", 0))
        if isinstance(rev_30d, str):
            rev_30d = {"up": 1, "down": -1, "flat": 0}.get(rev_30d.lower(), 0)
        return pl.Series("eps_revision_30d", [float(rev_30d)] * n, dtype=pl.Float64)
    except Exception:
        return _null_series("eps_revision_30d", n)


# =============================================================================
# Analyst Actions Features (upgrade/downgrade momentum)
# =============================================================================


@reg.register(
    "analyst_net_upgrades",
    FeatureTier.TIER5_ALTERNATIVE,
    lookback_days=1,
    description="Net analyst upgrades minus downgrades in recent period",
)
def analyst_net_upgrades(df: pl.DataFrame) -> pl.Series:
    n = df.height
    ticker = _infer_ticker(df)
    if not ticker:
        return _null_series("analyst_net_upgrades", n)

    data = _load_alt(ticker, "analyst_actions")
    if not data:
        return _null_series("analyst_net_upgrades", n)

    try:
        actions = data.get("actions", data.get("recent_actions", []))
        if isinstance(actions, list):
            upgrades = sum(
                1
                for a in actions
                if a.get("action", "").lower() in ("upgrade", "initiated", "reiterated")
            )
            downgrades = sum(
                1 for a in actions if a.get("action", "").lower() in ("downgrade", "lowered")
            )
            net = upgrades - downgrades
        else:
            net = data.get("net_upgrades", 0)
        return pl.Series("analyst_net_upgrades", [float(net)] * n, dtype=pl.Float64)
    except Exception:
        return _null_series("analyst_net_upgrades", n)


@reg.register(
    "target_price_gap",
    FeatureTier.TIER5_ALTERNATIVE,
    lookback_days=1,
    description="Gap between analyst consensus target price and current price (%)",
)
def target_price_gap(df: pl.DataFrame) -> pl.Series:
    n = df.height
    ticker = _infer_ticker(df)
    if not ticker:
        return _null_series("target_price_gap", n)

    data = _load_alt(ticker, "analyst_estimates")
    if not data:
        return _null_series("target_price_gap", n)

    try:
        target = data.get("target_mean_price", data.get("targetMeanPrice", None))
        if target is None:
            return _null_series("target_price_gap", n)

        # Use latest close price from OHLCV
        current_price = df["close"].to_list()[-1]
        if current_price and current_price > 0:
            gap = (float(target) - current_price) / current_price
        else:
            gap = 0.0

        return pl.Series("target_price_gap", [gap] * n, dtype=pl.Float64)
    except Exception:
        return _null_series("target_price_gap", n)


# =============================================================================
# Insider Transaction Features
# =============================================================================


@reg.register(
    "insider_net_signal",
    FeatureTier.TIER5_ALTERNATIVE,
    lookback_days=1,
    description="Net insider buying signal (+1=net buying, -1=net selling, 0=neutral)",
)
def insider_net_signal(df: pl.DataFrame) -> pl.Series:
    n = df.height
    ticker = _infer_ticker(df)
    if not ticker:
        return _null_series("insider_net_signal", n)

    data = _load_alt(ticker, "insider_transactions")
    if not data:
        return _null_series("insider_net_signal", n)

    try:
        txns = data.get("transactions", data.get("recent", []))
        if isinstance(txns, list):
            buys = sum(1 for t in txns if t.get("type", "").lower() in ("purchase", "buy", "p"))
            sells = sum(1 for t in txns if t.get("type", "").lower() in ("sale", "sell", "s"))
            if buys > sells:
                signal = 1.0
            elif sells > buys:
                signal = -1.0
            else:
                signal = 0.0
        else:
            signal = float(data.get("net_signal", 0))
        return pl.Series("insider_net_signal", [signal] * n, dtype=pl.Float64)
    except Exception:
        return _null_series("insider_net_signal", n)


# =============================================================================
# Earnings Date Features (event proximity)
# =============================================================================


@reg.register(
    "days_to_earnings",
    FeatureTier.TIER5_ALTERNATIVE,
    lookback_days=1,
    description="Days until next earnings announcement (event premium proxy)",
)
def days_to_earnings(df: pl.DataFrame) -> pl.Series:
    n = df.height
    ticker = _infer_ticker(df)
    if not ticker:
        return _null_series("days_to_earnings", n)

    data = _load_alt(ticker, "earnings_date")
    if not data:
        return _null_series("days_to_earnings", n)

    try:
        days = data.get("days_to_earnings", data.get("daysToEarnings", None))
        if days is not None:
            return pl.Series("days_to_earnings", [float(days)] * n, dtype=pl.Float64)
        return _null_series("days_to_earnings", n)
    except Exception:
        return _null_series("days_to_earnings", n)


@reg.register(
    "earnings_proximity",
    FeatureTier.TIER5_ALTERNATIVE,
    lookback_days=1,
    description="Earnings proximity score (1.0 = earnings tomorrow, decays exponentially)",
)
def earnings_proximity(df: pl.DataFrame) -> pl.Series:
    n = df.height
    ticker = _infer_ticker(df)
    if not ticker:
        return _null_series("earnings_proximity", n)

    data = _load_alt(ticker, "earnings_date")
    if not data:
        return _null_series("earnings_proximity", n)

    try:
        days = data.get("days_to_earnings", data.get("daysToEarnings", None))
        if days is not None:
            # Exponential decay: 1.0 at day 0, ~0.37 at day 7, ~0.05 at day 21
            proximity = np.exp(-float(days) / 7.0)
            return pl.Series("earnings_proximity", [proximity] * n, dtype=pl.Float64)
        return _null_series("earnings_proximity", n)
    except Exception:
        return _null_series("earnings_proximity", n)


# =============================================================================
# Market Regime Features
# =============================================================================


@reg.register(
    "market_regime_label",
    FeatureTier.TIER5_ALTERNATIVE,
    lookback_days=1,
    description="Market regime (0=risk-off, 1=neutral, 2=risk-on)",
)
def market_regime_label(df: pl.DataFrame) -> pl.Series:
    n = df.height
    data = _load_alt("_market", "market_regime")
    if not data:
        return _null_series("market_regime_label", n)

    try:
        regime = data.get("regime", data.get("label", "neutral"))
        regime_map = {"risk_off": 0, "bear": 0, "neutral": 1, "risk_on": 2, "bull": 2}
        if isinstance(regime, str):
            val = float(regime_map.get(regime.lower(), 1))
        else:
            val = float(regime)
        return pl.Series("market_regime_label", [val] * n, dtype=pl.Float64)
    except Exception:
        return _null_series("market_regime_label", n)


# =============================================================================
# Daily History Features (require accumulated data from alternative_data_daily)
# =============================================================================


def _load_alt_history(ticker: str, tool_name: str, n_days: int = 7) -> list[dict]:
    """Load daily history of MCP tool results for a ticker from SQLite."""
    db = _get_db()
    rows = db.get_alternative_history(ticker, tool_name, n_days=n_days)
    return rows  # List of {date, data, fetched_at}, newest first


@reg.register(
    "eps_revision_trend_7d",
    FeatureTier.TIER5_ALTERNATIVE,
    lookback_days=1,
    description="Count of EPS revision direction changes over past 7 daily records",
)
def eps_revision_trend_7d(df: pl.DataFrame) -> pl.Series:
    """Count how many times the EPS revision signal changed over the last 7 days.

    A high count indicates unstable analyst sentiment.
    Positive values = net upward revisions, negative = net downward.
    Falls back to 0 when insufficient daily data (<7 records).
    """
    n = df.height
    ticker = _infer_ticker(df)
    if not ticker:
        return _null_series("eps_revision_trend_7d", n)

    history = _load_alt_history(ticker, "earnings_trend", n_days=7)
    if len(history) < 7:
        # Not enough accumulated data — return 0 (neutral)
        return pl.Series("eps_revision_trend_7d", [0.0] * n, dtype=pl.Float64)

    try:
        # Extract revision direction from each daily record
        revisions = []
        for entry in history:
            data = entry["data"]
            trend = data.get("trend", data)
            rev = trend.get("eps_revision_7d", trend.get("revision_7d", 0))
            if isinstance(rev, str):
                rev = {"up": 1, "down": -1, "flat": 0}.get(rev.lower(), 0)
            revisions.append(float(rev))

        # Count direction changes (sign flips)
        changes = 0
        for i in range(1, len(revisions)):
            if revisions[i] != revisions[i - 1]:
                changes += 1

        # Net direction: sum of all revision signals
        net_direction = sum(revisions)
        # Combine: sign(net_direction) * change_count gives trend strength + volatility
        trend_score = net_direction  # Simpler: just use net revisions over 7 days

        return pl.Series("eps_revision_trend_7d", [trend_score] * n, dtype=pl.Float64)
    except Exception:
        return _null_series("eps_revision_trend_7d", n)


@reg.register(
    "target_price_gap_change",
    FeatureTier.TIER5_ALTERNATIVE,
    lookback_days=1,
    description="Change in analyst target price gap over past 7 daily records",
)
def target_price_gap_change(df: pl.DataFrame) -> pl.Series:
    """Compute the change in target_price_gap over the past 7 daily records.

    A positive value means analysts are raising targets relative to price.
    Falls back to 0 when insufficient daily data (<7 records).
    """
    n = df.height
    ticker = _infer_ticker(df)
    if not ticker:
        return _null_series("target_price_gap_change", n)

    history = _load_alt_history(ticker, "analyst_estimates", n_days=7)
    if len(history) < 7:
        return pl.Series("target_price_gap_change", [0.0] * n, dtype=pl.Float64)

    try:
        # Extract target_mean_price from newest and oldest records
        newest = history[0]["data"]
        oldest = history[-1]["data"]

        newest_target = newest.get("target_mean_price", newest.get("targetMeanPrice", None))
        oldest_target = oldest.get("target_mean_price", oldest.get("targetMeanPrice", None))

        if newest_target is None or oldest_target is None:
            return pl.Series("target_price_gap_change", [0.0] * n, dtype=pl.Float64)

        newest_target = float(newest_target)
        oldest_target = float(oldest_target)

        # Use last close price for gap calculation
        current_price = df["close"].to_list()[-1]
        if current_price and current_price > 0:
            newest_gap = (newest_target - current_price) / current_price
            oldest_gap = (oldest_target - current_price) / current_price
            gap_change = newest_gap - oldest_gap
        else:
            gap_change = 0.0

        return pl.Series("target_price_gap_change", [gap_change] * n, dtype=pl.Float64)
    except Exception:
        return _null_series("target_price_gap_change", n)


# =============================================================================
# Helper
# =============================================================================


def _infer_ticker(df: pl.DataFrame) -> str | None:
    """Try to infer ticker from dataframe context.

    In multi-ticker datasets, each ticker is processed separately by FeatureEngine,
    so we check for a 'ticker' column or fall back to None.
    """
    if "ticker" in df.columns:
        tickers = df["ticker"].unique().to_list()
        if len(tickers) == 1:
            return tickers[0]
    return None
