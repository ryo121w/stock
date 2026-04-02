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
