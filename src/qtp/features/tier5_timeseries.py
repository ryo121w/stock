"""Tier 5 time-series: Time-varying proxies for alternative data signals.

Problem: Original Tier5 features (eps_revision, analyst_net_upgrades, etc.) are
static snapshots fetched once from SQLite and broadcast identically across ALL
historical dates. The model cannot learn temporal patterns from constant columns.

Solution: Derive time-series features from OHLCV + macro data that PROXY the
same economic signals as Tier5, but vary naturally over time.

Mapping:
  days_to_earnings     -> earnings_proximity_cycle (quarterly sine cycle)
  eps_revision_*       -> price_earnings_momentum  (excess momentum)
  analyst_net_upgrades -> analyst_sentiment_proxy   (volume*return / vol)
  insider_net_signal   -> insider_signal_proxy      (-1 * 52w range position)
  market_regime_label  -> regime_proxy              (risk-adjusted market mom.)
"""

from __future__ import annotations

import numpy as np
import polars as pl
import structlog

from qtp.features.registry import FeatureRegistry, FeatureTier

reg = FeatureRegistry.instance()
logger = structlog.get_logger()


# =============================================================================
# Earnings Proximity Cycle (proxy for days_to_earnings / earnings_proximity)
# =============================================================================


@reg.register(
    "earnings_proximity_cycle",
    FeatureTier.TIER5_ALTERNATIVE,
    lookback_days=1,
    description="Cyclical earnings proximity proxy: sin(2*pi*day_index/63) capturing quarterly cycle",
)
def earnings_proximity_cycle(df: pl.DataFrame) -> pl.Series:
    """Most US stocks report quarterly (~63 trading days).

    A sine wave with period 63 captures the cyclical approach / retreat from
    earnings dates.  Value near +1 = mid-cycle peak, near -1 = trough.
    The model learns which phase of the cycle matters for returns.
    """
    n = df.height
    idx = np.arange(n, dtype=np.float64)
    cycle = np.sin(2.0 * np.pi * idx / 63.0)
    return pl.Series("earnings_proximity_cycle", cycle, dtype=pl.Float64)


@reg.register(
    "earnings_proximity_cycle_cos",
    FeatureTier.TIER5_ALTERNATIVE,
    lookback_days=1,
    description="Cosine component of quarterly earnings cycle (phase-shifted companion)",
)
def earnings_proximity_cycle_cos(df: pl.DataFrame) -> pl.Series:
    """Cosine companion so the model can reconstruct arbitrary phase offsets."""
    n = df.height
    idx = np.arange(n, dtype=np.float64)
    cycle = np.cos(2.0 * np.pi * idx / 63.0)
    return pl.Series("earnings_proximity_cycle_cos", cycle, dtype=pl.Float64)


# =============================================================================
# Price-Earnings Momentum (proxy for eps_revision)
# =============================================================================


@reg.register(
    "price_earnings_momentum",
    FeatureTier.TIER5_ALTERNATIVE,
    lookback_days=84,  # 63 + 21 lookback
    description="Excess momentum: ret_21d minus its 63-day rolling mean (EPS revision proxy)",
)
def price_earnings_momentum(df: pl.DataFrame) -> pl.Series:
    """If price is rising faster than its historical average, EPS estimates are
    likely being revised upward.  This captures the same information as
    eps_revision_7d / eps_revision_30d but varies over time.

    Feature = ret_21d - rolling_mean(ret_21d, 63)
    """
    ret_21d = df["close"].pct_change(21)
    rolling_avg = ret_21d.rolling_mean(63)
    excess = ret_21d - rolling_avg
    return excess.alias("price_earnings_momentum")


# =============================================================================
# Analyst Sentiment Proxy (proxy for analyst_net_upgrades)
# =============================================================================


@reg.register(
    "analyst_sentiment_proxy",
    FeatureTier.TIER5_ALTERNATIVE,
    lookback_days=25,
    description="Analyst upgrade proxy: (volume_ratio * ret_21d) / realized_vol_21d",
)
def analyst_sentiment_proxy(df: pl.DataFrame) -> pl.Series:
    """Stocks receiving analyst upgrades tend to exhibit:
      - Rising volume (institutional accumulation)
      - Rising price  (positive catalyst)
      - Lower volatility (conviction buying, not speculative)

    Feature = (volume_ratio_20d * ret_21d) / realized_vol_21d

    High values = strong, quiet uptrend = likely being upgraded.
    Low/negative = weak or volatile = likely being downgraded.
    """
    close = df["close"]
    volume = df["volume"]

    ret_21d = close.pct_change(21)
    vol_sma20 = volume.rolling_mean(20)
    volume_ratio = volume / vol_sma20

    daily_ret = close.pct_change(1)
    realized_vol = daily_ret.rolling_std(21) * (252.0**0.5)

    # Avoid division by zero
    safe_vol = realized_vol.fill_null(1.0)
    safe_vol = pl.when(safe_vol.abs() < 1e-8).then(1e-8).otherwise(safe_vol)

    sentiment = (volume_ratio * ret_21d) / safe_vol
    return sentiment.alias("analyst_sentiment_proxy")


# =============================================================================
# Insider Signal Proxy (proxy for insider_net_signal)
# =============================================================================


@reg.register(
    "insider_signal_proxy",
    FeatureTier.TIER5_ALTERNATIVE,
    lookback_days=252,
    description="Insider buying proxy: -1 * range_52w_position (insiders buy near lows)",
)
def insider_signal_proxy(df: pl.DataFrame) -> pl.Series:
    """Insiders buy when price is low relative to its range (value opportunity).
    Insiders sell when price is high (profit taking / diversification).

    Feature = -1 * (close - 52w_low) / (52w_high - 52w_low)

    Values near -1 = price at 52-week high = insiders likely selling
    Values near  0 = price at 52-week low  = insiders likely buying

    Works as a cross-sectional signal across stocks.
    """
    high_52w = df["high"].rolling_max(252)
    low_52w = df["low"].rolling_min(252)
    range_width = high_52w - low_52w

    # Avoid division by zero for flat stocks
    safe_range = pl.when(range_width.abs() < 1e-8).then(1e-8).otherwise(range_width)
    position = (df["close"] - low_52w) / safe_range

    return (-1.0 * position).alias("insider_signal_proxy")


# =============================================================================
# Regime Proxy (proxy for market_regime_label)
# =============================================================================


@reg.register(
    "regime_proxy",
    FeatureTier.TIER5_ALTERNATIVE,
    lookback_days=55,
    description="Market regime proxy: sp500_ret_21d / (vix_level / 20) — risk-adjusted market momentum",
)
def regime_proxy(df: pl.DataFrame) -> pl.Series:
    """Combines VIX level + S&P500 momentum into a continuous regime indicator.

    Feature = sp500_ret_21d / (vix_level / 20)

    - High positive = risk-on (strong market + low fear)
    - Near zero     = neutral / transitioning
    - Negative      = risk-off (weak market + high fear)

    Uses the same macro data cache as tier4_macro.
    """
    from qtp.features.tier4_macro import _load_macro_series

    n = df.height
    dates = df.select("date")

    # Load VIX
    vix_df = _load_macro_series("^VIX", "vix")
    if vix_df.height == 0:
        return pl.Series("regime_proxy", [None] * n, dtype=pl.Float64)

    # Load S&P 500
    sp_df = _load_macro_series("^GSPC", "sp500")
    if sp_df.height == 0:
        return pl.Series("regime_proxy", [None] * n, dtype=pl.Float64)

    # Merge VIX
    merged = dates.join(vix_df, on="date", how="left")
    vix_vals = merged["vix"].forward_fill()

    # Merge S&P 500
    merged2 = dates.join(sp_df, on="date", how="left")
    sp_vals = merged2["sp500"].forward_fill()
    sp_ret_21d = sp_vals.pct_change(21)

    # Normalize VIX: divide by 20 (long-term average) so VIX=20 -> divisor=1
    vix_norm = vix_vals / 20.0

    # Avoid division by zero
    safe_vix = pl.when(vix_norm.abs() < 1e-8).then(1e-8).otherwise(vix_norm)

    regime = sp_ret_21d / safe_vix
    return regime.alias("regime_proxy")
