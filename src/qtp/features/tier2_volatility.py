"""Tier 2: Volatility and volume features."""

from __future__ import annotations

import polars as pl

from qtp.features.registry import FeatureRegistry, FeatureTier

reg = FeatureRegistry.instance()


# --- Realized Volatility ---


@reg.register(
    "realized_vol_21d",
    FeatureTier.TIER2_VOLATILITY,
    lookback_days=25,
    description="21-day realized volatility (annualized)",
)
def realized_vol_21d(df: pl.DataFrame) -> pl.Series:
    ret = df["close"].pct_change(1)
    return (ret.rolling_std(21) * (252**0.5)).alias("realized_vol_21d")


@reg.register(
    "realized_vol_63d",
    FeatureTier.TIER2_VOLATILITY,
    lookback_days=68,
    description="63-day realized volatility (annualized)",
)
def realized_vol_63d(df: pl.DataFrame) -> pl.Series:
    ret = df["close"].pct_change(1)
    return (ret.rolling_std(63) * (252**0.5)).alias("realized_vol_63d")


# --- ATR ---


@reg.register(
    "atr_14",
    FeatureTier.TIER2_VOLATILITY,
    lookback_days=18,
    description="Average True Range (14-day)",
)
def atr_14(df: pl.DataFrame) -> pl.Series:
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    # Element-wise max of three series
    tr = pl.max_horizontal(tr1, tr2, tr3)
    return tr.rolling_mean(14).alias("atr_14")


@reg.register(
    "natr_14",
    FeatureTier.TIER2_VOLATILITY,
    lookback_days=18,
    description="Normalized ATR (ATR / Close)",
)
def natr_14(df: pl.DataFrame) -> pl.Series:
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pl.max_horizontal(high - low, (high - prev_close).abs(), (low - prev_close).abs())
    atr = tr.rolling_mean(14)
    return (atr / df["close"]).alias("natr_14")


# --- Bollinger Bands ---


@reg.register(
    "bbands_pct_b",
    FeatureTier.TIER2_VOLATILITY,
    lookback_days=25,
    description="Bollinger Bands %B (position within bands)",
)
def bbands_pct_b(df: pl.DataFrame) -> pl.Series:
    sma = df["close"].rolling_mean(20)
    std = df["close"].rolling_std(20)
    upper = sma + 2 * std
    lower = sma - 2 * std
    pct_b = (df["close"] - lower) / (upper - lower)
    return pct_b.alias("bbands_pct_b")


@reg.register(
    "bbands_width",
    FeatureTier.TIER2_VOLATILITY,
    lookback_days=25,
    description="Bollinger Bands width (normalized)",
)
def bbands_width(df: pl.DataFrame) -> pl.Series:
    sma = df["close"].rolling_mean(20)
    std = df["close"].rolling_std(20)
    width = (4 * std) / sma
    return width.alias("bbands_width")


# --- Volume ---


@reg.register(
    "volume_ratio_20d",
    FeatureTier.TIER2_VOLATILITY,
    lookback_days=25,
    description="Volume / SMA(volume, 20)",
)
def volume_ratio_20d(df: pl.DataFrame) -> pl.Series:
    sma_vol = df["volume"].rolling_mean(20)
    return (df["volume"] / sma_vol).alias("volume_ratio_20d")


@reg.register(
    "volume_change_5d",
    FeatureTier.TIER2_VOLATILITY,
    lookback_days=10,
    description="5-day volume change rate",
)
def volume_change_5d(df: pl.DataFrame) -> pl.Series:
    return df["volume"].pct_change(5).alias("volume_change_5d")


# --- Drawdown ---


@reg.register(
    "max_drawdown_63d",
    FeatureTier.TIER2_VOLATILITY,
    lookback_days=68,
    description="Maximum drawdown over 63 days",
)
def max_drawdown_63d(df: pl.DataFrame) -> pl.Series:
    close = df["close"]
    rolling_max = close.rolling_max(63)
    drawdown = (close - rolling_max) / rolling_max
    return drawdown.alias("max_drawdown_63d")
