"""Tier 3: Fundamental features — PER, ROE, earnings data via yfinance snapshot."""

from __future__ import annotations

import polars as pl

from qtp.features.registry import FeatureRegistry, FeatureTier

reg = FeatureRegistry.instance()


@reg.register("log_volume_avg_60d", FeatureTier.TIER3_FUNDAMENTAL, lookback_days=65,
              description="Log of 60-day average volume (liquidity proxy)")
def log_volume_avg_60d(df: pl.DataFrame) -> pl.Series:
    import numpy as np
    avg_vol = df["volume"].rolling_mean(60)
    return avg_vol.log().alias("log_volume_avg_60d")


@reg.register("price_to_52w_high", FeatureTier.TIER3_FUNDAMENTAL, lookback_days=252,
              description="Price relative to 52-week high")
def price_to_52w_high(df: pl.DataFrame) -> pl.Series:
    high_52w = df["high"].rolling_max(252)
    return (df["close"] / high_52w).alias("price_to_52w_high")


@reg.register("price_to_52w_low", FeatureTier.TIER3_FUNDAMENTAL, lookback_days=252,
              description="Price relative to 52-week low")
def price_to_52w_low(df: pl.DataFrame) -> pl.Series:
    low_52w = df["low"].rolling_min(252)
    return (df["close"] / low_52w).alias("price_to_52w_low")


@reg.register("range_52w_position", FeatureTier.TIER3_FUNDAMENTAL, lookback_days=252,
              description="Position within 52-week range (0=low, 1=high)")
def range_52w_position(df: pl.DataFrame) -> pl.Series:
    high_52w = df["high"].rolling_max(252)
    low_52w = df["low"].rolling_min(252)
    return ((df["close"] - low_52w) / (high_52w - low_52w)).alias("range_52w_position")


@reg.register("gap_overnight", FeatureTier.TIER3_FUNDAMENTAL, lookback_days=2,
              description="Overnight gap (open vs previous close)")
def gap_overnight(df: pl.DataFrame) -> pl.Series:
    prev_close = df["close"].shift(1)
    return ((df["open"] - prev_close) / prev_close).alias("gap_overnight")


@reg.register("intraday_range", FeatureTier.TIER3_FUNDAMENTAL, lookback_days=1,
              description="Intraday range normalized by close")
def intraday_range(df: pl.DataFrame) -> pl.Series:
    return ((df["high"] - df["low"]) / df["close"]).alias("intraday_range")


@reg.register("upper_shadow_ratio", FeatureTier.TIER3_FUNDAMENTAL, lookback_days=1,
              description="Upper shadow relative to full range (selling pressure)")
def upper_shadow_ratio(df: pl.DataFrame) -> pl.Series:
    body_top = pl.max_horizontal(df["open"], df["close"])
    full_range = df["high"] - df["low"]
    upper_shadow = df["high"] - body_top
    return (upper_shadow / full_range).fill_nan(0.0).alias("upper_shadow_ratio")


@reg.register("lower_shadow_ratio", FeatureTier.TIER3_FUNDAMENTAL, lookback_days=1,
              description="Lower shadow relative to full range (buying pressure)")
def lower_shadow_ratio(df: pl.DataFrame) -> pl.Series:
    body_bottom = pl.min_horizontal(df["open"], df["close"])
    full_range = df["high"] - df["low"]
    lower_shadow = body_bottom - df["low"]
    return (lower_shadow / full_range).fill_nan(0.0).alias("lower_shadow_ratio")
