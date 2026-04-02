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


# --- Volume Profile ---


@reg.register(
    "volume_up_down_ratio",
    FeatureTier.TIER2_VOLATILITY,
    lookback_days=22,
    description="20-day ratio of volume on up-days vs down-days",
)
def volume_up_down_ratio(df: pl.DataFrame) -> pl.Series:
    up_mask = df["close"] > df["close"].shift(1)
    up_vol = (df["volume"] * up_mask.cast(pl.Float64)).rolling_sum(20)
    down_vol = (df["volume"] * (~up_mask).cast(pl.Float64)).rolling_sum(20)
    return (up_vol / down_vol).fill_nan(0.0).alias("volume_up_down_ratio")


@reg.register(
    "volume_breakout",
    FeatureTier.TIER2_VOLATILITY,
    lookback_days=22,
    description="Current volume / 20-day average volume",
)
def volume_breakout(df: pl.DataFrame) -> pl.Series:
    avg_vol = df["volume"].rolling_mean(20)
    return (df["volume"] / avg_vol).fill_nan(0.0).alias("volume_breakout")


@reg.register(
    "ad_line_slope",
    FeatureTier.TIER2_VOLATILITY,
    lookback_days=22,
    description="Accumulation/Distribution line 20-day slope (pct_change)",
)
def ad_line_slope(df: pl.DataFrame) -> pl.Series:
    hl_range = df["high"] - df["low"]
    clv = (
        pl.when(hl_range == 0)
        .then(0.0)
        .otherwise(((df["close"] - df["low"]) - (df["high"] - df["close"])) / hl_range)
    )
    ad = (clv * df["volume"]).cum_sum()
    return ad.pct_change(20).fill_nan(0.0).fill_null(0.0).alias("ad_line_slope")


@reg.register(
    "obv_slope_20d",
    FeatureTier.TIER2_VOLATILITY,
    lookback_days=25,
    description="On-Balance Volume 20-day rate of change",
)
def obv_slope_20d(df: pl.DataFrame) -> pl.Series:
    direction = pl.when(df["close"] > df["close"].shift(1)).then(1).otherwise(-1)
    obv = (direction * df["volume"]).cum_sum()
    return obv.pct_change(20).fill_nan(0.0).fill_null(0.0).alias("obv_slope_20d")


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
