"""Tier 1: Momentum features — strongest predictive power."""

from __future__ import annotations

import polars as pl

from qtp.features.registry import FeatureRegistry, FeatureTier

reg = FeatureRegistry.instance()


# --- Returns ---


@reg.register(
    "ret_1d", FeatureTier.TIER1_MOMENTUM, lookback_days=2, description="1-day simple return"
)
def ret_1d(df: pl.DataFrame) -> pl.Series:
    return df["close"].pct_change(1).alias("ret_1d")


@reg.register(
    "ret_5d", FeatureTier.TIER1_MOMENTUM, lookback_days=6, description="5-day return (1 week)"
)
def ret_5d(df: pl.DataFrame) -> pl.Series:
    return df["close"].pct_change(5).alias("ret_5d")


@reg.register(
    "ret_21d", FeatureTier.TIER1_MOMENTUM, lookback_days=22, description="21-day return (1 month)"
)
def ret_21d(df: pl.DataFrame) -> pl.Series:
    return df["close"].pct_change(21).alias("ret_21d")


@reg.register(
    "ret_63d", FeatureTier.TIER1_MOMENTUM, lookback_days=64, description="63-day return (3 months)"
)
def ret_63d(df: pl.DataFrame) -> pl.Series:
    return df["close"].pct_change(63).alias("ret_63d")


# --- RSI ---


@reg.register(
    "rsi_14",
    FeatureTier.TIER1_MOMENTUM,
    lookback_days=30,
    description="RSI(14) - Relative Strength Index",
)
def rsi_14(df: pl.DataFrame) -> pl.Series:
    close = df["close"]
    delta = close.diff()
    gain = delta.clip(lower_bound=0).rolling_mean(14)
    loss = (-delta.clip(upper_bound=0)).rolling_mean(14)
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.alias("rsi_14")


# --- MACD ---


@reg.register(
    "macd_hist",
    FeatureTier.TIER1_MOMENTUM,
    lookback_days=35,
    description="MACD Histogram (12-26 EMA diff - Signal 9)",
)
def macd_hist(df: pl.DataFrame) -> pl.Series:
    close = df["close"]
    ema12 = close.ewm_mean(span=12, adjust=False)
    ema26 = close.ewm_mean(span=26, adjust=False)
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm_mean(span=9, adjust=False)
    return (macd_line - signal_line).alias("macd_hist")


@reg.register(
    "macd_signal", FeatureTier.TIER1_MOMENTUM, lookback_days=35, description="MACD Signal line"
)
def macd_signal(df: pl.DataFrame) -> pl.Series:
    close = df["close"]
    ema12 = close.ewm_mean(span=12, adjust=False)
    ema26 = close.ewm_mean(span=26, adjust=False)
    macd_line = ema12 - ema26
    return macd_line.ewm_mean(span=9, adjust=False).alias("macd_signal")


# --- SMA Distance ---


@reg.register(
    "dist_sma20",
    FeatureTier.TIER1_MOMENTUM,
    lookback_days=25,
    description="Distance from SMA(20) as ratio",
)
def dist_sma20(df: pl.DataFrame) -> pl.Series:
    sma = df["close"].rolling_mean(20)
    return ((df["close"] - sma) / sma).alias("dist_sma20")


@reg.register(
    "dist_sma50",
    FeatureTier.TIER1_MOMENTUM,
    lookback_days=55,
    description="Distance from SMA(50) as ratio",
)
def dist_sma50(df: pl.DataFrame) -> pl.Series:
    sma = df["close"].rolling_mean(50)
    return ((df["close"] - sma) / sma).alias("dist_sma50")


@reg.register(
    "dist_sma200",
    FeatureTier.TIER1_MOMENTUM,
    lookback_days=210,
    description="Distance from SMA(200) as ratio",
)
def dist_sma200(df: pl.DataFrame) -> pl.Series:
    sma = df["close"].rolling_mean(200)
    return ((df["close"] - sma) / sma).alias("dist_sma200")


# --- Momentum / ROC ---


@reg.register(
    "roc_10", FeatureTier.TIER1_MOMENTUM, lookback_days=11, description="Rate of Change (10-day)"
)
def roc_10(df: pl.DataFrame) -> pl.Series:
    return df["close"].pct_change(10).alias("roc_10")


@reg.register(
    "roc_20", FeatureTier.TIER1_MOMENTUM, lookback_days=21, description="Rate of Change (20-day)"
)
def roc_20(df: pl.DataFrame) -> pl.Series:
    return df["close"].pct_change(20).alias("roc_20")
