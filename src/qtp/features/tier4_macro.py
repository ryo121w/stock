"""Tier 4: Macro / cross-asset features — VIX, yield curve, market breadth."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl
import structlog

from qtp.features.registry import FeatureRegistry, FeatureTier

reg = FeatureRegistry.instance()
logger = structlog.get_logger()

# Cache for macro data (loaded once per session)
_macro_cache: dict[str, pl.DataFrame] = {}


def _load_macro_series(symbol: str, col_name: str) -> pl.DataFrame:
    """Load macro data from pre-fetched parquet or fetch live."""
    if symbol in _macro_cache:
        return _macro_cache[symbol]

    cache_path = Path("data/raw/macro") / f"{symbol.replace('^', '')}.parquet"
    if cache_path.exists():
        df = pl.read_parquet(cache_path)
        _macro_cache[symbol] = df
        return df

    # Fetch live via yfinance
    try:
        import pandas as pd
        import yfinance as yf
        start = (date.today() - timedelta(days=1890)).isoformat()
        df_pd = yf.download(symbol, start=start, progress=False)
        if isinstance(df_pd.columns, pd.MultiIndex):
            df_pd.columns = df_pd.columns.get_level_values(0)
        df_pd = df_pd.reset_index()
        df_pd.columns = [str(c).lower().strip() for c in df_pd.columns]
        df = pl.from_pandas(df_pd[["date", "close"]].dropna())
        if df["date"].dtype != pl.Date:
            df = df.with_columns(pl.col("date").cast(pl.Date))
        df = df.rename({"close": col_name})

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(cache_path)
        _macro_cache[symbol] = df
        logger.info("macro_data_fetched", symbol=symbol, rows=df.height)
        return df
    except Exception as e:
        logger.warning("macro_fetch_failed", symbol=symbol, error=str(e))
        return pl.DataFrame()


@reg.register("vix_level", FeatureTier.TIER4_MACRO, lookback_days=1,
              description="VIX fear index level")
def vix_level(df: pl.DataFrame) -> pl.Series:
    vix = _load_macro_series("^VIX", "vix")
    if vix.height == 0:
        return pl.Series("vix_level", [None] * df.height, dtype=pl.Float64)
    merged = df.select("date").join(vix, on="date", how="left")
    # Forward-fill for missing dates
    return merged["vix"].forward_fill().alias("vix_level")


@reg.register("vix_change_5d", FeatureTier.TIER4_MACRO, lookback_days=6,
              description="VIX 5-day change")
def vix_change_5d(df: pl.DataFrame) -> pl.Series:
    vix = _load_macro_series("^VIX", "vix")
    if vix.height == 0:
        return pl.Series("vix_change_5d", [None] * df.height, dtype=pl.Float64)
    merged = df.select("date").join(vix, on="date", how="left")
    vix_filled = merged["vix"].forward_fill()
    return vix_filled.pct_change(5).alias("vix_change_5d")


@reg.register("yield_10y", FeatureTier.TIER4_MACRO, lookback_days=1,
              description="US 10-year Treasury yield level")
def yield_10y(df: pl.DataFrame) -> pl.Series:
    tnx = _load_macro_series("^TNX", "yield_10y")
    if tnx.height == 0:
        return pl.Series("yield_10y", [None] * df.height, dtype=pl.Float64)
    merged = df.select("date").join(tnx, on="date", how="left")
    return merged["yield_10y"].forward_fill().alias("yield_10y")


@reg.register("yield_10y_change_21d", FeatureTier.TIER4_MACRO, lookback_days=22,
              description="10Y yield 21-day change (rate of change)")
def yield_10y_change_21d(df: pl.DataFrame) -> pl.Series:
    tnx = _load_macro_series("^TNX", "yield_10y")
    if tnx.height == 0:
        return pl.Series("yield_10y_change_21d", [None] * df.height, dtype=pl.Float64)
    merged = df.select("date").join(tnx, on="date", how="left")
    y = merged["yield_10y"].forward_fill()
    return (y - y.shift(21)).alias("yield_10y_change_21d")


@reg.register("sp500_ret_21d", FeatureTier.TIER4_MACRO, lookback_days=22,
              description="S&P 500 21-day return (market momentum)")
def sp500_ret_21d(df: pl.DataFrame) -> pl.Series:
    spy = _load_macro_series("^GSPC", "sp500")
    if spy.height == 0:
        return pl.Series("sp500_ret_21d", [None] * df.height, dtype=pl.Float64)
    merged = df.select("date").join(spy, on="date", how="left")
    sp = merged["sp500"].forward_fill()
    return sp.pct_change(21).alias("sp500_ret_21d")


@reg.register("sp500_dist_sma50", FeatureTier.TIER4_MACRO, lookback_days=55,
              description="S&P 500 distance from SMA(50)")
def sp500_dist_sma50(df: pl.DataFrame) -> pl.Series:
    spy = _load_macro_series("^GSPC", "sp500")
    if spy.height == 0:
        return pl.Series("sp500_dist_sma50", [None] * df.height, dtype=pl.Float64)
    merged = df.select("date").join(spy, on="date", how="left")
    sp = merged["sp500"].forward_fill()
    sma = sp.rolling_mean(50)
    return ((sp - sma) / sma).alias("sp500_dist_sma50")
