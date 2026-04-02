"""Cross-sectional features: relative comparisons across tickers on the same date."""

from __future__ import annotations

import polars as pl
import structlog

logger = structlog.get_logger()

# Columns that this module may produce
CROSS_SECTIONAL_FEATURES = [
    "relative_strength_21d",
    "momentum_rank",
    "relative_volume",
    "volatility_rank",
    "cross_sectional_dispersion",
]


def compute_cross_sectional_features(dataset: pl.DataFrame) -> pl.DataFrame:
    """Add cross-sectional features to a multi-ticker dataset.

    Must be called AFTER build_multi_ticker_dataset joins all tickers.
    Requires 'date', 'ticker', and feature columns to exist.

    Each feature uses .over("date") to compute cross-sectional statistics,
    comparing a ticker's value against all other tickers on the same day.
    """
    if "date" not in dataset.columns or "ticker" not in dataset.columns:
        logger.warning("cross_sectional_skip", reason="missing date or ticker column")
        return dataset

    n_tickers = dataset["ticker"].n_unique()
    if n_tickers < 2:
        logger.info("cross_sectional_skip", reason="need >=2 tickers", n_tickers=n_tickers)
        return dataset

    added: list[str] = []

    # 1. Relative momentum: ticker ret_21d minus cross-sectional mean
    if "ret_21d" in dataset.columns:
        dataset = dataset.with_columns(
            (pl.col("ret_21d") - pl.col("ret_21d").mean().over("date")).alias(
                "relative_strength_21d"
            )
        )
        added.append("relative_strength_21d")

    # 2. Momentum rank: percentile rank within each date (0=weakest, 1=strongest)
    if "ret_21d" in dataset.columns:
        dataset = dataset.with_columns(
            (pl.col("ret_21d").rank().over("date") / pl.col("ret_21d").count().over("date")).alias(
                "momentum_rank"
            )
        )
        added.append("momentum_rank")

    # 3. Volume relative: ticker volume ratio vs cross-sectional mean
    if "volume_ratio_20d" in dataset.columns:
        dataset = dataset.with_columns(
            (pl.col("volume_ratio_20d") - pl.col("volume_ratio_20d").mean().over("date")).alias(
                "relative_volume"
            )
        )
        added.append("relative_volume")

    # 4. Volatility rank: low vol = defensive, high vol = speculative
    if "realized_vol_21d" in dataset.columns:
        dataset = dataset.with_columns(
            (
                pl.col("realized_vol_21d").rank().over("date")
                / pl.col("realized_vol_21d").count().over("date")
            ).alias("volatility_rank")
        )
        added.append("volatility_rank")

    # 5. Dispersion: cross-sectional std of returns (high = stock-picking market)
    #    Fill nulls (dates with <2 tickers where std is undefined) with 0
    if "ret_21d" in dataset.columns:
        dataset = dataset.with_columns(
            pl.col("ret_21d").std().over("date").fill_null(0.0).alias("cross_sectional_dispersion")
        )
        added.append("cross_sectional_dispersion")

    logger.info("cross_sectional_features_added", features=added, count=len(added))
    return dataset
