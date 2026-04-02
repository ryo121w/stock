"""Feature computation engine with strict anti-leakage guarantees.

CRITICAL DESIGN:
- Features and labels are computed by SEPARATE methods
- shift(-1) appears ONLY in compute_label
- Feature functions receive data truncated to as_of
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import structlog

from qtp.data.fetchers.base import Market
from qtp.data.storage import ParquetStorage
from qtp.features.registry import FeatureRegistry

logger = structlog.get_logger()


class FeatureEngine:
    def __init__(self, registry: FeatureRegistry, storage: ParquetStorage):
        self.registry = registry
        self.storage = storage

    def compute_features(
        self,
        ticker: str,
        market: Market,
        as_of: date,
        tiers: list[int] | None = None,
        use_all_data: bool = False,
    ) -> pl.DataFrame:
        """Compute all registered features for a ticker as of a date.

        Returns DataFrame: [date, feature_1, ..., feature_N]
        Every row's features use ONLY data available on that date.

        Args:
            use_all_data: If True, use all available data (for training).
                         If False, only load max_lookback * 1.8 days (for prediction).
        """
        ohlcv = self.storage.load_ohlcv(ticker, market, as_of=as_of)

        if not use_all_data:
            max_lb = self.registry.max_lookback()
            start_date = as_of - timedelta(days=int(max_lb * 1.8))
            ohlcv = ohlcv.filter(pl.col("date") >= start_date)

        if ohlcv.height == 0:
            logger.warning("no_data_for_features", ticker=ticker, as_of=str(as_of))
            return pl.DataFrame()

        features = [f for f in self.registry.by_tiers(tiers)] if tiers else self.registry.all_features()

        result = ohlcv.select("date")
        for feat_def in features:
            try:
                series = feat_def.compute_fn(ohlcv)
                result = result.with_columns(series)
            except Exception as e:
                logger.error("feature_computation_failed",
                             feature=feat_def.name, ticker=ticker, error=str(e))
                result = result.with_columns(pl.lit(None).alias(feat_def.name))

        # Drop warmup rows (nulls from lookback)
        result = result.drop_nulls()

        logger.info("computed_features", ticker=ticker, rows=result.height,
                     n_features=len(result.columns) - 1)
        return result

    def compute_label(
        self,
        ticker: str,
        market: Market,
        as_of: date,
        horizon: int = 1,
    ) -> pl.DataFrame:
        """Compute target labels. SEPARATE from features to prevent leakage.

        Returns: [date, label_direction (1/0), label_magnitude (float)]
        - label_direction: 1 if close[T+horizon] > close[T], else 0
        - label_magnitude: (close[T+horizon] - close[T]) / close[T]

        NOTE: shift(-horizon) is used here and ONLY here in the entire codebase.
        """
        ohlcv = self.storage.load_ohlcv(ticker, market, as_of=as_of)

        labels = ohlcv.with_columns([
            (pl.col("close").shift(-horizon) > pl.col("close"))
            .cast(pl.Int8).alias("label_direction"),
            ((pl.col("close").shift(-horizon) - pl.col("close")) / pl.col("close"))
            .alias("label_magnitude"),
        ]).select(["date", "label_direction", "label_magnitude"])

        # Drop rows without labels (last `horizon` rows)
        labels = labels.filter(pl.col("label_direction").is_not_null())
        return labels

    def build_dataset(
        self,
        ticker: str,
        market: Market,
        as_of: date,
        tiers: list[int] | None = None,
        horizon: int = 1,
    ) -> pl.DataFrame:
        """Join features + labels on date. Computed independently to prevent leakage."""
        features = self.compute_features(ticker, market, as_of=as_of, tiers=tiers,
                                          use_all_data=True)
        labels = self.compute_label(ticker, market, as_of=as_of, horizon=horizon)

        if features.height == 0:
            return pl.DataFrame()

        dataset = features.join(labels, on="date", how="inner")
        logger.info("built_dataset", ticker=ticker, rows=dataset.height)
        return dataset

    def build_multi_ticker_dataset(
        self,
        tickers: list[str],
        market: Market,
        as_of: date,
        tiers: list[int] | None = None,
        horizon: int = 1,
    ) -> pl.DataFrame:
        """Build combined dataset across multiple tickers."""
        frames: list[pl.DataFrame] = []
        for ticker in tickers:
            try:
                ds = self.build_dataset(ticker, market, as_of, tiers, horizon)
                if ds.height > 0:
                    ds = ds.with_columns(pl.lit(ticker).alias("ticker"))
                    frames.append(ds)
            except Exception as e:
                logger.error("build_dataset_failed", ticker=ticker, error=str(e))

        if not frames:
            return pl.DataFrame()

        return pl.concat(frames, how="diagonal_relaxed")
