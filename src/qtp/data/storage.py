"""Parquet storage layer via Polars."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import structlog

from qtp.data.fetchers.base import Market

logger = structlog.get_logger()


class ParquetStorage:
    """Read/write Parquet files partitioned by market and ticker."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

    def save_ohlcv(self, ticker: str, market: Market, df: pl.DataFrame) -> Path:
        path = self.base_dir / "raw" / market.value / f"{ticker}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)
        logger.debug("saved_ohlcv", path=str(path), rows=df.height)
        return path

    def load_ohlcv(
        self, ticker: str, market: Market, as_of: date | None = None
    ) -> pl.DataFrame:
        path = self.base_dir / "raw" / market.value / f"{ticker}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"No OHLCV data for {ticker} ({market.value}): {path}")
        df = pl.read_parquet(path)
        if as_of:
            df = df.filter(pl.col("date") <= as_of)
        return df.sort("date")

    def save_features(self, ticker: str, market: Market, df: pl.DataFrame) -> Path:
        path = self.base_dir / "processed" / market.value / f"{ticker}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)
        logger.debug("saved_features", path=str(path), rows=df.height)
        return path

    def load_features(
        self, ticker: str, market: Market, as_of: date | None = None
    ) -> pl.DataFrame:
        path = self.base_dir / "processed" / market.value / f"{ticker}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"No feature data for {ticker} ({market.value}): {path}")
        df = pl.read_parquet(path)
        if as_of:
            df = df.filter(pl.col("date") <= as_of)
        return df.sort("date")

    def ohlcv_exists(self, ticker: str, market: Market) -> bool:
        path = self.base_dir / "raw" / market.value / f"{ticker}.parquet"
        return path.exists()

    def list_tickers(self, market: Market) -> list[str]:
        raw_dir = self.base_dir / "raw" / market.value
        if not raw_dir.exists():
            return []
        return [p.stem for p in raw_dir.glob("*.parquet")]
