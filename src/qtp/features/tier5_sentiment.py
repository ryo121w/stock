"""Tier 5: News sentiment features from keyword-scored headlines.

Data source: alternative_data_daily table, tool='news_sentiment'
Populated by: src/qtp/data/fetchers/news_sentiment.py

Features:
  - news_sentiment_avg: average headline sentiment score [-1, +1]
  - news_volume: number of recent news articles (attention proxy)

Both fall back to 0.0 when no data is available.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import structlog

from qtp.features.registry import FeatureRegistry, FeatureTier

reg = FeatureRegistry.instance()
logger = structlog.get_logger()

# Lazy DB singleton (same pattern as tier5_alternative.py)
_db = None


def _get_db():
    global _db
    if _db is None:
        from qtp.data.database import QTPDatabase

        _db = QTPDatabase(Path("data/qtp.db"))
    return _db


def _infer_ticker(df: pl.DataFrame) -> str | None:
    if "ticker" in df.columns:
        tickers = df["ticker"].unique().to_list()
        if len(tickers) == 1:
            return tickers[0]
    return None


def _load_news_sentiment(ticker: str) -> dict | None:
    """Load latest news_sentiment data from alternative_data_daily."""
    db = _get_db()
    rows = db.get_alternative_history(ticker, "news_sentiment", n_days=1)
    if rows:
        return rows[0]["data"]
    return None


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


@reg.register(
    "news_sentiment_avg",
    FeatureTier.TIER5_ALTERNATIVE,
    lookback_days=1,
    description="Average keyword-based news headline sentiment [-1, +1]",
)
def news_sentiment_avg(df: pl.DataFrame) -> pl.Series:
    n = df.height
    ticker = _infer_ticker(df)
    if not ticker:
        return pl.Series("news_sentiment_avg", [0.0] * n, dtype=pl.Float64)

    data = _load_news_sentiment(ticker)
    if not data:
        return pl.Series("news_sentiment_avg", [0.0] * n, dtype=pl.Float64)

    val = float(data.get("sentiment_avg", 0.0))
    return pl.Series("news_sentiment_avg", [val] * n, dtype=pl.Float64)


@reg.register(
    "news_volume",
    FeatureTier.TIER5_ALTERNATIVE,
    lookback_days=1,
    description="Number of recent news articles (attention/interest proxy)",
)
def news_volume(df: pl.DataFrame) -> pl.Series:
    n = df.height
    ticker = _infer_ticker(df)
    if not ticker:
        return pl.Series("news_volume", [0.0] * n, dtype=pl.Float64)

    data = _load_news_sentiment(ticker)
    if not data:
        return pl.Series("news_volume", [0.0] * n, dtype=pl.Float64)

    val = float(data.get("news_volume", 0))
    return pl.Series("news_volume", [val] * n, dtype=pl.Float64)
