"""Tier 5: Finnhub-based features from daily accumulated data.

These features become active after ~30 days of daily accumulation.
Before that, they return 0.0 (neutral, no signal).

Data source: alternative_data_daily table, tools:
- finnhub_reco_trends: analyst recommendation consensus
- finnhub_news: company news article count

Features compute rolling windows from daily records.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import structlog

from qtp.features.registry import FeatureRegistry, FeatureTier

reg = FeatureRegistry.instance()
logger = structlog.get_logger()

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


def _load_reco_history(ticker: str, n_days: int = 30) -> list[dict]:
    """Load accumulated recommendation trend data."""
    db = _get_db()
    return db.get_alternative_history(ticker, "finnhub_reco_trends", n_days=n_days)


def _load_news_history(ticker: str, n_days: int = 7) -> list[dict]:
    """Load accumulated news data."""
    db = _get_db()
    return db.get_alternative_history(ticker, "finnhub_news", n_days=n_days)


def _compute_consensus_score(reco: dict) -> float:
    """Compute consensus score from recommendation trends.

    Score = strongBuy*2 + buy*1 + hold*0 - sell*1 - strongSell*2
    Normalized by total analysts.
    """
    trends = reco.get("trends", [])
    if not trends:
        return 0.0

    latest = trends[0] if isinstance(trends, list) else trends
    if not isinstance(latest, dict):
        return 0.0

    sb = latest.get("strongBuy", 0) or 0
    b = latest.get("buy", 0) or 0
    h = latest.get("hold", 0) or 0
    s = latest.get("sell", 0) or 0
    ss = latest.get("strongSell", 0) or 0

    total = sb + b + h + s + ss
    if total == 0:
        return 0.0

    raw_score = sb * 2 + b * 1 + h * 0 - s * 1 - ss * 2
    # Normalize to [-1, 1] range
    max_possible = total * 2
    return raw_score / max_possible if max_possible > 0 else 0.0


# =============================================================================
# Feature: analyst_consensus_score
# =============================================================================


@reg.register(
    "analyst_consensus_score",
    FeatureTier.TIER5_ALTERNATIVE,
    lookback_days=1,
    description="Analyst recommendation consensus score [-1=all sell, +1=all buy]",
)
def analyst_consensus_score(df: pl.DataFrame) -> pl.Series:
    """Latest analyst consensus from Finnhub recommendation trends.

    Returns a static value (latest snapshot) until daily accumulation
    enables time-series computation.
    """
    n = df.height
    ticker = _infer_ticker(df)
    if not ticker or ticker.endswith(".T"):
        return pl.Series("analyst_consensus_score", [0.0] * n, dtype=pl.Float64)

    history = _load_reco_history(ticker, n_days=1)
    if not history:
        return pl.Series("analyst_consensus_score", [0.0] * n, dtype=pl.Float64)

    score = _compute_consensus_score(history[0].get("data", {}))
    return pl.Series("analyst_consensus_score", [score] * n, dtype=pl.Float64)


# =============================================================================
# Feature: analyst_consensus_change
# =============================================================================


@reg.register(
    "analyst_consensus_change",
    FeatureTier.TIER5_ALTERNATIVE,
    lookback_days=1,
    description="Change in analyst consensus over 7 days (requires 7+ days accumulation)",
)
def analyst_consensus_change(df: pl.DataFrame) -> pl.Series:
    """7-day change in analyst consensus score.

    Returns 0.0 until 7+ days of accumulation exist.
    """
    n = df.height
    ticker = _infer_ticker(df)
    if not ticker or ticker.endswith(".T"):
        return pl.Series("analyst_consensus_change", [0.0] * n, dtype=pl.Float64)

    history = _load_reco_history(ticker, n_days=7)
    if len(history) < 2:
        return pl.Series("analyst_consensus_change", [0.0] * n, dtype=pl.Float64)

    newest_score = _compute_consensus_score(history[-1].get("data", {}))
    oldest_score = _compute_consensus_score(history[0].get("data", {}))
    change = newest_score - oldest_score

    return pl.Series("analyst_consensus_change", [change] * n, dtype=pl.Float64)


# =============================================================================
# Feature: news_count_7d
# =============================================================================


@reg.register(
    "news_count_7d",
    FeatureTier.TIER5_ALTERNATIVE,
    lookback_days=1,
    description="Number of news articles in last 7 days (attention proxy)",
)
def news_count_7d(df: pl.DataFrame) -> pl.Series:
    """Total news article count from Finnhub over 7 days.

    High news volume = more attention = potentially more volatility.
    """
    n = df.height
    ticker = _infer_ticker(df)
    if not ticker or ticker.endswith(".T"):
        return pl.Series("news_count_7d", [0.0] * n, dtype=pl.Float64)

    history = _load_news_history(ticker, n_days=7)
    if not history:
        return pl.Series("news_count_7d", [0.0] * n, dtype=pl.Float64)

    total = 0
    for entry in history:
        data = entry.get("data", {})
        if isinstance(data, dict):
            articles = data.get("articles", [])
            total += len(articles) if isinstance(articles, list) else 0

    # Log scale: 0 articles → 0, 10 → 1, 100 → 2, 1000 → 3
    import math

    log_count = math.log10(total + 1) if total > 0 else 0.0
    return pl.Series("news_count_7d", [log_count] * n, dtype=pl.Float64)
