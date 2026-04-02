"""News sentiment scoring using keyword-based approach (zero LLM cost).

Uses yfinance's built-in news feed to get recent headlines, then scores
them with simple positive/negative keyword matching.

Data flow:
  yfinance ticker.news → keyword scorer → SQLite (alternative_data_daily)
"""

from __future__ import annotations

import re
from datetime import datetime

import structlog

from qtp.data.database import QTPDatabase

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Keyword dictionaries
# ---------------------------------------------------------------------------

POSITIVE_KEYWORDS: list[str] = [
    "beat",
    "upgrade",
    "strong",
    "growth",
    "record",
    "surge",
    "rally",
    "outperform",
    "bullish",
    "buy",
    "exceed",
    "raise",
    "positive",
    "profit",
    "gain",
    "soar",
    "jump",
    "rise",
    "boost",
    "top",
    "high",
]

NEGATIVE_KEYWORDS: list[str] = [
    "miss",
    "downgrade",
    "weak",
    "decline",
    "cut",
    "fall",
    "crash",
    "underperform",
    "bearish",
    "sell",
    "layoff",
    "lawsuit",
    "loss",
    "warn",
    "slump",
    "plunge",
    "risk",
    "drop",
    "slide",
    "tumble",
    "fear",
    "concern",
    "low",
]

# Pre-compile patterns that match stems + common inflections (e.g. fall/falls/falling)
_POS_PATTERNS = [re.compile(rf"\b{kw}\w*\b", re.IGNORECASE) for kw in POSITIVE_KEYWORDS]
_NEG_PATTERNS = [re.compile(rf"\b{kw}\w*\b", re.IGNORECASE) for kw in NEGATIVE_KEYWORDS]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_headline(headline: str) -> float:
    """Score a single headline on [-1, +1] using keyword matching.

    Returns:
        Float in range [-1, +1].
        +1 = strongly positive, -1 = strongly negative, 0 = neutral.
    """
    if not headline:
        return 0.0

    pos_hits = sum(1 for pat in _POS_PATTERNS if pat.search(headline))
    neg_hits = sum(1 for pat in _NEG_PATTERNS if pat.search(headline))

    total = pos_hits + neg_hits
    if total == 0:
        return 0.0

    # Net score normalized to [-1, +1]
    raw = (pos_hits - neg_hits) / total
    return max(-1.0, min(1.0, raw))


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------


def fetch_news_sentiment(
    ticker: str,
    db: QTPDatabase,
    date_str: str | None = None,
) -> dict:
    """Fetch news headlines via yfinance, score them, and save to SQLite.

    Args:
        ticker: Ticker symbol (e.g. "AAPL").
        db: QTPDatabase instance.
        date_str: ISO date (YYYY-MM-DD). Defaults to today.

    Returns:
        Dict with keys: sentiment_avg, news_volume, headlines (list of scored items).
    """
    import yfinance as yf

    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    try:
        tk = yf.Ticker(ticker)
        news_items = tk.news or []
    except Exception as e:
        logger.warning("yfinance_news_fetch_failed", ticker=ticker, error=str(e))
        news_items = []

    scored: list[dict] = []
    for item in news_items:
        # yfinance may nest title under 'content' dict or at top level
        content = item.get("content", item)
        title = content.get("title", "") if isinstance(content, dict) else item.get("title", "")
        if not title:
            continue
        score = score_headline(title)
        scored.append({"title": title, "score": score})

    if scored:
        sentiment_avg = sum(s["score"] for s in scored) / len(scored)
    else:
        sentiment_avg = 0.0

    result = {
        "sentiment_avg": round(sentiment_avg, 4),
        "news_volume": len(scored),
        "headlines": scored,
    }

    # Persist to alternative_data_daily table
    db.upsert_alternative_daily(
        ticker=ticker,
        tool="news_sentiment",
        data=result,
        date=date_str,
    )

    logger.info(
        "news_sentiment_fetched",
        ticker=ticker,
        date=date_str,
        n_articles=len(scored),
        sentiment_avg=result["sentiment_avg"],
    )
    return result
