"""CNN Fear & Greed Index fetcher.

Provides market sentiment scores (0-100) with ~1 year of daily history.
No API key required.
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger()

_fg_cache: dict | None = None


def fetch_fear_greed() -> dict:
    """Fetch current Fear & Greed data.

    Returns:
        {"score": 19.3, "rating": "extreme fear",
         "history": {"1w": 14.5, "1m": 31.6, "3m": 45.2, "6m": 54.0, "1y": 12.1},
         "indicators": {...}}
    """
    global _fg_cache
    if _fg_cache is not None:
        return _fg_cache

    try:
        import fear_greed

        data = fear_greed.get()
        _fg_cache = data
        logger.info("fear_greed_fetched", score=data.get("score"), rating=data.get("rating"))
        return data
    except Exception as e:
        logger.warning("fear_greed_fetch_failed", error=str(e))
        return {"score": 50.0, "rating": "neutral", "history": {}, "indicators": {}}


def fetch_fear_greed_history() -> list[dict]:
    """Fetch Fear & Greed daily history (~1 year).

    Returns list sorted oldest-first:
    [{"date": "2025-04-03", "score": 45.2}, ...]
    """
    try:
        import fear_greed

        history = fear_greed.get_history()
        # history is a list of HistoricalPoint with date and score
        result = []
        if isinstance(history, list):
            for point in history:
                if hasattr(point, "date") and hasattr(point, "score"):
                    d = point.date
                    if hasattr(d, "date"):
                        d = d.date()
                    result.append({"date": str(d), "score": float(point.score)})
                elif isinstance(point, dict):
                    result.append(
                        {
                            "date": str(point.get("date", "")),
                            "score": float(point.get("score", 50)),
                        }
                    )

        # Sort oldest first
        result.sort(key=lambda x: x["date"])
        logger.info("fear_greed_history_fetched", points=len(result))
        return result
    except Exception as e:
        logger.warning("fear_greed_history_failed", error=str(e))
        return []


def clear_cache() -> None:
    global _fg_cache
    _fg_cache = None
