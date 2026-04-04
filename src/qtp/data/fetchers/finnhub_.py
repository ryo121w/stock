"""Finnhub API fetcher for analyst data, EPS estimates, and news.

Free tier: 60 API calls/minute. No credit card required.
Register at https://finnhub.io/register to get an API key.

Set FINNHUB_API_KEY environment variable or pass to constructor.
If no key is set, all methods return empty results (graceful degradation).
"""

from __future__ import annotations

import os
import time
from datetime import date, timedelta

import structlog

logger = structlog.get_logger()

_client = None
_last_call_time = 0.0


def _get_client():
    """Lazy-init Finnhub client. Returns None if no API key."""
    global _client
    if _client is not None:
        return _client

    api_key = os.environ.get("FINNHUB_API_KEY", "")
    if not api_key:
        logger.warning(
            "finnhub_no_api_key", hint="Set FINNHUB_API_KEY env var (free at finnhub.io)"
        )
        return None

    try:
        import finnhub

        _client = finnhub.Client(api_key=api_key)
        logger.info("finnhub_client_initialized")
        return _client
    except Exception as e:
        logger.warning("finnhub_init_failed", error=str(e))
        return None


def _rate_limit():
    """Simple rate limiter: max ~55 calls/min (free tier = 60/min)."""
    global _last_call_time
    elapsed = time.time() - _last_call_time
    min_interval = 1.1  # ~55 calls/min
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _last_call_time = time.time()


def fetch_recommendation_trends(ticker: str) -> dict:
    """Analyst recommendation trends (strongBuy/buy/hold/sell/strongSell).

    Returns:
        {"trends": [{"period": "2026-04-01", "strongBuy": 10, "buy": 15, ...}, ...]}
    """
    client = _get_client()
    if client is None:
        return {}

    try:
        _rate_limit()
        data = client.recommendation_trends(ticker)
        logger.info("finnhub_reco_trends", ticker=ticker, periods=len(data))
        return {"trends": data}
    except Exception as e:
        logger.warning("finnhub_reco_trends_failed", ticker=ticker, error=str(e))
        return {}


def fetch_price_target(ticker: str) -> dict:
    """Analyst consensus price target.

    Returns:
        {"targetHigh": 250, "targetLow": 180, "targetMean": 215, "targetMedian": 210,
         "lastUpdated": "2026-03-15"}
    """
    client = _get_client()
    if client is None:
        return {}

    try:
        _rate_limit()
        data = client.price_target(ticker)
        return {
            "targetHigh": data.get("targetHigh"),
            "targetLow": data.get("targetLow"),
            "targetMean": data.get("targetMean"),
            "targetMedian": data.get("targetMedian"),
            "lastUpdated": data.get("lastUpdated"),
        }
    except Exception as e:
        logger.warning("finnhub_price_target_failed", ticker=ticker, error=str(e))
        return {}


def fetch_upgrade_downgrade(ticker: str, months: int = 6) -> list[dict]:
    """Analyst upgrade/downgrade history.

    Returns list of actions:
    [{"date": "2026-03-01", "company": "Goldman Sachs", "action": "upgrade",
      "fromGrade": "neutral", "toGrade": "buy"}, ...]
    """
    client = _get_client()
    if client is None:
        return []

    try:
        _rate_limit()
        from_date = (date.today() - timedelta(days=months * 30)).isoformat()
        to_date = date.today().isoformat()
        data = client.upgrade_downgrade(symbol=ticker, _from=from_date, to=to_date)

        actions = []
        for item in data or []:
            actions.append(
                {
                    "date": item.get("gradeDate", ""),
                    "company": item.get("company", ""),
                    "action": item.get("action", ""),
                    "fromGrade": item.get("fromGrade", ""),
                    "toGrade": item.get("toGrade", ""),
                }
            )

        logger.info("finnhub_upgrades", ticker=ticker, actions=len(actions))
        return actions
    except Exception as e:
        logger.warning("finnhub_upgrades_failed", ticker=ticker, error=str(e))
        return []


def fetch_company_news(ticker: str, days_back: int = 7) -> list[dict]:
    """Company news articles.

    Returns list of articles:
    [{"date": "2026-04-01", "headline": "...", "source": "Reuters",
      "summary": "...", "url": "...", "category": "company"}, ...]
    """
    client = _get_client()
    if client is None:
        return []

    try:
        _rate_limit()
        from_date = (date.today() - timedelta(days=days_back)).isoformat()
        to_date = date.today().isoformat()
        data = client.company_news(ticker, _from=from_date, to=to_date)

        articles = []
        for item in (data or [])[:50]:  # Cap at 50
            articles.append(
                {
                    "date": item.get("datetime", ""),
                    "headline": item.get("headline", ""),
                    "source": item.get("source", ""),
                    "summary": item.get("summary", ""),
                    "url": item.get("url", ""),
                    "category": item.get("category", ""),
                }
            )

        logger.info("finnhub_news", ticker=ticker, articles=len(articles))
        return articles
    except Exception as e:
        logger.warning("finnhub_news_failed", ticker=ticker, error=str(e))
        return []


def fetch_eps_estimates(ticker: str, freq: str = "quarterly") -> list[dict]:
    """EPS consensus estimates.

    Returns list of estimates:
    [{"period": "2026-Q1", "epsAvg": 1.52, "epsHigh": 1.65, "epsLow": 1.40,
      "numberAnalysts": 25}, ...]
    """
    client = _get_client()
    if client is None:
        return []

    try:
        _rate_limit()
        data = client.company_eps_estimates(ticker, freq=freq)

        estimates = []
        for item in data or []:
            estimates.append(
                {
                    "period": item.get("period", ""),
                    "epsAvg": item.get("epsAvg"),
                    "epsHigh": item.get("epsHigh"),
                    "epsLow": item.get("epsLow"),
                    "numberAnalysts": item.get("numberAnalysts"),
                }
            )

        logger.info("finnhub_eps_estimates", ticker=ticker, periods=len(estimates))
        return estimates
    except Exception as e:
        logger.warning("finnhub_eps_failed", ticker=ticker, error=str(e))
        return []


def is_available() -> bool:
    """Check if Finnhub API key is configured."""
    return bool(os.environ.get("FINNHUB_API_KEY", ""))
