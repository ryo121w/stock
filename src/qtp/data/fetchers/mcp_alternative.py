"""Alternative data fetcher — aggregates EDGAR, Finnhub, and Fear & Greed.

Replaces the old MCP-based fetcher (mcp_client.py was never implemented).
Now uses direct API calls via EdgarTools, Finnhub, and fear-greed library.

The cached data is consumed by tier5_alternative.py to compute features.
"""

from __future__ import annotations

import structlog

from qtp.data.database import QTPDatabase

logger = structlog.get_logger()


def fetch_alternative_data(
    ticker: str,
    db: QTPDatabase,
    force_refresh: bool = False,
    max_age_hours: int = 24,
) -> dict[str, dict]:
    """Fetch all alternative data for a ticker.

    Uses:
    - EdgarTools for insider transactions (US stocks only)
    - Finnhub for analyst recommendations and news
    - Fear & Greed for market sentiment

    Results are cached in SQLite (alternative_data table).
    """
    results = {}

    # EDGAR insider (US stocks only)
    if not ticker.endswith(".T"):
        if not force_refresh:
            cached = db.get_alternative_fresh(ticker, "edgar_insider", max_age_hours)
            if cached:
                results["insider_transactions"] = cached
        if "insider_transactions" not in results:
            try:
                from qtp.data.fetchers.edgar_ import fetch_insider_transactions

                txns = fetch_insider_transactions(ticker, months=6, max_filings=30)
                data = {"transactions": txns, "count": len(txns)}
                db.upsert_alternative(ticker, "edgar_insider", data, max_age_hours)
                results["insider_transactions"] = data
            except Exception as e:
                logger.warning("edgar_alt_failed", ticker=ticker, error=str(e))

    # Finnhub recommendation trends
    if not force_refresh:
        cached = db.get_alternative_fresh(ticker, "finnhub_reco_trends", max_age_hours)
        if cached:
            results["analyst_actions"] = cached
    if "analyst_actions" not in results:
        try:
            from qtp.data.fetchers.finnhub_ import fetch_recommendation_trends

            data = fetch_recommendation_trends(ticker)
            if data:
                db.upsert_alternative(ticker, "finnhub_reco_trends", data, max_age_hours)
                results["analyst_actions"] = data
        except Exception as e:
            logger.debug("finnhub_reco_alt_failed", ticker=ticker, error=str(e))

    # Fear & Greed (market-level)
    if not force_refresh:
        cached = db.get_alternative_fresh("_market", "fear_greed", max_age_hours)
        if cached:
            results["market_regime"] = cached
    if "market_regime" not in results:
        try:
            from qtp.data.fetchers.fear_greed_ import fetch_fear_greed

            data = fetch_fear_greed()
            if data:
                db.upsert_alternative("_market", "fear_greed", data, max_age_hours)
                results["market_regime"] = data
        except Exception as e:
            logger.debug("fear_greed_alt_failed", error=str(e))

    return results


def load_alternative_data(ticker: str, db: QTPDatabase) -> dict[str, dict]:
    """Load cached alternative data for a ticker from SQLite (no fetching)."""
    results = {}

    tool_map = {
        "edgar_insider": "insider_transactions",
        "finnhub_reco_trends": "analyst_actions",
        "finnhub_news": "company_news",
    }

    for db_tool, result_key in tool_map.items():
        data = db.get_alternative(ticker, db_tool)
        if data:
            results[result_key] = data

    # Market-level data
    fg = db.get_alternative("_market", "fear_greed")
    if fg:
        results["market_regime"] = fg

    return results
