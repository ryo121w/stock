"""Fetch alternative data via stock-tools MCP and cache in SQLite.

This fetcher calls MCP tools (earnings_trend, analyst_actions, insider_transactions,
analyst_estimates, earnings_date) and stores results in qtp.db.

The cached data is consumed by tier5_alternative.py to compute features.
"""

from __future__ import annotations

import structlog

from qtp.data.database import QTPDatabase

logger = structlog.get_logger()

MCP_TOOLS = {
    "earnings_trend": "mcp__stock-tools__fetch_earnings_trend",
    "analyst_actions": "mcp__stock-tools__fetch_analyst_actions",
    "insider_transactions": "mcp__stock-tools__fetch_insider_transactions",
    "analyst_estimates": "mcp__stock-tools__fetch_analyst_estimates",
    "earnings_date": "mcp__stock-tools__fetch_earnings_date",
    "market_regime": "mcp__stock-tools__fetch_market_regime",
}


def fetch_alternative_data(
    ticker: str,
    db: QTPDatabase,
    force_refresh: bool = False,
    max_age_hours: int = 24,
) -> dict[str, dict]:
    """Fetch all alternative data for a ticker via MCP tools.

    Returns dict of {tool_name: response_data}.
    Results are cached in SQLite (alternative_data table).
    """
    results = {}

    for tool_name, mcp_tool_id in MCP_TOOLS.items():
        # For market_regime, use "_market" as ticker key
        cache_ticker = "_market" if tool_name == "market_regime" else ticker

        # Use cache if fresh
        if not force_refresh:
            cached = db.get_alternative_fresh(cache_ticker, tool_name, max_age_hours)
            if cached:
                results[tool_name] = cached
                continue

        # Fetch via MCP tool
        try:
            data = _call_mcp_tool(mcp_tool_id, ticker, tool_name)
            if data:
                db.upsert_alternative(cache_ticker, tool_name, data, max_age_hours)
                results[tool_name] = data
                logger.info("alt_data_fetched", ticker=ticker, tool=tool_name)
            else:
                logger.warning("alt_data_empty", ticker=ticker, tool=tool_name)
        except Exception as e:
            logger.warning("alt_data_failed", ticker=ticker, tool=tool_name, error=str(e))

    return results


def load_alternative_data(ticker: str, db: QTPDatabase) -> dict[str, dict]:
    """Load cached alternative data for a ticker from SQLite (no fetching)."""
    results = {}

    for tool_name in MCP_TOOLS:
        cache_ticker = "_market" if tool_name == "market_regime" else ticker
        data = db.get_alternative(cache_ticker, tool_name)
        if data:
            results[tool_name] = data

    return results


def _call_mcp_tool(tool_id: str, ticker: str, tool_name: str) -> dict | None:
    """Call an MCP tool and return parsed JSON response."""
    try:
        from qtp.data.fetchers.mcp_client import call_stock_tool

        return call_stock_tool(tool_id, {"ticker": ticker})
    except ImportError:
        logger.debug(
            "mcp_client_not_available",
            tool=tool_name,
            msg="Using cached data only. Install mcp_client for live fetching.",
        )
        return None
