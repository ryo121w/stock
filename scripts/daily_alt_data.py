#!/usr/bin/env python3
"""Daily alternative data accumulation.

Fetches and stores alternative data from multiple sources:
- EDGAR: Insider transactions (SEC Form 4)
- Fear & Greed: CNN market sentiment index
- Finnhub: Analyst recommendations, price targets, upgrades, news, EPS estimates

Run daily via cron or `make alt-data`.
Data is stored in SQLite (alternative_data + alternative_data_daily tables).
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import structlog

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from qtp.data.database import QTPDatabase  # noqa: E402

logger = structlog.get_logger()


def load_tickers() -> list[str]:
    """Load tickers from Phase5 config."""
    import yaml

    config_path = project_root / "configs" / "phase5_optimized.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["universe"]["tickers"]


def fetch_edgar(db: QTPDatabase, tickers: list[str]) -> None:
    """Fetch insider transactions from SEC EDGAR."""
    from qtp.data.fetchers.edgar_ import clear_cache, fetch_insider_transactions

    clear_cache()
    today = date.today().isoformat()

    for ticker in tickers:
        if ticker.endswith(".T"):
            continue  # EDGAR = US only

        try:
            txns = fetch_insider_transactions(ticker, months=6, max_filings=30)
            buys = sum(1 for t in txns if t["type"] == "BUY")
            sells = sum(1 for t in txns if t["type"] == "SELL")

            data = {
                "transactions": txns,
                "buy_count": buys,
                "sell_count": sells,
                "net_signal": 1 if buys > sells else (-1 if sells > buys else 0),
            }

            db.upsert_alternative(ticker, "edgar_insider", data)
            db.upsert_alternative_daily(ticker, "edgar_insider", today, data)
            logger.info("edgar_saved", ticker=ticker, buys=buys, sells=sells)
        except Exception as e:
            logger.warning("edgar_failed", ticker=ticker, error=str(e))


def fetch_fear_greed(db: QTPDatabase) -> None:
    """Fetch CNN Fear & Greed Index."""
    from qtp.data.fetchers.fear_greed_ import clear_cache, fetch_fear_greed

    clear_cache()
    today = date.today().isoformat()

    try:
        data = fetch_fear_greed()
        db.upsert_alternative("_market", "fear_greed", data)
        db.upsert_alternative_daily("_market", "fear_greed", today, data)
        logger.info("fear_greed_saved", score=data.get("score"), rating=data.get("rating"))
    except Exception as e:
        logger.warning("fear_greed_failed", error=str(e))


def fetch_finnhub(db: QTPDatabase, tickers: list[str]) -> None:
    """Fetch analyst data from Finnhub (requires FINNHUB_API_KEY)."""
    from qtp.data.fetchers.finnhub_ import (
        fetch_company_news,
        fetch_eps_estimates,
        fetch_price_target,
        fetch_recommendation_trends,
        fetch_upgrade_downgrade,
        is_available,
    )

    if not is_available():
        logger.warning("finnhub_skipped", reason="FINNHUB_API_KEY not set")
        return

    today = date.today().isoformat()

    for ticker in tickers:
        if ticker.endswith(".T"):
            continue  # Finnhub = primarily US stocks

        tools = [
            ("finnhub_reco_trends", fetch_recommendation_trends),
            ("finnhub_price_target", fetch_price_target),
            ("finnhub_upgrades", fetch_upgrade_downgrade),
            ("finnhub_news", fetch_company_news),
            ("finnhub_eps_estimates", fetch_eps_estimates),
        ]

        for tool_name, fetch_fn in tools:
            try:
                data = fetch_fn(ticker)
                if data:
                    db.upsert_alternative(ticker, tool_name, data)
                    db.upsert_alternative_daily(ticker, tool_name, today, data)
            except Exception as e:
                logger.warning("finnhub_tool_failed", ticker=ticker, tool=tool_name, error=str(e))


def main():
    db = QTPDatabase(project_root / "data" / "qtp.db")
    tickers = load_tickers()

    logger.info("daily_alt_data_start", tickers=len(tickers), date=date.today().isoformat())

    # Phase 1: Market-level data
    fetch_fear_greed(db)

    # Phase 2: EDGAR insider (no rate limit, but slow)
    fetch_edgar(db, tickers)

    # Phase 3: Finnhub (rate-limited, requires API key)
    fetch_finnhub(db, tickers)

    logger.info("daily_alt_data_complete")


if __name__ == "__main__":
    main()
