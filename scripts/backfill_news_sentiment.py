#!/usr/bin/env python
"""Backfill news sentiment data for all pipeline tickers.

Uses yfinance's news feed to get currently-available headlines,
scores them with keyword-based sentiment, and saves to SQLite.

Usage:
    .venv/bin/python scripts/backfill_news_sentiment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from qtp.data.database import QTPDatabase  # noqa: E402
from qtp.data.fetchers.news_sentiment import fetch_news_sentiment  # noqa: E402

# Same tickers as config.yaml
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "UNH"]

DB_PATH = project_root / "data" / "qtp.db"


def main() -> None:
    db = QTPDatabase(DB_PATH)
    print(f"Database: {DB_PATH}")
    print(f"Tickers:  {', '.join(TICKERS)}")
    print("-" * 60)

    results = {}
    for ticker in TICKERS:
        try:
            result = fetch_news_sentiment(ticker, db)
            results[ticker] = result
            print(
                f"  {ticker:6s} | articles={result['news_volume']:3d} | "
                f"sentiment={result['sentiment_avg']:+.4f}"
            )
        except Exception as e:
            print(f"  {ticker:6s} | ERROR: {e}")
            results[ticker] = None

    # Summary
    print("-" * 60)
    ok = sum(1 for v in results.values() if v is not None)
    total_articles = sum(v["news_volume"] for v in results.values() if v is not None)
    print(f"Done: {ok}/{len(TICKERS)} tickers, {total_articles} total articles scored.")


if __name__ == "__main__":
    main()
