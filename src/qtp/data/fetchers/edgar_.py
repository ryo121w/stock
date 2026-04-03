"""SEC EDGAR insider transaction fetcher via EdgarTools.

Provides complete Form 4 transaction history directly from SEC.
No API key required, no rate limits, decades of history available.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import structlog

logger = structlog.get_logger()

# Session-level cache to avoid redundant SEC calls
_insider_cache: dict[str, list[dict]] = {}


def _ensure_identity():
    """Set SEC EDGAR identity (required by SEC fair access policy)."""
    try:
        from edgar import set_identity

        set_identity("QTP Pipeline research@example.com")
    except Exception:
        pass


def _load_disk_cache(ticker: str) -> list[dict] | None:
    """Load cached EDGAR data from disk (survives session restarts)."""
    import json

    cache_dir = Path("data/cache/edgar")
    cache_file = cache_dir / f"{ticker.replace('.', '_')}.json"
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text())
            # Check freshness: 7 days max
            from datetime import datetime

            cached_at = datetime.fromisoformat(data.get("cached_at", "2000-01-01"))
            if (datetime.now() - cached_at).days <= 7:
                return data.get("transactions", [])
        except Exception:
            pass
    return None


def _save_disk_cache(ticker: str, transactions: list[dict]) -> None:
    """Save EDGAR data to disk cache."""
    import json
    from datetime import datetime

    cache_dir = Path("data/cache/edgar")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{ticker.replace('.', '_')}.json"
    cache_file.write_text(
        json.dumps(
            {
                "cached_at": datetime.now().isoformat(),
                "transactions": transactions,
            }
        )
    )


def fetch_insider_transactions(ticker: str, months: int = 6, max_filings: int = 50) -> list[dict]:
    """Fetch insider transactions from SEC EDGAR Form 4 filings.

    Returns list of transactions sorted by date (oldest first):
    [{"date": "2025-01-15", "insider": "John Doe", "position": "CEO",
      "type": "BUY"/"SELL", "shares": 10000, "price": 150.0, "value": 1500000}, ...]
    """
    cache_key = f"{ticker}:{months}"
    if cache_key in _insider_cache:
        return _insider_cache[cache_key]

    # Try disk cache first (survives session restarts)
    disk_cached = _load_disk_cache(ticker)
    if disk_cached is not None:
        _insider_cache[cache_key] = disk_cached
        logger.info("edgar_disk_cache_hit", ticker=ticker, transactions=len(disk_cached))
        return disk_cached

    _ensure_identity()

    try:
        from edgar import Company

        company = Company(ticker)
        filings = company.get_filings(form="4")

        transactions = []
        cutoff = date.today() - timedelta(days=months * 30)

        for filing in filings[:max_filings]:
            # Skip filings older than cutoff
            filing_date = filing.filing_date
            if hasattr(filing_date, "date"):
                filing_date = filing_date.date()
            if isinstance(filing_date, str):
                filing_date = date.fromisoformat(filing_date)
            if filing_date < cutoff:
                break

            try:
                form4 = filing.obj()

                insider = getattr(form4, "insider_name", "Unknown")
                position = getattr(form4, "position", "Unknown")

                # Process purchases
                purchases = getattr(form4, "common_stock_purchases", None)
                if purchases is not None and not purchases.empty:
                    for _, row in purchases.iterrows():
                        transactions.append(
                            {
                                "date": str(row.get("Date", filing_date)),
                                "insider": insider,
                                "position": position,
                                "type": "BUY",
                                "shares": int(row.get("Shares", 0)),
                                "price": float(row.get("Price", 0)) if row.get("Price") else 0.0,
                                "value": int(row.get("Shares", 0))
                                * float(row.get("Price", 0) or 0),
                            }
                        )

                # Process sales
                sales = getattr(form4, "common_stock_sales", None)
                if sales is not None and not sales.empty:
                    for _, row in sales.iterrows():
                        transactions.append(
                            {
                                "date": str(row.get("Date", filing_date)),
                                "insider": insider,
                                "position": position,
                                "type": "SELL",
                                "shares": int(row.get("Shares", 0)),
                                "price": float(row.get("Price", 0)) if row.get("Price") else 0.0,
                                "value": int(row.get("Shares", 0))
                                * float(row.get("Price", 0) or 0),
                            }
                        )

            except Exception as e:
                logger.debug("form4_parse_error", filing_date=str(filing_date), error=str(e))
                continue

        # Sort oldest first
        transactions.sort(key=lambda t: t["date"])
        _insider_cache[cache_key] = transactions
        _save_disk_cache(ticker, transactions)

        logger.info(
            "edgar_insider_fetched",
            ticker=ticker,
            transactions=len(transactions),
            months=months,
        )
        return transactions

    except Exception as e:
        logger.warning("edgar_fetch_failed", ticker=ticker, error=str(e))
        _insider_cache[cache_key] = []
        return []


def clear_cache() -> None:
    _insider_cache.clear()
