#!/usr/bin/env python3
"""Grade past predictions by fetching actual prices.

For each ungraded prediction:
1. Fetch the actual close price on prediction_date (start)
2. Fetch the actual close price on prediction_date + horizon (end)
3. Calculate actual return and whether direction was correct
4. Update the prediction record in SQLite

Run this daily (or after each trading period) to build accuracy history.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import structlog
import yfinance as yf

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from qtp.data.database import QTPDatabase  # noqa: E402

logger = structlog.get_logger()


def fetch_price(ticker: str, target_date: str) -> float | None:
    """Fetch close price for a ticker on a specific date."""
    try:
        d = date.fromisoformat(target_date)
        start = d - timedelta(days=5)  # Buffer for weekends/holidays
        end = d + timedelta(days=5)

        df = yf.download(ticker, start=start.isoformat(), end=end.isoformat(), progress=False)
        if df.empty:
            return None

        # Find closest trading day on or after target
        df.index = df.index.tz_localize(None) if df.index.tz else df.index
        mask = df.index.date >= d
        if mask.any():
            row = df[mask].iloc[0]
        else:
            row = df.iloc[-1]

        close = row["Close"]
        if hasattr(close, "item"):
            return float(close.item())
        return float(close)
    except Exception as e:
        logger.warning("price_fetch_failed", ticker=ticker, date=target_date, error=str(e))
        return None


def grade_all(db: QTPDatabase):
    """Grade all ungraded predictions."""
    ungraded = db.get_ungraded_predictions()
    if not ungraded:
        print("No ungraded predictions found.")
        return

    today = date.today()
    print(f"Found {len(ungraded)} ungraded predictions.\n")

    graded = 0
    skipped = 0
    for pred in ungraded:
        ticker = pred["ticker"]
        pred_date = pred["prediction_date"]
        horizon = pred["horizon"]

        end_date = date.fromisoformat(pred_date) + timedelta(days=horizon)

        # Skip if end date is in the future (can't grade yet)
        if end_date > today:
            skipped += 1
            continue

        print(f"  Grading {ticker} {pred_date} (h={horizon})...", end=" ")

        price_start = fetch_price(ticker, pred_date)
        price_end = fetch_price(ticker, end_date.isoformat())

        if price_start is None or price_end is None:
            print("SKIP (price unavailable)")
            skipped += 1
            continue

        db.grade_prediction(pred["id"], price_start, price_end)

        actual_ret = (price_end - price_start) / price_start
        correct = (pred["direction"] == 1 and actual_ret > 0) or (
            pred["direction"] == 0 and actual_ret <= 0
        )
        status = "CORRECT" if correct else "WRONG"
        print(
            f"{status} (pred={'UP' if pred['direction'] == 1 else 'DOWN'} conf={pred['confidence']:.1%}, "
            f"actual={actual_ret:+.2%})"
        )
        graded += 1

    print(f"\nGraded: {graded}, Skipped: {skipped} (future or unavailable)")


def print_report(db: QTPDatabase):
    """Print accuracy report."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
    except ImportError:
        console = None

    summary = db.get_accuracy_summary()
    if not summary or not summary.get("total"):
        print("\nNo graded predictions yet.")
        return

    print(f"\n{'=' * 60}")
    print("  PREDICTION ACCURACY REPORT")
    print(f"{'=' * 60}")
    print(f"  Total graded: {summary['total']}")
    print(f"  Correct:      {summary['correct']} ({summary['accuracy']:.1%})")
    print(f"  Avg return:   {summary['avg_return']:+.3%}")
    if summary.get("avg_win"):
        print(f"  Avg win:      {summary['avg_win']:+.3%}")
    if summary.get("avg_loss"):
        print(f"  Avg loss:     {summary['avg_loss']:+.3%}")

    # By confidence bucket
    by_conf = db.get_accuracy_by_confidence()
    if by_conf:
        print("\n  By Confidence:")
        if console:
            t = Table(show_lines=False)
            t.add_column("Bucket")
            t.add_column("Total", justify="right")
            t.add_column("Correct", justify="right")
            t.add_column("Accuracy", justify="right", style="green")
            t.add_column("Avg Return", justify="right")
            for b in by_conf:
                t.add_row(
                    b["bucket"],
                    str(b["total"]),
                    str(b["correct"]),
                    f"{b['accuracy_pct']}%",
                    f"{b['avg_return_pct']}%",
                )
            console.print(t)
        else:
            for b in by_conf:
                print(
                    f"    {b['bucket']}: {b['accuracy_pct']}% accuracy "
                    f"({b['correct']}/{b['total']}, avg ret: {b['avg_return_pct']}%)"
                )

    # By ticker
    by_ticker = db.get_accuracy_by_ticker()
    if by_ticker:
        print("\n  By Ticker:")
        if console:
            t = Table(show_lines=False)
            t.add_column("Ticker", style="cyan")
            t.add_column("Total", justify="right")
            t.add_column("Correct", justify="right")
            t.add_column("Accuracy", justify="right", style="green")
            t.add_column("Avg Return", justify="right")
            for b in by_ticker:
                t.add_row(
                    b["ticker"],
                    str(b["total"]),
                    str(b["correct"]),
                    f"{b['accuracy_pct']}%",
                    f"{b['avg_return_pct']}%",
                )
            console.print(t)
        else:
            for b in by_ticker:
                print(
                    f"    {b['ticker']}: {b['accuracy_pct']}% accuracy "
                    f"({b['correct']}/{b['total']})"
                )

    print(f"\n{'=' * 60}")


def main():
    db = QTPDatabase(project_root / "data" / "qtp.db")

    if "--report" in sys.argv:
        print_report(db)
        return

    grade_all(db)
    print_report(db)


if __name__ == "__main__":
    main()
