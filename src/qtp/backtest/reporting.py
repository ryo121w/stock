"""Report generation with QuantStats."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import structlog

logger = structlog.get_logger()


class ReportGenerator:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_tearsheet(
        self,
        returns: pd.Series,
        benchmark_ticker: str = "SPY",
        title: str = "QTP Strategy",
    ) -> Path:
        """Generate full QuantStats HTML tear sheet."""
        import quantstats as qs
        import yfinance as yf

        # Fetch benchmark
        benchmark = yf.download(
            benchmark_ticker,
            start=returns.index[0],
            progress=False,
        )["Close"].pct_change().dropna()

        report_path = self.output_dir / f"tearsheet_{datetime.now():%Y%m%d_%H%M%S}.html"
        qs.reports.html(
            returns,
            benchmark=benchmark,
            output=str(report_path),
            title=title,
        )
        logger.info("tearsheet_generated", path=str(report_path))
        return report_path
