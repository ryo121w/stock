"""yfinance data fetcher implementation (development use)."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import polars as pl
import structlog
import yfinance as yf

from qtp.data.fetchers.base import DataFetcher, FetchRequest, Market

logger = structlog.get_logger()


class YFinanceFetcher(DataFetcher):
    """Fetch OHLCV and basic fundamentals from Yahoo Finance via yfinance."""

    def fetch_ohlcv(self, request: FetchRequest) -> pl.DataFrame:
        suffix = ".T" if request.market == Market.JP else ""
        symbol = f"{request.ticker}{suffix}"

        logger.info("fetching_ohlcv", symbol=symbol, start=str(request.start_date),
                     end=str(request.end_date))

        # yfinance end is exclusive, so add 1 day
        df_pd = yf.download(
            symbol,
            start=request.start_date.isoformat(),
            end=(request.end_date + timedelta(days=1)).isoformat(),
            auto_adjust=request.adjust,
            progress=False,
        )

        if df_pd.empty:
            logger.warning("no_data_returned", symbol=symbol)
            return pl.DataFrame(schema={
                "date": pl.Date, "open": pl.Float64, "high": pl.Float64,
                "low": pl.Float64, "close": pl.Float64, "volume": pl.Float64,
            })

        # Handle multi-level columns from yfinance >= 1.2
        if isinstance(df_pd.columns, pd.MultiIndex):
            df_pd.columns = df_pd.columns.get_level_values(0)

        # Deduplicate column names (yfinance may return duplicate "Price" level)
        df_pd = df_pd.loc[:, ~df_pd.columns.duplicated()]
        df_pd = df_pd.reset_index()

        # Normalize column names to lowercase
        df_pd.columns = [str(c).lower().strip() for c in df_pd.columns]

        # Ensure volume is float for Polars compatibility
        df_pd["volume"] = df_pd["volume"].astype(float)
        df = pl.from_pandas(df_pd[["date", "open", "high", "low", "close", "volume"]])

        # Cast date column
        if df["date"].dtype != pl.Date:
            df = df.with_columns(pl.col("date").cast(pl.Date))

        # ANTI-LEAKAGE: enforce end_date boundary
        df = df.filter(pl.col("date") <= request.end_date)
        df = df.sort("date")

        logger.info("fetched_ohlcv", symbol=symbol, rows=df.height)
        return df

    def fetch_fundamentals(self, ticker: str, market: Market, as_of: date) -> pl.DataFrame:
        suffix = ".T" if market == Market.JP else ""
        symbol = f"{ticker}{suffix}"

        try:
            info = yf.Ticker(symbol).info
        except Exception as e:
            logger.warning("fundamentals_fetch_failed", symbol=symbol, error=str(e))
            return pl.DataFrame()

        row = {
            "ticker": ticker,
            "as_of": as_of,
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "pb_ratio": info.get("priceToBook"),
            "peg_ratio": info.get("pegRatio"),
            "roe": info.get("returnOnEquity"),
            "revenue_growth": info.get("revenueGrowth"),
            "operating_margin": info.get("operatingMargins"),
            "profit_margin": info.get("profitMargins"),
            "debt_to_equity": info.get("debtToEquity"),
            "free_cashflow": info.get("freeCashflow"),
            "market_cap": info.get("marketCap"),
            "dividend_yield": info.get("dividendYield"),
        }
        return pl.DataFrame([row])

    def name(self) -> str:
        return "yfinance"
