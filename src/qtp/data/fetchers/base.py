"""Abstract data fetcher with enforced point-in-time correctness."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from datetime import date
from enum import Enum

import polars as pl


class Market(str, Enum):
    US = "us"
    JP = "jp"


@dataclass(frozen=True)
class FetchRequest:
    ticker: str
    market: Market
    start_date: date
    end_date: date      # INCLUSIVE upper bound = "as-of" date
    adjust: bool = True


class DataFetcher(abc.ABC):
    """Point-in-time data fetcher.

    ANTI-LEAKAGE CONTRACT: end_date/as_of is the maximum date for any returned data.
    Implementations MUST filter WHERE date <= end_date.
    """

    @abc.abstractmethod
    def fetch_ohlcv(self, request: FetchRequest) -> pl.DataFrame:
        """Return columns: [date, open, high, low, close, volume].

        - Dates are business days only
        - Sorted ascending by date
        - No data after request.end_date
        """
        ...

    @abc.abstractmethod
    def fetch_fundamentals(self, ticker: str, market: Market, as_of: date) -> pl.DataFrame:
        """Return point-in-time fundamentals available as of `as_of`.

        Uses report_date (filing date), NOT period_end_date.
        """
        ...

    @abc.abstractmethod
    def name(self) -> str:
        ...
