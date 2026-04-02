"""Stock universe management."""

from __future__ import annotations

from dataclasses import dataclass

from qtp.config import UniverseConfig
from qtp.data.fetchers.base import Market


@dataclass(frozen=True)
class StockInfo:
    ticker: str
    name: str
    sector: str
    market: Market


class Universe:
    """Manages the set of stocks to analyze."""

    def __init__(self, config: UniverseConfig):
        self.market = Market(config.market)
        self._tickers = config.tickers
        self.min_avg_daily_volume = config.min_avg_daily_volume

    def tickers(self) -> list[str]:
        return list(self._tickers)

    def __len__(self) -> int:
        return len(self._tickers)

    def __iter__(self):
        return iter(self._tickers)
