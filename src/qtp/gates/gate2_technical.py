"""Gate 2: Technical confirmation gate.

Computes RSI(14), MACD direction, SMA200 position from yfinance OHLCV data
(same data the QTP pipeline uses).  Eliminates overheated or down-trending
tickers.
"""

from __future__ import annotations

import polars as pl
import structlog

from qtp.gates import GateResult

logger = structlog.get_logger()


class Gate2_Technical:
    """Technical indicators gate -- filters overheated / downtrending tickers."""

    RSI_OVERHEAT = 75  # RSI > 75 = instant fail
    PASS_SCORE = 40  # Minimum score to pass

    def evaluate(self, ticker: str, ohlcv: pl.DataFrame) -> GateResult:
        """Run Gate 2 evaluation.

        Args:
            ticker: Ticker symbol.
            ohlcv: OHLCV DataFrame with columns [date, open, high, low, close, volume].
                   Must have at least 200 rows for SMA200.
        """
        if ohlcv.height < 26:
            return GateResult(
                gate="Technical",
                passed=False,
                score=0.0,
                reason="Insufficient data (< 26 rows)",
            )

        rsi = self._compute_rsi(ohlcv, period=14)
        macd_improving = self._is_macd_improving(ohlcv)
        above_sma200 = self._is_above_sma200(ohlcv)

        # Instant fail: RSI > 75
        if rsi is not None and rsi > self.RSI_OVERHEAT:
            return GateResult(
                gate="Technical",
                passed=False,
                score=max(0, 100 - rsi),
                reason=f"RSI {rsi:.0f} -- overheated (> {self.RSI_OVERHEAT})",
                data={"rsi": rsi, "macd_improving": macd_improving, "above_sma200": above_sma200},
            )

        # Score calculation (0-100)
        score = 50.0
        if rsi is not None:
            score += (50 - rsi) * 0.5  # Lower RSI = more room to grow (max +25)
        score += 15 if macd_improving else -10
        score += 10 if above_sma200 else -15
        score = max(0.0, min(100.0, score))

        passed = score >= self.PASS_SCORE

        reason_parts = []
        if rsi is not None:
            reason_parts.append(f"RSI={rsi:.0f}")
        else:
            reason_parts.append("RSI=N/A")
        reason_parts.append(f"MACD={'up' if macd_improving else 'down'}")
        reason_parts.append(f"SMA200={'above' if above_sma200 else 'below'}")

        warnings: list[str] = []
        if rsi is not None and rsi > 65:
            warnings.append(f"RSI {rsi:.0f} approaching overbought")
        if not above_sma200:
            warnings.append("Below SMA200 -- potential downtrend")

        return GateResult(
            gate="Technical",
            passed=passed,
            score=score,
            reason=", ".join(reason_parts),
            warnings=warnings,
            data={"rsi": rsi, "macd_improving": macd_improving, "above_sma200": above_sma200},
        )

    # ------------------------------------------------------------------
    # Technical indicators (reuse QTP feature logic)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_rsi(df: pl.DataFrame, period: int = 14) -> float | None:
        """Compute RSI(period) and return the latest value."""
        if df.height < period + 1:
            return None
        close = df["close"]
        delta = close.diff()
        gain = delta.clip(lower_bound=0).rolling_mean(period)
        loss = (-delta.clip(upper_bound=0)).rolling_mean(period)
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        last = rsi.to_list()[-1]
        if last is None:
            return None
        return float(last)

    @staticmethod
    def _is_macd_improving(df: pl.DataFrame) -> bool:
        """True if MACD histogram is increasing (momentum improving)."""
        if df.height < 35:
            return False
        close = df["close"]
        ema12 = close.ewm_mean(span=12, adjust=False)
        ema26 = close.ewm_mean(span=26, adjust=False)
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm_mean(span=9, adjust=False)
        hist = (macd_line - signal_line).to_list()
        # Compare last two histogram values
        if len(hist) < 2 or hist[-1] is None or hist[-2] is None:
            return False
        return hist[-1] > hist[-2]

    @staticmethod
    def _is_above_sma200(df: pl.DataFrame) -> bool:
        """True if latest close is above SMA(200)."""
        if df.height < 200:
            return True  # Not enough data -- give benefit of the doubt
        sma = df["close"].rolling_mean(200).to_list()[-1]
        latest_close = df["close"].to_list()[-1]
        if sma is None or latest_close is None:
            return True
        return latest_close > sma
