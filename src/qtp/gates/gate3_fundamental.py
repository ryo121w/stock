"""Gate 3: Fundamental check gate.

Validates fundamental health using pre-fetched MCP data:
  - yahoo_quote (revenue growth, earnings growth, ROE, current price)
  - earnings_trend (EPS revision signal: UPGRADE / NEUTRAL / DOWNGRADE)
  - analyst_estimates (target price consensus)

Instant-fail conditions:
  - EPS trend is DOWNGRADE
  - Analyst target price < current price
"""

from __future__ import annotations

import structlog

from qtp.gates import GateResult

logger = structlog.get_logger()


class Gate3_Fundamental:
    """Fundamental check gate -- filters deteriorating businesses."""

    PASS_SCORE = 35  # Minimum score to pass

    def evaluate(
        self,
        ticker: str,
        yahoo_quote: dict | None = None,
        earnings_trend: dict | None = None,
        analyst_estimates: dict | None = None,
    ) -> GateResult:
        """Run Gate 3 evaluation.

        Args:
            ticker: Ticker symbol.
            yahoo_quote: Dict from fetch_yahoo_quote MCP tool.
            earnings_trend: Dict from fetch_earnings_trend MCP tool.
            analyst_estimates: Dict from fetch_analyst_estimates MCP tool.
        """
        if yahoo_quote is None:
            return GateResult(
                gate="Fundamental",
                passed=False,
                score=0.0,
                reason="No yahoo_quote data available",
            )

        # Extract fields with safe defaults
        current_price = _safe_float(
            yahoo_quote.get("regularMarketPrice") or yahoo_quote.get("price")
        )
        revenue_growth = _safe_float(yahoo_quote.get("revenueGrowth"), default=0.0) * 100  # pct
        earnings_growth = (
            _safe_float(
                yahoo_quote.get("earningsGrowth") or yahoo_quote.get("earningsQuarterlyGrowth"),
                default=0.0,
            )
            * 100
        )
        roe = _safe_float(yahoo_quote.get("returnOnEquity"), default=0.0) * 100  # pct

        # --- EPS trend signal ---
        eps_signal = "NEUTRAL"
        if earnings_trend:
            eps_signal = _extract_eps_signal(earnings_trend)

        # --- Analyst target price ---
        target_price: float | None = None
        if analyst_estimates:
            target_price = _safe_float(
                analyst_estimates.get("targetMeanPrice")
                or analyst_estimates.get("target_mean_price")
                or analyst_estimates.get("targetPrice")
            )

        # ===============================================================
        # Instant-fail conditions
        # ===============================================================

        if eps_signal == "DOWNGRADE":
            return GateResult(
                gate="Fundamental",
                passed=False,
                score=10.0,
                reason="EPS DOWNGRADE detected",
                data=self._build_data(
                    revenue_growth, earnings_growth, roe, eps_signal, target_price, current_price
                ),
            )

        if target_price is not None and current_price is not None and target_price < current_price:
            return GateResult(
                gate="Fundamental",
                passed=False,
                score=15.0,
                reason=f"Target price {target_price:.2f} < current {current_price:.2f}",
                data=self._build_data(
                    revenue_growth, earnings_growth, roe, eps_signal, target_price, current_price
                ),
            )

        # ===============================================================
        # Score calculation
        # ===============================================================

        score = 50.0
        score += min(20.0, revenue_growth * 0.5)  # Revenue growth bonus (max +20)
        score += min(15.0, earnings_growth * 0.3)  # Earnings growth bonus (max +15)
        score += 10.0 if eps_signal == "UPGRADE" else 0.0
        score += 5.0 if roe > 15 else -5.0
        score = max(0.0, min(100.0, score))

        passed = score >= self.PASS_SCORE

        # Build reason
        reason_parts = [f"rev_growth={revenue_growth:+.1f}%"]
        reason_parts.append(f"earn_growth={earnings_growth:+.1f}%")
        reason_parts.append(f"EPS={eps_signal}")
        reason_parts.append(f"ROE={roe:.1f}%")
        if target_price is not None and current_price is not None:
            upside = (target_price - current_price) / current_price * 100
            reason_parts.append(f"upside={upside:+.1f}%")

        warnings: list[str] = []
        if revenue_growth < 0:
            warnings.append(f"Revenue declining ({revenue_growth:+.1f}%)")
        if roe < 10:
            warnings.append(f"Low ROE ({roe:.1f}%)")
        if earnings_growth < 0:
            warnings.append(f"Earnings declining ({earnings_growth:+.1f}%)")

        return GateResult(
            gate="Fundamental",
            passed=passed,
            score=score,
            reason=", ".join(reason_parts),
            warnings=warnings,
            data=self._build_data(
                revenue_growth, earnings_growth, roe, eps_signal, target_price, current_price
            ),
        )

    @staticmethod
    def _build_data(
        revenue_growth: float,
        earnings_growth: float,
        roe: float,
        eps_signal: str,
        target_price: float | None,
        current_price: float | None,
    ) -> dict:
        return {
            "revenue_growth": revenue_growth,
            "earnings_growth": earnings_growth,
            "roe": roe,
            "eps_signal": eps_signal,
            "target_price": target_price,
            "current_price": current_price,
        }


# =====================================================================
# Helper utilities
# =====================================================================


def _safe_float(value, default: float | None = None) -> float | None:
    """Safely convert a value to float."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_eps_signal(earnings_trend: dict) -> str:
    """Extract EPS revision signal from earnings_trend MCP data.

    Looks for common keys produced by fetch_earnings_trend:
      - "signal" (direct)
      - "eps_trend" / "epsTrend" with revision data
      - "trend" list with revision counts

    Returns: "UPGRADE", "NEUTRAL", or "DOWNGRADE"
    """
    # Direct signal field
    signal = earnings_trend.get("signal")
    if signal and isinstance(signal, str):
        upper = signal.upper()
        if upper in ("UPGRADE", "DOWNGRADE", "NEUTRAL"):
            return upper

    # Look at revision counts
    up = earnings_trend.get("upRevisions") or earnings_trend.get("up_revisions") or 0
    down = earnings_trend.get("downRevisions") or earnings_trend.get("down_revisions") or 0

    # Nested trend data
    trend_data = earnings_trend.get("trend") or earnings_trend.get("eps_trend") or []
    if isinstance(trend_data, list):
        for entry in trend_data:
            if isinstance(entry, dict):
                up += int(entry.get("earningsEstimateNumberOfUpRevisions", 0) or 0)
                down += int(entry.get("earningsEstimateNumberOfDownRevisions", 0) or 0)

    if down > up and down > 0:
        return "DOWNGRADE"
    if up > down and up > 0:
        return "UPGRADE"
    return "NEUTRAL"
