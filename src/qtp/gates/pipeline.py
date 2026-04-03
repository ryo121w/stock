"""7-Gate evaluation pipeline orchestrator.

Usage
-----
    from qtp.gates.pipeline import GatePipeline

    pipeline = GatePipeline()
    verdict = pipeline.evaluate(
        ticker="AAPL",
        magi_votes={"melchior": "BUY", "balthasar": "HOLD", "casper": "BUY"},
        sentiment_data={"analyst_all_buy": False, "board_pessimistic": True},
    )
    print(verdict.verdict, verdict.score)

Gates 1-3 require external data (DB, OHLCV, MCP responses).  The pipeline
accepts pre-computed GateResults for those gates via ``gate1_result``,
``gate2_result``, ``gate3_result`` keyword arguments.  When not supplied, the
pipeline skips those gates and computes the integrated score from whichever
gates *are* present.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from qtp.gates import FinalVerdict, GateResult
from qtp.gates.gate4_magi import Gate4_MAGI
from qtp.gates.gate5_sentiment import Gate5_Sentiment
from qtp.gates.gate6_integration import Gate6_Integration
from qtp.gates.gate7_verdict import Gate7_Verdict, VerdictCache

logger = logging.getLogger(__name__)


class GatePipeline:
    """7-gate evaluation pipeline."""

    def __init__(
        self,
        cache_db: str | Path | None = None,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.verdict_cache = VerdictCache(db_path=cache_db)
        self.integrator = Gate6_Integration(weights=weights)
        self.judge = Gate7_Verdict()

    # ------------------------------------------------------------------
    def evaluate(
        self,
        ticker: str,
        *,
        gate1_result: GateResult | None = None,
        gate2_result: GateResult | None = None,
        gate3_result: GateResult | None = None,
        magi_votes: dict[str, str] | None = None,
        sentiment_data: dict[str, Any] | None = None,
        current_price: float | None = None,
        force: bool = False,
    ) -> FinalVerdict:
        """Run all 7 gates sequentially. Stop early if a gate fails.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol.
        gate1_result : GateResult | None
            Pre-computed Gate 1 (QTP) result.
        gate2_result : GateResult | None
            Pre-computed Gate 2 (Technical) result.
        gate3_result : GateResult | None
            Pre-computed Gate 3 (Fundamental) result.
        magi_votes : dict | None
            Pre-computed MAGI votes (Gate 4).
        sentiment_data : dict | None
            Sentiment flags (Gate 5).
        current_price : float | None
            Current stock price for position sizing.
        force : bool
            If True, ignore the verdict cache.
        """

        # ── Check cache first (14-day lock) ──
        if not force:
            cached = self.verdict_cache.get(ticker)
            if cached is not None and not self.verdict_cache.should_re_evaluate(cached):
                logger.info(
                    "Returning cached verdict for %s (locked until %s)",
                    ticker,
                    cached.locked_until,
                )
                return cached

        results: dict[str, GateResult] = {}

        # ── Gate 1: QTP ──
        if gate1_result is not None:
            results["qtp"] = gate1_result
            if not gate1_result.passed:
                return self._early_exit(
                    ticker, "AVOID", results, f"Gate1 fail: {gate1_result.reason}", current_price
                )

        # ── Gate 2: Technical ──
        if gate2_result is not None:
            results["technical"] = gate2_result
            if not gate2_result.passed:
                return self._early_exit(
                    ticker, "AVOID", results, f"Gate2 fail: {gate2_result.reason}", current_price
                )

        # ── Gate 3: Fundamental ──
        if gate3_result is not None:
            results["fundamental"] = gate3_result
            if not gate3_result.passed:
                return self._early_exit(
                    ticker, "AVOID", results, f"Gate3 fail: {gate3_result.reason}", current_price
                )

        # ── Gate 4: MAGI ──
        if magi_votes is not None:
            g4 = Gate4_MAGI().evaluate(magi_votes)
            results["magi"] = g4
            if not g4.passed:
                return self._early_exit(
                    ticker, "HOLD", results, f"Gate4 fail: {g4.reason}", current_price
                )

        # ── Gate 5: Sentiment ──
        if sentiment_data is not None:
            g5 = Gate5_Sentiment().evaluate(sentiment_data)
            results["sentiment"] = g5

        # ── Gate 6: Integration ──
        integrated = self.integrator.calculate(results)

        # ── Gate 7: Verdict ──
        verdict = self.judge.judge(integrated, results, ticker=ticker, current_price=current_price)

        # ── Save to cache ──
        self.verdict_cache.put(ticker, verdict)

        return verdict

    # ------------------------------------------------------------------
    def _early_exit(
        self,
        ticker: str,
        verdict_label: str,
        results: dict[str, GateResult],
        reason: str,
        current_price: float | None,
    ) -> FinalVerdict:
        """Build a FinalVerdict for early gate failure."""
        integrated = self.integrator.calculate(results) if results else 0.0
        return FinalVerdict(
            verdict=verdict_label,
            score=integrated,
            allocation=0.0,
            entry_price=current_price,
            stop_loss=round(current_price * 0.85, 2) if current_price else None,
            target_price=None,
            locked_until=None,
            reason=reason,
            gate_results=results,
            ticker=ticker,
        )
