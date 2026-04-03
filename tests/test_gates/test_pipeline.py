"""Tests for the GatePipeline orchestrator.

Since Gates 1-3 may not be implemented yet, we test the pipeline with
only Gates 4-7 (MAGI + Sentiment + Integration + Verdict).
"""

import tempfile

from qtp.gates import FinalVerdict
from qtp.gates.pipeline import GatePipeline


class TestGatePipeline:
    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.pipeline = GatePipeline(cache_db=self.tmp.name)

    def test_full_run_buy(self):
        verdict = self.pipeline.evaluate(
            "AAPL",
            magi_votes={"melchior": "BUY", "balthasar": "BUY", "casper": "BUY"},
            sentiment_data={"board_pessimistic": True},
            current_price=150.0,
        )
        assert isinstance(verdict, FinalVerdict)
        assert verdict.ticker == "AAPL"
        # 3-0 BUY (score=95) + pessimistic sentiment (score=80)
        # With only magi+sentiment, weights renormalise
        assert verdict.score > 50

    def test_full_run_avoid_on_magi_fail(self):
        verdict = self.pipeline.evaluate(
            "BAD",
            magi_votes={"melchior": "AVOID", "balthasar": "AVOID", "casper": "AVOID"},
        )
        assert verdict.verdict == "HOLD"  # MAGI fail → early exit as HOLD
        assert verdict.allocation == 0.0

    def test_cache_returns_locked_verdict(self):
        """Second call should return cached result within lock period."""
        v1 = self.pipeline.evaluate(
            "MSFT",
            magi_votes={"melchior": "BUY", "balthasar": "BUY", "casper": "HOLD"},
            sentiment_data={},
        )
        v2 = self.pipeline.evaluate(
            "MSFT",
            magi_votes={"melchior": "AVOID", "balthasar": "AVOID", "casper": "AVOID"},
            sentiment_data={},
        )
        # v2 should be the cached v1 (not re-evaluated)
        assert v2.verdict == v1.verdict
        assert v2.score == v1.score

    def test_force_bypasses_cache(self):
        v1 = self.pipeline.evaluate(
            "GOOG",
            magi_votes={"melchior": "BUY", "balthasar": "BUY", "casper": "BUY"},
            sentiment_data={},
        )
        v2 = self.pipeline.evaluate(
            "GOOG",
            magi_votes={"melchior": "AVOID", "balthasar": "AVOID", "casper": "AVOID"},
            sentiment_data={},
            force=True,
        )
        # force=True should produce a different verdict
        assert v2.verdict != v1.verdict

    def test_no_magi_no_sentiment(self):
        """Pipeline should work even without optional gates."""
        verdict = self.pipeline.evaluate("TSLA")
        assert isinstance(verdict, FinalVerdict)

    def test_stop_loss_set(self):
        verdict = self.pipeline.evaluate(
            "NVDA",
            magi_votes={"melchior": "BUY", "balthasar": "BUY", "casper": "BUY"},
            current_price=200.0,
        )
        assert verdict.stop_loss == 170.0  # 200 * 0.85

    def test_entry_price_propagated(self):
        verdict = self.pipeline.evaluate(
            "AMZN",
            magi_votes={"melchior": "BUY", "balthasar": "BUY", "casper": "BUY"},
            current_price=180.0,
        )
        assert verdict.entry_price == 180.0
