"""Tests for Gate 7: Final verdict + VerdictCache."""

import tempfile
from datetime import date, timedelta

from qtp.gates import FinalVerdict, GateResult
from qtp.gates.gate7_verdict import Gate7_Verdict, VerdictCache


class TestGate7Verdict:
    def setup_method(self):
        self.judge = Gate7_Verdict()
        self.base_results = {
            "qtp": GateResult(gate="QTP", passed=True, score=80),
            "magi": GateResult(gate="MAGI", passed=True, score=75),
        }

    def test_strong_buy(self):
        v = self.judge.judge(85.0, self.base_results)
        assert v.verdict == "STRONG_BUY"

    def test_buy(self):
        v = self.judge.judge(70.0, self.base_results)
        assert v.verdict == "BUY"

    def test_watch(self):
        v = self.judge.judge(55.0, self.base_results)
        assert v.verdict == "WATCH"

    def test_hold(self):
        v = self.judge.judge(40.0, self.base_results)
        assert v.verdict == "HOLD"

    def test_avoid(self):
        v = self.judge.judge(20.0, self.base_results)
        assert v.verdict == "AVOID"

    def test_boundary_80(self):
        v = self.judge.judge(80.0, self.base_results)
        assert v.verdict == "STRONG_BUY"

    def test_boundary_65(self):
        v = self.judge.judge(65.0, self.base_results)
        assert v.verdict == "BUY"

    def test_boundary_50(self):
        v = self.judge.judge(50.0, self.base_results)
        assert v.verdict == "WATCH"

    def test_boundary_35(self):
        v = self.judge.judge(35.0, self.base_results)
        assert v.verdict == "HOLD"

    def test_price_targets(self):
        v = self.judge.judge(85.0, self.base_results, current_price=100.0)
        assert v.entry_price == 100.0
        assert v.stop_loss == 85.0  # -15%
        assert v.target_price == 120.0  # default +20%

    def test_locked_until_14_days(self):
        v = self.judge.judge(85.0, self.base_results)
        assert v.locked_until == date.today() + timedelta(days=14)

    def test_allocation_strong_buy(self):
        v = self.judge.judge(85.0, self.base_results)
        assert v.allocation == 0.10  # no sentiment gate → no adjustment

    def test_allocation_avoid_is_zero(self):
        v = self.judge.judge(20.0, self.base_results)
        assert v.allocation == 0.0

    def test_sentiment_adjusts_allocation(self):
        results = {
            **self.base_results,
            "sentiment": GateResult(gate="Sentiment", passed=True, score=35),
        }
        v = self.judge.judge(85.0, results)
        # STRONG_BUY base = 0.10, factor = 35/70 = 0.5
        assert v.allocation == 0.05

    def test_fundamental_target_price(self):
        results = {
            **self.base_results,
            "fundamental": GateResult(
                gate="Fundamental", passed=True, score=70, data={"target_price": 150.0}
            ),
        }
        v = self.judge.judge(85.0, results, current_price=100.0)
        assert v.target_price == 150.0


class TestVerdictCache:
    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.cache = VerdictCache(db_path=self.tmp.name)

    def test_put_and_get(self):
        fv = FinalVerdict(
            verdict="BUY",
            score=72.0,
            allocation=0.05,
            locked_until=date.today() + timedelta(days=14),
            ticker="AAPL",
        )
        self.cache.put("AAPL", fv)
        cached = self.cache.get("AAPL")
        assert cached is not None
        assert cached.verdict == "BUY"
        assert cached.score == 72.0

    def test_get_missing(self):
        assert self.cache.get("NOPE") is None

    def test_should_re_evaluate_not_expired(self):
        fv = FinalVerdict(
            verdict="BUY",
            score=72.0,
            allocation=0.05,
            locked_until=date.today() + timedelta(days=7),
        )
        assert self.cache.should_re_evaluate(fv) is False

    def test_should_re_evaluate_expired(self):
        fv = FinalVerdict(
            verdict="BUY",
            score=72.0,
            allocation=0.05,
            locked_until=date.today() - timedelta(days=1),
        )
        assert self.cache.should_re_evaluate(fv) is True

    def test_should_re_evaluate_no_lock(self):
        fv = FinalVerdict(verdict="BUY", score=72.0, allocation=0.05, locked_until=None)
        assert self.cache.should_re_evaluate(fv) is True

    def test_invalidate(self):
        fv = FinalVerdict(
            verdict="BUY",
            score=72.0,
            allocation=0.05,
            locked_until=date.today() + timedelta(days=14),
            ticker="AAPL",
        )
        self.cache.put("AAPL", fv)
        self.cache.invalidate("AAPL")
        assert self.cache.get("AAPL") is None
