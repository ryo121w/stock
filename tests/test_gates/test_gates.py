"""Tests for the 7-gate evaluation system."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from qtp.gates import GateResult

# ============================================================================
# GateResult dataclass
# ============================================================================


class TestGateResult:
    def test_creation(self):
        r = GateResult(gate="test", passed=True, score=75.0, reason="ok")
        assert r.passed
        assert r.score == 75.0
        assert r.gate == "test"
        assert r.reason == "ok"

    def test_defaults(self):
        r = GateResult(gate="g", passed=False, score=0.0)
        assert r.reason == ""
        assert r.warnings == []
        assert r.data == {}

    def test_with_warnings(self):
        r = GateResult(
            gate="g",
            passed=False,
            score=30.0,
            reason="bad",
            warnings=["low confidence", "direction DOWN"],
        )
        assert len(r.warnings) == 2

    def test_with_data(self):
        r = GateResult(
            gate="g",
            passed=True,
            score=80.0,
            reason="ok",
            data={"confidence": 0.72, "direction": 1},
        )
        assert r.data["confidence"] == 0.72


# ============================================================================
# Helpers
# ============================================================================


def _make_test_db(tmp_path: Path):
    """Create a test database with schema."""
    from qtp.data.database import QTPDatabase

    return QTPDatabase(tmp_path / "test.db")


def _insert_prediction(
    db,
    ticker: str,
    direction: int,
    confidence: float,
    prediction_date: str = "2025-04-01",
    is_correct: int | None = None,
    actual_return: float | None = None,
):
    """Insert a test prediction into the database."""
    with db._conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO predictions
               (ticker, prediction_date, direction, confidence, horizon, model_version,
                actual_return, is_correct, graded_at)
               VALUES (?, ?, ?, ?, 1, 'test_v1', ?, ?, ?)""",
            (
                ticker,
                prediction_date,
                direction,
                confidence,
                actual_return,
                is_correct,
                "2025-04-02" if is_correct is not None else None,
            ),
        )


def _make_ohlcv(n: int = 250, rsi_target: float | None = None) -> pl.DataFrame:
    """Generate synthetic OHLCV data.

    Args:
        n: Number of rows.
        rsi_target: If set, adjust the final prices to approximate this RSI.
    """
    np.random.seed(42)
    dates = pl.date_range(date(2024, 1, 2), date(2024, 12, 31), eager=True)[:n]

    if rsi_target is not None and rsi_target > 70:
        # Simulate strong uptrend for high RSI
        close = 100 + np.cumsum(np.abs(np.random.randn(n)) * 0.5)
    elif rsi_target is not None and rsi_target < 30:
        # Simulate strong downtrend for low RSI
        close = 100 - np.cumsum(np.abs(np.random.randn(n)) * 0.3)
        close = np.maximum(close, 10)
    else:
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        close = np.maximum(close, 10)

    return pl.DataFrame(
        {
            "date": dates,
            "open": close + np.random.randn(n) * 0.3,
            "high": close + abs(np.random.randn(n) * 0.5),
            "low": close - abs(np.random.randn(n) * 0.5),
            "close": close,
            "volume": np.random.randint(1_000_000, 10_000_000, n).astype(float),
        }
    )


# ============================================================================
# Gate 1: QTP Quantitative Model
# ============================================================================


class TestGate1QTP:
    def test_high_confidence_passes(self, tmp_path):
        """conf=0.70, direction=UP, hist_accuracy=0.60 -> pass."""
        from qtp.gates.gate1_qtp import Gate1_QTP

        db = _make_test_db(tmp_path)
        _insert_prediction(db, "AAPL", direction=1, confidence=0.70)
        for i in range(10):
            _insert_prediction(
                db,
                "AAPL",
                direction=1,
                confidence=0.65,
                prediction_date=f"2025-03-{10 + i:02d}",
                is_correct=1 if i < 6 else 0,
                actual_return=0.02 if i < 6 else -0.01,
            )

        gate = Gate1_QTP(db)
        r = gate.evaluate("AAPL")
        assert r.passed
        assert r.score == 70.0
        assert r.gate == "QTP"

    def test_low_confidence_fails(self, tmp_path):
        """conf=0.50 -> fail (below 55% threshold)."""
        from qtp.gates.gate1_qtp import Gate1_QTP

        db = _make_test_db(tmp_path)
        _insert_prediction(db, "TSLA", direction=1, confidence=0.50)

        gate = Gate1_QTP(db)
        r = gate.evaluate("TSLA")
        assert not r.passed
        assert any("Confidence" in w for w in r.warnings)

    def test_low_historical_accuracy_fails(self, tmp_path):
        """conf=0.80, hist_accuracy=0.50 -> fail (MSFT case)."""
        from qtp.gates.gate1_qtp import Gate1_QTP

        db = _make_test_db(tmp_path)
        _insert_prediction(db, "MSFT", direction=1, confidence=0.80)
        for i in range(10):
            _insert_prediction(
                db,
                "MSFT",
                direction=1,
                confidence=0.65,
                prediction_date=f"2025-03-{10 + i:02d}",
                is_correct=1 if i < 5 else 0,
                actual_return=0.02 if i < 5 else -0.02,
            )

        gate = Gate1_QTP(db)
        r = gate.evaluate("MSFT")
        assert not r.passed
        assert any("accuracy" in w.lower() for w in r.warnings)

    def test_no_prediction_fails(self, tmp_path):
        """No prediction data -> fail."""
        from qtp.gates.gate1_qtp import Gate1_QTP

        db = _make_test_db(tmp_path)
        gate = Gate1_QTP(db)
        r = gate.evaluate("UNKNOWN")
        assert not r.passed
        assert r.score == 0.0
        assert "No prediction" in r.reason

    def test_down_direction_fails(self, tmp_path):
        """direction=DOWN even with high confidence -> fail."""
        from qtp.gates.gate1_qtp import Gate1_QTP

        db = _make_test_db(tmp_path)
        _insert_prediction(db, "SPY", direction=0, confidence=0.75)

        gate = Gate1_QTP(db)
        r = gate.evaluate("SPY")
        assert not r.passed
        assert any("DOWN" in w for w in r.warnings)


# ============================================================================
# Gate 2: Technical Analysis (requires OHLCV DataFrame)
# ============================================================================


class TestGate2Technical:
    def test_rsi_above_75_fails(self):
        """Strong uptrend OHLCV (RSI high) should fail or reduce score."""
        from qtp.gates.gate2_technical import Gate2_Technical

        # Generate strong uptrend data -> high RSI
        ohlcv = _make_ohlcv(250, rsi_target=80)
        gate = Gate2_Technical()
        r = gate.evaluate("TEST", ohlcv)
        # Either fails entirely or has a low score
        assert not r.passed or r.score < 50

    def test_normal_rsi_passes(self):
        """Normal OHLCV data (moderate RSI) -> pass."""
        from qtp.gates.gate2_technical import Gate2_Technical

        ohlcv = _make_ohlcv(250)
        gate = Gate2_Technical()
        r = gate.evaluate("TEST", ohlcv)
        # With random walk data, RSI should be moderate -> pass
        assert r.gate == "Technical"
        # Score should be computed (not zero)
        assert r.score > 0

    def test_insufficient_data_fails(self):
        """Less than 26 rows -> fail."""
        from qtp.gates.gate2_technical import Gate2_Technical

        ohlcv = _make_ohlcv(20)
        gate = Gate2_Technical()
        r = gate.evaluate("TEST", ohlcv)
        assert not r.passed
        assert "Insufficient" in r.reason

    def test_rsi_computation(self):
        """Verify RSI is computed and stored in data."""
        from qtp.gates.gate2_technical import Gate2_Technical

        ohlcv = _make_ohlcv(100)
        gate = Gate2_Technical()
        r = gate.evaluate("TEST", ohlcv)
        assert "rsi" in r.data
        rsi = r.data["rsi"]
        if rsi is not None:
            assert 0 <= rsi <= 100


# ============================================================================
# Gate 3: Fundamental Analysis (uses MCP data dicts)
# ============================================================================


class TestGate3Fundamental:
    def test_eps_downgrade_fails(self):
        """EPS downgrade trend should fail."""
        from qtp.gates.gate3_fundamental import Gate3_Fundamental

        gate = Gate3_Fundamental()
        r = gate.evaluate(
            ticker="BAD",
            yahoo_quote={
                "regularMarketPrice": 100,
                "revenueGrowth": 0.05,
                "earningsGrowth": -0.1,
                "returnOnEquity": 0.12,
            },
            earnings_trend={"signal": "DOWNGRADE"},
        )
        assert not r.passed
        assert "DOWNGRADE" in r.reason

    def test_target_below_current_fails(self):
        """Analyst target below current price -> fail (Japanese shipping stock case)."""
        from qtp.gates.gate3_fundamental import Gate3_Fundamental

        gate = Gate3_Fundamental()
        r = gate.evaluate(
            ticker="9101.T",
            yahoo_quote={
                "regularMarketPrice": 5000,
                "revenueGrowth": 0.02,
                "earningsGrowth": 0.05,
                "returnOnEquity": 0.15,
            },
            analyst_estimates={"targetMeanPrice": 4000},
        )
        assert not r.passed
        assert "Target price" in r.reason or r.score < 20

    def test_good_fundamentals_pass(self):
        """Positive EPS revision, target above current -> pass."""
        from qtp.gates.gate3_fundamental import Gate3_Fundamental

        gate = Gate3_Fundamental()
        r = gate.evaluate(
            ticker="GOOD",
            yahoo_quote={
                "regularMarketPrice": 100,
                "revenueGrowth": 0.25,
                "earningsGrowth": 0.30,
                "returnOnEquity": 0.20,
            },
            earnings_trend={"signal": "UPGRADE"},
            analyst_estimates={"targetMeanPrice": 150},
        )
        assert r.passed
        assert r.score >= 60

    def test_no_data_fails(self):
        """No yahoo_quote -> fail."""
        from qtp.gates.gate3_fundamental import Gate3_Fundamental

        gate = Gate3_Fundamental()
        r = gate.evaluate(ticker="NODATA")
        assert not r.passed
        assert r.score == 0.0


# ============================================================================
# Gate 4: MAGI System (Evangelion-style 3-module consensus)
# ============================================================================


class TestGate4MAGI:
    def test_three_zero_buy(self):
        """All 3 modules BUY -> strong pass."""
        from qtp.gates.gate4_magi import Gate4_MAGI

        r = Gate4_MAGI().evaluate({"melchior": "BUY", "balthasar": "BUY", "casper": "BUY"})
        assert r.passed
        assert r.score >= 90

    def test_two_one_buy(self):
        """2 BUY + 1 HOLD -> pass with moderate score."""
        from qtp.gates.gate4_magi import Gate4_MAGI

        r = Gate4_MAGI().evaluate({"melchior": "BUY", "balthasar": "HOLD", "casper": "BUY"})
        assert r.passed
        assert 70 <= r.score < 90

    def test_two_one_avoid(self):
        """2 AVOID + 1 BUY -> fail."""
        from qtp.gates.gate4_magi import Gate4_MAGI

        r = Gate4_MAGI().evaluate({"melchior": "AVOID", "balthasar": "AVOID", "casper": "BUY"})
        assert not r.passed

    def test_three_zero_avoid(self):
        """All 3 modules AVOID -> strong fail."""
        from qtp.gates.gate4_magi import Gate4_MAGI

        r = Gate4_MAGI().evaluate({"melchior": "AVOID", "balthasar": "AVOID", "casper": "AVOID"})
        assert not r.passed
        assert r.score <= 10

    def test_mixed_hold(self):
        """1 BUY + 1 HOLD + 1 AVOID -> borderline, should not pass."""
        from qtp.gates.gate4_magi import Gate4_MAGI

        r = Gate4_MAGI().evaluate({"melchior": "BUY", "balthasar": "HOLD", "casper": "AVOID"})
        assert not r.passed or r.score < 70


# ============================================================================
# Gate 5: Sentiment (Contrarian filter)
# ============================================================================


class TestGate5Sentiment:
    def test_always_passes(self):
        """Gate 5 always passes but reduces score for euphoria."""
        from qtp.gates.gate5_sentiment import Gate5_Sentiment

        r = Gate5_Sentiment().evaluate({"analyst_all_buy": True, "board_euphoric": True})
        assert r.passed  # Always passes (soft gate)
        assert r.score < 70  # But score is reduced due to euphoria

    def test_pessimistic_board_bonus(self):
        """Contrarian: pessimistic board sentiment gives bonus score."""
        from qtp.gates.gate5_sentiment import Gate5_Sentiment

        r = Gate5_Sentiment().evaluate({"board_pessimistic": True})
        assert r.score > 70  # Contrarian bonus

    def test_neutral_sentiment(self):
        """Neutral sentiment -> middle score."""
        from qtp.gates.gate5_sentiment import Gate5_Sentiment

        r = Gate5_Sentiment().evaluate({})
        assert r.passed
        assert 40 <= r.score <= 80


# ============================================================================
# Gate 6: Integration (Weighted composite)
# ============================================================================


class TestGate6Integration:
    def test_weighted_calculation(self):
        """Integration gate computes weighted average of prior gate scores."""
        from qtp.gates.gate6_integration import Gate6_Integration

        prior_results = [
            GateResult(gate="QTP", passed=True, score=80.0, reason="ok"),
            GateResult(gate="Technical", passed=True, score=70.0, reason="ok"),
            GateResult(gate="Fundamental", passed=True, score=75.0, reason="ok"),
            GateResult(gate="MAGI", passed=True, score=90.0, reason="ok"),
            GateResult(gate="Sentiment", passed=True, score=60.0, reason="ok"),
        ]
        gate = Gate6_Integration()
        r = gate.evaluate(prior_results)
        assert r.passed
        assert 60 <= r.score <= 100

    def test_any_hard_fail_blocks(self):
        """If a hard gate (1-4) failed, integration should fail."""
        from qtp.gates.gate6_integration import Gate6_Integration

        prior_results = [
            GateResult(gate="QTP", passed=False, score=30.0, reason="low conf"),
            GateResult(gate="Technical", passed=True, score=70.0, reason="ok"),
            GateResult(gate="Fundamental", passed=True, score=75.0, reason="ok"),
            GateResult(gate="MAGI", passed=True, score=90.0, reason="ok"),
            GateResult(gate="Sentiment", passed=True, score=60.0, reason="ok"),
        ]
        gate = Gate6_Integration()
        r = gate.evaluate(prior_results)
        assert not r.passed

    def test_calculate_method(self):
        """Direct calculate() returns weighted score."""
        from qtp.gates.gate6_integration import Gate6_Integration

        gate = Gate6_Integration()
        results_dict = {
            "qtp": GateResult(gate="QTP", passed=True, score=100.0, reason="ok"),
            "technical": GateResult(gate="Technical", passed=True, score=100.0, reason="ok"),
            "fundamental": GateResult(gate="Fundamental", passed=True, score=100.0, reason="ok"),
            "magi": GateResult(gate="MAGI", passed=True, score=100.0, reason="ok"),
            "sentiment": GateResult(gate="Sentiment", passed=True, score=100.0, reason="ok"),
        }
        score = gate.calculate(results_dict)
        assert score == 100.0


# ============================================================================
# Gate 7: Final Verdict
# ============================================================================


class TestGate7Verdict:
    def test_strong_buy_threshold(self):
        """High integration score -> STRONG_BUY."""
        from qtp.gates.gate7_verdict import Gate7_Verdict

        gate = Gate7_Verdict()
        integration = GateResult(gate="Integration", passed=True, score=85.0, reason="ok")
        r = gate.evaluate(integration)
        assert r.passed
        assert r.score >= 80
        assert "BUY" in r.reason.upper()

    def test_avoid_threshold(self):
        """Low integration score -> AVOID."""
        from qtp.gates.gate7_verdict import Gate7_Verdict

        gate = Gate7_Verdict()
        integration = GateResult(gate="Integration", passed=False, score=25.0, reason="weak")
        r = gate.evaluate(integration)
        assert not r.passed
        assert "AVOID" in r.reason.upper() or r.score < 40

    def test_hold_zone(self):
        """Moderate integration score -> HOLD or WATCH."""
        from qtp.gates.gate7_verdict import Gate7_Verdict

        gate = Gate7_Verdict()
        integration = GateResult(gate="Integration", passed=True, score=55.0, reason="ok")
        r = gate.evaluate(integration)
        # HOLD/WATCH zone: passes, moderate score
        assert r.passed
        assert 35 <= r.score <= 65

    def test_judge_method(self):
        """Full judge() produces FinalVerdict with allocation."""
        from qtp.gates import FinalVerdict
        from qtp.gates.gate7_verdict import Gate7_Verdict

        gate = Gate7_Verdict()
        gate_results = {
            "qtp": GateResult(gate="QTP", passed=True, score=80.0, reason="ok"),
            "sentiment": GateResult(gate="Sentiment", passed=True, score=70.0, reason="ok"),
        }
        verdict = gate.judge(85.0, gate_results, ticker="AAPL", current_price=150.0)
        assert isinstance(verdict, FinalVerdict)
        assert verdict.verdict == "STRONG_BUY"
        assert verdict.allocation > 0
        assert verdict.stop_loss is not None
        assert verdict.target_price is not None
