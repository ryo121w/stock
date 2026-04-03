"""Tests for Gate 1: QTP quantitative model gate."""

from __future__ import annotations

import pytest

from qtp.data.database import QTPDatabase
from qtp.gates import GateResult
from qtp.gates.gate1_qtp import Gate1_QTP


@pytest.fixture
def db(tmp_path):
    return QTPDatabase(tmp_path / "test.db")


@pytest.fixture
def gate(db):
    return Gate1_QTP(db)


class TestGate1_QTP:
    def test_no_prediction_fails(self, gate):
        result = gate.evaluate("AAPL")
        assert isinstance(result, GateResult)
        assert result.passed is False
        assert result.score == 0.0
        assert "No prediction" in result.reason

    def test_high_confidence_up_passes(self, db, gate):
        # Insert a high-confidence UP prediction
        db.save_prediction("AAPL", "2026-04-01", direction=1, confidence=0.75, model_version="v1")
        # Insert some graded predictions (good accuracy)
        for i in range(10):
            db.save_prediction(
                "AAPL", f"2026-03-{10 + i:02d}", direction=1, confidence=0.7, model_version="v1"
            )
        # Grade them -- 7/10 correct
        with db._conn() as conn:
            rows = conn.execute(
                "SELECT id FROM predictions WHERE ticker='AAPL' AND prediction_date < '2026-04-01'"
            ).fetchall()
            for idx, row in enumerate(rows):
                conn.execute(
                    "UPDATE predictions SET is_correct=?, graded_at='2026-04-01' WHERE id=?",
                    (1 if idx < 7 else 0, row["id"]),
                )

        result = gate.evaluate("AAPL")
        assert result.passed is True
        assert result.score == 75.0  # conf * 100
        assert result.data["confidence"] == 0.75
        assert result.data["direction"] == 1
        assert result.data["historical_accuracy"] == 0.7

    def test_low_confidence_fails(self, db, gate):
        db.save_prediction("AAPL", "2026-04-01", direction=1, confidence=0.50, model_version="v1")
        result = gate.evaluate("AAPL")
        assert result.passed is False
        assert result.score == 50.0

    def test_direction_down_fails(self, db, gate):
        db.save_prediction("AAPL", "2026-04-01", direction=0, confidence=0.80, model_version="v1")
        result = gate.evaluate("AAPL")
        assert result.passed is False
        assert "DOWN" in result.warnings[0]

    def test_low_historical_accuracy_fails(self, db, gate):
        # High confidence but bad historical accuracy
        db.save_prediction("AAPL", "2026-04-01", direction=1, confidence=0.85, model_version="v1")
        # 10 graded predictions, only 5 correct (50%)
        for i in range(10):
            db.save_prediction(
                "AAPL", f"2026-03-{10 + i:02d}", direction=1, confidence=0.7, model_version="v1"
            )
        with db._conn() as conn:
            rows = conn.execute(
                "SELECT id FROM predictions WHERE ticker='AAPL' AND prediction_date < '2026-04-01'"
            ).fetchall()
            for idx, row in enumerate(rows):
                conn.execute(
                    "UPDATE predictions SET is_correct=?, graded_at='2026-04-01' WHERE id=?",
                    (1 if idx < 5 else 0, row["id"]),
                )

        result = gate.evaluate("AAPL")
        assert result.passed is False
        assert result.data["historical_accuracy"] == 0.5

    def test_no_graded_data_uses_neutral(self, db, gate):
        # Only ungraded prediction -- historical accuracy defaults to 0.5
        db.save_prediction("AAPL", "2026-04-01", direction=1, confidence=0.60, model_version="v1")
        result = gate.evaluate("AAPL")
        # 0.5 < 0.53 threshold -- should fail
        assert result.passed is False
        assert result.data["historical_accuracy"] == 0.5
