"""Tests for gate_evaluations table in QTPDatabase."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from qtp.data.database import QTPDatabase


@pytest.fixture
def db(tmp_path):
    return QTPDatabase(tmp_path / "test.db")


class TestGateEvaluations:
    def test_save_and_retrieve(self, db):
        db.save_gate_evaluation(
            "AAPL",
            "2026-04-01",
            gate1_score=75.0,
            gate1_passed=True,
            gate2_score=60.0,
            gate2_passed=True,
            gate3_score=55.0,
            gate3_passed=True,
            integrated_score=65.0,
            final_verdict="BUY",
            allocation=0.05,
            locked_until="2026-04-15",
        )
        result = db.get_cached_verdict("AAPL")
        assert result is not None
        assert result["ticker"] == "AAPL"
        assert result["gate1_score"] == 75.0
        assert result["gate1_passed"] == 1
        assert result["final_verdict"] == "BUY"
        assert result["allocation"] == 0.05

    def test_upsert_updates(self, db):
        db.save_gate_evaluation("AAPL", "2026-04-01", gate1_score=50.0, gate1_passed=False)
        db.save_gate_evaluation("AAPL", "2026-04-01", gate1_score=75.0, gate1_passed=True)
        result = db.get_cached_verdict("AAPL")
        assert result["gate1_score"] == 75.0

    def test_no_evaluation_returns_none(self, db):
        result = db.get_cached_verdict("AAPL")
        assert result is None

    def test_locked_flag(self, db):
        future = (date.today() + timedelta(days=14)).isoformat()
        db.save_gate_evaluation("AAPL", "2026-04-01", final_verdict="BUY", locked_until=future)
        result = db.get_cached_verdict("AAPL")
        assert result["is_locked"] is True

    def test_expired_lock(self, db):
        past = (date.today() - timedelta(days=1)).isoformat()
        db.save_gate_evaluation("AAPL", "2026-04-01", final_verdict="WATCH", locked_until=past)
        result = db.get_cached_verdict("AAPL")
        assert result["is_locked"] is False

    def test_most_recent_evaluation_returned(self, db):
        db.save_gate_evaluation("AAPL", "2026-03-01", final_verdict="WATCH")
        db.save_gate_evaluation("AAPL", "2026-04-01", final_verdict="BUY")
        result = db.get_cached_verdict("AAPL")
        assert result["final_verdict"] == "BUY"
        assert result["evaluation_date"] == "2026-04-01"

    def test_table_exists(self, db):
        with db._conn() as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='gate_evaluations'"
            ).fetchone()
        assert row is not None
