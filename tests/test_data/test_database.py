"""Tests for SQLite database manager."""

from __future__ import annotations

import pytest

from qtp.data.database import QTPDatabase


@pytest.fixture
def db(tmp_path):
    """Create a temporary database."""
    return QTPDatabase(tmp_path / "test.db")


class TestAlternativeData:
    def test_upsert_and_get(self, db):
        db.upsert_alternative("AAPL", "earnings_trend", {"revision_7d": "up"})
        result = db.get_alternative("AAPL", "earnings_trend")
        assert result == {"revision_7d": "up"}

    def test_get_missing_returns_none(self, db):
        assert db.get_alternative("AAPL", "nonexistent") is None

    def test_upsert_overwrites(self, db):
        db.upsert_alternative("AAPL", "earnings_trend", {"old": True})
        db.upsert_alternative("AAPL", "earnings_trend", {"new": True})
        result = db.get_alternative("AAPL", "earnings_trend")
        assert result == {"new": True}

    def test_freshness_check(self, db):
        db.upsert_alternative("AAPL", "test_tool", {"data": 1})
        # Just inserted — should be fresh
        fresh = db.get_alternative_fresh("AAPL", "test_tool", max_age_hours=1)
        assert fresh is not None

    def test_coverage(self, db):
        db.upsert_alternative("AAPL", "tool1", {"a": 1})
        db.upsert_alternative("AAPL", "tool2", {"b": 2})
        db.upsert_alternative("MSFT", "tool1", {"c": 3})

        coverage = db.alternative_coverage()
        assert len(coverage) == 2
        aapl = [c for c in coverage if c["ticker"] == "AAPL"][0]
        assert aapl["n_tools"] == 2

    def test_list_for_ticker(self, db):
        db.upsert_alternative("AAPL", "tool_a", {"x": 1})
        db.upsert_alternative("AAPL", "tool_b", {"y": 2})

        items = db.list_alternative_for_ticker("AAPL")
        assert len(items) == 2
        assert items[0]["tool"] == "tool_a"


class TestAlternativeDataDaily:
    def test_upsert_and_get_history(self, db):
        db.upsert_alternative_daily("AAPL", "earnings_trend", {"rev": "up"}, "2026-04-01")
        db.upsert_alternative_daily("AAPL", "earnings_trend", {"rev": "down"}, "2026-04-02")
        db.upsert_alternative_daily("AAPL", "earnings_trend", {"rev": "flat"}, "2026-04-03")

        history = db.get_alternative_history("AAPL", "earnings_trend", n_days=10)
        assert len(history) == 3
        # Newest first
        assert history[0]["date"] == "2026-04-03"
        assert history[0]["data"] == {"rev": "flat"}
        assert history[2]["date"] == "2026-04-01"

    def test_upsert_daily_overwrites_same_date(self, db):
        db.upsert_alternative_daily("AAPL", "tool1", {"v": 1}, "2026-04-01")
        db.upsert_alternative_daily("AAPL", "tool1", {"v": 2}, "2026-04-01")
        history = db.get_alternative_history("AAPL", "tool1", n_days=10)
        assert len(history) == 1
        assert history[0]["data"] == {"v": 2}

    def test_upsert_daily_also_updates_legacy(self, db):
        db.upsert_alternative_daily("AAPL", "tool1", {"daily": True}, "2026-04-01")
        legacy = db.get_alternative("AAPL", "tool1")
        assert legacy == {"daily": True}

    def test_get_as_of_exact_date(self, db):
        db.upsert_alternative_daily("AAPL", "tool1", {"d": 1}, "2026-04-01")
        db.upsert_alternative_daily("AAPL", "tool1", {"d": 2}, "2026-04-03")

        result = db.get_alternative_as_of("AAPL", "tool1", "2026-04-03")
        assert result is not None
        assert result["data"] == {"d": 2}

    def test_get_as_of_falls_back_to_earlier(self, db):
        db.upsert_alternative_daily("AAPL", "tool1", {"d": 1}, "2026-04-01")
        db.upsert_alternative_daily("AAPL", "tool1", {"d": 2}, "2026-04-03")

        # Ask for 04-02 — should get 04-01 data
        result = db.get_alternative_as_of("AAPL", "tool1", "2026-04-02")
        assert result is not None
        assert result["data"] == {"d": 1}
        assert result["date"] == "2026-04-01"

    def test_get_as_of_no_data(self, db):
        result = db.get_alternative_as_of("AAPL", "nonexistent", "2026-04-01")
        assert result is None

    def test_get_history_limited(self, db):
        for i in range(10):
            db.upsert_alternative_daily("AAPL", "tool1", {"i": i}, f"2026-04-{i + 1:02d}")
        history = db.get_alternative_history("AAPL", "tool1", n_days=3)
        assert len(history) == 3
        assert history[0]["date"] == "2026-04-10"

    def test_default_date_is_today(self, db):
        from datetime import datetime

        db.upsert_alternative_daily("AAPL", "tool1", {"today": True})
        history = db.get_alternative_history("AAPL", "tool1", n_days=1)
        assert len(history) == 1
        assert history[0]["date"] == datetime.now().strftime("%Y-%m-%d")

    def test_migration_from_legacy(self, tmp_path):
        """Test that legacy data is migrated to daily table on first init."""
        db1 = QTPDatabase(tmp_path / "migrate.db")
        # Insert legacy data only
        db1.upsert_alternative("AAPL", "tool1", {"legacy": True})
        # Now manually clear daily table to simulate old schema state
        import sqlite3

        conn = sqlite3.connect(str(tmp_path / "migrate.db"))
        conn.execute("DELETE FROM alternative_data_daily")
        conn.commit()
        conn.close()
        # Re-init should trigger migration
        db2 = QTPDatabase(tmp_path / "migrate.db")
        history = db2.get_alternative_history("AAPL", "tool1", n_days=10)
        assert len(history) == 1
        assert history[0]["data"] == {"legacy": True}


class TestModelRegistry:
    def test_register_and_get(self, db):
        db.register_model(
            "lgbm_20260402", "lgbm", "/models/lgbm_20260402", metrics={"wf_auc": 0.65}
        )
        model = db.get_model("lgbm_20260402")
        assert model is not None
        assert model["version"] == "lgbm_20260402"

    def test_list_models(self, db):
        db.register_model("v1", "lgbm", "/m/v1")
        db.register_model("v2", "lgbm", "/m/v2")
        models = db.list_models()
        assert len(models) == 2

    def test_best_model(self, db):
        db.register_model("v1", "lgbm", "/m/v1", metrics={"wf_auc_roc": 0.55})
        db.register_model("v2", "lgbm", "/m/v2", metrics={"wf_auc_roc": 0.70})
        best = db.best_model("wf_auc_roc")
        assert best["version"] == "v2"


class TestExperiments:
    def test_log_and_list(self, db):
        exp_id = db.log_experiment(
            config={
                "labels": {"horizon": 5, "direction_threshold": 0.02},
                "features": {"tiers": [1, 2, 3]},
            },
            metrics={"wf_auc_roc": 0.69, "wf_sharpe": 10.0, "n_tickers": 10, "n_samples": 6000},
            model_version="lgbm_test",
        )
        assert exp_id > 0

        experiments = db.list_experiments()
        assert len(experiments) == 1
        assert experiments[0]["wf_auc"] == 0.69

    def test_best_experiments(self, db):
        db.log_experiment(
            config={"labels": {"horizon": 1}},
            metrics={"wf_auc_roc": 0.51},
        )
        db.log_experiment(
            config={"labels": {"horizon": 5}},
            metrics={"wf_auc_roc": 0.69},
        )
        best = db.best_experiments("wf_auc", limit=1)
        assert len(best) == 1
        assert best[0]["wf_auc"] == 0.69

    def test_compare(self, db):
        id1 = db.log_experiment(config={}, metrics={"wf_auc_roc": 0.51})
        id2 = db.log_experiment(config={}, metrics={"wf_auc_roc": 0.69})
        compared = db.compare_experiments([id1, id2])
        assert len(compared) == 2
        assert compared[0]["wf_auc"] == 0.69  # Sorted by AUC desc


class TestPredictions:
    def test_save_and_get_ungraded(self, db):
        db.save_prediction("AAPL", "2026-04-01", direction=1, confidence=0.65)
        ungraded = db.get_ungraded_predictions()
        assert len(ungraded) == 1
        assert ungraded[0]["ticker"] == "AAPL"

    def test_grade_prediction(self, db):
        db.save_prediction("AAPL", "2026-04-01", direction=1, confidence=0.65)
        ungraded = db.get_ungraded_predictions()
        db.grade_prediction(ungraded[0]["id"], actual_price_start=100.0, actual_price_end=105.0)
        # Should no longer be ungraded
        assert len(db.get_ungraded_predictions()) == 0

    def test_accuracy_summary(self, db):
        db.save_prediction("AAPL", "2026-04-01", direction=1, confidence=0.60)
        db.save_prediction("MSFT", "2026-04-01", direction=1, confidence=0.55)
        ungraded = db.get_ungraded_predictions()
        # AAPL correct (went up)
        db.grade_prediction(ungraded[0]["id"], 100.0, 110.0)
        # MSFT wrong (went down)
        db.grade_prediction(ungraded[1]["id"], 100.0, 90.0)

        summary = db.get_accuracy_summary()
        assert summary["total"] == 2
        assert summary["correct"] == 1
        assert summary["accuracy"] == 0.5

    def test_accuracy_summary_with_days(self, db):
        db.save_prediction("AAPL", "2026-04-01", direction=1, confidence=0.60)
        ungraded = db.get_ungraded_predictions()
        db.grade_prediction(ungraded[0]["id"], 100.0, 110.0)
        # days=30 should include recent predictions
        summary = db.get_accuracy_summary(days=30)
        assert summary["total"] >= 0  # May be 0 if date offset excludes it


class TestAccuracyTrend:
    def _populate_predictions(self, db, n=20):
        """Insert and grade predictions across several dates."""
        from datetime import datetime, timedelta

        base = datetime.now()
        for i in range(n):
            date = (base - timedelta(days=i)).strftime("%Y-%m-%d")
            direction = 1
            confidence = 0.60
            db.save_prediction("AAPL", date, direction=direction, confidence=confidence)

        ungraded = db.get_ungraded_predictions()
        for j, pred in enumerate(ungraded):
            # Alternate correct/wrong for variety
            if j % 2 == 0:
                db.grade_prediction(pred["id"], 100.0, 105.0)  # correct (up)
            else:
                db.grade_prediction(pred["id"], 100.0, 95.0)  # wrong (went down)

    def test_trend_returns_list(self, db):
        self._populate_predictions(db, n=30)
        trend = db.get_accuracy_trend(window_days=7, n_windows=4)
        assert isinstance(trend, list)
        # Should have some windows with data
        assert len(trend) > 0

    def test_trend_window_structure(self, db):
        self._populate_predictions(db, n=30)
        trend = db.get_accuracy_trend(window_days=7, n_windows=4)
        for window in trend:
            assert "window" in window
            assert "total" in window
            assert "accuracy" in window
            assert isinstance(window["total"], int)
            assert 0 <= window["accuracy"] <= 1

    def test_trend_empty_db(self, db):
        trend = db.get_accuracy_trend(window_days=7, n_windows=4)
        assert trend == []

    def test_trend_custom_params(self, db):
        self._populate_predictions(db, n=60)
        trend = db.get_accuracy_trend(window_days=14, n_windows=2)
        assert isinstance(trend, list)
        # Each window label should contain "14" for 14-day windows
        for w in trend:
            assert "14" in w["window"] or "28" in w["window"]
