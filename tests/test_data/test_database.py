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
