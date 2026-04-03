"""Tests for Gate 3: Fundamental check gate."""

from __future__ import annotations

import pytest

from qtp.gates import GateResult
from qtp.gates.gate3_fundamental import Gate3_Fundamental, _extract_eps_signal


@pytest.fixture
def gate():
    return Gate3_Fundamental()


def _make_quote(
    price: float = 150.0,
    revenue_growth: float = 0.10,
    earnings_growth: float = 0.15,
    roe: float = 0.20,
) -> dict:
    return {
        "regularMarketPrice": price,
        "revenueGrowth": revenue_growth,
        "earningsGrowth": earnings_growth,
        "returnOnEquity": roe,
    }


class TestGate3_Fundamental:
    def test_no_data_fails(self, gate):
        result = gate.evaluate("AAPL")
        assert isinstance(result, GateResult)
        assert result.passed is False

    def test_healthy_stock_passes(self, gate):
        quote = _make_quote()
        estimates = {"targetMeanPrice": 180.0}
        earnings = {"signal": "UPGRADE"}
        result = gate.evaluate(
            "AAPL", yahoo_quote=quote, earnings_trend=earnings, analyst_estimates=estimates
        )
        assert result.passed is True
        assert result.score > 50

    def test_eps_downgrade_instant_fail(self, gate):
        quote = _make_quote()
        earnings = {"signal": "DOWNGRADE"}
        result = gate.evaluate("AAPL", yahoo_quote=quote, earnings_trend=earnings)
        assert result.passed is False
        assert "DOWNGRADE" in result.reason

    def test_target_below_price_instant_fail(self, gate):
        quote = _make_quote(price=150.0)
        estimates = {"targetMeanPrice": 130.0}
        result = gate.evaluate("AAPL", yahoo_quote=quote, analyst_estimates=estimates)
        assert result.passed is False
        assert "Target price" in result.reason

    def test_declining_revenue_warning(self, gate):
        quote = _make_quote(revenue_growth=-0.05)
        estimates = {"targetMeanPrice": 180.0}
        result = gate.evaluate("AAPL", yahoo_quote=quote, analyst_estimates=estimates)
        assert any("Revenue declining" in w for w in result.warnings)

    def test_low_roe_warning(self, gate):
        quote = _make_quote(roe=0.05)
        estimates = {"targetMeanPrice": 180.0}
        result = gate.evaluate("AAPL", yahoo_quote=quote, analyst_estimates=estimates)
        assert any("Low ROE" in w for w in result.warnings)

    def test_score_clamped_0_100(self, gate):
        # Extreme negative values
        quote = _make_quote(revenue_growth=-0.50, earnings_growth=-0.50, roe=0.01)
        estimates = {"targetMeanPrice": 180.0}
        result = gate.evaluate("AAPL", yahoo_quote=quote, analyst_estimates=estimates)
        assert 0 <= result.score <= 100

    def test_data_dict_populated(self, gate):
        quote = _make_quote()
        result = gate.evaluate("AAPL", yahoo_quote=quote)
        assert "revenue_growth" in result.data
        assert "eps_signal" in result.data

    def test_no_analyst_estimates_still_works(self, gate):
        quote = _make_quote()
        result = gate.evaluate("AAPL", yahoo_quote=quote)
        # Should still produce a score without analyst data
        assert result.score > 0


class TestExtractEpsSignal:
    def test_direct_signal(self):
        assert _extract_eps_signal({"signal": "UPGRADE"}) == "UPGRADE"
        assert _extract_eps_signal({"signal": "DOWNGRADE"}) == "DOWNGRADE"
        assert _extract_eps_signal({"signal": "NEUTRAL"}) == "NEUTRAL"

    def test_revision_counts(self):
        assert _extract_eps_signal({"upRevisions": 5, "downRevisions": 1}) == "UPGRADE"
        assert _extract_eps_signal({"upRevisions": 1, "downRevisions": 5}) == "DOWNGRADE"

    def test_nested_trend(self):
        data = {
            "trend": [
                {
                    "earningsEstimateNumberOfUpRevisions": 3,
                    "earningsEstimateNumberOfDownRevisions": 0,
                },
                {
                    "earningsEstimateNumberOfUpRevisions": 2,
                    "earningsEstimateNumberOfDownRevisions": 0,
                },
            ]
        }
        assert _extract_eps_signal(data) == "UPGRADE"

    def test_empty_returns_neutral(self):
        assert _extract_eps_signal({}) == "NEUTRAL"
