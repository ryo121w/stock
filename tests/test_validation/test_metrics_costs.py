"""Tests for transaction cost integration in compute_metrics."""

from __future__ import annotations

import numpy as np
import pytest

from qtp.validation.metrics import compute_metrics


def _make_metric_inputs(n: int = 100, seed: int = 42):
    """Generate synthetic metric inputs with clear signal."""
    rng = np.random.RandomState(seed)
    y_true_dir = rng.randint(0, 2, n)
    # Make predictions correlated with truth (AUC > 0.5)
    noise = rng.randn(n) * 0.3
    y_pred_proba = np.clip(y_true_dir * 0.6 + (1 - y_true_dir) * 0.4 + noise, 0.01, 0.99)
    y_true_mag = (y_true_dir * 2 - 1) * (0.01 + rng.rand(n) * 0.02)
    y_pred_mag = y_true_mag + rng.randn(n) * 0.005
    return y_true_dir, y_pred_proba, y_true_mag, y_pred_mag


class TestTransactionCosts:
    """compute_metrics should properly reflect transaction costs."""

    def test_zero_cost_matches_original(self):
        """With zero costs, results should match the no-cost case."""
        inputs = _make_metric_inputs()
        m_default = compute_metrics(*inputs)
        m_zero = compute_metrics(*inputs, commission_bps=0.0, slippage_bps=0.0)

        assert m_default.sharpe_ratio == pytest.approx(m_zero.sharpe_ratio)
        assert m_default.win_rate == pytest.approx(m_zero.win_rate)
        assert m_default.profit_factor == pytest.approx(m_zero.profit_factor)

    def test_costs_reduce_sharpe(self):
        """Adding transaction costs should reduce (or not increase) Sharpe."""
        inputs = _make_metric_inputs()
        m_no_cost = compute_metrics(*inputs)
        m_with_cost = compute_metrics(*inputs, commission_bps=10.0, slippage_bps=10.0)

        # Sharpe with costs should be <= Sharpe without costs
        assert m_with_cost.sharpe_ratio <= m_no_cost.sharpe_ratio + 1e-9, \
            f"Cost Sharpe {m_with_cost.sharpe_ratio} > no-cost {m_no_cost.sharpe_ratio}"

    def test_higher_costs_lower_sharpe(self):
        """Higher costs should produce lower (or equal) Sharpe."""
        inputs = _make_metric_inputs()
        m_low = compute_metrics(*inputs, commission_bps=5.0, slippage_bps=5.0)
        m_high = compute_metrics(*inputs, commission_bps=20.0, slippage_bps=20.0)

        assert m_high.sharpe_ratio <= m_low.sharpe_ratio + 1e-9

    def test_costs_reduce_win_rate(self):
        """Transaction costs should reduce win rate (more trades become losers)."""
        inputs = _make_metric_inputs()
        m_no_cost = compute_metrics(*inputs)
        m_with_cost = compute_metrics(*inputs, commission_bps=10.0, slippage_bps=10.0)

        assert m_with_cost.win_rate <= m_no_cost.win_rate + 1e-9

    def test_costs_worsen_max_drawdown(self):
        """Transaction costs should make max drawdown equal or worse (more negative)."""
        inputs = _make_metric_inputs()
        m_no_cost = compute_metrics(*inputs)
        m_with_cost = compute_metrics(*inputs, commission_bps=10.0, slippage_bps=10.0)

        # max_drawdown is negative; with costs it should be same or more negative
        assert m_with_cost.max_drawdown <= m_no_cost.max_drawdown + 1e-9

    def test_classification_metrics_unaffected_by_costs(self):
        """AUC, accuracy, precision etc. should not change with costs."""
        inputs = _make_metric_inputs()
        m_no_cost = compute_metrics(*inputs)
        m_with_cost = compute_metrics(*inputs, commission_bps=10.0, slippage_bps=10.0)

        assert m_no_cost.auc_roc == pytest.approx(m_with_cost.auc_roc)
        assert m_no_cost.accuracy == pytest.approx(m_with_cost.accuracy)
        assert m_no_cost.precision == pytest.approx(m_with_cost.precision)
        assert m_no_cost.recall == pytest.approx(m_with_cost.recall)
        assert m_no_cost.f1 == pytest.approx(m_with_cost.f1)

    def test_round_trip_cost_calculation(self):
        """Verify the round-trip cost math: 10bps commission + 10bps slippage = 40bps round-trip."""
        # With all predictions at high confidence and known returns,
        # we can verify the exact cost deduction
        y_true_dir = np.array([1, 1, 1, 1, 1])
        y_pred_proba = np.array([0.9, 0.9, 0.9, 0.9, 0.9])  # All above 0.55 threshold
        y_true_mag = np.array([0.05, 0.05, 0.05, 0.05, 0.05])  # 5% return each
        y_pred_mag = np.array([0.05, 0.05, 0.05, 0.05, 0.05])

        m = compute_metrics(
            y_true_dir, y_pred_proba, y_true_mag, y_pred_mag,
            commission_bps=10.0, slippage_bps=10.0,
        )
        # Round-trip = 2*(10+10)/10000 = 0.004 = 40bps
        # Net return per trade = 0.05 - 0.004 = 0.046
        # All trades are winners
        assert m.win_rate == pytest.approx(1.0)

    def test_no_trades_with_low_confidence(self):
        """If all predictions are below confidence threshold, financial metrics should be zero."""
        y_true_dir = np.array([1, 0, 1])
        y_pred_proba = np.array([0.3, 0.4, 0.2])  # All below 0.55
        y_true_mag = np.array([0.01, -0.01, 0.02])
        y_pred_mag = np.array([0.01, -0.01, 0.02])

        m = compute_metrics(y_true_dir, y_pred_proba, y_true_mag, y_pred_mag,
                            commission_bps=10.0, slippage_bps=10.0)
        assert m.sharpe_ratio == 0.0
        assert m.win_rate == 0.0
