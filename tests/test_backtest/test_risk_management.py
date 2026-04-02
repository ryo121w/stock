"""Tests for TradeManager risk management rules."""

from __future__ import annotations

import pytest

from qtp.backtest.risk_management import ExitSignal, TradeManager


@pytest.fixture
def manager():
    return TradeManager(
        stop_loss_pct=-0.02,
        take_profit_pct=0.05,
        trailing_stop_pct=0.03,
        max_hold_days=10,
    )


class TestStopLoss:
    def test_triggers_at_threshold(self, manager):
        """Stop-loss triggers when PnL hits -2%."""
        result = manager.check_exit(
            entry_price=100.0, current_price=98.0, peak_price=100.0, days_held=1
        )
        assert result is not None
        assert result.reason == "stop_loss"
        assert result.pnl == pytest.approx(-0.02)

    def test_triggers_below_threshold(self, manager):
        """Stop-loss triggers when PnL exceeds -2% (e.g. gap down)."""
        result = manager.check_exit(
            entry_price=100.0, current_price=95.0, peak_price=100.0, days_held=1
        )
        assert result is not None
        assert result.reason == "stop_loss"
        assert result.pnl == pytest.approx(-0.05)

    def test_no_trigger_above_threshold(self, manager):
        """No stop-loss when loss is within threshold."""
        result = manager.check_exit(
            entry_price=100.0, current_price=99.0, peak_price=100.0, days_held=1
        )
        # -1% loss, within -2% threshold -> no exit
        assert result is None


class TestTakeProfit:
    def test_triggers_at_threshold(self, manager):
        """Take-profit triggers when PnL hits +5%."""
        result = manager.check_exit(
            entry_price=100.0, current_price=105.0, peak_price=105.0, days_held=3
        )
        assert result is not None
        assert result.reason == "take_profit"
        assert result.pnl == pytest.approx(0.05)

    def test_triggers_above_threshold(self, manager):
        """Take-profit triggers when PnL exceeds +5%."""
        result = manager.check_exit(
            entry_price=100.0, current_price=108.0, peak_price=108.0, days_held=3
        )
        assert result is not None
        assert result.reason == "take_profit"
        assert result.pnl == pytest.approx(0.08)

    def test_no_trigger_below_threshold(self, manager):
        """No take-profit when gain is below threshold."""
        result = manager.check_exit(
            entry_price=100.0, current_price=103.0, peak_price=103.0, days_held=3
        )
        assert result is None


class TestTrailingStop:
    def test_triggers_after_runup_and_pullback(self, manager):
        """Trailing stop triggers: ran up 4% then pulled back >3% from peak."""
        # Entry=100, peak=104 (4% gain), current=100.8 (drawdown from peak ~3.08%)
        result = manager.check_exit(
            entry_price=100.0, current_price=100.8, peak_price=104.0, days_held=5
        )
        assert result is not None
        assert result.reason == "trailing_stop"

    def test_no_trigger_without_sufficient_runup(self, manager):
        """Trailing stop does NOT trigger if peak gain is < 2%."""
        # Entry=100, peak=101.5 (only 1.5% gain), current=99 (big drop from peak)
        result = manager.check_exit(
            entry_price=100.0, current_price=99.0, peak_price=101.5, days_held=5
        )
        # peak_pnl = 1.5% < 2%, so trailing stop not active -> no exit
        # Also not stop-loss (-1% > -2%)
        assert result is None

    def test_no_trigger_small_pullback(self, manager):
        """Trailing stop does NOT trigger if pullback from peak is < 3%."""
        # Entry=100, peak=104 (4% gain), current=101.5 (drawdown ~2.4%)
        result = manager.check_exit(
            entry_price=100.0, current_price=101.5, peak_price=104.0, days_held=5
        )
        assert result is None


class TestMaxHold:
    def test_triggers_at_max_days(self, manager):
        """Max hold triggers at exactly max_hold_days."""
        result = manager.check_exit(
            entry_price=100.0, current_price=101.0, peak_price=101.0, days_held=10
        )
        assert result is not None
        assert result.reason == "max_hold"

    def test_triggers_beyond_max_days(self, manager):
        """Max hold triggers when exceeding max_hold_days."""
        result = manager.check_exit(
            entry_price=100.0, current_price=101.0, peak_price=101.0, days_held=15
        )
        assert result is not None
        assert result.reason == "max_hold"

    def test_no_trigger_before_max_days(self, manager):
        """No max hold trigger before max_hold_days."""
        result = manager.check_exit(
            entry_price=100.0, current_price=101.0, peak_price=101.0, days_held=9
        )
        assert result is None


class TestNoExit:
    def test_within_all_bounds(self, manager):
        """No exit when price is within all thresholds."""
        result = manager.check_exit(
            entry_price=100.0, current_price=101.0, peak_price=101.0, days_held=3
        )
        assert result is None

    def test_flat_price(self, manager):
        """No exit when price is unchanged."""
        result = manager.check_exit(
            entry_price=100.0, current_price=100.0, peak_price=100.0, days_held=1
        )
        assert result is None


class TestPriorityOrder:
    def test_stop_loss_takes_priority_over_max_hold(self, manager):
        """When both stop-loss and max-hold would trigger, stop-loss wins (checked first)."""
        result = manager.check_exit(
            entry_price=100.0, current_price=97.0, peak_price=100.0, days_held=15
        )
        assert result is not None
        assert result.reason == "stop_loss"

    def test_take_profit_takes_priority_over_max_hold(self, manager):
        """Take-profit checked before max-hold."""
        result = manager.check_exit(
            entry_price=100.0, current_price=106.0, peak_price=106.0, days_held=15
        )
        assert result is not None
        assert result.reason == "take_profit"


class TestExitSignalDataclass:
    def test_fields(self):
        sig = ExitSignal(reason="stop_loss", pnl=-0.02)
        assert sig.reason == "stop_loss"
        assert sig.pnl == -0.02

    def test_equality(self):
        a = ExitSignal(reason="take_profit", pnl=0.05)
        b = ExitSignal(reason="take_profit", pnl=0.05)
        assert a == b
