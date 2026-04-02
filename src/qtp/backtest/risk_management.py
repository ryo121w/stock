"""Trade-level risk management: stop-loss, take-profit, trailing stop."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExitSignal:
    """Represents an exit decision with reason and current PnL."""

    reason: str  # "stop_loss", "take_profit", "trailing_stop", "max_hold", "signal_exit"
    pnl: float  # Current PnL at exit


class TradeManager:
    """Intra-holding-period risk management.

    Checks daily OHLCV against stop-loss, take-profit, trailing stop,
    and max holding period rules.

    Parameters
    ----------
    stop_loss_pct : float
        Maximum allowed loss (negative, e.g. -0.02 = -2%).
    take_profit_pct : float
        Target profit to lock in (e.g. 0.05 = +5%).
    trailing_stop_pct : float
        Drawdown from peak that triggers exit (e.g. 0.03 = 3% from peak).
    max_hold_days : int
        Maximum days to hold a position.
    """

    def __init__(
        self,
        stop_loss_pct: float = -0.02,
        take_profit_pct: float = 0.05,
        trailing_stop_pct: float = 0.03,
        max_hold_days: int = 10,
    ):
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.max_hold_days = max_hold_days

    def check_exit(
        self,
        entry_price: float,
        current_price: float,
        peak_price: float,
        days_held: int,
    ) -> ExitSignal | None:
        """Check whether an open position should be exited.

        Parameters
        ----------
        entry_price : float
            Price at which the position was entered.
        current_price : float
            Current (or closing) price.
        peak_price : float
            Highest price observed since entry.
        days_held : int
            Number of trading days the position has been held.

        Returns
        -------
        ExitSignal or None
            An ExitSignal if exit is triggered, None to keep holding.
        """
        pnl = (current_price - entry_price) / entry_price

        # Stop-loss: cut losses early
        if pnl <= self.stop_loss_pct:
            return ExitSignal("stop_loss", pnl)

        # Take-profit: lock in gains
        if pnl >= self.take_profit_pct:
            return ExitSignal("take_profit", pnl)

        # Trailing stop: protect gains after meaningful run-up
        peak_pnl = (peak_price - entry_price) / entry_price
        if peak_pnl > 0.02:  # Only activate after 2% gain
            drawdown_from_peak = (current_price - peak_price) / peak_price
            if drawdown_from_peak < -self.trailing_stop_pct:
                return ExitSignal("trailing_stop", pnl)

        # Max holding period
        if days_held >= self.max_hold_days:
            return ExitSignal("max_hold", pnl)

        return None
