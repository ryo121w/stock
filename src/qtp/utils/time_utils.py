"""Timezone-aware date helpers."""

from __future__ import annotations

from datetime import date, timedelta


def trading_days_back(from_date: date, n_days: int, padding_factor: float = 1.8) -> date:
    """Estimate calendar days needed to cover n trading days (accounting for weekends/holidays)."""
    calendar_days = int(n_days * padding_factor)
    return from_date - timedelta(days=calendar_days)
