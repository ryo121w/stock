"""Tier 5: CNN Fear & Greed Index features.

Uses ~1 year of daily historical Fear & Greed scores.
Each row gets the score for that date (point-in-time safe).

Data source: CNN Fear & Greed API via fear-greed Python library (free, no key)
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import structlog

from qtp.features.registry import FeatureRegistry, FeatureTier

reg = FeatureRegistry.instance()
logger = structlog.get_logger()

_fg_history: list[dict] | None = None


def _get_history() -> list[dict]:
    """Get Fear & Greed history. Cached per session."""
    global _fg_history
    if _fg_history is not None:
        return _fg_history

    try:
        from qtp.data.fetchers.fear_greed_ import fetch_fear_greed_history

        _fg_history = fetch_fear_greed_history()
        return _fg_history
    except Exception as e:
        logger.warning("fear_greed_history_failed", error=str(e))
        _fg_history = []
        return []


def _parse_date(d) -> date | None:
    if isinstance(d, date):
        return d
    if hasattr(d, "date"):
        return d.date()
    try:
        return date.fromisoformat(str(d)[:10])
    except (ValueError, TypeError):
        return None


def _build_date_map(history: list[dict]) -> dict[date, float]:
    """Build date → score lookup from history."""
    result = {}
    for point in history:
        d = _parse_date(point.get("date"))
        if d:
            result[d] = point["score"]
    return result


def _lookup_score(date_map: dict[date, float], target: date, max_lookback: int = 5) -> float:
    """Find score for target date. If not found, look back up to N days."""
    for offset in range(max_lookback + 1):
        d = target - timedelta(days=offset)
        if d in date_map:
            return date_map[d]
    return 50.0  # neutral default


# =============================================================================
# Feature: fear_greed_score
# =============================================================================


@reg.register(
    "fear_greed_score",
    FeatureTier.TIER5_ALTERNATIVE,
    lookback_days=1,
    description="CNN Fear & Greed Index (0=extreme fear, 100=extreme greed) per date",
)
def fear_greed_score(df: pl.DataFrame) -> pl.Series:
    """Map Fear & Greed score to each OHLCV date."""
    n = df.height
    history = _get_history()
    if not history:
        return pl.Series("fear_greed_score", [50.0] * n, dtype=pl.Float64)

    date_map = _build_date_map(history)
    dates = df["date"].to_list()

    result = []
    for d in dates:
        d = _parse_date(d)
        if d is None:
            result.append(50.0)
            continue
        # Normalize to 0-1 range for consistency with other features
        score = _lookup_score(date_map, d) / 100.0
        result.append(score)

    return pl.Series("fear_greed_score", result, dtype=pl.Float64)


# =============================================================================
# Feature: fear_greed_change_7d
# =============================================================================


@reg.register(
    "fear_greed_change_7d",
    FeatureTier.TIER5_ALTERNATIVE,
    lookback_days=7,
    description="Fear & Greed change over 7 days (momentum of sentiment)",
)
def fear_greed_change_7d(df: pl.DataFrame) -> pl.Series:
    """7-day change in Fear & Greed score."""
    n = df.height
    history = _get_history()
    if not history:
        return pl.Series("fear_greed_change_7d", [0.0] * n, dtype=pl.Float64)

    date_map = _build_date_map(history)
    dates = df["date"].to_list()

    result = []
    for d in dates:
        d = _parse_date(d)
        if d is None:
            result.append(0.0)
            continue
        current = _lookup_score(date_map, d)
        past = _lookup_score(date_map, d - timedelta(days=7))
        # Normalize change to fraction
        result.append((current - past) / 100.0)

    return pl.Series("fear_greed_change_7d", result, dtype=pl.Float64)


def clear_cache() -> None:
    global _fg_history
    _fg_history = None
