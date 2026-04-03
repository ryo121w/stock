"""Gate system types and shared definitions for the 7-gate pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass
class GateResult:
    """Result of a single gate evaluation."""

    gate: str
    passed: bool
    score: float  # 0-100
    reason: str = ""
    warnings: list[str] = field(default_factory=list)
    data: dict = field(default_factory=dict)  # Extra data for downstream gates


@dataclass
class FinalVerdict:
    """Output of the full 7-gate pipeline."""

    verdict: str  # STRONG_BUY / BUY / WATCH / HOLD / AVOID
    score: float  # integrated score 0-100
    allocation: float  # recommended portfolio weight (0.0-1.0)
    entry_price: float | None = None
    stop_loss: float | None = None
    target_price: float | None = None
    locked_until: date | None = None
    reason: str = ""
    gate_results: dict[str, GateResult] = field(default_factory=dict)
    ticker: str = ""
