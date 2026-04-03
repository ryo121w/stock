"""Gate 7: Final verdict with anti-flip locking.

Converts the integrated score into a verdict label, allocation, and price
targets.  Includes VerdictCache for 14-day lock to prevent flip-flopping.
"""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from pathlib import Path

from qtp.gates import FinalVerdict, GateResult

# Ordered from highest threshold to lowest so the first match wins.
THRESHOLDS: list[tuple[str, float]] = [
    ("STRONG_BUY", 80),
    ("BUY", 65),
    ("WATCH", 50),
    ("HOLD", 35),
    ("AVOID", 0),
]

# Allocation lookup per verdict
_ALLOC: dict[str, float] = {
    "STRONG_BUY": 0.10,
    "BUY": 0.05,
    "WATCH": 0.00,
    "HOLD": 0.00,
    "AVOID": 0.00,
}


class Gate7_Verdict:
    """Produce a FinalVerdict from the integrated score."""

    def judge(
        self,
        integrated_score: float,
        gate_results: dict[str, GateResult],
        ticker: str = "",
        current_price: float | None = None,
    ) -> FinalVerdict:
        """Map integrated score → verdict + position sizing."""

        # Determine verdict label
        verdict_label = "AVOID"
        for label, threshold in THRESHOLDS:
            if integrated_score >= threshold:
                verdict_label = label
                break

        # Base allocation from verdict
        allocation = _ALLOC.get(verdict_label, 0.0)

        # Adjust allocation by sentiment factor (base=70)
        sentiment = gate_results.get("sentiment")
        if sentiment is not None and allocation > 0:
            factor = sentiment.score / 70.0
            allocation = round(allocation * factor, 4)

        # Price targets (simple heuristics when current_price is available)
        entry_price: float | None = current_price
        stop_loss: float | None = None
        target_price: float | None = None
        if current_price is not None:
            stop_loss = round(current_price * 0.85, 2)  # -15%
            # Pull target from fundamental gate details if present
            fundamental = gate_results.get("fundamental")
            if fundamental and fundamental.data.get("target_price"):
                target_price = fundamental.data["target_price"]
            else:
                target_price = round(current_price * 1.20, 2)  # default +20%

        locked_until = date.today() + timedelta(days=14)

        return FinalVerdict(
            verdict=verdict_label,
            score=integrated_score,
            allocation=allocation,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price=target_price,
            locked_until=locked_until,
            reason=self._build_reason(verdict_label, integrated_score, gate_results),
            gate_results=gate_results,
            ticker=ticker,
        )

    # ------------------------------------------------------------------
    # Compatibility: Phase-1 tests call evaluate(GateResult) -> GateResult
    # ------------------------------------------------------------------

    def evaluate(self, integration_result: GateResult) -> GateResult:
        """Compatibility wrapper that takes an integration GateResult."""
        score = integration_result.score

        verdict_label = "AVOID"
        for label, threshold in THRESHOLDS:
            if score >= threshold:
                verdict_label = label
                break

        passed = score >= 35  # HOLD or better

        return GateResult(
            gate="Verdict",
            passed=passed,
            score=score,
            reason=f"{verdict_label} (score={score:.1f})",
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _build_reason(verdict: str, score: float, results: dict[str, GateResult]) -> str:
        parts = [f"{verdict} (score={score:.1f})"]
        for name, r in results.items():
            status = "PASS" if r.passed else "FAIL"
            parts.append(f"  {r.gate}: {status} {r.score:.0f} - {r.reason}")
        return "\n".join(parts)


# =====================================================================
# VerdictCache — 14-day lock to prevent flip-flopping
# =====================================================================

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS verdict_cache (
    ticker TEXT PRIMARY KEY,
    verdict TEXT NOT NULL,
    score REAL NOT NULL,
    allocation REAL NOT NULL,
    entry_price REAL,
    stop_loss REAL,
    target_price REAL,
    locked_until TEXT NOT NULL,
    reason TEXT,
    evaluated_at TEXT NOT NULL
)
"""


class VerdictCache:
    """SQLite-backed verdict cache with 14-day lock."""

    LOCK_DAYS = 14

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_path = Path.home() / ".qtp" / "verdict_cache.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE_TABLE)

    # ------------------------------------------------------------------
    def get(self, ticker: str) -> FinalVerdict | None:
        """Return cached verdict or None."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT verdict, score, allocation, entry_price, stop_loss, "
                "target_price, locked_until, reason FROM verdict_cache WHERE ticker = ?",
                (ticker,),
            ).fetchone()
        if row is None:
            return None
        return FinalVerdict(
            verdict=row[0],
            score=row[1],
            allocation=row[2],
            entry_price=row[3],
            stop_loss=row[4],
            target_price=row[5],
            locked_until=date.fromisoformat(row[6]) if row[6] else None,
            reason=row[7] or "",
            ticker=ticker,
        )

    # ------------------------------------------------------------------
    def put(self, ticker: str, verdict: FinalVerdict) -> None:
        """Insert or replace cached verdict."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO verdict_cache "
                "(ticker, verdict, score, allocation, entry_price, stop_loss, "
                "target_price, locked_until, reason, evaluated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    ticker,
                    verdict.verdict,
                    verdict.score,
                    verdict.allocation,
                    verdict.entry_price,
                    verdict.stop_loss,
                    verdict.target_price,
                    verdict.locked_until.isoformat() if verdict.locked_until else None,
                    verdict.reason,
                    date.today().isoformat(),
                ),
            )

    # ------------------------------------------------------------------
    def should_re_evaluate(self, cached: FinalVerdict) -> bool:
        """Return True when the cached verdict should be refreshed."""
        if cached.locked_until is None:
            return True
        return date.today() >= cached.locked_until

    # ------------------------------------------------------------------
    def invalidate(self, ticker: str) -> None:
        """Force removal (e.g. after earnings or major news)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM verdict_cache WHERE ticker = ?", (ticker,))
