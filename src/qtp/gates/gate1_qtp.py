"""Gate 1: QTP quantitative model score gate.

Checks ML model prediction confidence, direction, and historical accuracy.
Eliminates tickers the model is not confident about.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from qtp.data.database import QTPDatabase
from qtp.gates import GateResult

logger = structlog.get_logger()


@dataclass
class _Prediction:
    """Lightweight prediction snapshot for gate evaluation."""

    direction: int  # 1=up, 0=down
    confidence: float  # 0-1


class Gate1_QTP:
    """Quantitative model gate -- first filter in the pipeline."""

    PASS_THRESHOLD = 0.55  # Confidence >= 55% to pass
    HIST_ACCURACY_THRESHOLD = 0.53  # Historical accuracy >= 53%

    def __init__(self, db: QTPDatabase):
        self.db = db

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, ticker: str) -> GateResult:
        """Run Gate 1 evaluation for *ticker*.

        Steps:
          1. Get latest prediction from the predictions table.
          2. Calculate historical accuracy from graded predictions.
          3. Pass only if confidence >= 55%, direction == UP, and
             historical accuracy >= 53%.
        """
        prediction = self._get_latest_prediction(ticker)
        if prediction is None:
            return GateResult(
                gate="QTP",
                passed=False,
                score=0.0,
                reason="No prediction available",
            )

        historical_accuracy = self._get_historical_accuracy(ticker)

        passed = (
            prediction.confidence >= self.PASS_THRESHOLD
            and prediction.direction == 1  # UP
            and historical_accuracy >= self.HIST_ACCURACY_THRESHOLD
        )

        score = prediction.confidence * 100  # 0-100

        # Build human-readable reason
        parts: list[str] = []
        parts.append(f"conf={prediction.confidence:.1%}")
        parts.append(f"dir={'UP' if prediction.direction == 1 else 'DOWN'}")
        parts.append(f"hist_acc={historical_accuracy:.1%}")

        warnings: list[str] = []
        if prediction.direction != 1:
            warnings.append("Direction is DOWN")
        if prediction.confidence < self.PASS_THRESHOLD:
            warnings.append(f"Confidence {prediction.confidence:.1%} < {self.PASS_THRESHOLD:.0%}")
        if historical_accuracy < self.HIST_ACCURACY_THRESHOLD:
            warnings.append(
                f"Historical accuracy {historical_accuracy:.1%} < {self.HIST_ACCURACY_THRESHOLD:.0%}"
            )

        return GateResult(
            gate="QTP",
            passed=passed,
            score=score,
            reason=", ".join(parts),
            warnings=warnings,
            data={
                "confidence": prediction.confidence,
                "direction": prediction.direction,
                "historical_accuracy": historical_accuracy,
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_latest_prediction(self, ticker: str) -> _Prediction | None:
        """Fetch the most recent prediction for *ticker* from SQLite."""
        with self.db._conn() as conn:
            row = conn.execute(
                """SELECT direction, confidence
                   FROM predictions
                   WHERE ticker = ?
                   ORDER BY prediction_date DESC
                   LIMIT 1""",
                (ticker,),
            ).fetchone()
        if row is None:
            return None
        return _Prediction(direction=row["direction"], confidence=row["confidence"])

    def _get_historical_accuracy(self, ticker: str) -> float:
        """Calculate historical accuracy from graded predictions.

        Returns the fraction of correct predictions (0.0-1.0).
        If no graded data exists, returns 0.5 (neutral -- coin flip).
        """
        with self.db._conn() as conn:
            row = conn.execute(
                """SELECT COUNT(*) as total, SUM(is_correct) as correct
                   FROM predictions
                   WHERE ticker = ? AND graded_at IS NOT NULL""",
                (ticker,),
            ).fetchone()
        if row is None or row["total"] == 0:
            return 0.5  # No data -- assume coin flip
        return row["correct"] / row["total"]
