"""Gate 5: Sentiment confirmation.

This gate never blocks — it only adjusts the score.
Extreme optimism is a contrarian warning; pessimism is a contrarian bonus.
"""

from __future__ import annotations

from qtp.gates import GateResult

_BASE_SCORE = 70.0


class Gate5_Sentiment:
    """Market sentiment confirmation gate."""

    def evaluate(self, sentiment_data: dict) -> GateResult:
        """Evaluate sentiment signals and return an always-passing GateResult.

        Parameters
        ----------
        sentiment_data : dict
            Keys:
              analyst_all_buy (bool) – all analysts rate BUY → contrarian warning
              board_euphoric  (bool) – bulletin board is extremely bullish
              board_pessimistic (bool) – bulletin board is pessimistic → contrarian bonus

        Returns
        -------
        GateResult  (always passed=True)
        """
        score = _BASE_SCORE
        warnings: list[str] = []

        if sentiment_data.get("analyst_all_buy", False):
            warnings.append("All analysts are BUY (contrarian risk)")
            score -= 15

        if sentiment_data.get("board_euphoric", False):
            warnings.append("Board sentiment is extremely bullish")
            score -= 20

        if sentiment_data.get("board_pessimistic", False):
            score += 10  # contrarian bonus

        score = max(0.0, min(100.0, score))

        return GateResult(
            gate="Sentiment",
            passed=True,  # This gate always passes
            score=score,
            reason=f"Sentiment score={score:.0f}"
            + (f", warnings: {'; '.join(warnings)}" if warnings else ""),
            warnings=warnings,
            data=sentiment_data,
        )
