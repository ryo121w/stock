"""Gate 4: MAGI 3-body review (simplified).

Takes pre-computed MAGI votes and converts them to a GateResult.
Actual agent spawning happens in the skill layer, not here.
"""

from __future__ import annotations

from qtp.gates import GateResult


class Gate4_MAGI:
    """MAGI 3-body review gate (simplified vote counter)."""

    def evaluate(self, magi_votes: dict[str, str]) -> GateResult:
        """Tally pre-computed MAGI votes and produce a GateResult.

        Parameters
        ----------
        magi_votes : dict
            e.g. {"melchior": "BUY", "balthasar": "HOLD", "casper": "BUY"}
            Each value must be one of: "BUY", "HOLD", "AVOID"

        Returns
        -------
        GateResult
        """
        votes = [v.upper() for v in magi_votes.values()]
        buy_count = votes.count("BUY")
        avoid_count = votes.count("AVOID")
        hold_count = votes.count("HOLD")

        # Scoring matrix per spec
        if buy_count == 3:
            score = 95.0
            passed = True
        elif buy_count == 2:
            score = 75.0
            passed = True
        elif buy_count == 1 and avoid_count <= 1 and hold_count >= 1:
            # 1-1-1 split
            score = 50.0
            passed = True
        elif avoid_count == 2:
            score = 25.0
            passed = False
        elif avoid_count == 3:
            score = 5.0
            passed = False
        else:
            # Fallback for other combos (e.g. 1 BUY, 2 HOLD)
            score = 50.0 + buy_count * 15 - avoid_count * 20
            score = max(0.0, min(100.0, score))
            passed = score >= 50.0

        vote_str = ", ".join(f"{k}={v}" for k, v in magi_votes.items())
        reason = f"Votes: {vote_str} (BUY={buy_count}, HOLD={hold_count}, AVOID={avoid_count})"

        return GateResult(
            gate="MAGI",
            passed=passed,
            score=score,
            reason=reason,
            data={"votes": magi_votes, "buy": buy_count, "hold": hold_count, "avoid": avoid_count},
        )
