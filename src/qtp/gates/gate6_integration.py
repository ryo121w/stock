"""Gate 6: Integrated score calculation.

Computes the weighted sum of all gate scores.
"""

from __future__ import annotations

from qtp.gates import GateResult

DEFAULT_WEIGHTS: dict[str, float] = {
    "qtp": 0.25,
    "technical": 0.15,
    "fundamental": 0.25,
    "magi": 0.25,
    "sentiment": 0.10,
}


class Gate6_Integration:
    """Weighted integration of gate scores."""

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or dict(DEFAULT_WEIGHTS)

    def calculate(self, gate_results: dict[str, GateResult]) -> float:
        """Return integrated score (0-100) as weighted sum of gate scores.

        Gates not present in *gate_results* are skipped; their weight is
        redistributed proportionally among the gates that *are* present.
        """
        active_weights: dict[str, float] = {}
        for name, result in gate_results.items():
            if name in self.weights:
                active_weights[name] = self.weights[name]

        if not active_weights:
            return 0.0

        # Normalise weights so they sum to 1.0 even when gates are missing
        weight_sum = sum(active_weights.values())
        if weight_sum == 0:
            return 0.0

        total = 0.0
        for name, w in active_weights.items():
            normalised_w = w / weight_sum
            total += gate_results[name].score * normalised_w

        return round(total, 2)

    # ------------------------------------------------------------------
    # Compatibility: Phase-1 tests call evaluate(list[GateResult])
    # ------------------------------------------------------------------

    # Map gate name -> weight key
    _GATE_KEY_MAP: dict[str, str] = {
        "QTP": "qtp",
        "Technical": "technical",
        "Fundamental": "fundamental",
        "MAGI": "magi",
        "Sentiment": "sentiment",
    }

    def evaluate(self, prior_results: list[GateResult]) -> GateResult:
        """Compatibility wrapper that accepts a list of GateResult.

        If any hard gate (1-4) failed, the integration result is marked failed.
        """
        results_dict: dict[str, GateResult] = {}
        for r in prior_results:
            key = self._GATE_KEY_MAP.get(r.gate, r.gate.lower())
            results_dict[key] = r

        score = self.calculate(results_dict)

        # A hard gate failure (gates 1-4) blocks integration
        hard_gates = {"qtp", "technical", "fundamental", "magi"}
        any_hard_fail = any(not r.passed for name, r in results_dict.items() if name in hard_gates)

        passed = not any_hard_fail and score >= 35

        return GateResult(
            gate="Integration",
            passed=passed,
            score=score,
            reason=f"Integrated score={score:.1f}, hard_fail={any_hard_fail}",
        )
