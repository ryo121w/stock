"""Tests for Gate 6: Integration (weighted score)."""

from qtp.gates import GateResult
from qtp.gates.gate6_integration import Gate6_Integration


class TestGate6Integration:
    def test_all_gates_equal_score(self):
        """All gates at 80 → integrated should be 80."""
        results = {
            "qtp": GateResult(gate="QTP", passed=True, score=80),
            "technical": GateResult(gate="Technical", passed=True, score=80),
            "fundamental": GateResult(gate="Fundamental", passed=True, score=80),
            "magi": GateResult(gate="MAGI", passed=True, score=80),
            "sentiment": GateResult(gate="Sentiment", passed=True, score=80),
        }
        score = Gate6_Integration().calculate(results)
        assert score == 80.0

    def test_weighted_calculation(self):
        """Verify weights are applied correctly."""
        results = {
            "qtp": GateResult(gate="QTP", passed=True, score=100),
            "technical": GateResult(gate="Technical", passed=True, score=0),
            "fundamental": GateResult(gate="Fundamental", passed=True, score=100),
            "magi": GateResult(gate="MAGI", passed=True, score=0),
            "sentiment": GateResult(gate="Sentiment", passed=True, score=0),
        }
        score = Gate6_Integration().calculate(results)
        # qtp(100*0.25) + tech(0*0.15) + fund(100*0.25) + magi(0*0.25) + sent(0*0.10) = 50
        assert score == 50.0

    def test_missing_gates_renormalize(self):
        """When some gates are missing, weights should renormalize."""
        results = {
            "qtp": GateResult(gate="QTP", passed=True, score=100),
        }
        score = Gate6_Integration().calculate(results)
        # Only qtp present, weight renormalises to 1.0 → score = 100
        assert score == 100.0

    def test_custom_weights(self):
        results = {
            "qtp": GateResult(gate="QTP", passed=True, score=100),
            "magi": GateResult(gate="MAGI", passed=True, score=0),
        }
        custom = {"qtp": 0.50, "magi": 0.50}
        score = Gate6_Integration(weights=custom).calculate(results)
        assert score == 50.0

    def test_empty_results(self):
        score = Gate6_Integration().calculate({})
        assert score == 0.0

    def test_unknown_gate_ignored(self):
        """Gates not in the weight map are ignored."""
        results = {
            "unknown_gate": GateResult(gate="Unknown", passed=True, score=100),
        }
        score = Gate6_Integration().calculate(results)
        assert score == 0.0
