"""Tests for Gate 4: MAGI vote counter."""

from qtp.gates.gate4_magi import Gate4_MAGI


class TestGate4MAGI:
    def setup_method(self):
        self.gate = Gate4_MAGI()

    def test_unanimous_buy(self):
        result = self.gate.evaluate({"melchior": "BUY", "balthasar": "BUY", "casper": "BUY"})
        assert result.passed is True
        assert result.score == 95.0

    def test_majority_buy(self):
        result = self.gate.evaluate({"melchior": "BUY", "balthasar": "HOLD", "casper": "BUY"})
        assert result.passed is True
        assert result.score == 75.0

    def test_split_111(self):
        result = self.gate.evaluate({"melchior": "BUY", "balthasar": "HOLD", "casper": "AVOID"})
        assert result.passed is True
        assert result.score == 50.0

    def test_majority_avoid(self):
        result = self.gate.evaluate({"melchior": "AVOID", "balthasar": "AVOID", "casper": "BUY"})
        assert result.passed is False
        assert result.score == 25.0

    def test_unanimous_avoid(self):
        result = self.gate.evaluate({"melchior": "AVOID", "balthasar": "AVOID", "casper": "AVOID"})
        assert result.passed is False
        assert result.score == 5.0

    def test_case_insensitive(self):
        result = self.gate.evaluate({"melchior": "buy", "balthasar": "Buy", "casper": "BUY"})
        assert result.passed is True
        assert result.score == 95.0

    def test_gate_name(self):
        result = self.gate.evaluate({"melchior": "BUY", "balthasar": "BUY", "casper": "BUY"})
        assert result.gate == "MAGI"

    def test_data_contains_vote_counts(self):
        result = self.gate.evaluate({"melchior": "BUY", "balthasar": "HOLD", "casper": "AVOID"})
        assert result.data["buy"] == 1
        assert result.data["hold"] == 1
        assert result.data["avoid"] == 1
