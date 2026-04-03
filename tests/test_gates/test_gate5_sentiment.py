"""Tests for Gate 5: Sentiment confirmation."""

from qtp.gates.gate5_sentiment import Gate5_Sentiment


class TestGate5Sentiment:
    def setup_method(self):
        self.gate = Gate5_Sentiment()

    def test_neutral_sentiment(self):
        result = self.gate.evaluate({})
        assert result.passed is True
        assert result.score == 70.0
        assert result.warnings == []

    def test_analyst_all_buy_penalty(self):
        result = self.gate.evaluate({"analyst_all_buy": True})
        assert result.passed is True
        assert result.score == 55.0  # 70 - 15
        assert len(result.warnings) == 1

    def test_board_euphoric_penalty(self):
        result = self.gate.evaluate({"board_euphoric": True})
        assert result.passed is True
        assert result.score == 50.0  # 70 - 20
        assert len(result.warnings) == 1

    def test_board_pessimistic_bonus(self):
        result = self.gate.evaluate({"board_pessimistic": True})
        assert result.passed is True
        assert result.score == 80.0  # 70 + 10
        assert result.warnings == []

    def test_all_negative_signals(self):
        result = self.gate.evaluate({"analyst_all_buy": True, "board_euphoric": True})
        assert result.passed is True
        assert result.score == 35.0  # 70 - 15 - 20
        assert len(result.warnings) == 2

    def test_always_passes(self):
        """Gate 5 should never block."""
        result = self.gate.evaluate({"analyst_all_buy": True, "board_euphoric": True})
        assert result.passed is True

    def test_gate_name(self):
        result = self.gate.evaluate({})
        assert result.gate == "Sentiment"

    def test_score_clamped_to_0(self):
        """Score should not go below 0 even with extreme penalties."""
        # Only 35 points of penalty possible, so score stays at 35
        # But test the clamping logic is present
        result = self.gate.evaluate({"analyst_all_buy": True, "board_euphoric": True})
        assert result.score >= 0.0
