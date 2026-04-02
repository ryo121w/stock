"""Bridge between QTP predictions and Claude Code investment skills."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import structlog

from qtp.models.base import PredictionResult

logger = structlog.get_logger()


@dataclass
class ActionableSignal:
    """Output format compatible with Claude Code invest skill ecosystem."""

    ticker: str
    market: str
    signal_date: str
    direction: str              # "LONG" | "NEUTRAL"
    confidence: str             # "HIGH" | "MEDIUM" | "LOW"
    confidence_score: float
    expected_return_pct: float
    model_version: str
    top_features: list[dict] = field(default_factory=list)

    def to_markdown(self) -> str:
        emoji = "\U0001f7e2" if self.direction == "LONG" else "\u26aa"
        features_str = ", ".join(f["name"] for f in self.top_features[:5])
        return (
            f"### {emoji} {self.ticker} ({self.market.upper()})\n"
            f"| Item | Value |\n|------|-------|\n"
            f"| Direction | {self.direction} |\n"
            f"| Confidence | {self.confidence} ({self.confidence_score:.1%}) |\n"
            f"| Expected Return | {self.expected_return_pct:+.2%} |\n"
            f"| Model | {self.model_version} |\n\n"
            f"**Top Drivers**: {features_str}\n"
        )


class ClaudeBridge:
    """Export QTP predictions for Claude Code skill consumption."""

    def export_signals(
        self,
        predictions: list[PredictionResult],
        market: str,
        output_dir: Path,
    ) -> Path:
        signals = [self._to_actionable(p, market) for p in predictions]
        path = output_dir / "latest_signals.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([asdict(s) for s in signals], indent=2, default=str))
        logger.info("signals_exported", path=str(path), count=len(signals))
        return path

    def export_markdown_report(
        self,
        predictions: list[PredictionResult],
        market: str,
        output_dir: Path,
    ) -> Path:
        signals = [self._to_actionable(p, market) for p in predictions]
        lines = [f"## QTP Signals ({date.today()})\n"]
        for s in sorted(signals, key=lambda x: x.confidence_score, reverse=True):
            lines.append(s.to_markdown())
        path = output_dir / "latest_signals.md"
        path.write_text("\n".join(lines))
        return path

    @staticmethod
    def _to_actionable(pred: PredictionResult, market: str) -> ActionableSignal:
        if pred.direction_proba >= 0.70:
            confidence = "HIGH"
        elif pred.direction_proba >= 0.55:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        return ActionableSignal(
            ticker=pred.ticker,
            market=market,
            signal_date=pred.prediction_date.isoformat(),
            direction="LONG" if pred.direction == 1 else "NEUTRAL",
            confidence=confidence,
            confidence_score=pred.direction_proba,
            expected_return_pct=pred.magnitude,
            model_version=pred.model_version,
        )
