"""Evaluation metrics combining classification and financial metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, precision_score, recall_score, roc_auc_score,
)


@dataclass
class EvaluationMetrics:
    # Classification
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    log_loss_val: float

    # Financial
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float

    def summary(self) -> dict:
        return {
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "auc_roc": round(self.auc_roc, 4),
            "sharpe": round(self.sharpe_ratio, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 4),
        }


def compute_metrics(
    y_true_direction: np.ndarray,
    y_pred_proba: np.ndarray,
    y_true_magnitude: np.ndarray,
    y_pred_magnitude: np.ndarray,
    commission_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> EvaluationMetrics:
    """Compute classification + financial metrics.

    Parameters
    ----------
    commission_bps : float
        One-way commission in basis points (e.g., 10 = 10bps = 0.10%).
        Round-trip cost = 2 * commission_bps.
    slippage_bps : float
        One-way slippage in basis points.
        Round-trip cost = 2 * slippage_bps.
    """
    y_pred_direction = (y_pred_proba >= 0.5).astype(int)

    # Total round-trip transaction cost as a fraction
    round_trip_cost = 2 * (commission_bps + slippage_bps) / 10_000

    # Classification metrics
    acc = accuracy_score(y_true_direction, y_pred_direction)
    prec = precision_score(y_true_direction, y_pred_direction, zero_division=0)
    rec = recall_score(y_true_direction, y_pred_direction, zero_division=0)
    f1 = f1_score(y_true_direction, y_pred_direction, zero_division=0)

    try:
        auc = roc_auc_score(y_true_direction, y_pred_proba)
    except ValueError:
        auc = 0.5

    try:
        ll = log_loss(y_true_direction, y_pred_proba)
    except ValueError:
        ll = float("inf")

    # Financial metrics (simulate trading the predicted direction)
    trade_mask = y_pred_proba >= 0.55  # Only trade high-confidence
    if trade_mask.sum() > 0:
        gross_returns = y_true_magnitude[trade_mask] * np.where(
            y_pred_direction[trade_mask] == 1, 1, 0
        )
        # Deduct transaction costs for every trade
        trade_returns = gross_returns - round_trip_cost

        wins = trade_returns[trade_returns > 0]
        losses = trade_returns[trade_returns < 0]

        win_rate = len(wins) / len(trade_returns) if len(trade_returns) > 0 else 0
        profit_factor = (
            abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else float("inf")
        )

        # Sharpe (annualized)
        if trade_returns.std() > 0:
            sharpe = (trade_returns.mean() / trade_returns.std()) * (252 ** 0.5)
        else:
            sharpe = 0.0

        # Max drawdown
        cumulative = (1 + trade_returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min() if len(drawdown) > 0 else 0.0
    else:
        win_rate = 0.0
        profit_factor = 0.0
        sharpe = 0.0
        max_dd = 0.0

    return EvaluationMetrics(
        accuracy=acc, precision=prec, recall=rec, f1=f1,
        auc_roc=auc, log_loss_val=ll,
        sharpe_ratio=sharpe, max_drawdown=max_dd,
        win_rate=win_rate, profit_factor=profit_factor,
    )
