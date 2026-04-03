#!/usr/bin/env python3
"""Validate the gate system against historical (backfilled) predictions.

Loads graded predictions from the database, retroactively computes Gate 1
and Gate 2 scores for each, then compares actual returns for predictions
that would-have-passed vs would-have-failed the gates.

Usage:
    .venv/bin/python scripts/validate_gates.py
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import structlog  # noqa: E402

from qtp.data.database import QTPDatabase  # noqa: E402

logger = structlog.get_logger()


def load_graded_predictions(db: QTPDatabase) -> list[dict]:
    """Load all graded predictions from the database."""
    with db._conn() as conn:
        rows = conn.execute(
            """SELECT ticker, prediction_date, direction, confidence,
                      predicted_magnitude, actual_return, is_correct,
                      model_version
               FROM predictions
               WHERE graded_at IS NOT NULL
               ORDER BY prediction_date"""
        ).fetchall()
    return [dict(r) for r in rows]


def compute_gate1_score(pred: dict, all_preds: list[dict]) -> dict:
    """Retroactively compute Gate 1 (QTP) score for a prediction.

    Gate 1 checks:
      - confidence >= 55%
      - direction == UP (1)
      - historical accuracy >= 53%
    """
    confidence = pred["confidence"]
    direction = pred["direction"]
    ticker = pred["ticker"]
    pred_date = pred["prediction_date"]

    # Calculate historical accuracy: all graded predictions for this ticker
    # that occurred BEFORE this prediction date
    history = [
        p
        for p in all_preds
        if p["ticker"] == ticker
        and p["prediction_date"] < pred_date
        and p["is_correct"] is not None
    ]

    if len(history) == 0:
        hist_accuracy = 0.5  # No history -> assume coin flip
    else:
        hist_accuracy = sum(1 for p in history if p["is_correct"]) / len(history)

    passed = confidence >= 0.55 and direction == 1 and hist_accuracy >= 0.53

    return {
        "gate": "G1_QTP",
        "passed": passed,
        "score": confidence * 100,
        "confidence": confidence,
        "direction": direction,
        "hist_accuracy": hist_accuracy,
        "n_history": len(history),
    }


def compute_gate2_score(pred: dict) -> dict:
    """Retroactively compute a simplified Gate 2 (Technical) score.

    Without real-time technical data, we use confidence + direction as proxy.
    High confidence + UP direction = higher technical score.
    """
    confidence = pred["confidence"]
    direction = pred["direction"]

    # Proxy: high confidence + UP direction -> better technical score
    if direction == 1 and confidence >= 0.60:
        score = 70 + (confidence - 0.60) * 100
        passed = True
    elif direction == 1:
        score = 50 + confidence * 30
        passed = score >= 55
    else:
        score = 30
        passed = False

    return {
        "gate": "G2_Technical",
        "passed": passed,
        "score": min(score, 100),
    }


def main():
    db_path = project_root / "data" / "qtp.db"
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        print("Run 'make backfill' first to generate historical predictions.")
        sys.exit(1)

    db = QTPDatabase(db_path)
    all_preds = load_graded_predictions(db)

    if not all_preds:
        print("No graded predictions found in database.")
        print("Run 'make backfill' first to generate historical predictions.")
        sys.exit(1)

    print(f"Loaded {len(all_preds)} graded predictions")
    print()

    # Compute gate scores for each prediction
    gate_passed: list[dict] = []  # Would have passed all gates
    gate_failed: list[dict] = []  # Would have failed at least one gate

    for pred in all_preds:
        g1 = compute_gate1_score(pred, all_preds)
        g2 = compute_gate2_score(pred)

        both_passed = g1["passed"] and g2["passed"]

        entry = {
            **pred,
            "g1_passed": g1["passed"],
            "g1_score": g1["score"],
            "g1_hist_acc": g1["hist_accuracy"],
            "g2_passed": g2["passed"],
            "g2_score": g2["score"],
        }

        if both_passed:
            gate_passed.append(entry)
        else:
            gate_failed.append(entry)

    # Calculate statistics
    def calc_stats(entries: list[dict]) -> dict:
        if not entries:
            return {
                "count": 0,
                "accuracy": 0.0,
                "avg_return": 0.0,
                "avg_confidence": 0.0,
                "positive_returns": 0.0,
            }
        n = len(entries)
        correct = sum(1 for e in entries if e["is_correct"])
        returns = [e["actual_return"] for e in entries if e["actual_return"] is not None]
        avg_return = sum(returns) / len(returns) if returns else 0.0
        positive = sum(1 for r in returns if r > 0) / len(returns) if returns else 0.0
        avg_conf = sum(e["confidence"] for e in entries) / n
        return {
            "count": n,
            "accuracy": correct / n if n > 0 else 0.0,
            "avg_return": avg_return,
            "avg_confidence": avg_conf,
            "positive_returns": positive,
        }

    passed_stats = calc_stats(gate_passed)
    failed_stats = calc_stats(gate_failed)

    # Print comparison table
    print("=" * 72)
    print("  GATE VALIDATION: Would-Have-Passed vs Would-Have-Failed")
    print("=" * 72)
    print()
    print(f"  {'Metric':<28s}  {'PASSED Gates':>14s}  {'FAILED Gates':>14s}  {'Delta':>10s}")
    print(f"  {'-' * 28}  {'-' * 14}  {'-' * 14}  {'-' * 10}")

    rows = [
        ("Count", f"{passed_stats['count']}", f"{failed_stats['count']}", ""),
        (
            "Direction Accuracy",
            f"{passed_stats['accuracy']:.1%}",
            f"{failed_stats['accuracy']:.1%}",
            f"{passed_stats['accuracy'] - failed_stats['accuracy']:+.1%}",
        ),
        (
            "Avg Return",
            f"{passed_stats['avg_return']:+.3%}",
            f"{failed_stats['avg_return']:+.3%}",
            f"{passed_stats['avg_return'] - failed_stats['avg_return']:+.3%}",
        ),
        (
            "Positive Return %",
            f"{passed_stats['positive_returns']:.1%}",
            f"{failed_stats['positive_returns']:.1%}",
            f"{passed_stats['positive_returns'] - failed_stats['positive_returns']:+.1%}",
        ),
        (
            "Avg Confidence",
            f"{passed_stats['avg_confidence']:.1%}",
            f"{failed_stats['avg_confidence']:.1%}",
            f"{passed_stats['avg_confidence'] - failed_stats['avg_confidence']:+.1%}",
        ),
    ]

    for label, passed_val, failed_val, delta in rows:
        print(f"  {label:<28s}  {passed_val:>14s}  {failed_val:>14s}  {delta:>10s}")

    print()
    print("=" * 72)

    # Gate-specific breakdown
    print()
    print("  Gate 1 (QTP) Breakdown:")
    g1_passed = [e for e in all_preds if compute_gate1_score(e, all_preds)["passed"]]
    g1_failed = [e for e in all_preds if not compute_gate1_score(e, all_preds)["passed"]]
    g1p_stats = calc_stats(g1_passed)
    g1f_stats = calc_stats(g1_failed)
    print(
        f"    Passed: {g1p_stats['count']} predictions, "
        f"accuracy={g1p_stats['accuracy']:.1%}, avg_return={g1p_stats['avg_return']:+.3%}"
    )
    print(
        f"    Failed: {g1f_stats['count']} predictions, "
        f"accuracy={g1f_stats['accuracy']:.1%}, avg_return={g1f_stats['avg_return']:+.3%}"
    )

    print()
    print("  Gate 2 (Technical proxy) Breakdown:")
    g2_passed_list = [e for e in all_preds if compute_gate2_score(e)["passed"]]
    g2_failed_list = [e for e in all_preds if not compute_gate2_score(e)["passed"]]
    g2p_stats = calc_stats(g2_passed_list)
    g2f_stats = calc_stats(g2_failed_list)
    print(
        f"    Passed: {g2p_stats['count']} predictions, "
        f"accuracy={g2p_stats['accuracy']:.1%}, avg_return={g2p_stats['avg_return']:+.3%}"
    )
    print(
        f"    Failed: {g2f_stats['count']} predictions, "
        f"accuracy={g2f_stats['accuracy']:.1%}, avg_return={g2f_stats['avg_return']:+.3%}"
    )

    print()

    # Verdict
    delta_acc = passed_stats["accuracy"] - failed_stats["accuracy"]
    delta_ret = passed_stats["avg_return"] - failed_stats["avg_return"]

    if delta_acc > 0.03 and delta_ret > 0:
        print("  VERDICT: Gates are EFFECTIVE -- passed predictions outperform failed ones.")
    elif delta_acc > 0 or delta_ret > 0:
        print("  VERDICT: Gates show MARGINAL benefit -- some improvement but needs tuning.")
    else:
        print("  VERDICT: Gates NOT YET EFFECTIVE -- thresholds may need recalibration.")

    print()


if __name__ == "__main__":
    main()
