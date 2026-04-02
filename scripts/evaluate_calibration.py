#!/usr/bin/env python3
"""Evaluate probability calibration on Phase 3 dataset.

Walk-forward CV with calibrator trained on validation data WITHIN each fold:
  1. For each fold: split into train / val / test
  2. Train ensemble on train data
  3. Get val predictions -> fit calibrator on val
  4. Apply calibrator to test predictions
  5. Compare accuracy by confidence bucket: BEFORE vs AFTER calibration
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import structlog

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

# Import feature definitions (triggers registration)
import qtp.features.tier1_momentum  # noqa: E402, F401
import qtp.features.tier2_volatility  # noqa: E402, F401
import qtp.features.tier3_fundamental  # noqa: E402, F401
import qtp.features.tier4_macro  # noqa: E402, F401
import qtp.features.tier5_alternative  # noqa: E402, F401
from qtp.config import PipelineConfig  # noqa: E402
from qtp.data.fetchers.base import Market  # noqa: E402
from qtp.data.storage import ParquetStorage  # noqa: E402
from qtp.features.engine import FeatureEngine  # noqa: E402
from qtp.features.registry import FeatureRegistry  # noqa: E402
from qtp.models.calibration import ProbabilityCalibrator  # noqa: E402
from qtp.models.ensemble import WeightedEnsemble  # noqa: E402
from qtp.models.lgbm import LGBMPipeline  # noqa: E402
from qtp.models.xgb import XGBPipeline  # noqa: E402

logger = structlog.get_logger()

# --- Configuration ---
STEP_SIZE = 63  # Walk-forward step (quarterly)
VAL_SIZE = 63  # Validation window for calibrator fitting
MIN_TRAIN = 504  # Minimum training samples
PURGE_GAP = 5  # Gap between train and val/test to prevent leakage
TEST_SIZE = 63  # Test window size

CONFIDENCE_BUCKETS = [
    ("50-55%", 0.50, 0.55),
    ("55-60%", 0.55, 0.60),
    ("60-65%", 0.60, 0.65),
    ("65-70%", 0.65, 0.70),
    ("70%+", 0.70, 1.01),
]


def load_best_params() -> dict:
    """Load best ensemble params from Optuna tuning."""
    params_path = project_root / "configs" / "best_params.json"
    if params_path.exists():
        return json.loads(params_path.read_text())
    return {}


def build_ensemble(best_params: dict) -> WeightedEnsemble:
    """Create ensemble with best params."""
    if best_params:
        lgbm = LGBMPipeline(clf_params=best_params.get("lgbm_clf_params"))
        xgb_model = XGBPipeline(clf_params=best_params.get("xgb_clf_params"))
        lgbm_weight = best_params.get("lgbm_weight", 0.63)
    else:
        lgbm = LGBMPipeline()
        xgb_model = XGBPipeline()
        lgbm_weight = 0.63

    return WeightedEnsemble(
        [
            (lgbm, lgbm_weight),
            (xgb_model, 1.0 - lgbm_weight),
        ]
    )


def load_dataset(config: PipelineConfig) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """Load Phase 3 dataset."""
    storage = ParquetStorage(project_root / config.data.storage_dir)
    engine = FeatureEngine(FeatureRegistry.instance(), storage)
    market = Market(config.universe.market)

    dataset = engine.build_multi_ticker_dataset(
        tickers=config.universe.tickers,
        market=market,
        as_of=date.today(),
        tiers=config.features.tiers,
        horizon=config.labels.horizon,
        direction_threshold=config.labels.direction_threshold,
    )

    # Sort by date for time-series ordering
    dataset = dataset.sort("date")

    label_cols = ["label_direction", "label_magnitude", "date", "ticker"]
    feature_cols = [c for c in dataset.columns if c not in label_cols]

    X = dataset.select(feature_cols)

    # Replace inf/-inf with NaN, then fill NaN with 0
    X = X.with_columns(
        [
            pl.when(pl.col(c).is_infinite())
            .then(None)
            .otherwise(pl.col(c))
            .fill_nan(None)
            .fill_null(0.0)
            .alias(c)
            for c in X.columns
        ]
    )

    y_dir = dataset["label_direction"].to_numpy()
    y_mag = dataset["label_magnitude"].to_numpy()

    return X, y_dir, y_mag


def compute_bucket_accuracy(
    probas: np.ndarray, actuals: np.ndarray, buckets: list[tuple[str, float, float]]
) -> dict[str, dict]:
    """Compute accuracy by confidence bucket."""
    results = {}
    # Convert to directional confidence: max(p, 1-p)
    confidence = np.maximum(probas, 1.0 - probas)
    predictions = (probas >= 0.5).astype(int)

    for name, lo, hi in buckets:
        mask = (confidence >= lo) & (confidence < hi)
        n = mask.sum()
        if n > 0:
            correct = (predictions[mask] == actuals[mask]).sum()
            accuracy = correct / n
        else:
            correct = 0
            accuracy = float("nan")
        results[name] = {"n": int(n), "correct": int(correct), "accuracy": accuracy}

    # Overall for conf >= 55%
    mask_55 = confidence >= 0.55
    n_55 = mask_55.sum()
    if n_55 > 0:
        correct_55 = (predictions[mask_55] == actuals[mask_55]).sum()
        results["55%+ total"] = {
            "n": int(n_55),
            "correct": int(correct_55),
            "accuracy": correct_55 / n_55,
        }

    # Overall for conf >= 65%
    mask_65 = confidence >= 0.65
    n_65 = mask_65.sum()
    if n_65 > 0:
        correct_65 = (predictions[mask_65] == actuals[mask_65]).sum()
        results["65%+ total"] = {
            "n": int(n_65),
            "correct": int(correct_65),
            "accuracy": correct_65 / n_65,
        }

    return results


def run_walk_forward_calibration(
    X: pl.DataFrame,
    y_dir: np.ndarray,
    y_mag: np.ndarray,
    best_params: dict,
    calibration_method: str = "isotonic",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Walk-forward CV with calibration.

    For each fold:
      train = [0, train_end)
      val   = [val_start, val_end)   -- calibrator trained here
      test  = [test_start, test_end) -- calibrator applied here

    Returns (raw_probas, calibrated_probas, actual_labels, fold_ids) for test sets only.
    """
    n = len(X)
    y_mag_series = pl.Series(y_mag)

    all_raw = []
    all_calibrated = []
    all_actual = []
    all_folds = []

    fold = 0
    # We need: MIN_TRAIN for training + PURGE_GAP + VAL_SIZE + PURGE_GAP + TEST_SIZE
    test_start = MIN_TRAIN + PURGE_GAP + VAL_SIZE + PURGE_GAP

    while test_start + TEST_SIZE <= n:
        test_end = test_start + TEST_SIZE

        # Validation window: just before test (with purge gap)
        val_end = test_start - PURGE_GAP
        val_start = val_end - VAL_SIZE

        # Training window: everything before val (with purge gap)
        train_end = val_start - PURGE_GAP

        if train_end < MIN_TRAIN:
            test_start += STEP_SIZE
            continue

        # --- Train model on [0, train_end) ---
        train_X = X[:train_end]
        train_y_dir = pl.Series(y_dir[:train_end])
        train_y_mag = y_mag_series[:train_end]

        ensemble = build_ensemble(best_params)
        ensemble.fit(train_X, train_y_dir, train_y_mag)

        # --- Get validation predictions for calibrator ---
        val_X = X[val_start:val_end]
        val_y_dir = y_dir[val_start:val_end]
        val_raw_proba = np.array(ensemble.predict_proba(val_X))

        # --- Fit calibrator on validation data ---
        calibrator = ProbabilityCalibrator(method=calibration_method)
        calibrator.fit(val_raw_proba, val_y_dir)

        # --- Get test predictions ---
        test_X = X[test_start:test_end]
        test_y_dir = y_dir[test_start:test_end]
        test_raw_proba = np.array(ensemble.predict_proba(test_X))

        # --- Apply calibration ---
        test_cal_proba = calibrator.transform(test_raw_proba)

        all_raw.append(test_raw_proba)
        all_calibrated.append(test_cal_proba)
        all_actual.append(test_y_dir)
        all_folds.append(np.full(len(test_y_dir), fold))

        # Per-fold summary
        raw_acc = ((test_raw_proba >= 0.5).astype(int) == test_y_dir).mean()
        cal_acc = ((test_cal_proba >= 0.5).astype(int) == test_y_dir).mean()
        print(
            f"  Fold {fold}: train={train_end}, val={val_start}-{val_end}, "
            f"test={test_start}-{test_end} | "
            f"raw_acc={raw_acc:.3f}, cal_acc={cal_acc:.3f}, "
            f"cal_samples={calibrator.n_train_samples}"
        )

        fold += 1
        test_start += STEP_SIZE

    if not all_raw:
        raise RuntimeError("No folds produced. Check dataset size and parameters.")

    return (
        np.concatenate(all_raw),
        np.concatenate(all_calibrated),
        np.concatenate(all_actual),
        np.concatenate(all_folds),
    )


def print_comparison_table(raw_buckets: dict, cal_buckets: dict) -> str:
    """Print BEFORE vs AFTER calibration comparison."""
    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("  CALIBRATION EVALUATION — BEFORE vs AFTER")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"  {'Bucket':<14} {'':>3} {'BEFORE':>20}  {'AFTER':>20}  {'Delta':>8}")
    lines.append(f"  {'':<14} {'':>3} {'acc (n)':>20}  {'acc (n)':>20}  {'':>8}")
    lines.append(f"  {'-' * 14} {'---':>3} {'-' * 20}  {'-' * 20}  {'-' * 8}")

    for bucket_name in list(CONFIDENCE_BUCKETS) + [
        ("55%+ total", 0, 0),
        ("65%+ total", 0, 0),
    ]:
        name = bucket_name[0] if isinstance(bucket_name, tuple) else bucket_name
        raw = raw_buckets.get(name, {"n": 0, "accuracy": float("nan")})
        cal = cal_buckets.get(name, {"n": 0, "accuracy": float("nan")})

        raw_acc = raw["accuracy"]
        cal_acc = cal["accuracy"]
        raw_str = (
            f"{raw_acc:6.1%} ({raw['n']:>4})"
            if not np.isnan(raw_acc)
            else f"{'N/A':>6} ({raw['n']:>4})"
        )
        cal_str = (
            f"{cal_acc:6.1%} ({cal['n']:>4})"
            if not np.isnan(cal_acc)
            else f"{'N/A':>6} ({cal['n']:>4})"
        )

        if not np.isnan(raw_acc) and not np.isnan(cal_acc):
            delta = cal_acc - raw_acc
            delta_str = f"{delta:+6.1%}"
        else:
            delta_str = "  N/A"

        sep = "  " if name not in ("55%+ total", "65%+ total") else "  "
        lines.append(f"  {name:<14} {'':>3} {raw_str:>20}{sep}{cal_str:>20}  {delta_str:>8}")

    lines.append("")
    lines.append("=" * 80)
    return "\n".join(lines)


def print_calibration_curve(
    raw_proba: np.ndarray, cal_proba: np.ndarray, actual: np.ndarray
) -> str:
    """Show calibration curve: predicted probability vs actual hit rate."""
    lines = []
    lines.append("")
    lines.append("  CALIBRATION CURVE (predicted vs actual)")
    lines.append(
        f"  {'Pred Range':<14} {'Raw Pred':>10} {'Raw Act':>10} {'Cal Pred':>10} {'Cal Act':>10}"
    )
    lines.append(f"  {'-' * 14} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")

    bins = np.arange(0.3, 0.75, 0.05)
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        label = f"{lo:.0%}-{hi:.0%}"

        # Raw
        raw_mask = (raw_proba >= lo) & (raw_proba < hi)
        if raw_mask.sum() > 5:
            raw_pred_mean = raw_proba[raw_mask].mean()
            raw_act_mean = actual[raw_mask].mean()
            raw_pred_str = f"{raw_pred_mean:.3f}"
            raw_act_str = f"{raw_act_mean:.3f}"
        else:
            raw_pred_str = "---"
            raw_act_str = "---"

        # Calibrated
        cal_mask = (cal_proba >= lo) & (cal_proba < hi)
        if cal_mask.sum() > 5:
            cal_pred_mean = cal_proba[cal_mask].mean()
            cal_act_mean = actual[cal_mask].mean()
            cal_pred_str = f"{cal_pred_mean:.3f}"
            cal_act_str = f"{cal_act_mean:.3f}"
        else:
            cal_pred_str = "---"
            cal_act_str = "---"

        lines.append(
            f"  {label:<14} {raw_pred_str:>10} {raw_act_str:>10} {cal_pred_str:>10} {cal_act_str:>10}"
        )

    lines.append("")

    # Expected Calibration Error (ECE)
    raw_ece = _compute_ece(raw_proba, actual, n_bins=10)
    cal_ece = _compute_ece(cal_proba, actual, n_bins=10)
    lines.append("  Expected Calibration Error (ECE):")
    lines.append(f"    Raw:        {raw_ece:.4f}")
    lines.append(f"    Calibrated: {cal_ece:.4f}")
    if cal_ece < raw_ece:
        lines.append(
            f"    Improvement: {raw_ece - cal_ece:.4f} ({(raw_ece - cal_ece) / raw_ece:.0%} better)"
        )
    else:
        lines.append(f"    Degradation: {cal_ece - raw_ece:+.4f}")
    lines.append("")

    return "\n".join(lines)


def _compute_ece(proba: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (proba >= bin_edges[i]) & (proba < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        avg_pred = proba[mask].mean()
        avg_actual = actual[mask].mean()
        ece += mask.sum() / len(proba) * abs(avg_pred - avg_actual)
    return ece


def main():
    t0 = time.monotonic()
    print("=" * 80)
    print("  PROBABILITY CALIBRATION EVALUATION")
    print("  Phase 3 dataset (h10_t3%, 7 tickers)")
    print("=" * 80)

    # Load config
    config = PipelineConfig.from_yamls(
        project_root / "configs" / "default.yaml",
        project_root / "configs" / "phase3_best.yaml",
    )
    print(f"\nConfig: h{config.labels.horizon}_t{config.labels.direction_threshold}")
    print(f"Tickers: {config.universe.tickers}")
    print(f"Walk-forward: step={STEP_SIZE}, val={VAL_SIZE}, test={TEST_SIZE}")

    # Load best ensemble params
    best_params = load_best_params()
    if best_params:
        print(
            f"Ensemble: LGBM {best_params.get('lgbm_weight', 0.63):.1%} + XGB {best_params.get('xgb_weight', 0.37):.1%}"
        )
    else:
        print("Ensemble: default params (no best_params.json found)")

    # Load dataset
    print("\nLoading dataset...")
    X, y_dir, y_mag = load_dataset(config)
    print(f"Dataset: {X.height} rows, {len(X.columns)} features")
    print(f"Label balance: {y_dir.mean():.2%} positive")

    # --- Run calibration evaluation (Isotonic) ---
    print("\n--- Walk-Forward with Isotonic Calibration ---")
    raw_proba, cal_proba, actual, folds = run_walk_forward_calibration(
        X, y_dir, y_mag, best_params, calibration_method="isotonic"
    )
    print(f"\nTotal test samples: {len(actual)}, Folds: {int(folds.max()) + 1}")

    # Compute accuracy by confidence bucket
    raw_buckets = compute_bucket_accuracy(raw_proba, actual, CONFIDENCE_BUCKETS)
    cal_buckets = compute_bucket_accuracy(cal_proba, actual, CONFIDENCE_BUCKETS)

    # Print comparison
    comparison = print_comparison_table(raw_buckets, cal_buckets)
    print(comparison)

    # Print calibration curve
    curve = print_calibration_curve(raw_proba, cal_proba, actual)
    print(curve)

    # --- Also test Platt Scaling for comparison ---
    print("\n--- Walk-Forward with Platt Scaling (Sigmoid) ---")
    raw_proba_s, cal_proba_s, actual_s, folds_s = run_walk_forward_calibration(
        X, y_dir, y_mag, best_params, calibration_method="sigmoid"
    )

    cal_buckets_s = compute_bucket_accuracy(cal_proba_s, actual_s, CONFIDENCE_BUCKETS)
    print("\n  Platt Scaling results:")
    for name in ["55%+ total", "65%+ total"]:
        b = cal_buckets_s.get(name, {"n": 0, "accuracy": float("nan")})
        if not np.isnan(b["accuracy"]):
            print(f"    {name}: {b['accuracy']:.1%} ({b['n']} samples)")

    sigmoid_ece = _compute_ece(cal_proba_s, actual_s, n_bins=10)
    isotonic_ece = _compute_ece(cal_proba, actual, n_bins=10)
    print(f"\n  ECE comparison: Isotonic={isotonic_ece:.4f} vs Sigmoid={sigmoid_ece:.4f}")
    winner = "Isotonic" if isotonic_ece <= sigmoid_ece else "Sigmoid"
    print(f"  Winner: {winner}")

    # --- Distribution summary ---
    print("\n--- Probability Distribution ---")
    raw_conf = np.maximum(raw_proba, 1.0 - raw_proba)
    cal_conf = np.maximum(cal_proba, 1.0 - cal_proba)
    print(
        f"  Raw:        mean={raw_conf.mean():.3f}, std={raw_conf.std():.3f}, "
        f"max={raw_conf.max():.3f}, >65%={(raw_conf >= 0.65).sum()}"
    )
    print(
        f"  Calibrated: mean={cal_conf.mean():.3f}, std={cal_conf.std():.3f}, "
        f"max={cal_conf.max():.3f}, >65%={(cal_conf >= 0.65).sum()}"
    )

    # Save report
    report_dir = project_root / "data" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "calibration_evaluation.txt"

    report_lines = [
        comparison,
        curve,
        f"\nMethod comparison: Isotonic ECE={isotonic_ece:.4f}, Sigmoid ECE={sigmoid_ece:.4f}",
        f"Recommended: {winner}",
    ]
    report_path.write_text("\n".join(report_lines))
    print(f"\nReport saved to: {report_path}")

    duration = time.monotonic() - t0
    print(f"\nTotal time: {duration:.0f}s ({duration / 60:.1f} min)")


if __name__ == "__main__":
    main()
