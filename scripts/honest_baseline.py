#!/usr/bin/env python3
"""Honest Baseline: Compare CV/OOS metrics before and after Phase 1.5 fixes.

Runs:
1. Walk-Forward CV with transaction costs (the honest evaluation)
2. PurgedKFold CV with transaction costs (for comparison)
3. Outputs a side-by-side comparison table

Expected result: AUC drops from inflated 0.676 to honest 0.52-0.55 range.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import structlog

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from qtp.config import PipelineConfig
from qtp.data.fetchers.base import Market
from qtp.data.storage import ParquetStorage
from qtp.features.engine import FeatureEngine
from qtp.features.registry import FeatureRegistry
from qtp.models.lgbm import LGBMPipeline
from qtp.validation.metrics import compute_metrics, EvaluationMetrics
from qtp.validation.purged_kfold import PurgedKFold
from qtp.validation.walk_forward import ExpandingWindowCV

# Import feature definitions
import qtp.features.tier1_momentum  # noqa: F401
import qtp.features.tier2_volatility  # noqa: F401

logger = structlog.get_logger()


def load_dataset(config: PipelineConfig) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """Load feature + label dataset."""
    storage = ParquetStorage(project_root / config.data.storage_dir)
    engine = FeatureEngine(FeatureRegistry.instance(), storage)
    market = Market(config.universe.market)

    dataset = engine.build_multi_ticker_dataset(
        tickers=config.universe.tickers,
        market=market,
        as_of=date.today(),
        tiers=config.features.tiers,
        horizon=config.labels.horizon,
    )

    label_cols = ["label_direction", "label_magnitude", "date", "ticker"]
    feature_cols = [c for c in dataset.columns if c not in label_cols]
    X = dataset.select(feature_cols)
    y_dir = dataset["label_direction"].to_numpy()
    y_mag = dataset["label_magnitude"].to_numpy()

    return X, y_dir, y_mag


def run_walk_forward(
    X: pl.DataFrame, y_dir: np.ndarray, y_mag: np.ndarray,
    config: PipelineConfig, cost_bps: float,
) -> list[EvaluationMetrics]:
    """Run Walk-Forward CV (honest, time-respecting)."""
    cv = ExpandingWindowCV(
        min_train_size=config.validation.walk_forward_train_days,
        test_size=config.validation.walk_forward_test_days,
        step_size=config.validation.walk_forward_step_days,
        purge_gap=config.validation.dev_cv_purge_days,
    )

    metrics_list = []
    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X.to_pandas().values)):
        model = LGBMPipeline()
        model.fit(X[train_idx], pl.Series(y_dir[train_idx]), pl.Series(y_mag[train_idx]))

        pred_proba = np.array(model.predict_proba(X[test_idx]))
        pred_mag = np.array(model.predict_magnitude(X[test_idx]))

        m = compute_metrics(
            y_dir[test_idx], pred_proba, y_mag[test_idx], pred_mag,
            commission_bps=cost_bps / 2, slippage_bps=cost_bps / 2,
        )
        metrics_list.append(m)
        print(f"  WF Fold {fold_i}: AUC={m.auc_roc:.4f}  Sharpe={m.sharpe_ratio:.4f}  "
              f"Train={len(train_idx)}  Test={len(test_idx)}")

    return metrics_list


def run_purged_kfold(
    X: pl.DataFrame, y_dir: np.ndarray, y_mag: np.ndarray,
    config: PipelineConfig, cost_bps: float,
) -> list[EvaluationMetrics]:
    """Run PurgedKFold CV (auxiliary, for comparison)."""
    cv = PurgedKFold(
        n_splits=config.validation.dev_cv_splits,
        purge_days=config.validation.dev_cv_purge_days,
    )

    metrics_list = []
    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X.to_pandas().values)):
        model = LGBMPipeline()
        model.fit(X[train_idx], pl.Series(y_dir[train_idx]), pl.Series(y_mag[train_idx]))

        pred_proba = np.array(model.predict_proba(X[test_idx]))
        pred_mag = np.array(model.predict_magnitude(X[test_idx]))

        m = compute_metrics(
            y_dir[test_idx], pred_proba, y_mag[test_idx], pred_mag,
            commission_bps=cost_bps / 2, slippage_bps=cost_bps / 2,
        )
        metrics_list.append(m)
        print(f"  PKF Fold {fold_i}: AUC={m.auc_roc:.4f}  Sharpe={m.sharpe_ratio:.4f}")

    return metrics_list


def format_comparison(
    wf_metrics: list[EvaluationMetrics],
    pkf_metrics: list[EvaluationMetrics],
) -> str:
    """Format side-by-side comparison table."""
    def avg(metrics_list, attr):
        return np.mean([getattr(m, attr) for m in metrics_list])

    def std(metrics_list, attr):
        vals = [getattr(m, attr) for m in metrics_list]
        return np.std(vals) if len(vals) > 1 else 0.0

    lines = [
        "",
        "=" * 72,
        "  HONEST BASELINE — Phase 1.5 Comparison",
        "=" * 72,
        "",
        "  Before fixes (Phase 1 reported): AUC=0.676, Sharpe=5.79, OOS=-82%",
        "",
        f"  {'Metric':<20} {'Walk-Forward':>18} {'PurgedKFold':>18}",
        f"  {'-'*20} {'-'*18} {'-'*18}",
    ]

    for attr, label in [
        ("auc_roc", "AUC-ROC"),
        ("accuracy", "Accuracy"),
        ("sharpe_ratio", "Sharpe (w/ cost)"),
        ("max_drawdown", "Max Drawdown"),
        ("win_rate", "Win Rate"),
        ("profit_factor", "Profit Factor"),
    ]:
        wf_val = avg(wf_metrics, attr)
        wf_sd = std(wf_metrics, attr)
        pkf_val = avg(pkf_metrics, attr)
        pkf_sd = std(pkf_metrics, attr)
        lines.append(
            f"  {label:<20} {wf_val:>8.4f} ±{wf_sd:.4f}  {pkf_val:>8.4f} ±{pkf_sd:.4f}"
        )

    lines.extend([
        "",
        f"  Walk-Forward folds: {len(wf_metrics)}",
        f"  PurgedKFold folds:  {len(pkf_metrics)}",
        "",
        "  Interpretation:",
        "  - Walk-Forward is the HONEST metric (no lookahead bias)",
        "  - If WF AUC ≈ 0.50-0.55, the model has minimal edge",
        "  - If WF Sharpe < 0.5, alpha is marginal after costs",
        "  - PurgedKFold numbers will likely be higher (optimistic bias)",
        "",
        "=" * 72,
    ])
    return "\n".join(lines)


def main():
    config_path = project_root / "configs" / "default.yaml"
    if config_path.exists():
        config = PipelineConfig.from_yaml(config_path)
    else:
        config = PipelineConfig()

    print("Loading dataset...")
    X, y_dir, y_mag = load_dataset(config)
    print(f"Dataset: {X.height} rows, {len(X.columns)} features")
    print(f"Label balance: {y_dir.mean():.2%} positive (direction=1)")
    print()

    # Transaction cost: 20bps round-trip (10bps commission + 10bps slippage)
    COST_BPS = 10.0  # one-way; doubled internally for round-trip

    print("Running Walk-Forward CV (honest evaluation)...")
    wf_metrics = run_walk_forward(X, y_dir, y_mag, config, COST_BPS)
    print()

    print("Running PurgedKFold CV (comparison)...")
    pkf_metrics = run_purged_kfold(X, y_dir, y_mag, config, COST_BPS)

    report = format_comparison(wf_metrics, pkf_metrics)
    print(report)

    # Save report
    output_path = project_root / "data" / "reports" / "honest_baseline.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"\nReport saved to {output_path}")


if __name__ == "__main__":
    main()
