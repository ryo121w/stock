#!/usr/bin/env python3
"""Feature Importance Analysis & Selection.

1. Loads dataset with Phase 3 config (h10_t3%, 7 tickers, Tier 1-5)
2. Trains LGBMPipeline on 80% of data
3. Computes permutation importance on held-out 20%
4. Ranks features by importance
5. Tests accuracy with top-N features (N=10, 15, 20, 25, 30, all)
6. For each N, runs mini walk-forward (step=126) and reports accuracy at conf 55%+
7. Prints clear comparison table
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402

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
from qtp.models.lgbm import LGBMPipeline  # noqa: E402
from qtp.validation.walk_forward import ExpandingWindowCV  # noqa: E402


def load_config() -> PipelineConfig:
    """Load Phase 3 config."""
    default_path = project_root / "configs" / "default.yaml"
    p3_path = project_root / "configs" / "phase3_best.yaml"
    if p3_path.exists() and default_path.exists():
        return PipelineConfig.from_yamls(default_path, p3_path)
    if p3_path.exists():
        return PipelineConfig.from_yaml(p3_path)
    return PipelineConfig.from_yaml(default_path)


def build_dataset(config: PipelineConfig) -> pl.DataFrame:
    """Build multi-ticker dataset."""
    storage = ParquetStorage(project_root / config.data.storage_dir)
    engine = FeatureEngine(FeatureRegistry.instance(), storage)
    market = Market(config.universe.market)
    return engine.build_multi_ticker_dataset(
        tickers=config.universe.tickers,
        market=market,
        as_of=date.today(),
        tiers=config.features.tiers,
        horizon=config.labels.horizon,
        direction_threshold=config.labels.direction_threshold,
    )


def split_features_labels(dataset: pl.DataFrame):
    """Split dataset into X, y_dir, y_mag, and feature column names."""
    label_cols = ["label_direction", "label_magnitude", "date", "ticker"]
    feature_cols = [c for c in dataset.columns if c not in label_cols]
    X = dataset.select(feature_cols)
    y_dir = dataset["label_direction"].to_numpy()
    y_mag = dataset["label_magnitude"].to_numpy()
    return X, y_dir, y_mag, feature_cols


def compute_permutation_importance(
    model: LGBMPipeline,
    X_test: pl.DataFrame,
    y_test: np.ndarray,
    n_repeats: int = 10,
) -> dict[str, float]:
    """Compute permutation importance on held-out test set."""
    X_pd = X_test.to_pandas()

    result = permutation_importance(
        model.clf,
        X_pd,
        y_test,
        n_repeats=n_repeats,
        random_state=42,
        scoring="accuracy",
        n_jobs=-1,
    )

    importance = {}
    for i, col in enumerate(X_test.columns):
        importance[col] = result.importances_mean[i]

    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


def run_walk_forward_with_features(
    dataset: pl.DataFrame,
    feature_subset: list[str],
    config: PipelineConfig,
    conf_threshold: float = 0.55,
    step_size: int = 126,
) -> dict:
    """Run mini walk-forward CV with a subset of features.

    Returns dict with accuracy, n_trades, trade_accuracy, n_folds.
    """
    X = dataset.select(feature_subset)
    y_dir = dataset["label_direction"].to_numpy()
    y_mag = dataset["label_magnitude"].to_numpy()

    cv = ExpandingWindowCV(
        min_train_size=config.validation.walk_forward_train_days,
        test_size=config.validation.walk_forward_test_days,
        step_size=step_size,
        purge_gap=config.validation.dev_cv_purge_days,
    )

    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    all_y_mag_true = []
    n_folds = 0

    for train_idx, test_idx in cv.split(X.to_pandas().values):
        model = LGBMPipeline()
        model.fit(
            X[train_idx],
            pl.Series(y_dir[train_idx]),
            pl.Series(y_mag[train_idx]),
        )

        pred_proba = np.array(model.predict_proba(X[test_idx]))
        pred_dir = (pred_proba >= 0.5).astype(int)

        all_y_true.extend(y_dir[test_idx].tolist())
        all_y_pred.extend(pred_dir.tolist())
        all_y_proba.extend(pred_proba.tolist())
        all_y_mag_true.extend(y_mag[test_idx].tolist())
        n_folds += 1

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_proba = np.array(all_y_proba)
    all_y_mag_true = np.array(all_y_mag_true)

    # Overall accuracy
    overall_acc = accuracy_score(all_y_true, all_y_pred) if len(all_y_true) > 0 else 0.0

    # Accuracy at confidence threshold
    conf_mask = all_y_proba >= conf_threshold
    n_trades = int(conf_mask.sum())
    if n_trades > 0:
        trade_acc = accuracy_score(all_y_true[conf_mask], all_y_pred[conf_mask])
        # PnL: long when predicted up with high confidence
        trade_returns = all_y_mag_true[conf_mask] * np.where(all_y_pred[conf_mask] == 1, 1, -1)
        cost_per_trade = 2 * (config.backtest.commission_pct + config.backtest.slippage_pct)
        net_returns = trade_returns - cost_per_trade
        win_rate = (net_returns > 0).mean()
        mean_return = net_returns.mean() * 100  # in percent
    else:
        trade_acc = 0.0
        win_rate = 0.0
        mean_return = 0.0

    return {
        "n_features": len(feature_subset),
        "n_folds": n_folds,
        "overall_acc": overall_acc,
        "n_trades": n_trades,
        "trade_acc": trade_acc,
        "win_rate": win_rate,
        "mean_return_pct": mean_return,
        "total_predictions": len(all_y_true),
    }


def main():
    print("=" * 72)
    print("  FEATURE IMPORTANCE ANALYSIS & SELECTION")
    print("=" * 72)

    # 1. Load config and dataset
    print("\n[1/4] Loading Phase 3 config and dataset...")
    config = load_config()
    dataset = build_dataset(config)

    X, y_dir, y_mag, feature_cols = split_features_labels(dataset)
    print(f"  Dataset: {X.height} rows, {len(feature_cols)} features")
    print(f"  Tickers: {config.universe.tickers}")
    print(f"  Label balance: {y_dir.mean():.2%} positive")

    # 2. Train on 80%, compute permutation importance on 20%
    print("\n[2/4] Training model and computing permutation importance...")
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_dir_train, y_dir_test = y_dir[:split_idx], y_dir[split_idx:]
    y_mag_train = y_mag[:split_idx]

    model = LGBMPipeline()
    model.fit(X_train, pl.Series(y_dir_train), pl.Series(y_mag_train))

    importance = compute_permutation_importance(model, X_test, y_dir_test, n_repeats=10)

    # 3. Print feature ranking
    print("\n" + "=" * 72)
    print("  FEATURE IMPORTANCE RANKING (Permutation Importance)")
    print("=" * 72)
    print(f"  {'Rank':<5} {'Feature':<40} {'Importance':>12}")
    print(f"  {'-' * 5} {'-' * 40} {'-' * 12}")
    for rank, (feat, imp) in enumerate(importance.items(), 1):
        marker = " *" if imp < 0 else ""
        print(f"  {rank:<5} {feat:<40} {imp:>12.6f}{marker}")

    # Identify noise features (negative or zero importance)
    noise_features = [f for f, imp in importance.items() if imp <= 0]
    useful_features = [f for f, imp in importance.items() if imp > 0]
    print(f"\n  Noise features (importance <= 0): {len(noise_features)}")
    if noise_features:
        print(f"    {', '.join(noise_features)}")
    print(f"  Useful features (importance > 0): {len(useful_features)}")

    # 4. Test accuracy with top-N features via mini walk-forward
    print("\n" + "=" * 72)
    print("  WALK-FORWARD ACCURACY BY FEATURE COUNT (step=126, conf>=55%)")
    print("=" * 72)

    sorted_features = list(importance.keys())  # already sorted by importance desc
    n_values = [10, 15, 20, 25, 30, len(sorted_features)]
    # Remove duplicates and values > total features
    n_values = sorted(set(min(n, len(sorted_features)) for n in n_values))

    results = []
    for n in n_values:
        subset = sorted_features[:n]
        label = f"Top {n}" if n < len(sorted_features) else f"All ({n})"
        print(f"\n  Testing {label} features...", flush=True)
        r = run_walk_forward_with_features(
            dataset, subset, config, conf_threshold=0.55, step_size=126
        )
        r["label"] = label
        results.append(r)
        print(
            f"    Folds={r['n_folds']}  OOS_Acc={r['overall_acc']:.4f}  "
            f"Trades={r['n_trades']}  Trade_Acc={r['trade_acc']:.4f}  "
            f"WinRate={r['win_rate']:.4f}  MeanRet={r['mean_return_pct']:.4f}%"
        )

    # Print comparison table
    print("\n" + "=" * 72)
    print("  COMPARISON TABLE")
    print("=" * 72)
    header = (
        f"  {'Features':<14} {'N_Feat':>6} {'OOS_Acc':>8} {'Trades':>7} "
        f"{'Trade_Acc':>10} {'WinRate':>8} {'MeanRet%':>9}"
    )
    print(header)
    print(f"  {'-' * 14} {'-' * 6} {'-' * 8} {'-' * 7} {'-' * 10} {'-' * 8} {'-' * 9}")
    for r in results:
        print(
            f"  {r['label']:<14} {r['n_features']:>6} {r['overall_acc']:>8.4f} "
            f"{r['n_trades']:>7} {r['trade_acc']:>10.4f} {r['win_rate']:>8.4f} "
            f"{r['mean_return_pct']:>9.4f}"
        )

    # Find best configuration
    best = max(results, key=lambda r: r["trade_acc"] if r["n_trades"] >= 10 else 0)
    print(
        f"\n  Best: {best['label']} (Trade Acc={best['trade_acc']:.4f}, "
        f"WinRate={best['win_rate']:.4f}, {best['n_trades']} trades)"
    )

    # Output recommended feature list
    best_n = best["n_features"]
    recommended_features = sorted_features[:best_n]
    print(f"\n  Recommended features ({best_n}):")
    for f in recommended_features:
        print(f"    - {f}")

    # Save results for config generation
    output_path = project_root / "data" / "reports" / "feature_selection.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Feature Selection Results",
        f"Date: {date.today()}",
        f"Best N: {best_n}",
        f"Best Trade Acc: {best['trade_acc']:.4f}",
        "",
        "Recommended features:",
    ]
    lines.extend([f"  - {f}" for f in recommended_features])
    lines.append("")
    lines.append("Full ranking:")
    for rank, (feat, imp) in enumerate(importance.items(), 1):
        lines.append(f"  {rank}. {feat}: {imp:.6f}")
    output_path.write_text("\n".join(lines))
    print(f"\n  Report saved to {output_path}")

    print("\n" + "=" * 72)
    return recommended_features, results


if __name__ == "__main__":
    main()
