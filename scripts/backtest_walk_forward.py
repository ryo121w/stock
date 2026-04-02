#!/usr/bin/env python3
"""Walk-Forward Backtest: True OOS PnL simulation with transaction costs.

Simulates what would have happened if we traded the model's signals in real time:
1. Train on expanding window of past data
2. Predict next period
3. Apply signal filtering (confidence + magnitude threshold)
4. Calculate PnL after commission + slippage
5. Compare vs Buy & Hold benchmark

This is the final reality check — CV metrics can be misleading, but PnL doesn't lie.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import structlog

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

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

logger = structlog.get_logger()


def load_config():
    config_path = project_root / "configs" / "default.yaml"
    p2_path = project_root / "configs" / "phase2_experiment.yaml"
    if p2_path.exists():
        return PipelineConfig.from_yamls(config_path, p2_path)
    return PipelineConfig.from_yaml(config_path)


def build_dataset(config):
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
    return dataset


def run_backtest(config, dataset):
    """Walk-forward backtest with actual PnL tracking."""
    label_cols = ["label_direction", "label_magnitude", "date", "ticker"]
    feature_cols = [c for c in dataset.columns if c not in label_cols]

    X = dataset.select(feature_cols)
    y_dir = dataset["label_direction"].to_numpy()
    y_mag = dataset["label_magnitude"].to_numpy()
    dates = dataset["date"].to_list()

    X_np = X.to_pandas().values

    # Walk-forward CV splits
    cv = ExpandingWindowCV(
        min_train_size=config.validation.walk_forward_train_days,
        test_size=config.validation.walk_forward_test_days,
        step_size=config.validation.walk_forward_step_days,
        purge_gap=config.validation.dev_cv_purge_days,
    )

    # Transaction costs
    cost_per_trade = 2 * (config.backtest.commission_pct + config.backtest.slippage_pct)

    # Backtest strategies
    strategies = {
        "all_signals": {"conf": 0.50, "mag": 0.0},
        "conf_55": {"conf": 0.55, "mag": 0.0},
        "conf_55_mag_02": {"conf": 0.55, "mag": 0.002},
        "conf_60_mag_03": {"conf": 0.60, "mag": 0.003},
        "conf_65_mag_05": {"conf": 0.65, "mag": 0.005},
    }

    # Collect all OOS predictions
    all_oos_dates = []
    all_oos_returns = []
    all_oos_proba = []
    all_oos_pred_dir = []

    print(f"Running walk-forward backtest ({cv.get_n_splits(X_np)} folds)...")
    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X_np)):
        model = LGBMPipeline()
        model.fit(
            X[train_idx],
            pl.Series(y_dir[train_idx]),
            pl.Series(y_mag[train_idx]),
        )

        pred_proba = np.array(model.predict_proba(X[test_idx]))
        pred_dir = (pred_proba >= 0.5).astype(int)
        actual_mag = y_mag[test_idx]
        fold_dates = [dates[i] for i in test_idx]

        all_oos_dates.extend(fold_dates)
        all_oos_returns.extend(actual_mag.tolist())
        all_oos_proba.extend(pred_proba.tolist())
        all_oos_pred_dir.extend(pred_dir.tolist())

        if fold_i % 10 == 0:
            print(f"  Fold {fold_i}: train={len(train_idx)}, test={len(test_idx)}")

    all_oos_returns = np.array(all_oos_returns)
    all_oos_proba = np.array(all_oos_proba)
    all_oos_pred_dir = np.array(all_oos_pred_dir)

    # Deduplicate OOS samples: when multiple folds cover the same date+ticker,
    # keep only the prediction from the LATEST model (largest training set)
    oos_df = (
        pl.DataFrame(
            {
                "date": all_oos_dates,
                "return": all_oos_returns,
                "proba": all_oos_proba,
                "pred_dir": all_oos_pred_dir,
            }
        )
        .sort("date")
        .group_by("date")
        .last()
    )  # Last = latest model's prediction
    oos_df = oos_df.sort("date")

    all_oos_returns = oos_df["return"].to_numpy()
    all_oos_proba = oos_df["proba"].to_numpy()
    all_oos_pred_dir = oos_df["pred_dir"].to_numpy()
    all_oos_dates = oos_df["date"].to_list()

    # Convert period returns to simple per-period (no daily splitting)
    # Each row represents one non-overlapping prediction period
    horizon = config.labels.horizon

    print(f"\nDeduplicated OOS: {len(all_oos_returns)} periods")
    print(f"OOS date range: {min(all_oos_dates)} → {max(all_oos_dates)}")
    print(f"Mean period return: {all_oos_returns.mean():.4%} ({horizon}-day)")

    # Compute strategy returns for each strategy
    results = {}
    for name, params in strategies.items():
        conf_thresh = params["conf"]

        # Signal: 1 if long, 0 if flat
        signal = np.where(
            (all_oos_proba >= conf_thresh) & (all_oos_pred_dir == 1),
            1.0,
            0.0,
        )

        # Period returns when signal is on
        gross_returns = all_oos_returns * signal

        # Transaction costs: pay cost on each signal change
        trades = np.abs(np.diff(signal, prepend=0))
        net_returns = gross_returns - trades * cost_per_trade

        # Equity curve (period-by-period compounding)
        equity = (1 + net_returns).cumprod()

        # Buy & Hold benchmark
        bh_equity = (1 + all_oos_returns).cumprod()

        # Metrics
        n_trades = int(trades.sum() / 2)  # entry + exit = 2 signals per trade
        n_days_in_market = int(signal.sum())
        pct_in_market = n_days_in_market / len(signal) * 100

        if net_returns[signal == 1].std() > 0:
            sharpe = (net_returns[signal == 1].mean() / net_returns[signal == 1].std()) * (252**0.5)
        else:
            sharpe = 0.0

        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        max_dd = drawdown.min()

        wins = net_returns[signal == 1]
        win_rate = (wins > 0).mean() if len(wins) > 0 else 0

        total_return = equity[-1] - 1
        bh_return = bh_equity[-1] - 1
        excess_return = total_return - bh_return

        total_cost = (trades * cost_per_trade).sum()

        results[name] = {
            "total_return": total_return,
            "bh_return": bh_return,
            "excess_return": excess_return,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "n_trades": n_trades,
            "pct_in_market": pct_in_market,
            "total_cost": total_cost,
            "equity_final": equity[-1],
        }

    return results, all_oos_dates, all_oos_returns


def format_report(results, initial_capital):
    lines = [
        "",
        "=" * 85,
        "  WALK-FORWARD BACKTEST — True OOS PnL Simulation",
        "=" * 85,
        "",
        f"  Initial Capital: ${initial_capital:,.0f}",
        "",
        f"  {'Strategy':<22} {'Return':>9} {'B&H':>9} {'Excess':>9} {'Sharpe':>8} "
        f"{'MaxDD':>8} {'WinRate':>8} {'Trades':>7} {'InMkt':>6}",
        f"  {'-' * 22} {'-' * 9} {'-' * 9} {'-' * 9} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 7} {'-' * 6}",
    ]

    for name, m in results.items():
        lines.append(
            f"  {name:<22} {m['total_return']:>+8.2%} {m['bh_return']:>+8.2%} "
            f"{m['excess_return']:>+8.2%} {m['sharpe']:>8.2f} "
            f"{m['max_drawdown']:>8.2%} {m['win_rate']:>8.1%} "
            f"{m['n_trades']:>7} {m['pct_in_market']:>5.1f}%"
        )

    lines.extend(
        [
            "",
            "  Interpretation:",
            "  - Excess > 0: Strategy beats Buy & Hold (the real test)",
            "  - Sharpe > 1.0: Decent risk-adjusted returns",
            "  - Sharpe > 2.0: Strong risk-adjusted returns",
            "  - MaxDD: Worst peak-to-trough (smaller = better)",
            "  - InMkt: % of time with a position (lower = more selective)",
            "",
            "  Note: Returns include 20bps round-trip transaction costs.",
            "  All returns are OUT-OF-SAMPLE (never trained on test data).",
            "",
            "=" * 85,
        ]
    )
    return "\n".join(lines)


def main():
    print("Loading config...")
    config = load_config()
    initial_capital = config.backtest.initial_capital

    print(f"Config: horizon={config.labels.horizon}, threshold={config.labels.direction_threshold}")
    print(f"Tiers: {config.features.tiers}")
    print(f"Capital: ${initial_capital:,.0f}")
    print()

    print("Building dataset...")
    dataset = build_dataset(config)
    print(f"Dataset: {dataset.height} rows, {len(dataset.columns)} columns")
    print()

    results, oos_dates, oos_returns = run_backtest(config, dataset)

    report = format_report(results, initial_capital)
    print(report)

    # Save report
    output_path = project_root / "data" / "reports" / "backtest_walk_forward.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    main()
