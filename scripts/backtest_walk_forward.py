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
    p3_path = project_root / "configs" / "phase3_best.yaml"
    p2_path = project_root / "configs" / "phase2_experiment.yaml"
    if p3_path.exists():
        return PipelineConfig.from_yamls(config_path, p3_path)
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

    # Collect all OOS predictions (keep ticker information)
    all_oos_dates = []
    all_oos_tickers = []
    all_oos_returns = []
    all_oos_proba = []
    all_oos_pred_dir = []

    tickers = dataset["ticker"].to_list()

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
        fold_tickers = [tickers[i] for i in test_idx]

        all_oos_dates.extend(fold_dates)
        all_oos_tickers.extend(fold_tickers)
        all_oos_returns.extend(actual_mag.tolist())
        all_oos_proba.extend(pred_proba.tolist())
        all_oos_pred_dir.extend(pred_dir.tolist())

        if fold_i % 10 == 0:
            print(f"  Fold {fold_i}: train={len(train_idx)}, test={len(test_idx)}")

    # Build DataFrame with date + ticker (no deduplication by date alone)
    oos_df = pl.DataFrame(
        {
            "date": all_oos_dates,
            "ticker": all_oos_tickers,
            "return": np.array(all_oos_returns),
            "proba": np.array(all_oos_proba),
            "pred_dir": np.array(all_oos_pred_dir),
        }
    )

    # Deduplicate: when multiple folds cover the same date+ticker,
    # keep only the prediction from the LATEST model (largest training set)
    oos_df = oos_df.sort("date").group_by(["date", "ticker"]).last().sort("date", "ticker")

    horizon = config.labels.horizon
    unique_dates = sorted(oos_df["date"].unique().to_list())
    unique_tickers = sorted(oos_df["ticker"].unique().to_list())

    print(
        f"\nDeduplicated OOS: {oos_df.height} rows ({len(unique_tickers)} tickers x {len(unique_dates)} dates)"
    )
    print(f"OOS date range: {unique_dates[0]} → {unique_dates[-1]}")
    print(f"Mean period return: {oos_df['return'].mean():.4%} ({horizon}-day)")

    # Non-overlapping periods: subsample every horizon-th date so that
    # 10-day returns don't overlap when compounded.
    non_overlap_dates = unique_dates[::horizon]
    oos_nolap = oos_df.filter(pl.col("date").is_in(non_overlap_dates))
    n_periods = len(non_overlap_dates)
    print(f"Non-overlapping periods: {n_periods} (every {horizon}th date)")

    # Compute strategy returns as equal-weight portfolio
    results = {}
    for name, params in strategies.items():
        conf_thresh = params["conf"]

        # Add signal column: 1 if long, 0 if flat
        strat_df = oos_nolap.with_columns(
            pl.when((pl.col("proba") >= conf_thresh) & (pl.col("pred_dir") == 1))
            .then(1.0)
            .otherwise(0.0)
            .alias("signal")
        )

        # Compute per-ticker costs: charge cost when signal changes
        strat_df = strat_df.sort("ticker", "date")
        strat_df = strat_df.with_columns(
            pl.col("signal")
            .diff()
            .abs()
            .fill_null(pl.col("signal"))  # first row: cost if entering
            .over("ticker")
            .alias("trade_flag")
        )

        # Per-row net return = signal * return - trade_flag * cost
        strat_df = strat_df.with_columns(
            (pl.col("signal") * pl.col("return") - pl.col("trade_flag") * cost_per_trade).alias(
                "net_return"
            )
        )

        # Portfolio return per date = mean of net_return across tickers
        portfolio_daily = (
            strat_df.group_by("date")
            .agg(
                pl.col("net_return").mean().alias("port_return"),
                pl.col("signal").mean().alias("avg_signal"),
                pl.col("trade_flag").sum().alias("trades_today"),
            )
            .sort("date")
        )

        port_returns = portfolio_daily["port_return"].to_numpy()
        avg_signals = portfolio_daily["avg_signal"].to_numpy()

        # Equity curve (each period is horizon-day, non-overlapping)
        equity = (1 + port_returns).cumprod()

        # Buy & Hold benchmark = equal-weight portfolio of all tickers
        bh_daily = (
            oos_nolap.group_by("date").agg(pl.col("return").mean().alias("bh_return")).sort("date")
        )
        bh_returns = bh_daily["bh_return"].to_numpy()
        bh_equity = (1 + bh_returns).cumprod()

        # Metrics
        total_trades_flag = strat_df["trade_flag"].sum()
        n_trades = int(total_trades_flag / 2)  # entry + exit = 2 per round-trip
        pct_in_market = float(avg_signals.mean()) * 100

        active_returns = port_returns[avg_signals > 0]
        if len(active_returns) > 0 and active_returns.std() > 0:
            # Annualize: periods_per_year = 252 / horizon
            periods_per_year = 252 / horizon
            sharpe = (active_returns.mean() / active_returns.std()) * (periods_per_year**0.5)
        else:
            sharpe = 0.0

        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        max_dd = drawdown.min()

        win_rate = (active_returns > 0).mean() if len(active_returns) > 0 else 0

        total_return = equity[-1] - 1
        bh_return = bh_equity[-1] - 1
        excess_return = total_return - bh_return

        total_cost = float(strat_df["trade_flag"].sum()) * cost_per_trade

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

    return results, unique_dates, oos_df["return"].to_numpy()


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
