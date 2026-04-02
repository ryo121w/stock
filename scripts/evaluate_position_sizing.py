#!/usr/bin/env python3
"""Evaluate AdvancedPositionSizer vs FixedSizer in walk-forward backtest.

Compares:
  1. FixedSizer  — original confidence-scaled sizing (PositionSizer)
  2. AdvancedSizer — Half-Kelly + volatility targeting + portfolio cap

Metrics: Total Return, Sharpe, MaxDD, Avg Position Size
Uses Phase 3 config (h10_t3%, 7 tickers).
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import qtp.features.tier1_momentum  # noqa: E402, F401
import qtp.features.tier2_volatility  # noqa: E402, F401
import qtp.features.tier3_fundamental  # noqa: E402, F401
import qtp.features.tier4_macro  # noqa: E402, F401
import qtp.features.tier5_alternative  # noqa: E402, F401
from qtp.backtest.signals import AdvancedPositionSizer, PositionSizer  # noqa: E402
from qtp.config import PipelineConfig  # noqa: E402
from qtp.data.fetchers.base import Market  # noqa: E402
from qtp.data.storage import ParquetStorage  # noqa: E402
from qtp.features.engine import FeatureEngine  # noqa: E402
from qtp.features.registry import FeatureRegistry  # noqa: E402
from qtp.models.lgbm import LGBMPipeline  # noqa: E402
from qtp.validation.walk_forward import ExpandingWindowCV  # noqa: E402


def load_config():
    config_path = project_root / "configs" / "default.yaml"
    p3_path = project_root / "configs" / "phase3_best.yaml"
    return PipelineConfig.from_yamls(config_path, p3_path)


def build_dataset(config):
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


def compute_ticker_volatility(dataset: pl.DataFrame, window: int = 63) -> dict[str, pl.DataFrame]:
    """Compute annualized rolling volatility per ticker (63-day std * sqrt(252)).

    Returns a dict: ticker -> DataFrame(date, ann_volatility).
    """
    vol_map: dict[str, pl.DataFrame] = {}
    for ticker in dataset["ticker"].unique().to_list():
        tk_df = dataset.filter(pl.col("ticker") == ticker).sort("date")
        # Use label_magnitude as the period return proxy (forward return).
        # For volatility of past returns, compute from close-to-close returns
        # stored in the feature columns. Use ret_5d / 5 as a daily return proxy,
        # or fall back to label_magnitude.
        if "ret_5d" in tk_df.columns:
            daily_proxy = tk_df["ret_5d"] / 5.0
        else:
            daily_proxy = tk_df["label_magnitude"]

        rolling_std = daily_proxy.rolling_std(window)
        ann_vol = rolling_std * np.sqrt(252)

        vol_map[ticker] = pl.DataFrame(
            {
                "date": tk_df["date"],
                "ann_volatility": ann_vol,
            }
        )
    return vol_map


def run_comparison(config, dataset):
    """Walk-forward backtest comparing Fixed vs Advanced sizing."""

    label_cols = ["label_direction", "label_magnitude", "date", "ticker"]
    feature_cols = [c for c in dataset.columns if c not in label_cols]

    X = dataset.select(feature_cols)
    y_dir = dataset["label_direction"].to_numpy()
    y_mag = dataset["label_magnitude"].to_numpy()
    dates = dataset["date"].to_list()
    tickers = dataset["ticker"].to_list()

    X_np = X.to_pandas().values

    cv = ExpandingWindowCV(
        min_train_size=config.validation.walk_forward_train_days,
        test_size=config.validation.walk_forward_test_days,
        step_size=config.validation.walk_forward_step_days,
        purge_gap=config.validation.dev_cv_purge_days,
    )

    cost_per_trade = 2 * (config.backtest.commission_pct + config.backtest.slippage_pct)
    conf_thresh = 0.55
    horizon = config.labels.horizon

    # Collect OOS predictions
    all_oos = []
    print(f"Running walk-forward ({cv.get_n_splits(X_np)} folds)...")
    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X_np)):
        model = LGBMPipeline()
        model.fit(X[train_idx], pl.Series(y_dir[train_idx]), pl.Series(y_mag[train_idx]))
        pred_proba = np.array(model.predict_proba(X[test_idx]))
        pred_dir = (pred_proba >= 0.5).astype(int)

        for j, idx in enumerate(test_idx):
            all_oos.append(
                {
                    "date": dates[idx],
                    "ticker": tickers[idx],
                    "return": y_mag[idx],
                    "proba": pred_proba[j],
                    "pred_dir": int(pred_dir[j]),
                }
            )
        if fold_i % 10 == 0:
            print(f"  Fold {fold_i}: train={len(train_idx)}, test={len(test_idx)}")

    oos_df = pl.DataFrame(all_oos)
    oos_df = oos_df.sort("date").group_by(["date", "ticker"]).last().sort("date", "ticker")

    # Compute per-ticker annualized volatility
    vol_map = compute_ticker_volatility(dataset)

    # Join volatility to oos_df
    vol_frames = []
    for ticker, vdf in vol_map.items():
        vol_frames.append(vdf.with_columns(pl.lit(ticker).alias("ticker")))
    vol_all = pl.concat(vol_frames)
    oos_df = oos_df.join(vol_all, on=["date", "ticker"], how="left")
    # Fill missing volatility with cross-sectional median
    med_vol = oos_df["ann_volatility"].drop_nulls().median()
    oos_df = oos_df.with_columns(pl.col("ann_volatility").fill_null(med_vol))

    # Non-overlapping periods
    unique_dates = sorted(oos_df["date"].unique().to_list())
    non_overlap_dates = unique_dates[::horizon]
    oos_nolap = oos_df.filter(pl.col("date").is_in(non_overlap_dates))

    print(f"\nOOS: {oos_df.height} rows, non-overlapping periods: {len(non_overlap_dates)}")
    print(f"Date range: {unique_dates[0]} -> {unique_dates[-1]}")

    # ------- Strategy 1: Fixed Sizer (original) -------
    fixed_sizer = PositionSizer(
        max_position_pct=0.05, total_capital=config.backtest.initial_capital
    )

    # For Fixed: position weight = sizer.size(signal) / capital
    # which is: max_position_pct * scale, where scale = (conf - 0.5) / 0.5
    def fixed_weight(conf):
        scale = min(max((conf - 0.5) / 0.5, 0.0), 1.0)
        return fixed_sizer.max_position_pct * scale

    # ------- Strategy 2: Advanced Sizer -------
    adv_sizer = AdvancedPositionSizer(
        max_position_pct=0.05,
        max_portfolio_pct=0.30,
        target_volatility=0.15,
        kelly_fraction=0.5,
    )

    # Evaluate both strategies
    results = {}
    for sizer_name in ["FixedSizer", "AdvancedSizer"]:
        period_returns = []
        position_sizes = []

        for dt in non_overlap_dates:
            day_df = oos_nolap.filter(pl.col("date") == dt)
            signals = day_df.filter((pl.col("proba") >= conf_thresh) & (pl.col("pred_dir") == 1))

            if signals.height == 0:
                period_returns.append(0.0)
                continue

            # Compute weights
            weights = []
            current_exposure = 0.0
            for row in signals.iter_rows(named=True):
                if sizer_name == "FixedSizer":
                    w = fixed_weight(row["proba"])
                else:
                    w = adv_sizer.size(
                        confidence=row["proba"],
                        ticker_volatility=row["ann_volatility"],
                        current_exposure=current_exposure,
                    )
                    current_exposure += w
                weights.append(w)
                position_sizes.append(w)

            total_weight = sum(weights)
            if total_weight == 0:
                period_returns.append(0.0)
                continue

            # Weighted return (net of costs for each signal)
            weighted_ret = 0.0
            for w, row in zip(weights, signals.iter_rows(named=True)):
                net_ret = row["return"] - cost_per_trade
                weighted_ret += w * net_ret

            period_returns.append(weighted_ret)

        port_returns = np.array(period_returns)
        equity = (1 + port_returns).cumprod()

        # Metrics
        active = port_returns[port_returns != 0]
        if len(active) > 0 and active.std() > 0:
            periods_per_year = 252 / horizon
            sharpe = (active.mean() / active.std()) * (periods_per_year**0.5)
        else:
            sharpe = 0.0

        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        max_dd = drawdown.min()
        total_return = equity[-1] - 1
        avg_pos = float(np.mean(position_sizes)) if position_sizes else 0.0

        results[sizer_name] = {
            "total_return": total_return,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "avg_position_size": avg_pos,
            "n_signals": len(position_sizes),
            "equity_curve": equity,
        }

    return results


def format_report(results):
    lines = [
        "",
        "=" * 75,
        "  POSITION SIZING COMPARISON — Fixed vs Advanced (Half-Kelly + VolTarget)",
        "=" * 75,
        "",
        f"  {'Sizer':<20} {'Return':>10} {'Sharpe':>8} {'MaxDD':>10} {'AvgSize':>10} {'Signals':>8}",
        f"  {'-' * 20} {'-' * 10} {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 8}",
    ]

    for name, m in results.items():
        lines.append(
            f"  {name:<20} {m['total_return']:>+9.2%} {m['sharpe']:>8.2f} "
            f"{m['max_drawdown']:>+9.2%} {m['avg_position_size']:>9.4f} "
            f"{m['n_signals']:>8}"
        )

    lines.extend(
        [
            "",
            "  AdvancedSizer: Half-Kelly * vol-targeting, max 5% per position, 30% portfolio cap",
            "  FixedSizer:    confidence-scaled, max 5% per position, no portfolio cap",
            "",
            "=" * 75,
        ]
    )
    return "\n".join(lines)


def main():
    print("Loading Phase 3 config...")
    config = load_config()
    print(f"  Tickers: {config.universe.tickers}")
    print(f"  Horizon: {config.labels.horizon}, Threshold: {config.labels.direction_threshold}")
    print()

    print("Building dataset...")
    dataset = build_dataset(config)
    print(f"  {dataset.height} rows, {len(dataset.columns)} columns")
    print()

    results = run_comparison(config, dataset)
    report = format_report(results)
    print(report)

    output_path = project_root / "data" / "reports" / "position_sizing_comparison.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    main()
