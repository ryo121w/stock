#!/usr/bin/env python3
"""Evaluate risk management rules: stop-loss, take-profit, trailing stop.

Compares three strategies using walk-forward OOS predictions + daily OHLCV:
  1. No Risk Management — hold for full horizon (10 days)
  2. Stop-Loss Only — -2% stop-loss, no take-profit
  3. Full TradeManager — stop-loss + take-profit + trailing stop

For each OOS signal, loads daily OHLCV for that ticker during the holding
period and simulates day-by-day exit checks.

Metrics: Avg PnL, Win Rate, Max Single-Trade Loss, Total Return, Sharpe.
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
from qtp.backtest.risk_management import TradeManager  # noqa: E402
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
    if p3_path.exists():
        return PipelineConfig.from_yamls(config_path, p3_path)
    return PipelineConfig.from_yaml(config_path)


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


def load_ohlcv(tickers: list[str]) -> dict[str, pl.DataFrame]:
    """Load daily OHLCV for each ticker from raw parquet files."""
    ohlcv = {}
    raw_dir = project_root / "data" / "raw" / "us"
    for ticker in tickers:
        path = raw_dir / f"{ticker}.parquet"
        if path.exists():
            df = pl.read_parquet(path).sort("date")
            ohlcv[ticker] = df
    return ohlcv


def generate_oos_predictions(config, dataset):
    """Run walk-forward CV and collect OOS predictions."""
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

    all_oos = []
    n_folds = cv.get_n_splits(X_np)
    print(f"Running walk-forward ({n_folds} folds)...")
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
                    "actual_return": y_mag[idx],
                    "proba": pred_proba[j],
                    "pred_dir": int(pred_dir[j]),
                }
            )
        if fold_i % 10 == 0:
            print(f"  Fold {fold_i}/{n_folds}: train={len(train_idx)}, test={len(test_idx)}")

    oos_df = pl.DataFrame(all_oos)
    # De-duplicate: keep last prediction for each (date, ticker) pair
    oos_df = oos_df.sort("date").group_by(["date", "ticker"]).last().sort("date", "ticker")
    return oos_df


def simulate_trade(
    ohlcv_df: pl.DataFrame,
    entry_date: date,
    horizon: int,
    trade_manager: TradeManager | None,
) -> dict:
    """Simulate a single trade using daily OHLCV data.

    Parameters
    ----------
    ohlcv_df : pl.DataFrame
        Daily OHLCV for the ticker, sorted by date.
    entry_date : date
        Signal date (position entered at next day's open, or this day's close).
    horizon : int
        Maximum holding period in trading days.
    trade_manager : TradeManager or None
        If None, hold for full horizon (no risk management).

    Returns
    -------
    dict with keys: exit_pnl, exit_reason, days_held
    """
    # Find the entry point: use the close on entry_date
    mask = ohlcv_df["date"] >= entry_date
    future_bars = ohlcv_df.filter(mask)

    if future_bars.height < 2:
        return {"exit_pnl": 0.0, "exit_reason": "insufficient_data", "days_held": 0}

    entry_price = future_bars["close"][0]
    peak_price = entry_price

    # Walk through subsequent days
    max_days = min(horizon, future_bars.height - 1)

    for day_i in range(1, max_days + 1):
        row = future_bars.row(day_i, named=True)
        high = row["high"]
        low = row["low"]
        close = row["close"]

        # Update peak with intraday high
        peak_price = max(peak_price, high)

        if trade_manager is not None:
            # Check intraday low for stop-loss (worst case)
            exit_sig = trade_manager.check_exit(entry_price, low, peak_price, day_i)
            if exit_sig is not None and exit_sig.reason == "stop_loss":
                # Assume exit at stop-loss price (not the low)
                stop_price = entry_price * (1 + trade_manager.stop_loss_pct)
                actual_exit = max(low, stop_price)
                pnl = (actual_exit - entry_price) / entry_price
                return {"exit_pnl": pnl, "exit_reason": "stop_loss", "days_held": day_i}

            # Check intraday high for take-profit
            exit_sig = trade_manager.check_exit(entry_price, high, peak_price, day_i)
            if exit_sig is not None and exit_sig.reason == "take_profit":
                tp_price = entry_price * (1 + trade_manager.take_profit_pct)
                actual_exit = min(high, tp_price)
                pnl = (actual_exit - entry_price) / entry_price
                return {"exit_pnl": pnl, "exit_reason": "take_profit", "days_held": day_i}

            # Check close for trailing stop and max hold
            exit_sig = trade_manager.check_exit(entry_price, close, peak_price, day_i)
            if exit_sig is not None:
                pnl = (close - entry_price) / entry_price
                return {"exit_pnl": pnl, "exit_reason": exit_sig.reason, "days_held": day_i}

    # Held for full horizon — exit at close
    final_close = future_bars["close"][max_days]
    pnl = (final_close - entry_price) / entry_price
    return {"exit_pnl": pnl, "exit_reason": "horizon_end", "days_held": max_days}


def evaluate_strategy(
    oos_signals: pl.DataFrame,
    ohlcv: dict[str, pl.DataFrame],
    horizon: int,
    trade_manager: TradeManager | None,
    strategy_name: str,
) -> dict:
    """Evaluate a risk management strategy across all OOS signals."""
    trades = []

    for row in oos_signals.iter_rows(named=True):
        ticker = row["ticker"]
        entry_date = row["date"]

        if ticker not in ohlcv:
            continue

        result = simulate_trade(ohlcv[ticker], entry_date, horizon, trade_manager)
        result["ticker"] = ticker
        result["entry_date"] = entry_date
        result["actual_return"] = row["actual_return"]
        trades.append(result)

    if not trades:
        print(f"  {strategy_name}: No trades executed")
        return {}

    trades_df = pl.DataFrame(trades)
    pnls = trades_df["exit_pnl"].to_numpy()

    n_trades = len(pnls)
    avg_pnl = float(np.mean(pnls))
    med_pnl = float(np.median(pnls))
    win_rate = float(np.mean(pnls > 0))
    max_loss = float(np.min(pnls))
    max_gain = float(np.max(pnls))
    total_return = float(np.sum(pnls))
    std_pnl = float(np.std(pnls))
    sharpe = avg_pnl / std_pnl * np.sqrt(252 / horizon) if std_pnl > 0 else 0.0
    avg_days = float(trades_df["days_held"].mean())

    # Exit reason distribution
    exit_reasons = trades_df.group_by("exit_reason").len().sort("len", descending=True)

    print(f"\n{'=' * 60}")
    print(f"  Strategy: {strategy_name}")
    print(f"{'=' * 60}")
    print(f"  Trades:         {n_trades}")
    print(f"  Avg PnL:        {avg_pnl:+.4f} ({avg_pnl * 100:+.2f}%)")
    print(f"  Median PnL:     {med_pnl:+.4f} ({med_pnl * 100:+.2f}%)")
    print(f"  Win Rate:       {win_rate:.1%}")
    print(f"  Max Loss:       {max_loss:+.4f} ({max_loss * 100:+.2f}%)")
    print(f"  Max Gain:       {max_gain:+.4f} ({max_gain * 100:+.2f}%)")
    print(f"  Total Return:   {total_return:+.4f} ({total_return * 100:+.2f}%)")
    print(f"  Sharpe (ann.):  {sharpe:.2f}")
    print(f"  Avg Hold Days:  {avg_days:.1f}")
    print("\n  Exit Reasons:")
    for row in exit_reasons.iter_rows(named=True):
        pct = row["len"] / n_trades * 100
        print(f"    {row['exit_reason']:20s} {row['len']:5d} ({pct:.1f}%)")

    return {
        "strategy": strategy_name,
        "n_trades": n_trades,
        "avg_pnl": avg_pnl,
        "median_pnl": med_pnl,
        "win_rate": win_rate,
        "max_loss": max_loss,
        "max_gain": max_gain,
        "total_return": total_return,
        "sharpe": sharpe,
        "avg_days": avg_days,
    }


def main():
    print("=" * 60)
    print("  Risk Management Evaluation")
    print("=" * 60)

    # 1. Load config and build dataset
    config = load_config()
    horizon = config.labels.horizon
    print(f"\nConfig: horizon={horizon}, tickers={config.universe.tickers}")

    print("\nBuilding dataset...")
    dataset = build_dataset(config)
    print(f"Dataset: {dataset.height} rows, {dataset.width} columns")

    # 2. Generate OOS predictions via walk-forward
    oos_df = generate_oos_predictions(config, dataset)

    # 3. Filter to buy signals with sufficient confidence
    conf_thresh = 0.55
    signals = oos_df.filter((pl.col("proba") >= conf_thresh) & (pl.col("pred_dir") == 1))

    # Non-overlapping: sample every `horizon` days to avoid overlap
    unique_dates = sorted(signals["date"].unique().to_list())
    non_overlap_dates = unique_dates[::horizon]
    signals = signals.filter(pl.col("date").is_in(non_overlap_dates))

    print(f"\nOOS signals (conf >= {conf_thresh}): {signals.height}")
    print(f"Non-overlapping signal dates: {len(non_overlap_dates)}")

    # 4. Load daily OHLCV
    tickers = config.universe.tickers
    print(f"\nLoading OHLCV for {len(tickers)} tickers...")
    ohlcv = load_ohlcv(tickers)
    for tk, df in ohlcv.items():
        print(f"  {tk}: {df.height} bars ({df['date'].min()} to {df['date'].max()})")

    # 5. Evaluate three strategies
    strategies = [
        ("No Risk Mgmt (Hold Full Horizon)", None),
        (
            "Stop-Loss Only (-2%)",
            TradeManager(
                stop_loss_pct=-0.02,
                take_profit_pct=1.0,  # effectively disabled
                trailing_stop_pct=1.0,  # effectively disabled
                max_hold_days=horizon,
            ),
        ),
        (
            "Full TradeManager (SL -2%, TP +5%, TS 3%)",
            TradeManager(
                stop_loss_pct=-0.02,
                take_profit_pct=0.05,
                trailing_stop_pct=0.03,
                max_hold_days=horizon,
            ),
        ),
    ]

    results = []
    for name, tm in strategies:
        r = evaluate_strategy(signals, ohlcv, horizon, tm, name)
        if r:
            results.append(r)

    # 6. Summary comparison table
    if results:
        print("\n" + "=" * 80)
        print("  COMPARISON SUMMARY")
        print("=" * 80)
        header = f"{'Strategy':<42s} {'AvgPnL':>8s} {'WinRate':>8s} {'MaxLoss':>8s} {'Sharpe':>7s} {'AvgDays':>8s}"
        print(header)
        print("-" * 80)
        for r in results:
            print(
                f"{r['strategy']:<42s} "
                f"{r['avg_pnl'] * 100:>+7.2f}% "
                f"{r['win_rate']:>7.1%} "
                f"{r['max_loss'] * 100:>+7.2f}% "
                f"{r['sharpe']:>7.2f} "
                f"{r['avg_days']:>7.1f}"
            )
        print("=" * 80)


if __name__ == "__main__":
    main()
