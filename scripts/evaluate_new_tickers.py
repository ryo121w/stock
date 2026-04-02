#!/usr/bin/env python3
"""Evaluate candidate tickers for universe expansion (A1).

For each candidate ticker:
1. Check OHLCV data is available (fetch if not present)
2. Build features using the same config as existing tickers
3. Run mini walk-forward CV (step=126, 3 folds) with ensemble model
4. Report accuracy at conf 55%+ for each ticker
5. Recommend tickers with accuracy > 53%

Usage:
    .venv/bin/python scripts/evaluate_new_tickers.py
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import structlog

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Register all feature tiers
import qtp.features.tier1_momentum  # noqa: E402, F401
import qtp.features.tier2_volatility  # noqa: E402, F401
import qtp.features.tier3_fundamental  # noqa: E402, F401
import qtp.features.tier4_macro  # noqa: E402, F401
import qtp.features.tier5_alternative  # noqa: E402, F401
from qtp.data.fetchers.base import FetchRequest, Market  # noqa: E402
from qtp.data.fetchers.yfinance_ import YFinanceFetcher  # noqa: E402
from qtp.data.storage import ParquetStorage  # noqa: E402
from qtp.features.engine import FeatureEngine  # noqa: E402
from qtp.features.registry import FeatureRegistry  # noqa: E402
from qtp.models.ensemble import WeightedEnsemble  # noqa: E402
from qtp.models.lgbm import LGBMPipeline  # noqa: E402
from qtp.models.xgb import XGBPipeline  # noqa: E402

logger = structlog.get_logger()

# --- Configuration ---
CANDIDATE_TICKERS = ["COST", "PG", "LLY", "JNJ", "XOM", "CAT"]
EXISTING_TICKERS = ["MSFT", "GOOGL", "AMZN", "NVDA", "META", "JPM", "V"]

# Match phase3_selected.yaml / phase3_best.yaml settings
HORIZON = 10
DIRECTION_THRESHOLD = 0.03
TIERS = [1, 2, 3, 4, 5]
SELECTED_FEATURES = [
    "yield_10y",
    "yield_10y_change_21d",
    "price_to_52w_low",
    "vix_level",
    "sp500_dist_sma50",
    "dist_sma200",
    "sp500_ret_21d",
    "vix_change_5d",
    "atr_14",
    "realized_vol_63d",
    "dist_sma50",
    "realized_vol_21d",
    "macd_signal",
    "eps_revision_7d",
    "roc_20",
    "range_52w_position",
    "max_drawdown_63d",
    "ret_63d",
    "lower_shadow_ratio",
    "ret_5d",
]

# Mini walk-forward: step=126 (~6 months), 3 folds
WF_MIN_TRAIN = 504  # ~2 years
WF_STEP = 126  # ~6 months
WF_TEST = 126  # ~6 months per fold
WF_FOLDS = 3

CONF_THRESHOLD = 0.55
ACCURACY_THRESHOLD = 0.53  # Recommend if > 53%

MARKET = Market("us")


def ensure_data(ticker: str, storage: ParquetStorage) -> bool:
    """Fetch OHLCV data if not present. Returns True if data available."""
    raw_path = storage.base_dir / "raw" / "us" / f"{ticker}.parquet"
    if raw_path.exists():
        df = pl.read_parquet(raw_path)
        print(f"  {ticker}: data already exists ({df.height} rows)")
        return True

    print(f"  {ticker}: fetching from Yahoo Finance...")
    try:
        fetcher = YFinanceFetcher()
        today = date.today()
        start = today - timedelta(days=1890)
        req = FetchRequest(ticker=ticker, market=MARKET, start_date=start, end_date=today)
        df = fetcher.fetch_ohlcv(req)
        if df.height > 0:
            storage.save_ohlcv(ticker, MARKET, df)
            print(f"  {ticker}: fetched {df.height} rows")
            return True
        else:
            print(f"  {ticker}: no data returned!")
            return False
    except Exception as e:
        print(f"  {ticker}: fetch failed: {e}")
        return False


def build_ensemble() -> WeightedEnsemble:
    """Create a fresh XGB+LGBM ensemble (same as production)."""
    lgbm = LGBMPipeline()
    xgb_model = XGBPipeline()
    return WeightedEnsemble([(lgbm, 0.63), (xgb_model, 0.37)])


def evaluate_ticker(
    ticker: str,
    engine: FeatureEngine,
    include_existing: bool = False,
) -> dict:
    """Run mini walk-forward evaluation for a single ticker.

    If include_existing=True, trains on existing tickers + candidate
    (simulates actual usage). Otherwise evaluates ticker in isolation.
    """
    print(f"\n{'=' * 60}")
    print(f"  Evaluating: {ticker}")
    print(f"{'=' * 60}")

    # Build dataset for the candidate ticker
    try:
        ds = engine.build_dataset(
            ticker=ticker,
            market=MARKET,
            as_of=date.today(),
            tiers=TIERS,
            horizon=HORIZON,
            direction_threshold=DIRECTION_THRESHOLD,
        )
    except Exception as e:
        print(f"  ERROR building dataset: {e}")
        return {"ticker": ticker, "status": "error", "error": str(e)}

    if ds.height == 0:
        print("  No data rows after feature computation")
        return {"ticker": ticker, "status": "no_data"}

    ds = ds.with_columns(pl.lit(ticker).alias("ticker"))
    print(f"  Dataset: {ds.height} rows")

    # Also build existing ticker datasets for combined training
    all_frames = [ds]
    if include_existing:
        for ex_ticker in EXISTING_TICKERS:
            try:
                ex_ds = engine.build_dataset(
                    ticker=ex_ticker,
                    market=MARKET,
                    as_of=date.today(),
                    tiers=TIERS,
                    horizon=HORIZON,
                    direction_threshold=DIRECTION_THRESHOLD,
                )
                if ex_ds.height > 0:
                    ex_ds = ex_ds.with_columns(pl.lit(ex_ticker).alias("ticker"))
                    all_frames.append(ex_ds)
            except Exception:
                pass

    combined = pl.concat(all_frames, how="diagonal_relaxed").sort("date")

    # Select features (match phase3_selected)
    label_cols = ["label_direction", "label_magnitude", "date", "ticker"]
    available_features = [c for c in combined.columns if c not in label_cols]
    use_features = [f for f in SELECTED_FEATURES if f in available_features]
    if not use_features:
        use_features = [c for c in available_features if c not in label_cols]
    print(f"  Features: {len(use_features)}")

    # Filter to only candidate ticker rows for testing

    X_all = combined.select(use_features)
    y_dir_all = combined["label_direction"].to_numpy()
    dates_all = combined["date"].to_list()
    tickers_all = combined["ticker"].to_list()

    # Walk-forward on the candidate ticker
    # Use combined data for training, test only on candidate ticker
    n_rows = len(combined)

    if n_rows < WF_MIN_TRAIN + WF_TEST:
        print(f"  Not enough data ({n_rows} rows, need {WF_MIN_TRAIN + WF_TEST})")
        return {"ticker": ticker, "status": "insufficient_data", "rows": n_rows}

    total_preds = 0
    total_correct = 0
    conf_preds = 0
    conf_correct = 0
    fold_results = []

    i = WF_MIN_TRAIN
    fold = 0
    while i + WF_TEST <= n_rows and fold < WF_FOLDS:
        # Train on [0, i)
        train_X = X_all[:i]
        train_y = pl.Series(y_dir_all[:i])
        train_y_mag = pl.Series(combined["label_magnitude"].to_numpy()[:i])

        # Test on [i, i+WF_TEST) -- only candidate ticker rows
        test_end = min(i + WF_TEST, n_rows)
        test_X = X_all[i:test_end]
        test_y = y_dir_all[i:test_end]
        test_tickers = tickers_all[i:test_end]
        test_dates = dates_all[i:test_end]

        # Build and train ensemble
        model = build_ensemble()
        try:
            model.fit(train_X, train_y, train_y_mag)
        except Exception as e:
            print(f"  Fold {fold}: training failed: {e}")
            i += WF_STEP
            fold += 1
            continue

        # Predict
        pred_proba = np.array(model.predict_proba(test_X))
        pred_dir = (pred_proba >= 0.5).astype(int)

        # Filter to candidate ticker predictions only
        for j in range(len(test_tickers)):
            if test_tickers[j] != ticker:
                continue

            is_correct = int(pred_dir[j] == test_y[j])
            conf = float(pred_proba[j])
            total_preds += 1
            total_correct += is_correct

            if conf >= CONF_THRESHOLD or (1 - conf) >= CONF_THRESHOLD:
                # High confidence prediction (either direction)
                effective_correct = is_correct if conf >= 0.5 else (1 - is_correct)
                conf_preds += 1
                conf_correct += int(effective_correct)

        fold_acc = total_correct / total_preds * 100 if total_preds > 0 else 0
        fold_conf_acc = conf_correct / conf_preds * 100 if conf_preds > 0 else 0
        print(
            f"  Fold {fold}: train={i}, test={test_end - i}, "
            f"dates={test_dates[0]}..{test_dates[-1]}, "
            f"cumul_acc={fold_acc:.1f}%, conf55+_acc={fold_conf_acc:.1f}% "
            f"({conf_preds} trades)"
        )

        fold_results.append(
            {
                "fold": fold,
                "train_size": i,
                "test_size": test_end - i,
                "start_date": str(test_dates[0]),
                "end_date": str(test_dates[-1]),
            }
        )

        i += WF_STEP
        fold += 1

    overall_acc = total_correct / total_preds if total_preds > 0 else 0
    conf_acc = conf_correct / conf_preds if conf_preds > 0 else 0

    result = {
        "ticker": ticker,
        "status": "ok",
        "total_preds": total_preds,
        "total_correct": total_correct,
        "overall_accuracy": overall_acc,
        "conf55_preds": conf_preds,
        "conf55_correct": conf_correct,
        "conf55_accuracy": conf_acc,
        "folds": fold,
        "recommend": conf_acc > ACCURACY_THRESHOLD
        if conf_preds >= 10
        else overall_acc > ACCURACY_THRESHOLD,
    }

    print(f"\n  RESULT: {ticker}")
    print(f"    Overall accuracy: {overall_acc:.1%} ({total_correct}/{total_preds})")
    print(f"    Conf 55%+ accuracy: {conf_acc:.1%} ({conf_correct}/{conf_preds})")
    print(f"    Recommend: {'YES' if result['recommend'] else 'NO'}")

    return result


def main():
    print("=" * 60)
    print("  UNIVERSE EXPANSION: New Ticker Evaluation (A1)")
    print("=" * 60)
    print(f"  Candidates: {CANDIDATE_TICKERS}")
    print(f"  Existing:   {EXISTING_TICKERS}")
    print(f"  Config:     h={HORIZON}, threshold={DIRECTION_THRESHOLD}")
    print(
        f"  WF:         min_train={WF_MIN_TRAIN}, step={WF_STEP}, test={WF_TEST}, folds={WF_FOLDS}"
    )
    print(f"  Criteria:   conf 55%+ accuracy > {ACCURACY_THRESHOLD:.0%}")
    print()

    storage = ParquetStorage(project_root / "data")
    engine = FeatureEngine(FeatureRegistry.instance(), storage)

    # Step 1: Ensure data for all candidates
    print("--- Step 1: Data Check ---")
    data_ok = {}
    for ticker in CANDIDATE_TICKERS:
        data_ok[ticker] = ensure_data(ticker, storage)
    print()

    # Step 2: Evaluate each candidate
    print("--- Step 2: Walk-Forward Evaluation ---")
    results = []
    for ticker in CANDIDATE_TICKERS:
        if not data_ok.get(ticker):
            print(f"\n  Skipping {ticker}: no data available")
            results.append({"ticker": ticker, "status": "no_data"})
            continue
        result = evaluate_ticker(ticker, engine, include_existing=True)
        results.append(result)

    # Step 3: Summary
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    print(
        f"  {'Ticker':<8} {'Status':<12} {'Overall':>10} {'Conf55+':>10} {'Trades':>8} {'Recommend':>10}"
    )
    print(f"  {'-' * 8} {'-' * 12} {'-' * 10} {'-' * 10} {'-' * 8} {'-' * 10}")

    recommended = []
    for r in results:
        if r["status"] != "ok":
            print(f"  {r['ticker']:<8} {r['status']:<12}")
            continue
        print(
            f"  {r['ticker']:<8} {'ok':<12} "
            f"{r['overall_accuracy']:>9.1%} "
            f"{r['conf55_accuracy']:>9.1%} "
            f"{r['conf55_preds']:>8} "
            f"{'YES' if r['recommend'] else 'NO':>10}"
        )
        if r["recommend"]:
            recommended.append(r["ticker"])

    print(f"\n  Recommended tickers (accuracy > {ACCURACY_THRESHOLD:.0%}): {recommended}")
    print(f"  Proposed universe: {EXISTING_TICKERS + recommended}")
    print()

    return recommended


if __name__ == "__main__":
    recommended = main()
