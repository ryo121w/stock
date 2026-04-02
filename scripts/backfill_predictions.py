#!/usr/bin/env python3
"""Backfill historical predictions and immediately grade them.

Instead of waiting days for real predictions to accumulate:
1. Walk-forward through historical data
2. At each step, train on past data, predict the next period
3. Save prediction to SQLite
4. Immediately grade with actual prices (which we already have)

Result: Hundreds of graded predictions in minutes, ready for accuracy analysis.
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
from qtp.data.database import QTPDatabase  # noqa: E402
from qtp.data.fetchers.base import Market  # noqa: E402
from qtp.data.storage import ParquetStorage  # noqa: E402
from qtp.features.engine import FeatureEngine  # noqa: E402
from qtp.features.registry import FeatureRegistry  # noqa: E402
from qtp.models.lgbm import LGBMPipeline  # noqa: E402

logger = structlog.get_logger()


def load_config():
    config_path = project_root / "configs" / "default.yaml"
    p2_path = project_root / "configs" / "phase2_experiment.yaml"
    if p2_path.exists():
        return PipelineConfig.from_yamls(config_path, p2_path)
    return PipelineConfig.from_yaml(config_path)


def main():
    config = load_config()
    db = QTPDatabase(project_root / "data" / "qtp.db")
    storage = ParquetStorage(project_root / config.data.storage_dir)
    engine = FeatureEngine(FeatureRegistry.instance(), storage)
    market = Market(config.universe.market)
    horizon = config.labels.horizon
    threshold = config.labels.direction_threshold

    print(f"Config: horizon={horizon}, threshold={threshold}")
    print(f"Tickers: {config.universe.tickers}")
    print()

    # Build full dataset
    print("Building dataset...")
    dataset = engine.build_multi_ticker_dataset(
        tickers=config.universe.tickers,
        market=market,
        as_of=date.today(),
        tiers=config.features.tiers,
        horizon=horizon,
        direction_threshold=threshold,
    )

    label_cols = ["label_direction", "label_magnitude", "date", "ticker"]
    feature_cols = [c for c in dataset.columns if c not in label_cols]
    print(f"Dataset: {dataset.height} rows, {len(feature_cols)} features")

    # Sort by date for time-series ordering
    dataset = dataset.sort("date")
    dates = dataset["date"].to_list()
    tickers = dataset["ticker"].to_list()
    X = dataset.select(feature_cols)
    y_dir = dataset["label_direction"].to_numpy()
    y_mag = dataset["label_magnitude"].to_numpy()

    # Walk-forward: train on past, predict future, step forward
    min_train = config.validation.walk_forward_train_days
    step = config.validation.walk_forward_step_days
    test_size = config.validation.walk_forward_test_days

    total_predictions = 0
    total_correct = 0
    fold = 0

    print(f"\nWalk-forward backfill (min_train={min_train}, step={step}, test={test_size})...")
    print()

    i = min_train
    while i + test_size <= len(dataset):
        # Train on [0, i)
        train_X = X[:i]
        train_y_dir = pl.Series(y_dir[:i])
        train_y_mag = pl.Series(y_mag[:i])

        # Test on [i, i+test_size)
        test_end = min(i + test_size, len(dataset))
        test_X = X[i:test_end]
        test_dates = dates[i:test_end]
        test_tickers = tickers[i:test_end]
        test_y_dir = y_dir[i:test_end]
        test_y_mag = y_mag[i:test_end]

        # Train
        model = LGBMPipeline()
        model.fit(train_X, train_y_dir, train_y_mag)

        # Predict
        pred_proba = np.array(model.predict_proba(test_X))
        pred_mag = np.array(model.predict_magnitude(test_X))
        pred_dir = (pred_proba >= 0.5).astype(int)

        # Save each prediction and immediately grade
        fold_correct = 0
        for j in range(len(test_dates)):
            pred_date = test_dates[j]
            ticker = test_tickers[j]
            direction = int(pred_dir[j])
            confidence = float(pred_proba[j])
            magnitude = float(pred_mag[j])
            actual_return = float(test_y_mag[j])
            actual_dir = int(test_y_dir[j])

            # Save prediction
            db.save_prediction(
                ticker=ticker,
                prediction_date=str(pred_date),
                direction=direction,
                confidence=confidence,
                predicted_magnitude=magnitude,
                model_version=model.version,
                horizon=horizon,
            )

            # Immediately grade (we already know the actual return)
            is_correct = 1 if (direction == actual_dir) else 0
            with db._conn() as conn:
                conn.execute(
                    """UPDATE predictions SET
                         actual_return=?, is_correct=?, graded_at=CURRENT_TIMESTAMP
                       WHERE ticker=? AND prediction_date=? AND model_version=?""",
                    (actual_return, is_correct, ticker, str(pred_date), model.version),
                )

            fold_correct += is_correct

        fold_accuracy = fold_correct / len(test_dates) * 100
        total_predictions += len(test_dates)
        total_correct += fold_correct

        if fold % 5 == 0:
            print(
                f"  Fold {fold}: train={i}, test={len(test_dates)}, "
                f"accuracy={fold_accuracy:.1f}%, "
                f"dates={test_dates[0]}→{test_dates[-1]}"
            )

        fold += 1
        i += step

    print(f"\n{'=' * 60}")
    print("  BACKFILL COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total predictions: {total_predictions}")
    print(f"  Total correct:     {total_correct}")
    print(f"  Overall accuracy:  {total_correct / total_predictions:.1%}")
    print(f"  Folds:             {fold}")
    print(f"{'=' * 60}")

    # Print accuracy report
    print("\n--- Accuracy by Confidence ---")
    by_conf = db.get_accuracy_by_confidence()
    for b in by_conf:
        print(
            f"  {b['bucket']}: {b['accuracy_pct']}% ({b['correct']}/{b['total']}) "
            f"avg_ret={b['avg_return_pct']}%"
        )

    print("\n--- Accuracy by Ticker ---")
    by_ticker = db.get_accuracy_by_ticker()
    for b in by_ticker:
        print(
            f"  {b['ticker']}: {b['accuracy_pct']}% ({b['correct']}/{b['total']}) "
            f"avg_ret={b['avg_return_pct']}%"
        )


if __name__ == "__main__":
    main()
