#!/usr/bin/env python3
"""Evaluate LSTM model vs LGBM-only and LGBM+LSTM average ensemble.

Uses Phase 3 dataset (h10_t3%, 7 tickers) with walk-forward CV.
step=252 (fewer folds due to LSTM training time), min_train=504.
Reports accuracy at confidence >= 55%.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import date
from pathlib import Path

# Prevent OpenMP thread deadlock between LightGBM and PyTorch
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import polars as pl  # noqa: E402
import structlog  # noqa: E402

# Import feature definitions (triggers registration)
import qtp.features.tier1_momentum  # noqa: E402, F401
import qtp.features.tier2_volatility  # noqa: E402, F401
import qtp.features.tier3_fundamental  # noqa: E402, F401
import qtp.features.tier4_macro  # noqa: E402, F401
import qtp.features.tier5_alternative  # noqa: E402, F401
import qtp.features.tier5_timeseries  # noqa: E402, F401
from qtp.config import PipelineConfig  # noqa: E402
from qtp.data.fetchers.base import Market  # noqa: E402
from qtp.data.storage import ParquetStorage  # noqa: E402
from qtp.data.universe import Universe  # noqa: E402
from qtp.features.engine import FeatureEngine  # noqa: E402
from qtp.features.registry import FeatureRegistry  # noqa: E402
from qtp.models.lgbm import LGBMPipeline  # noqa: E402
from qtp.models.lstm_model import LSTMPipeline  # noqa: E402
from qtp.validation.walk_forward import ExpandingWindowCV  # noqa: E402

logger = structlog.get_logger()

# --- Config ---
CONF_THRESHOLD = 0.55
MIN_TRAIN = 504
STEP_SIZE = 252  # ~1 year steps (fewer folds, LSTM is slow)
TEST_SIZE = 63
PURGE_GAP = 5
SEQ_LEN = 20
LSTM_EPOCHS = 30  # Keep fast

PARAMS_PATH = project_root / "configs" / "best_params.json"


def load_dataset(config: PipelineConfig):
    """Load Phase 3 multi-ticker dataset."""
    storage = ParquetStorage(project_root / config.data.storage_dir)
    market = Market(config.universe.market)
    universe = Universe(config.universe)
    engine = FeatureEngine(FeatureRegistry.instance(), storage)

    dataset = engine.build_multi_ticker_dataset(
        tickers=universe.tickers(),
        market=market,
        as_of=date.today(),
        tiers=config.features.tiers,
        horizon=config.labels.horizon,
        direction_threshold=config.labels.direction_threshold,
    )

    label_cols = ["label_direction", "label_magnitude", "date", "ticker"]
    feature_cols = [c for c in dataset.columns if c not in label_cols]

    X = dataset.select(feature_cols)
    y_direction = dataset["label_direction"]
    y_magnitude = dataset["label_magnitude"]

    # Clean inf/NaN
    X = X.with_columns(
        [
            pl.when(pl.col(c).is_infinite()).then(None).otherwise(pl.col(c)).fill_null(0.0).alias(c)
            for c in X.columns
        ]
    )

    return X, y_direction, y_magnitude, feature_cols


def load_best_params() -> dict | None:
    if PARAMS_PATH.exists():
        return json.loads(PARAMS_PATH.read_text())
    return None


def evaluate_model_cv(
    model_name: str,
    model_factory,
    X: pl.DataFrame,
    y_dir: pl.Series,
    y_mag: pl.Series,
) -> dict:
    """Run walk-forward CV and return accuracy stats at conf >= 55%."""
    cv = ExpandingWindowCV(
        min_train_size=MIN_TRAIN,
        test_size=TEST_SIZE,
        step_size=STEP_SIZE,
        purge_gap=PURGE_GAP,
    )

    y_dir_np = y_dir.to_numpy()
    y_mag_np = y_mag.to_numpy()
    X_np = X.to_pandas().values

    all_probas = []
    all_actuals = []
    fold_accs = []

    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X_np)):
        t0 = time.monotonic()

        try:
            model = model_factory()
            model.fit(
                X[train_idx],
                pl.Series(y_dir_np[train_idx]),
                pl.Series(y_mag_np[train_idx]),
            )

            probas = np.array(model.predict_proba(X[test_idx]))
        except Exception as e:
            print(f"  {model_name} Fold {fold_i}: SKIPPED ({type(e).__name__}: {e})")
            continue

        actuals = y_dir_np[test_idx]

        # Accuracy at conf >= threshold
        conf_mask = np.abs(probas - 0.5) >= (CONF_THRESHOLD - 0.5)
        if conf_mask.sum() > 0:
            preds = (probas[conf_mask] >= 0.5).astype(int)
            acc = (preds == actuals[conf_mask]).mean()
            fold_accs.append(acc)

        all_probas.extend(probas.tolist())
        all_actuals.extend(actuals.tolist())

        elapsed = time.monotonic() - t0
        n_conf = int(conf_mask.sum()) if conf_mask.sum() > 0 else 0
        fold_acc = fold_accs[-1] if fold_accs else 0.0
        print(
            f"  {model_name} Fold {fold_i}: "
            f"acc@55%={fold_acc:.3f} ({n_conf} signals) "
            f"train={len(train_idx)} test={len(test_idx)} "
            f"[{elapsed:.1f}s]"
        )

    # Overall stats
    all_probas = np.array(all_probas)
    all_actuals = np.array(all_actuals)

    conf_mask = np.abs(all_probas - 0.5) >= (CONF_THRESHOLD - 0.5)
    n_conf = int(conf_mask.sum())

    if n_conf > 0:
        preds_conf = (all_probas[conf_mask] >= 0.5).astype(int)
        overall_acc = float((preds_conf == all_actuals[conf_mask]).mean())
    else:
        overall_acc = 0.0

    preds_all = (all_probas >= 0.5).astype(int)
    raw_acc = float((preds_all == all_actuals).mean())

    return {
        "model": model_name,
        "accuracy_55pct": overall_acc,
        "n_signals_55pct": n_conf,
        "raw_accuracy": raw_acc,
        "n_total": len(all_probas),
        "n_folds": len(fold_accs),
        "fold_accs": fold_accs,
        "mean_fold_acc": float(np.mean(fold_accs)) if fold_accs else 0.0,
        "std_fold_acc": float(np.std(fold_accs)) if len(fold_accs) > 1 else 0.0,
    }


class LGBMLSTMAverage:
    """Simple average of LGBM and LSTM predictions."""

    def __init__(self, lgbm_factory, lstm_factory):
        self.lgbm = lgbm_factory()
        self.lstm = lstm_factory()

    def fit(self, X, y_dir, y_mag):
        self.lgbm.fit(X, y_dir, y_mag)
        self.lstm.fit(X, y_dir, y_mag)

    def predict_proba(self, X):
        p_lgbm = np.array(self.lgbm.predict_proba(X))
        p_lstm = np.array(self.lstm.predict_proba(X))
        return ((p_lgbm + p_lstm) / 2.0).tolist()

    def predict_magnitude(self, X):
        m_lgbm = np.array(self.lgbm.predict_magnitude(X))
        m_lstm = np.array(self.lstm.predict_magnitude(X))
        return ((m_lgbm + m_lstm) / 2.0).tolist()


def main():
    t0_total = time.monotonic()
    print("=" * 72)
    print("  LSTM MODEL EVALUATION")
    print("=" * 72)

    # Load config
    config_default = project_root / "configs" / "default.yaml"
    config_phase3 = project_root / "configs" / "phase3_best.yaml"
    config = PipelineConfig.from_yamls(config_default, config_phase3)

    print(f"\nConfig: {config_phase3.name}")
    print(f"Tickers: {config.universe.tickers}")
    print(f"Horizon: {config.labels.horizon}, Threshold: {config.labels.direction_threshold}")
    print(f"Walk-Forward: min_train={MIN_TRAIN}, step={STEP_SIZE}, test={TEST_SIZE}")
    print(f"LSTM: seq_len={SEQ_LEN}, epochs={LSTM_EPOCHS}, hidden=64")
    print(f"Confidence threshold: {CONF_THRESHOLD}")

    # Load dataset
    print("\nLoading dataset...")
    X, y_dir, y_mag, feature_cols = load_dataset(config)
    print(f"Dataset: {X.height} rows, {len(feature_cols)} features")
    print(f"Label balance: {y_dir.to_numpy().mean():.2%} positive")

    # Load best params
    bp = load_best_params()
    lgbm_clf_params = bp["lgbm_clf_params"] if bp else None

    results = []

    # 1. LGBM-only
    print("\n--- 1. LGBM-only ---")

    def make_lgbm():
        return LGBMPipeline(clf_params=lgbm_clf_params)

    results.append(evaluate_model_cv("LGBM", make_lgbm, X, y_dir, y_mag))

    # 2. LSTM-only
    print(f"\n--- 2. LSTM-only (seq_len={SEQ_LEN}, epochs={LSTM_EPOCHS}) ---")

    n_feat = len(feature_cols)

    def make_lstm():
        return LSTMPipeline(
            n_features=n_feat,
            seq_len=SEQ_LEN,
            hidden_size=64,
            n_layers=2,
            dropout=0.3,
            lr=0.001,
            epochs=LSTM_EPOCHS,
            batch_size=64,
            patience=7,
        )

    results.append(evaluate_model_cv("LSTM", make_lstm, X, y_dir, y_mag))

    # 3. LGBM + LSTM average
    print("\n--- 3. LGBM + LSTM Average ---")

    def make_avg():
        return LGBMLSTMAverage(make_lgbm, make_lstm)

    results.append(evaluate_model_cv("LGBM+LSTM_Avg", make_avg, X, y_dir, y_mag))

    # --- Summary ---
    duration = time.monotonic() - t0_total
    print("\n" + "=" * 72)
    print("  RESULTS SUMMARY")
    print("=" * 72)
    print(
        f"\n  {'Model':<22} {'Acc@55%':>8} {'N_sig':>7} {'Raw Acc':>8} "
        f"{'Folds':>6} {'Mean+/-Std':>14}"
    )
    print(f"  {'-' * 22} {'-' * 8} {'-' * 7} {'-' * 8} {'-' * 6} {'-' * 14}")

    for r in results:
        mean_std = f"{r['mean_fold_acc']:.3f}+/-{r['std_fold_acc']:.3f}"
        print(
            f"  {r['model']:<22} {r['accuracy_55pct']:>7.1%} {r['n_signals_55pct']:>7} "
            f"{r['raw_accuracy']:>7.1%} {r['n_folds']:>6} {mean_std:>14}"
        )

    best = max(results, key=lambda r: r["accuracy_55pct"])
    print(f"\n  Best model: {best['model']} ({best['accuracy_55pct']:.1%} at conf 55%+)")
    print(f"\n  Total time: {duration:.0f}s ({duration / 60:.1f} min)")
    print("=" * 72)

    # Save results
    output_path = project_root / "data" / "reports" / "lstm_evaluation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = []
    for r in results:
        r_copy = dict(r)
        r_copy["fold_accs"] = [round(a, 4) for a in r_copy["fold_accs"]]
        r_copy["accuracy_55pct"] = round(r_copy["accuracy_55pct"], 4)
        r_copy["raw_accuracy"] = round(r_copy["raw_accuracy"], 4)
        r_copy["mean_fold_acc"] = round(r_copy["mean_fold_acc"], 4)
        r_copy["std_fold_acc"] = round(r_copy["std_fold_acc"], 4)
        serializable.append(r_copy)
    output_path.write_text(json.dumps(serializable, indent=2))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
