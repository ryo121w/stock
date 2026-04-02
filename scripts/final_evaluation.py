#!/usr/bin/env python3
"""Final evaluation: combine all 3 improvements."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datetime import date  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import qtp.features.tier1_momentum  # noqa: E402, F401
import qtp.features.tier2_volatility  # noqa: E402, F401
import qtp.features.tier3_fundamental  # noqa: E402, F401
import qtp.features.tier4_macro  # noqa: E402, F401
import qtp.features.tier5_alternative  # noqa: E402, F401
import qtp.features.tier5_timeseries  # noqa: E402, F401
from qtp.config import PipelineConfig  # noqa: E402
from qtp.data.fetchers.base import Market  # noqa: E402
from qtp.data.storage import ParquetStorage  # noqa: E402
from qtp.features.engine import FeatureEngine  # noqa: E402
from qtp.features.registry import FeatureRegistry  # noqa: E402
from qtp.models.lgbm import LGBMPipeline  # noqa: E402
from qtp.models.xgb import XGBPipeline  # noqa: E402

project_root = Path(__file__).parent.parent

# Load config
config = PipelineConfig.from_yamls(
    project_root / "configs/default.yaml",
    project_root / "configs/phase3_best.yaml",
)

# Load best params
best_params = json.loads((project_root / "configs/best_params.json").read_text())

# Load selected features
import yaml  # noqa: E402

with open(project_root / "configs/phase3_selected.yaml") as f:
    sel_yaml = yaml.safe_load(f)
selected_features = sel_yaml.get("features", {}).get("selected", [])

# Build dataset
storage = ParquetStorage(project_root / config.data.storage_dir)
engine = FeatureEngine(FeatureRegistry.instance(), storage)
market = Market(config.universe.market)

dataset = engine.build_multi_ticker_dataset(
    tickers=config.universe.tickers,
    market=market,
    as_of=date.today(),
    tiers=[1, 2, 3, 4, 5],
    horizon=10,
    direction_threshold=0.03,
)
dataset = dataset.sort("date")

label_cols = ["label_direction", "label_magnitude", "date", "ticker"]
all_feature_cols = [c for c in dataset.columns if c not in label_cols]

# Use selected features (intersect with available)
if selected_features:
    feature_cols = [c for c in selected_features if c in all_feature_cols]
    # Also add any tier5_timeseries features not in original selection
    tier5_ts = [
        "earnings_proximity_cycle",
        "earnings_proximity_cycle_cos",
        "price_earnings_momentum",
        "analyst_sentiment_proxy",
        "insider_signal_proxy",
        "regime_proxy",
    ]
    for f in tier5_ts:
        if f in all_feature_cols and f not in feature_cols:
            feature_cols.append(f)
else:
    feature_cols = all_feature_cols

print(f"Features: {len(feature_cols)} (selected + tier5_ts)")
print(f"Feature list: {feature_cols}")
print(f"Dataset: {dataset.height} rows")

X = dataset.select(feature_cols)
y_dir = dataset["label_direction"].to_numpy()
y_mag = dataset["label_magnitude"].to_numpy()

# Walk-forward with ensemble
min_train, step, test_size = 504, 63, 63
lgbm_weight = best_params.get("lgbm_weight", 0.369)
xgb_weight = 1 - lgbm_weight

lgbm_clf_params = best_params.get("lgbm_clf_params", {})
lgbm_reg_params = {k: v for k, v in lgbm_clf_params.items()}
lgbm_reg_params["objective"] = "regression"
if "verbose" not in lgbm_reg_params:
    lgbm_reg_params["verbose"] = -1
if "verbose" not in lgbm_clf_params:
    lgbm_clf_params["verbose"] = -1

xgb_clf_params = best_params.get("xgb_clf_params", {})
xgb_reg_params = {k: v for k, v in xgb_clf_params.items()}
xgb_reg_params["eval_metric"] = "rmse"
if "eval_metric" not in xgb_clf_params:
    xgb_clf_params["eval_metric"] = "logloss"
xgb_clf_params["verbosity"] = 0
xgb_reg_params["verbosity"] = 0
xgb_clf_params["early_stopping_rounds"] = 50
xgb_reg_params["early_stopping_rounds"] = 50

all_proba, all_pred, all_actual_dir, all_actual_mag = [], [], [], []
fold = 0
i = min_train

while i + test_size <= len(dataset):
    test_end = min(i + test_size, len(dataset))

    # Train LGBM
    lgbm = LGBMPipeline(clf_params=lgbm_clf_params, reg_params=lgbm_reg_params)
    lgbm.fit(X[:i], pl.Series(y_dir[:i]), pl.Series(y_mag[:i]))

    # Train XGB
    xgb_model = XGBPipeline(clf_params=xgb_clf_params, reg_params=xgb_reg_params)
    xgb_model.fit(X[:i], pl.Series(y_dir[:i]), pl.Series(y_mag[:i]))

    # Ensemble prediction
    lgbm_proba = np.array(lgbm.predict_proba(X[i:test_end]))
    xgb_proba = np.array(xgb_model.predict_proba(X[i:test_end]))
    ensemble_proba = lgbm_weight * lgbm_proba + xgb_weight * xgb_proba

    pred_dir = (ensemble_proba >= 0.5).astype(int)

    all_proba.extend(ensemble_proba.tolist())
    all_pred.extend(pred_dir.tolist())
    all_actual_dir.extend(y_dir[i:test_end].tolist())
    all_actual_mag.extend(y_mag[i:test_end].tolist())

    if fold % 10 == 0:
        acc = (pred_dir == y_dir[i:test_end]).mean()
        print(f"  Fold {fold}: train={i}, test={test_end - i}, acc={acc:.1%}")

    fold += 1
    i += step

proba = np.array(all_proba)
pred = np.array(all_pred)
actual = np.array(all_actual_dir)
mag = np.array(all_actual_mag)

# Results
print(f"\n{'=' * 70}")
print("  FINAL COMBINED EVALUATION")
print(f"  Features: top-20 selected + 6 tier5_timeseries = {len(feature_cols)} total")
print(f"  Model: XGB {xgb_weight:.0%} + LGBM {lgbm_weight:.0%} (Optuna tuned)")
print("  Config: h=10, t=3%, 7 tickers")
print(f"{'=' * 70}")

total = len(proba)
print(f"\n  Overall: {(pred == actual).mean():.1%} ({(pred == actual).sum()}/{total})")

for thresh_name, thresh in [
    ("50%+", 0.50),
    ("55%+", 0.55),
    ("60%+", 0.60),
    ("65%+", 0.65),
    ("70%+", 0.70),
]:
    mask = proba >= thresh
    n = mask.sum()
    if n > 0:
        acc = (pred[mask] == actual[mask]).mean()
        avg_ret = mag[mask].mean()
        print(
            f"  conf {thresh_name}: {acc:.1%} ({(pred[mask] == actual[mask]).sum()}/{n}) avg_ret={avg_ret:+.2%}"
        )
    else:
        print(f"  conf {thresh_name}: no signals")

# Compare with baseline (LGBM only, all features)
print("\n  --- Comparison ---")
print("  Phase 3 baseline (LGBM, 43 features):   56.7% overall, 61.3% at 55%+")
print("  This run (Ensemble, selected+tier5_ts):  see above")
print(f"{'=' * 70}")
