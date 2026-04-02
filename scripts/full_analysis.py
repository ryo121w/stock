"""Full analysis script: fetch → train → SHAP → Optuna → backtest report."""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qtp.config import PipelineConfig
from qtp.data.fetchers.base import FetchRequest, Market
from qtp.data.fetchers.yfinance_ import YFinanceFetcher
from qtp.data.storage import ParquetStorage
from qtp.data.universe import Universe
from qtp.data.validator import DataValidator
from qtp.features.engine import FeatureEngine
from qtp.features.registry import FeatureRegistry
import qtp.features.tier1_momentum  # noqa: F401
import qtp.features.tier2_volatility  # noqa: F401
from qtp.models.lgbm import LGBMPipeline
from qtp.models.versioning import ModelStore
from qtp.validation.metrics import compute_metrics
from qtp.validation.purged_kfold import PurgedKFold
from qtp.utils.logging_ import setup_logging

setup_logging("INFO")

import structlog
logger = structlog.get_logger()

# Load config
cfg = PipelineConfig.from_yaml(Path("configs/default.yaml"))
storage = ParquetStorage(Path(cfg.data.storage_dir))
market = Market(cfg.universe.market)
universe = Universe(cfg.universe)
model_store = ModelStore(Path(cfg.data.storage_dir) / "models")
feature_engine = FeatureEngine(FeatureRegistry.instance(), storage)
today = date.today()

# ============================================================
# Step 1: Fetch 5 years of data
# ============================================================
print("\n" + "=" * 60)
print("STEP 1: Fetching 5 years of OHLCV data")
print("=" * 60)

fetcher = YFinanceFetcher()
start = today - timedelta(days=cfg.data.history_days)

for ticker in universe:
    try:
        df = fetcher.fetch_ohlcv(FetchRequest(
            ticker=ticker, market=market,
            start_date=start, end_date=today,
        ))
        validator = DataValidator()
        result = validator.validate_ohlcv(df, as_of=today)
        if df.height > 0:
            storage.save_ohlcv(ticker, market, df)
            print(f"  ✓ {ticker}: {df.height} rows ({df['date'].min()} → {df['date'].max()})")
    except Exception as e:
        print(f"  ✗ {ticker}: {e}")

# ============================================================
# Step 2: Build training dataset
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Building training dataset (all data)")
print("=" * 60)

dataset = feature_engine.build_multi_ticker_dataset(
    tickers=universe.tickers(),
    market=market,
    as_of=today,
    tiers=cfg.features.tiers,
    horizon=cfg.labels.horizon,
)
print(f"  Dataset: {dataset.height} rows × {len(dataset.columns)} columns")
print(f"  Tickers: {dataset['ticker'].n_unique()}")
print(f"  Date range: {dataset['date'].min()} → {dataset['date'].max()}")

label_cols = ["label_direction", "label_magnitude", "date", "ticker"]
feature_cols = [c for c in dataset.columns if c not in label_cols]
X = dataset.select(feature_cols)
y_direction = dataset["label_direction"]
y_magnitude = dataset["label_magnitude"]

print(f"  Features: {len(feature_cols)}")
print(f"  Label balance: UP={y_direction.sum()}/{y_direction.len()} ({y_direction.mean():.1%})")

# ============================================================
# Step 3: Train with Purged K-Fold CV
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Training with 5-fold Purged K-Fold CV")
print("=" * 60)

cv = PurgedKFold(n_splits=5, purge_days=5)
X_np = X.to_pandas().values
y_dir_np = y_direction.to_numpy()
y_mag_np = y_magnitude.to_numpy()

fold_metrics = []
for fold_i, (train_idx, test_idx) in enumerate(cv.split(X_np)):
    fold_model = LGBMPipeline()
    fold_model.fit(
        X[train_idx], pl.Series(y_dir_np[train_idx]), pl.Series(y_mag_np[train_idx])
    )
    pred_proba = np.array(fold_model.predict_proba(X[test_idx]))
    pred_mag = np.array(fold_model.predict_magnitude(X[test_idx]))
    metrics = compute_metrics(y_dir_np[test_idx], pred_proba, y_mag_np[test_idx], pred_mag)
    fold_metrics.append(metrics)
    print(f"  Fold {fold_i+1}: acc={metrics.accuracy:.3f} auc={metrics.auc_roc:.3f} "
          f"sharpe={metrics.sharpe_ratio:.2f} win_rate={metrics.win_rate:.3f}")

avg = {
    "accuracy": np.mean([m.accuracy for m in fold_metrics]),
    "auc_roc": np.mean([m.auc_roc for m in fold_metrics]),
    "sharpe": np.mean([m.sharpe_ratio for m in fold_metrics]),
    "max_drawdown": np.mean([m.max_drawdown for m in fold_metrics]),
    "win_rate": np.mean([m.win_rate for m in fold_metrics]),
}
print(f"\n  AVG: acc={avg['accuracy']:.3f} auc={avg['auc_roc']:.3f} "
      f"sharpe={avg['sharpe']:.2f} win_rate={avg['win_rate']:.3f} max_dd={avg['max_drawdown']:.3f}")

# Train final model on all data
print("\n  Training final model on all data...")
final_model = LGBMPipeline()
final_model.fit(X, y_direction, y_magnitude)
version = model_store.save(final_model, metrics=avg)
print(f"  Model saved: {version}")

# ============================================================
# Step 4: SHAP Analysis
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: SHAP Feature Importance Analysis")
print("=" * 60)

import shap

explainer = shap.TreeExplainer(final_model.clf)
X_pd = X.to_pandas()
shap_values = explainer.shap_values(X_pd)

# For binary classification
if isinstance(shap_values, list):
    shap_values = shap_values[1]

mean_abs_shap = np.abs(shap_values).mean(axis=0)
total = mean_abs_shap.sum()
importance = sorted(zip(feature_cols, mean_abs_shap / total), key=lambda x: x[1], reverse=True)

print("\n  Feature Importance (mean |SHAP|):")
print(f"  {'Rank':<5} {'Feature':<25} {'Importance':>12} {'Bar'}")
print(f"  {'─'*5} {'─'*25} {'─'*12} {'─'*30}")
for i, (name, imp) in enumerate(importance):
    bar = "█" * int(imp * 200)
    print(f"  {i+1:<5} {name:<25} {imp:>11.1%} {bar}")

# Save SHAP summary plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, X_pd, show=False, max_display=21)
plt.tight_layout()
shap_path = Path("data/reports/shap_summary.png")
shap_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(shap_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  SHAP summary plot saved: {shap_path}")

# ============================================================
# Step 5: Optuna Hyperparameter Tuning
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Optuna Hyperparameter Tuning (50 trials)")
print("=" * 60)

import optuna
from sklearn.metrics import roc_auc_score
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    import lightgbm as lgb
    scores = []
    for train_idx, val_idx in cv.split(X_np):
        clf = lgb.LGBMClassifier(**params)
        clf.fit(
            X_np[train_idx], y_dir_np[train_idx],
            eval_set=[(X_np[val_idx], y_dir_np[val_idx])],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
        )
        pred = clf.predict_proba(X_np[val_idx])[:, 1]
        scores.append(roc_auc_score(y_dir_np[val_idx], pred))
    return np.mean(scores)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"\n  Best AUC-ROC: {study.best_value:.4f}")
print(f"  Best params:")
for k, v in study.best_params.items():
    print(f"    {k}: {v}")

# Train optimized model
print("\n  Training optimized model with best params...")
optimized_model = LGBMPipeline(clf_params={**study.best_params, "random_state": 42, "n_jobs": -1, "verbose": -1})
optimized_model.fit(X, y_direction, y_magnitude)

# Evaluate optimized model
opt_fold_metrics = []
for train_idx, test_idx in cv.split(X_np):
    opt_fold = LGBMPipeline(clf_params={**study.best_params, "random_state": 42, "n_jobs": -1, "verbose": -1})
    opt_fold.fit(X[train_idx], pl.Series(y_dir_np[train_idx]), pl.Series(y_mag_np[train_idx]))
    pred_proba = np.array(opt_fold.predict_proba(X[test_idx]))
    pred_mag = np.array(opt_fold.predict_magnitude(X[test_idx]))
    m = compute_metrics(y_dir_np[test_idx], pred_proba, y_mag_np[test_idx], pred_mag)
    opt_fold_metrics.append(m)

opt_avg = {
    "accuracy": np.mean([m.accuracy for m in opt_fold_metrics]),
    "auc_roc": np.mean([m.auc_roc for m in opt_fold_metrics]),
    "sharpe": np.mean([m.sharpe_ratio for m in opt_fold_metrics]),
    "max_drawdown": np.mean([m.max_drawdown for m in opt_fold_metrics]),
    "win_rate": np.mean([m.win_rate for m in opt_fold_metrics]),
}
print(f"\n  Optimized AVG: acc={opt_avg['accuracy']:.3f} auc={opt_avg['auc_roc']:.3f} "
      f"sharpe={opt_avg['sharpe']:.2f} win_rate={opt_avg['win_rate']:.3f}")
print(f"  vs Default:    acc={avg['accuracy']:.3f} auc={avg['auc_roc']:.3f} "
      f"sharpe={avg['sharpe']:.2f} win_rate={avg['win_rate']:.3f}")

# Save optimized model
opt_version = model_store.save(optimized_model, metrics=opt_avg)
print(f"  Optimized model saved: {opt_version}")

# Save best params
params_path = Path("data/reports/best_params.json")
params_path.write_text(json.dumps(study.best_params, indent=2))

# ============================================================
# Step 6: Backtest with Walk-Forward simulation
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Walk-Forward Backtest")
print("=" * 60)

# Simple walk-forward: train on first 80%, predict on last 20%
split_idx = int(dataset.height * 0.8)
# Sort by date for temporal split
dataset_sorted = dataset.sort("date")
X_sorted = dataset_sorted.select(feature_cols)
y_dir_sorted = dataset_sorted["label_direction"]
y_mag_sorted = dataset_sorted["label_magnitude"]

train_X = X_sorted[:split_idx]
test_X = X_sorted[split_idx:]
train_y_dir = y_dir_sorted[:split_idx]
test_y_dir = y_dir_sorted[split_idx:]
train_y_mag = y_mag_sorted[:split_idx]
test_y_mag = y_mag_sorted[split_idx:]

print(f"  Train: {train_X.height} rows, Test: {test_X.height} rows")

bt_model = LGBMPipeline(clf_params={**study.best_params, "random_state": 42, "n_jobs": -1, "verbose": -1})
bt_model.fit(train_X, train_y_dir, train_y_mag)

pred_proba = np.array(bt_model.predict_proba(test_X))
pred_mag = np.array(bt_model.predict_magnitude(test_X))

# Simulate trading: go long when confidence > threshold
threshold = cfg.backtest.confidence_threshold
pred_direction = (pred_proba >= threshold).astype(int)
actual_returns = test_y_mag.to_numpy()

# Strategy returns: go long when signal, else flat
strategy_returns = actual_returns * pred_direction
# Account for slippage + commission
cost_per_trade = cfg.backtest.commission_pct + cfg.backtest.slippage_pct
trades = np.diff(pred_direction, prepend=0)
trade_costs = np.abs(trades) * cost_per_trade
strategy_returns_net = strategy_returns - trade_costs

print(f"\n  Signals: {pred_direction.sum()} longs out of {len(pred_direction)} days")
print(f"  Trades: {np.abs(trades).sum():.0f}")

# Metrics
import pandas as pd
test_dates = dataset_sorted["date"][split_idx:].to_list()
returns_series = pd.Series(strategy_returns_net, index=pd.DatetimeIndex(test_dates), name="QTP Strategy")
buy_hold = pd.Series(actual_returns, index=pd.DatetimeIndex(test_dates), name="Buy & Hold")

cumulative_strat = (1 + returns_series).cumprod()
cumulative_bh = (1 + buy_hold).cumprod()

strat_total = cumulative_strat.iloc[-1] - 1
bh_total = cumulative_bh.iloc[-1] - 1

# Sharpe
if returns_series.std() > 0:
    sharpe = (returns_series.mean() / returns_series.std()) * (252 ** 0.5)
else:
    sharpe = 0
max_dd = (cumulative_strat / cumulative_strat.cummax() - 1).min()

wins = strategy_returns_net[strategy_returns_net > 0]
losses = strategy_returns_net[strategy_returns_net < 0]
active_trades = strategy_returns_net[pred_direction == 1]
if len(active_trades) > 0:
    win_rate = len(active_trades[active_trades > 0]) / len(active_trades)
else:
    win_rate = 0

print(f"\n  === Backtest Results (OOS) ===")
print(f"  Strategy Return: {strat_total:+.2%}")
print(f"  Buy & Hold:      {bh_total:+.2%}")
print(f"  Sharpe Ratio:    {sharpe:.2f}")
print(f"  Max Drawdown:    {max_dd:.2%}")
print(f"  Win Rate:        {win_rate:.1%}")
print(f"  Period: {test_dates[0]} → {test_dates[-1]}")

# Generate QuantStats tear sheet
print("\n  Generating QuantStats tear sheet...")
try:
    import quantstats as qs
    report_path = Path("data/reports/backtest_tearsheet.html")
    qs.reports.html(returns_series, benchmark=buy_hold, output=str(report_path),
                    title="QTP Strategy vs Buy & Hold")
    print(f"  Tear sheet saved: {report_path}")
except Exception as e:
    print(f"  Tear sheet generation failed: {e}")

# ============================================================
# Step 7: Generate final predictions with optimized model
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Final Predictions (Optimized Model)")
print("=" * 60)

from qtp.integration.claude_bridge import ClaudeBridge

predictions = []
from qtp.models.base import PredictionResult

for ticker in universe:
    try:
        features = feature_engine.compute_features(
            ticker, market, as_of=today, tiers=cfg.features.tiers
        )
        if features.height == 0:
            continue
        latest = features.tail(1)
        feat_cols = [c for c in latest.columns if c != "date"]
        X_latest = latest.select(feat_cols)
        proba = optimized_model.predict_proba(X_latest)[0]
        magnitude = optimized_model.predict_magnitude(X_latest)[0]

        # Get SHAP values for this prediction
        import shap
        expl = shap.TreeExplainer(optimized_model.clf)
        sv = expl.shap_values(X_latest.to_pandas())
        if isinstance(sv, list):
            sv = sv[1]
        top_shap = sorted(zip(feat_cols, sv[0]), key=lambda x: abs(x[1]), reverse=True)[:5]

        predictions.append(PredictionResult(
            ticker=ticker,
            prediction_date=today + timedelta(days=1),
            direction=1 if proba >= 0.5 else 0,
            direction_proba=proba,
            magnitude=magnitude,
            model_version=optimized_model.version,
            features_used=feat_cols,
        ))

        emoji = "🟢" if proba >= 0.55 else "⚪" if proba >= 0.5 else "🔴"
        conf = "HIGH" if proba >= 0.7 else "MEDIUM" if proba >= 0.55 else "LOW"
        print(f"  {emoji} {ticker:5s} {conf:6s} conf={proba:.1%} ret={magnitude:+.2%}")
        print(f"         Top drivers: {', '.join(f'{n}({v:+.3f})' for n, v in top_shap[:3])}")
    except Exception as e:
        print(f"  ✗ {ticker}: {e}")

# Export
bridge = ClaudeBridge()
output_dir = Path(cfg.reporting.output_dir)
bridge.export_signals(predictions, market.value, output_dir)
bridge.export_markdown_report(predictions, market.value, output_dir)

print(f"\n{'=' * 60}")
print("COMPLETE")
print(f"{'=' * 60}")
print(f"  Model: {optimized_model.version}")
print(f"  Signals: data/reports/latest_signals.json")
print(f"  SHAP plot: data/reports/shap_summary.png")
print(f"  Best params: data/reports/best_params.json")
print(f"  Backtest: data/reports/backtest_tearsheet.html")
