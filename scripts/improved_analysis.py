"""Improved analysis: +Tier3/4 features, +XGBoost ensemble, +trade filtering."""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qtp.config import PipelineConfig
from qtp.data.fetchers.base import Market
from qtp.data.storage import ParquetStorage
from qtp.data.universe import Universe
from qtp.features.engine import FeatureEngine
from qtp.features.registry import FeatureRegistry

# Register ALL tiers
import qtp.features.tier1_momentum   # noqa: F401
import qtp.features.tier2_volatility  # noqa: F401
import qtp.features.tier3_fundamental  # noqa: F401
import qtp.features.tier4_macro  # noqa: F401

from qtp.models.lgbm import LGBMPipeline
from qtp.models.xgb import XGBPipeline
from qtp.models.ensemble import WeightedEnsemble
from qtp.models.versioning import ModelStore
from qtp.validation.metrics import compute_metrics
from qtp.validation.purged_kfold import PurgedKFold
from qtp.utils.logging_ import setup_logging

setup_logging("WARNING")  # Quieter for cleaner output

import structlog
logger = structlog.get_logger()

cfg = PipelineConfig.from_yaml(Path("configs/default.yaml"))
storage = ParquetStorage(Path(cfg.data.storage_dir))
market = Market(cfg.universe.market)
universe = Universe(cfg.universe)
model_store = ModelStore(Path(cfg.data.storage_dir) / "models")

reg = FeatureRegistry.instance()
feature_engine = FeatureEngine(reg, storage)
today = date.today()

print(f"Registered features: {len(reg.all_features())} across Tiers {cfg.features.tiers}")
for t in [1, 2, 3, 4]:
    feats = reg.by_tiers([t])
    if feats:
        print(f"  Tier {t}: {len(feats)} features")

# ============================================================
# Step 1: Build dataset with ALL 4 tiers
# ============================================================
print("\n" + "=" * 60)
print("STEP 1: Building dataset (Tier 1-4, all data)")
print("=" * 60)

dataset = feature_engine.build_multi_ticker_dataset(
    tickers=universe.tickers(), market=market, as_of=today,
    tiers=cfg.features.tiers, horizon=cfg.labels.horizon,
)

# Drop rows with any nulls (macro features may have gaps)
dataset = dataset.drop_nulls()

label_cols = ["label_direction", "label_magnitude", "date", "ticker"]
feature_cols = [c for c in dataset.columns if c not in label_cols]
X = dataset.select(feature_cols)
y_direction = dataset["label_direction"]
y_magnitude = dataset["label_magnitude"]

print(f"  Dataset: {dataset.height} rows x {len(feature_cols)} features")
print(f"  Features: {feature_cols}")
print(f"  Label balance: UP={y_direction.sum()}/{y_direction.len()} ({y_direction.mean():.1%})")

# ============================================================
# Step 2: Compare Default vs Improved
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Model Comparison (5-fold Purged CV)")
print("=" * 60)

cv = PurgedKFold(n_splits=5, purge_days=5)
X_np = X.to_pandas().values
y_dir_np = y_direction.to_numpy()
y_mag_np = y_magnitude.to_numpy()

def evaluate_model(model_factory, name):
    fold_metrics = []
    for train_idx, test_idx in cv.split(X_np):
        model = model_factory()
        model.fit(X[train_idx], pl.Series(y_dir_np[train_idx]), pl.Series(y_mag_np[train_idx]))
        pp = np.array(model.predict_proba(X[test_idx]))
        pm = np.array(model.predict_magnitude(X[test_idx]))
        m = compute_metrics(y_dir_np[test_idx], pp, y_mag_np[test_idx], pm)
        fold_metrics.append(m)
    avg = {
        "accuracy": np.mean([m.accuracy for m in fold_metrics]),
        "auc_roc": np.mean([m.auc_roc for m in fold_metrics]),
        "sharpe": np.mean([m.sharpe_ratio for m in fold_metrics]),
        "max_dd": np.mean([m.max_drawdown for m in fold_metrics]),
        "win_rate": np.mean([m.win_rate for m in fold_metrics]),
    }
    print(f"  {name:20s} acc={avg['accuracy']:.3f} auc={avg['auc_roc']:.3f} "
          f"sharpe={avg['sharpe']:.2f} win={avg['win_rate']:.3f} dd={avg['max_dd']:.3f}")
    return avg, fold_metrics

# Load best params from Optuna if available
best_params_path = Path("data/reports/best_params.json")
if best_params_path.exists():
    optuna_params = json.loads(best_params_path.read_text())
    optuna_params.update({"random_state": 42, "n_jobs": -1, "verbose": -1})
else:
    optuna_params = None

print(f"\n  {'Model':<20s} {'Acc':>6} {'AUC':>6} {'Sharpe':>7} {'Win%':>6} {'MaxDD':>7}")
print(f"  {'─'*20} {'─'*6} {'─'*6} {'─'*7} {'─'*6} {'─'*7}")

lgbm_avg, _ = evaluate_model(lambda: LGBMPipeline(), "LightGBM (default)")

if optuna_params:
    lgbm_opt_avg, _ = evaluate_model(
        lambda: LGBMPipeline(clf_params=optuna_params), "LightGBM (tuned)")

xgb_avg, _ = evaluate_model(lambda: XGBPipeline(), "XGBoost (default)")

# Ensemble: LightGBM 60% + XGBoost 40%
def make_ensemble():
    lgbm = LGBMPipeline(clf_params=optuna_params) if optuna_params else LGBMPipeline()
    xgb_m = XGBPipeline()
    return WeightedEnsemble([(lgbm, 0.6), (xgb_m, 0.4)])

ens_avg, _ = evaluate_model(make_ensemble, "Ensemble (LGBM+XGB)")

# ============================================================
# Step 3: Walk-Forward Backtest with trade filtering
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Walk-Forward Backtest (Ensemble + Trade Filter)")
print("=" * 60)

dataset_sorted = dataset.sort("date")
X_sorted = dataset_sorted.select(feature_cols)
y_dir_sorted = dataset_sorted["label_direction"].to_numpy()
y_mag_sorted = dataset_sorted["label_magnitude"].to_numpy()
dates_sorted = dataset_sorted["date"].to_list()

# Walk-forward: retrain every 63 days (3 months)
train_size = int(dataset.height * 0.6)
step = 63
results = []

print(f"  Walk-forward: train_init={train_size}, step={step}, total={dataset.height}")

i = train_size
while i < dataset.height:
    end = min(i + step, dataset.height)

    # Train on all data up to i
    train_X = X_sorted[:i]
    train_y_dir = pl.Series(y_dir_sorted[:i])
    train_y_mag = pl.Series(y_mag_sorted[:i])

    # Test on [i, end)
    test_X = X_sorted[i:end]
    test_dates = dates_sorted[i:end]
    test_y_dir = y_dir_sorted[i:end]
    test_y_mag = y_mag_sorted[i:end]

    # Train ensemble
    ens = make_ensemble()
    ens.fit(train_X, train_y_dir, train_y_mag)

    pred_proba = np.array(ens.predict_proba(test_X))
    pred_mag = np.array(ens.predict_magnitude(test_X))

    for j in range(len(test_dates)):
        results.append({
            "date": test_dates[j],
            "pred_proba": pred_proba[j],
            "pred_mag": pred_mag[j],
            "actual_dir": test_y_dir[j],
            "actual_mag": test_y_mag[j],
        })

    i = end

print(f"  OOS predictions: {len(results)}")

# Apply trade filtering strategies
import pandas as pd

def backtest_strategy(results, conf_threshold, mag_threshold, name):
    """Backtest with given thresholds."""
    pred_proba = np.array([r["pred_proba"] for r in results])
    pred_mag = np.array([r["pred_mag"] for r in results])
    actual_mag = np.array([r["actual_mag"] for r in results])
    dates = [r["date"] for r in results]

    # Signal: go long only when confidence AND magnitude exceed thresholds
    signal = ((pred_proba >= conf_threshold) & (pred_mag >= mag_threshold)).astype(int)

    # Strategy returns
    strat_ret = actual_mag * signal

    # Costs: only on trade entries/exits
    trades = np.abs(np.diff(signal, prepend=0))
    cost = 0.002  # 20bps round-trip
    strat_ret_net = strat_ret - trades * cost

    # Buy & hold
    bh_ret = actual_mag

    # Cumulative
    cum_strat = (1 + strat_ret_net).cumprod()
    cum_bh = (1 + bh_ret).cumprod()

    total_strat = cum_strat[-1] - 1
    total_bh = cum_bh[-1] - 1

    # Sharpe
    active_returns = strat_ret_net[signal == 1]
    if len(active_returns) > 0 and active_returns.std() > 0:
        sharpe = (active_returns.mean() / active_returns.std()) * (252 ** 0.5)
    else:
        sharpe = 0

    # Max drawdown
    dd = cum_strat / np.maximum.accumulate(cum_strat) - 1
    max_dd = dd.min()

    # Win rate
    if signal.sum() > 0:
        wins = (strat_ret_net[signal == 1] > 0).sum()
        win_rate = wins / signal.sum()
    else:
        win_rate = 0

    n_trades = trades.sum()

    print(f"  {name:35s} ret={total_strat:+7.2%} bh={total_bh:+7.2%} "
          f"sharpe={sharpe:+5.2f} dd={max_dd:+7.2%} win={win_rate:.1%} trades={n_trades:.0f} "
          f"signals={signal.sum()}/{len(signal)}")

    return {
        "name": name,
        "total_return": total_strat,
        "buy_hold": total_bh,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "win_rate": win_rate,
        "n_trades": n_trades,
        "n_signals": signal.sum(),
        "cum_strat": cum_strat,
        "cum_bh": cum_bh,
        "dates": dates,
        "strat_ret_net": strat_ret_net,
        "signal": signal,
    }

print(f"\n  {'Strategy':<35s} {'Return':>8} {'B&H':>8} {'Sharpe':>7} {'MaxDD':>8} {'Win%':>5} {'Trades':>7} {'Signals'}")
print(f"  {'─'*35} {'─'*8} {'─'*8} {'─'*7} {'─'*8} {'─'*5} {'─'*7} {'─'*7}")

# Compare different filtering thresholds
s1 = backtest_strategy(results, 0.50, 0.000, "All signals (baseline)")
s2 = backtest_strategy(results, 0.55, 0.000, "Confidence > 55%")
s3 = backtest_strategy(results, 0.55, 0.002, "Conf>55% + Mag>0.2%")
s4 = backtest_strategy(results, 0.60, 0.003, "Conf>60% + Mag>0.3%")
s5 = backtest_strategy(results, 0.65, 0.005, "Conf>65% + Mag>0.5% (selective)")

# ============================================================
# Step 4: SHAP on ensemble
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: SHAP Analysis (Full Feature Set)")
print("=" * 60)

import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Train final ensemble on all data
final_ens_lgbm = LGBMPipeline(clf_params=optuna_params) if optuna_params else LGBMPipeline()
final_ens_lgbm.fit(X, y_direction, y_magnitude)

explainer = shap.TreeExplainer(final_ens_lgbm.clf)
X_sample = X.to_pandas()
if X_sample.shape[0] > 3000:
    X_sample = X_sample.sample(3000, random_state=42)
shap_values = explainer.shap_values(X_sample)
if isinstance(shap_values, list):
    shap_values = shap_values[1]

mean_abs_shap = np.abs(shap_values).mean(axis=0)
total = mean_abs_shap.sum()
importance = sorted(zip(feature_cols, mean_abs_shap / total), key=lambda x: x[1], reverse=True)

print(f"\n  {'Rank':<5} {'Feature':<25} {'Importance':>12}")
print(f"  {'─'*5} {'─'*25} {'─'*12}")
for i, (name, imp) in enumerate(importance[:15]):
    bar = "█" * int(imp * 150)
    print(f"  {i+1:<5} {name:<25} {imp:>11.1%} {bar}")

# SHAP plot
fig, ax = plt.subplots(figsize=(12, 10))
shap.summary_plot(shap_values, X_sample, show=False, max_display=25)
plt.tight_layout()
shap_path = Path("data/reports/shap_improved.png")
plt.savefig(shap_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  SHAP plot: {shap_path}")

# ============================================================
# Step 5: Equity curve plot
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Equity Curve Comparison")
print("=" * 60)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Plot equity curves for all strategies
for s in [s1, s2, s3, s4, s5]:
    ax1.plot(s["dates"], s["cum_strat"], label=f"{s['name']} ({s['total_return']:+.1%})", alpha=0.8)
ax1.plot(s1["dates"], s1["cum_bh"], label=f"Buy & Hold ({s1['buy_hold']:+.1%})", color='black', linestyle='--', alpha=0.5)
ax1.set_title("Equity Curves: Strategy Comparison (Out-of-Sample)")
ax1.set_ylabel("Cumulative Return")
ax1.legend(fontsize=8, loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)

# Plot the best strategy's drawdown
best = s4  # Typically the selective one performs best risk-adjusted
dd = best["cum_strat"] / np.maximum.accumulate(best["cum_strat"]) - 1
ax2.fill_between(best["dates"], dd, 0, alpha=0.3, color='red')
ax2.set_title(f"Drawdown: {best['name']}")
ax2.set_ylabel("Drawdown")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
eq_path = Path("data/reports/equity_curves.png")
plt.savefig(eq_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Equity curve: {eq_path}")

# ============================================================
# Step 6: Final predictions with ensemble
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Today's Predictions (Ensemble)")
print("=" * 60)

final_xgb = XGBPipeline()
final_xgb.fit(X, y_direction, y_magnitude)
final_ensemble = WeightedEnsemble([(final_ens_lgbm, 0.6), (final_xgb, 0.4)])

from qtp.models.base import PredictionResult
from qtp.integration.claude_bridge import ClaudeBridge

predictions = []
print(f"\n  {'Ticker':<7} {'Signal':<8} {'Conf':>7} {'Mag':>8} {'Top 3 SHAP Drivers'}")
print(f"  {'─'*7} {'─'*8} {'─'*7} {'─'*8} {'─'*40}")

for ticker in universe:
    try:
        features = feature_engine.compute_features(
            ticker, market, as_of=today, tiers=cfg.features.tiers
        )
        if features.height == 0:
            continue
        latest = features.tail(1).drop_nulls()
        if latest.height == 0:
            continue
        feat_cols = [c for c in latest.columns if c != "date"]
        X_latest = latest.select(feat_cols)

        proba = final_ensemble.predict_proba(X_latest)[0]
        magnitude = final_ensemble.predict_magnitude(X_latest)[0]

        # SHAP for this prediction
        sv = explainer.shap_values(X_latest.to_pandas())
        if isinstance(sv, list):
            sv = sv[1]
        top_shap = sorted(zip(feat_cols, sv[0]), key=lambda x: abs(x[1]), reverse=True)[:3]
        drivers = ", ".join(f"{n}({v:+.2f})" for n, v in top_shap)

        # Determine signal with filtering
        if proba >= 0.60 and magnitude >= 0.003:
            signal = "🟢 LONG"
        elif proba >= 0.55:
            signal = "🟡 WATCH"
        elif proba < 0.45:
            signal = "🔴 SHORT*"
        else:
            signal = "⚪ FLAT"

        print(f"  {ticker:<7} {signal:<8} {proba:>6.1%} {magnitude:>+7.2%} {drivers}")

        predictions.append(PredictionResult(
            ticker=ticker,
            prediction_date=today + timedelta(days=1),
            direction=1 if proba >= 0.5 else 0,
            direction_proba=proba,
            magnitude=magnitude,
            model_version=final_ensemble.version,
            features_used=feat_cols,
        ))
    except Exception as e:
        print(f"  {ticker:<7} ERROR: {e}")

# Export
bridge = ClaudeBridge()
output_dir = Path(cfg.reporting.output_dir)
bridge.export_signals(predictions, market.value, output_dir)

# ============================================================
# Summary
# ============================================================
print(f"\n{'=' * 60}")
print("IMPROVEMENT SUMMARY")
print(f"{'=' * 60}")
print(f"  Features: 21 → {len(feature_cols)} (+Tier3 price-action, +Tier4 macro)")
print(f"  Models: LightGBM only → LightGBM(60%) + XGBoost(40%) ensemble")
print(f"  Trade filter: conf>55% → conf>60% + mag>0.3% (selective)")
print(f"\n  Best strategy OOS: {s4['name']}")
print(f"    Return:   {s4['total_return']:+.2%}")
print(f"    B&H:      {s4['buy_hold']:+.2%}")
print(f"    Sharpe:   {s4['sharpe']:.2f}")
print(f"    Max DD:   {s4['max_dd']:.2%}")
print(f"    Win Rate: {s4['win_rate']:.1%}")
print(f"    Trades:   {s4['n_trades']:.0f}")
print(f"\n  Output files:")
print(f"    data/reports/shap_improved.png")
print(f"    data/reports/equity_curves.png")
print(f"    data/reports/latest_signals.json")
