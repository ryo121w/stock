"""LightGBM + XGBoost ensemble with Optuna hyperparameter tuning.

Loads Phase 3 config (h10_t3%, 7 tickers), runs 30 Optuna trials
optimizing ensemble accuracy at confidence >= 0.55 via 3-fold expanding window CV.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import optuna  # noqa: E402
import polars as pl  # noqa: E402
import structlog  # noqa: E402

# Import feature definitions (triggers registration)
import qtp.features.tier1_momentum  # noqa: E402, F401
import qtp.features.tier2_volatility  # noqa: E402, F401
import qtp.features.tier3_fundamental  # noqa: E402, F401
import qtp.features.tier4_macro  # noqa: E402, F401
import qtp.features.tier5_alternative  # noqa: E402, F401
from qtp.config import PipelineConfig  # noqa: E402
from qtp.data.storage import ParquetStorage  # noqa: E402
from qtp.data.universe import Universe  # noqa: E402
from qtp.features.engine import FeatureEngine  # noqa: E402
from qtp.features.registry import FeatureRegistry  # noqa: E402
from qtp.models.ensemble import WeightedEnsemble  # noqa: E402
from qtp.models.lgbm import LGBMPipeline  # noqa: E402
from qtp.models.xgb import XGBPipeline  # noqa: E402
from qtp.validation.walk_forward import ExpandingWindowCV  # noqa: E402

logger = structlog.get_logger()

PROJECT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_DIR / "configs" / "phase3_best.yaml"
N_TRIALS = 30
N_FOLDS = 3
STEP_SIZE = 252  # ~1 year steps for speed
CONF_THRESHOLD = 0.55


def load_dataset(config: PipelineConfig) -> tuple[pl.DataFrame, pl.Series, pl.Series, list[str]]:
    """Load and prepare the multi-ticker dataset."""
    storage = ParquetStorage(PROJECT_DIR / config.data.storage_dir)
    from qtp.data.fetchers.base import Market

    market = Market(config.universe.market)
    universe = Universe(config.universe)
    feature_engine = FeatureEngine(FeatureRegistry.instance(), storage)

    dataset = feature_engine.build_multi_ticker_dataset(
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

    return X, y_direction, y_magnitude, feature_cols


def evaluate_ensemble_cv(
    X: pl.DataFrame,
    y_direction: pl.Series,
    y_magnitude: pl.Series,
    lgbm_clf_params: dict,
    xgb_clf_params: dict,
    lgbm_weight: float,
) -> float:
    """Run 3-fold expanding window CV and return mean accuracy at conf >= 0.55."""
    cv = ExpandingWindowCV(
        min_train_size=504,
        test_size=63,
        step_size=STEP_SIZE,
        purge_gap=5,
    )

    y_dir_np = y_direction.to_numpy()
    y_mag_np = y_magnitude.to_numpy()
    X_np_for_split = X.to_pandas().values

    fold_accuracies = []
    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X_np_for_split)):
        if fold_i >= N_FOLDS:
            break

        X_train = X[train_idx]
        X_test = X[test_idx]
        y_dir_train = pl.Series(y_dir_np[train_idx])
        y_mag_train = pl.Series(y_mag_np[train_idx])
        y_dir_test = y_dir_np[test_idx]

        # Train LightGBM
        lgbm = LGBMPipeline(clf_params=lgbm_clf_params)
        lgbm.fit(X_train, y_dir_train, y_mag_train)

        # Train XGBoost
        xgb_model = XGBPipeline(clf_params=xgb_clf_params)
        xgb_model.fit(X_train, y_dir_train, y_mag_train)

        # Ensemble predictions
        ensemble = WeightedEnsemble(
            [
                (lgbm, lgbm_weight),
                (xgb_model, 1.0 - lgbm_weight),
            ]
        )
        probas = np.array(ensemble.predict_proba(X_test))

        # Accuracy at confidence >= threshold
        conf_mask = np.abs(probas - 0.5) >= (CONF_THRESHOLD - 0.5)
        if conf_mask.sum() > 0:
            preds = (probas[conf_mask] >= 0.5).astype(int)
            actual = y_dir_test[conf_mask]
            acc = (preds == actual).mean()
            fold_accuracies.append(float(acc))

    if not fold_accuracies:
        return 0.0
    return float(np.mean(fold_accuracies))


def objective(
    trial: optuna.Trial, X: pl.DataFrame, y_direction: pl.Series, y_magnitude: pl.Series
) -> float:
    """Optuna objective: maximize accuracy at conf >= 0.55."""

    # LightGBM hyperparams
    lgbm_lr = trial.suggest_float("lgbm_learning_rate", 0.005, 0.1, log=True)
    lgbm_max_depth = trial.suggest_int("lgbm_max_depth", 3, 8)
    lgbm_num_leaves = trial.suggest_int("lgbm_num_leaves", 15, 63)
    lgbm_min_child = trial.suggest_int("lgbm_min_child_samples", 10, 50)
    lgbm_subsample = trial.suggest_float("lgbm_subsample", 0.6, 1.0)
    lgbm_colsample = trial.suggest_float("lgbm_colsample_bytree", 0.5, 1.0)
    lgbm_reg_alpha = trial.suggest_float("lgbm_reg_alpha", 1e-3, 10.0, log=True)
    lgbm_reg_lambda = trial.suggest_float("lgbm_reg_lambda", 1e-3, 10.0, log=True)

    lgbm_clf_params = {
        "n_estimators": 1000,
        "learning_rate": lgbm_lr,
        "max_depth": lgbm_max_depth,
        "num_leaves": lgbm_num_leaves,
        "min_child_samples": lgbm_min_child,
        "subsample": lgbm_subsample,
        "colsample_bytree": lgbm_colsample,
        "reg_alpha": lgbm_reg_alpha,
        "reg_lambda": lgbm_reg_lambda,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    # XGBoost hyperparams
    xgb_lr = trial.suggest_float("xgb_learning_rate", 0.005, 0.1, log=True)
    xgb_max_depth = trial.suggest_int("xgb_max_depth", 3, 8)
    xgb_min_child_weight = trial.suggest_float("xgb_min_child_weight", 1.0, 10.0)
    xgb_subsample = trial.suggest_float("xgb_subsample", 0.6, 1.0)
    xgb_colsample = trial.suggest_float("xgb_colsample_bytree", 0.5, 1.0)
    xgb_reg_alpha = trial.suggest_float("xgb_reg_alpha", 1e-3, 10.0, log=True)
    xgb_reg_lambda = trial.suggest_float("xgb_reg_lambda", 1e-3, 10.0, log=True)

    xgb_clf_params = {
        "n_estimators": 800,
        "learning_rate": xgb_lr,
        "max_depth": xgb_max_depth,
        "min_child_weight": xgb_min_child_weight,
        "subsample": xgb_subsample,
        "colsample_bytree": xgb_colsample,
        "reg_alpha": xgb_reg_alpha,
        "reg_lambda": xgb_reg_lambda,
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "logloss",
        "verbosity": 0,
        "early_stopping_rounds": 50,
    }

    # Ensemble weight
    lgbm_weight = trial.suggest_float("lgbm_weight", 0.3, 0.8)

    acc = evaluate_ensemble_cv(
        X, y_direction, y_magnitude, lgbm_clf_params, xgb_clf_params, lgbm_weight
    )
    return acc


def run_default_baseline(X: pl.DataFrame, y_direction: pl.Series, y_magnitude: pl.Series) -> float:
    """Evaluate default params as baseline."""
    lgbm_defaults = LGBMPipeline._default_clf_params()
    xgb_defaults = XGBPipeline._default_clf_params()
    default_weight = 0.6  # from config default

    return evaluate_ensemble_cv(
        X, y_direction, y_magnitude, lgbm_defaults, xgb_defaults, default_weight
    )


def main():
    t0 = time.monotonic()
    print("=" * 70)
    print("LightGBM + XGBoost Ensemble Tuning with Optuna")
    print("=" * 70)

    # Load config
    config = PipelineConfig.from_yaml(CONFIG_PATH)
    print(f"\nConfig: {CONFIG_PATH.name}")
    print(f"Tickers: {config.universe.tickers}")
    print(f"Horizon: {config.labels.horizon}, Threshold: {config.labels.direction_threshold}")

    # Load dataset
    print("\nLoading dataset...")
    X, y_direction, y_magnitude, feature_cols = load_dataset(config)
    print(f"Dataset: {X.height} rows, {len(feature_cols)} features")

    # Baseline with default params
    print("\n--- Baseline (default params) ---")
    baseline_acc = run_default_baseline(X, y_direction, y_magnitude)
    print(f"Default ensemble accuracy (conf >= {CONF_THRESHOLD}): {baseline_acc:.4f}")

    # Optuna study
    print(f"\n--- Optuna Tuning ({N_TRIALS} trials, {N_FOLDS}-fold CV, step={STEP_SIZE}) ---")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", study_name="ensemble_tuning")

    def obj_wrapper(trial):
        return objective(trial, X, y_direction, y_magnitude)

    study.optimize(obj_wrapper, n_trials=N_TRIALS, show_progress_bar=True)

    # Results
    best = study.best_trial
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nBaseline accuracy (default params): {baseline_acc:.4f}")
    print(f"Best trial accuracy (tuned):        {best.value:.4f}")
    improvement = best.value - baseline_acc
    print(
        f"Improvement:                        {improvement:+.4f} ({improvement / max(baseline_acc, 1e-6) * 100:+.1f}%)"
    )

    print(f"\nBest trial #{best.number}:")
    print(f"  LightGBM weight: {best.params['lgbm_weight']:.3f}")
    print(f"  XGBoost weight:  {1 - best.params['lgbm_weight']:.3f}")
    print("\n  LightGBM params:")
    for k, v in sorted(best.params.items()):
        if k.startswith("lgbm_") and k != "lgbm_weight":
            print(f"    {k}: {v}")
    print("\n  XGBoost params:")
    for k, v in sorted(best.params.items()):
        if k.startswith("xgb_"):
            print(f"    {k}: {v}")

    # Save best params
    output = {
        "study_name": "ensemble_tuning",
        "n_trials": N_TRIALS,
        "n_folds": N_FOLDS,
        "step_size": STEP_SIZE,
        "confidence_threshold": CONF_THRESHOLD,
        "baseline_accuracy": round(baseline_acc, 6),
        "best_accuracy": round(best.value, 6),
        "improvement": round(improvement, 6),
        "best_trial_number": best.number,
        "lgbm_weight": best.params["lgbm_weight"],
        "xgb_weight": 1 - best.params["lgbm_weight"],
        "lgbm_clf_params": {
            "n_estimators": 1000,
            "learning_rate": best.params["lgbm_learning_rate"],
            "max_depth": best.params["lgbm_max_depth"],
            "num_leaves": best.params["lgbm_num_leaves"],
            "min_child_samples": best.params["lgbm_min_child_samples"],
            "subsample": best.params["lgbm_subsample"],
            "colsample_bytree": best.params["lgbm_colsample_bytree"],
            "reg_alpha": best.params["lgbm_reg_alpha"],
            "reg_lambda": best.params["lgbm_reg_lambda"],
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        },
        "xgb_clf_params": {
            "n_estimators": 800,
            "learning_rate": best.params["xgb_learning_rate"],
            "max_depth": best.params["xgb_max_depth"],
            "min_child_weight": best.params["xgb_min_child_weight"],
            "subsample": best.params["xgb_subsample"],
            "colsample_bytree": best.params["xgb_colsample_bytree"],
            "reg_alpha": best.params["xgb_reg_alpha"],
            "reg_lambda": best.params["xgb_reg_lambda"],
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "logloss",
            "verbosity": 0,
            "early_stopping_rounds": 50,
        },
    }

    output_path = PROJECT_DIR / "configs" / "best_params.json"
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nBest params saved to: {output_path}")

    duration = time.monotonic() - t0
    print(f"\nTotal time: {duration:.0f}s ({duration / 60:.1f} min)")


if __name__ == "__main__":
    main()
