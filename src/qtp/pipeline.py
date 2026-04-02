"""Pipeline orchestrator — wires fetch → features → train → backtest → predict."""

from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import structlog

from qtp.config import PipelineConfig
from qtp.data.database import QTPDatabase
from qtp.data.fetchers.base import FetchRequest, Market
from qtp.data.fetchers.yfinance_ import YFinanceFetcher
from qtp.data.storage import ParquetStorage
from qtp.data.universe import Universe
from qtp.data.validator import DataValidator
from qtp.features.engine import FeatureEngine
from qtp.features.registry import FeatureRegistry
from qtp.models.base import PredictionResult
from qtp.models.lgbm import LGBMPipeline
from qtp.models.versioning import ModelStore
from qtp.validation.metrics import EvaluationMetrics, compute_metrics
from qtp.validation.purged_kfold import PurgedKFold
from qtp.validation.walk_forward import ExpandingWindowCV

logger = structlog.get_logger()


class PipelineRunner:
    def __init__(self, config: PipelineConfig, project_dir: Path | None = None):
        self.config = config
        self.project_dir = project_dir or Path(".")
        self.storage = ParquetStorage(self.project_dir / config.data.storage_dir)
        self.validator = DataValidator()
        self.market = Market(config.universe.market)
        self.universe = Universe(config.universe)
        self.model_store = ModelStore(self.project_dir / config.data.storage_dir / "models")

        # Import feature definitions (triggers registration)
        import qtp.features.tier1_momentum  # noqa: F401
        import qtp.features.tier2_volatility  # noqa: F401
        import qtp.features.tier3_fundamental  # noqa: F401
        import qtp.features.tier4_macro  # noqa: F401
        import qtp.features.tier5_alternative  # noqa: F401
        import qtp.features.tier5_timeseries  # noqa: F401

        self.db = QTPDatabase(self.project_dir / config.data.storage_dir / "qtp.db")
        self.feature_engine = FeatureEngine(FeatureRegistry.instance(), self.storage)

    def _create_fetcher(self):
        if self.config.data.fetcher == "yfinance":
            return YFinanceFetcher()
        raise ValueError(f"Unknown fetcher: {self.config.data.fetcher}")

    def run_fetch(self) -> None:
        """Fetch OHLCV data for all tickers in the universe."""
        fetcher = self._create_fetcher()
        today = date.today()
        start = today - timedelta(days=self.config.data.history_days)

        for ticker in self.universe:
            logger.info("fetching", ticker=ticker)
            try:
                df = fetcher.fetch_ohlcv(
                    FetchRequest(
                        ticker=ticker,
                        market=self.market,
                        start_date=start,
                        end_date=today,
                    )
                )
                result = self.validator.validate_ohlcv(df, as_of=today)
                if not result.passed:
                    logger.warning("validation_failed", ticker=ticker, issues=result.issues)
                if df.height > 0:
                    self.storage.save_ohlcv(ticker, self.market, df)
            except Exception as e:
                logger.error("fetch_failed", ticker=ticker, error=str(e))

    def run_train(self, fast: bool = False) -> str:
        """Train model and return version string.

        Args:
            fast: If True, limit Walk-Forward CV to 3 folds for quick iteration.
        """
        t0 = time.monotonic()
        today = date.today()
        tiers = self.config.features.tiers

        # Build multi-ticker dataset
        dataset = self.feature_engine.build_multi_ticker_dataset(
            tickers=self.universe.tickers(),
            market=self.market,
            as_of=today,
            tiers=tiers,
            horizon=self.config.labels.horizon,
            direction_threshold=self.config.labels.direction_threshold,
        )

        if dataset.height == 0:
            raise RuntimeError("No data available for training")

        # Separate features, labels
        label_cols = ["label_direction", "label_magnitude", "date", "ticker"]
        feature_cols = [c for c in dataset.columns if c not in label_cols]

        X = dataset.select(feature_cols)
        y_direction = dataset["label_direction"]
        y_magnitude = dataset["label_magnitude"]

        logger.info(
            "training_dataset",
            rows=X.height,
            features=len(feature_cols),
            tickers=len(self.universe),
        )

        # Train
        model = LGBMPipeline()
        model.fit(X, y_direction, y_magnitude)

        # Primary evaluation: Expanding Window Walk-Forward CV
        wf_cv = ExpandingWindowCV(
            min_train_size=self.config.validation.walk_forward_train_days,
            test_size=self.config.validation.walk_forward_test_days,
            step_size=self.config.validation.walk_forward_step_days,
            purge_gap=self.config.validation.dev_cv_purge_days,
        )

        X_np = X.to_pandas().values
        y_dir_np = y_direction.to_numpy()
        y_mag_np = y_magnitude.to_numpy()

        commission_bps = self.config.backtest.commission_pct * 100  # to bps
        slippage_bps = self.config.backtest.slippage_pct * 100

        max_wf_folds = 3 if fast else None
        if fast:
            logger.info("fast_mode", max_wf_folds=max_wf_folds, skip_pkf=True)

        wf_metrics: list[EvaluationMetrics] = []
        for fold_i, (train_idx, test_idx) in enumerate(wf_cv.split(X_np)):
            if max_wf_folds and fold_i >= max_wf_folds:
                break
            fold_model = LGBMPipeline()
            fold_model.fit(
                X[train_idx],
                pl.Series(y_dir_np[train_idx]),
                pl.Series(y_mag_np[train_idx]),
            )
            pred_proba = np.array(fold_model.predict_proba(X[test_idx]))
            pred_mag = np.array(fold_model.predict_magnitude(X[test_idx]))

            metrics = compute_metrics(
                y_dir_np[test_idx],
                pred_proba,
                y_mag_np[test_idx],
                pred_mag,
                commission_bps=commission_bps,
                slippage_bps=slippage_bps,
            )
            wf_metrics.append(metrics)
            logger.info(
                "wf_fold",
                fold=fold_i,
                train=len(train_idx),
                test=len(test_idx),
                auc=round(metrics.auc_roc, 4),
                sharpe=round(metrics.sharpe_ratio, 4),
            )

        # Auxiliary: PurgedKFold (skip in fast mode)
        pkf_metrics: list[EvaluationMetrics] = []
        if not fast:
            pkf_cv = PurgedKFold(
                n_splits=self.config.validation.dev_cv_splits,
                purge_days=self.config.validation.dev_cv_purge_days,
            )
            for train_idx, test_idx in pkf_cv.split(X_np):
                fold_model = LGBMPipeline()
                fold_model.fit(
                    X[train_idx],
                    pl.Series(y_dir_np[train_idx]),
                    pl.Series(y_mag_np[train_idx]),
                )
                pred_proba = np.array(fold_model.predict_proba(X[test_idx]))
                pred_mag = np.array(fold_model.predict_magnitude(X[test_idx]))
                m = compute_metrics(
                    y_dir_np[test_idx],
                    pred_proba,
                    y_mag_np[test_idx],
                    pred_mag,
                    commission_bps=commission_bps,
                    slippage_bps=slippage_bps,
                )
                pkf_metrics.append(m)

        # Average CV metrics (Walk-Forward = primary)
        # Use nanmean to handle any residual nan values gracefully
        avg_metrics = {
            "wf_accuracy": float(np.nanmean([m.accuracy for m in wf_metrics])),
            "wf_auc_roc": float(np.nanmean([m.auc_roc for m in wf_metrics])),
            "wf_sharpe": float(np.nanmean([m.sharpe_ratio for m in wf_metrics])),
            "wf_max_drawdown": float(np.nanmean([m.max_drawdown for m in wf_metrics])),
            "wf_win_rate": float(np.nanmean([m.win_rate for m in wf_metrics])),
            "wf_n_folds": len(wf_metrics),
            "pkf_auc_roc": float(np.nanmean([m.auc_roc for m in pkf_metrics]))
            if pkf_metrics
            else 0.0,
            "pkf_sharpe": float(np.nanmean([m.sharpe_ratio for m in pkf_metrics]))
            if pkf_metrics
            else 0.0,
        }
        logger.info(
            "cv_results",
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in avg_metrics.items()},
        )

        # Save model
        version = self.model_store.save(model, metrics=avg_metrics)

        # Register in SQLite
        self.db.register_model(
            version=version,
            model_type="lgbm",
            model_path=str(self.project_dir / self.config.data.storage_dir / "models" / version),
            config=self.config.model_dump(),
            metrics={
                str(k): float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v
                for k, v in avg_metrics.items()
            },
            feature_names=feature_cols,
        )

        # Log experiment
        duration = time.monotonic() - t0
        avg_metrics["n_tickers"] = len(self.universe)
        avg_metrics["n_samples"] = X.height
        self.db.log_experiment(
            config=self.config.model_dump(),
            metrics={
                str(k): float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v
                for k, v in avg_metrics.items()
            },
            model_version=version,
            duration_seconds=duration,
        )
        logger.info("experiment_logged", duration=f"{duration:.1f}s")

        return version

    def run_predict(self, model_version: str | None = None) -> list[PredictionResult]:
        """Generate predictions for today."""
        today = date.today()
        model = (
            self.model_store.load(model_version)
            if model_version
            else self.model_store.load_latest()
        )

        predictions: list[PredictionResult] = []
        for ticker in self.universe:
            try:
                features = self.feature_engine.compute_features(
                    ticker,
                    self.market,
                    as_of=today,
                    tiers=self.config.features.tiers,
                )
                if features.height == 0:
                    continue

                # Use only the latest row for prediction
                latest = features.tail(1)
                feature_cols = [c for c in latest.columns if c != "date"]
                X = latest.select(feature_cols)

                proba = model.predict_proba(X)[0]
                magnitude = model.predict_magnitude(X)[0]

                predictions.append(
                    PredictionResult(
                        ticker=ticker,
                        prediction_date=today + timedelta(days=1),
                        direction=1 if proba >= 0.5 else 0,
                        direction_proba=proba,
                        magnitude=magnitude,
                        model_version=model.version,
                        features_used=feature_cols,
                    )
                )
            except Exception as e:
                logger.error("predict_failed", ticker=ticker, error=str(e))

        # Save predictions to SQLite for later grading
        for p in predictions:
            self.db.save_prediction(
                ticker=p.ticker,
                prediction_date=p.prediction_date.isoformat(),
                direction=p.direction,
                confidence=p.direction_proba,
                predicted_magnitude=p.magnitude,
                model_version=p.model_version,
                horizon=self.config.labels.horizon,
            )

        logger.info("predictions_generated", count=len(predictions), saved_to_db=True)
        return predictions

    def run_all(self, fast: bool = False) -> dict:
        """Full pipeline: fetch → train → predict."""
        logger.info("pipeline_start", fast=fast)
        self.run_fetch()
        version = self.run_train(fast=fast)
        predictions = self.run_predict(version)

        # Export signals for Claude Code integration
        from qtp.integration.claude_bridge import ClaudeBridge

        bridge = ClaudeBridge()
        output_dir = self.project_dir / self.config.reporting.output_dir
        bridge.export_signals(predictions, self.market.value, output_dir)
        bridge.export_markdown_report(predictions, self.market.value, output_dir)

        logger.info("pipeline_complete", model_version=version, predictions=len(predictions))
        return {
            "model_version": version,
            "predictions": predictions,
        }
