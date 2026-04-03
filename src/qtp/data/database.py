"""SQLite database manager for QTP metadata, alternative data, and experiment tracking.

Tables:
  - alternative_data: MCP tool responses cached per ticker/tool/date
  - model_registry: Model versions with metrics and config snapshots
  - experiments: Pipeline run history with config + CV metrics
  - feature_cache: Pre-computed feature values (optional, for expensive features)

OHLCV time-series data stays in Parquet (better for columnar analytics with Polars).
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

import structlog

logger = structlog.get_logger()

SCHEMA_VERSION = 3

SCHEMA_SQL = """
-- Alternative data from MCP tools (earnings_trend, analyst_actions, etc.)
-- Legacy table: single snapshot per ticker/tool (kept for backward compatibility)
CREATE TABLE IF NOT EXISTS alternative_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    tool TEXT NOT NULL,
    data JSON NOT NULL,
    fetched_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    UNIQUE(ticker, tool)
);
CREATE INDEX IF NOT EXISTS idx_alt_ticker ON alternative_data(ticker);
CREATE INDEX IF NOT EXISTS idx_alt_freshness ON alternative_data(fetched_at);

-- Daily alternative data accumulation (new: supports time-series analysis)
CREATE TABLE IF NOT EXISTS alternative_data_daily (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    tool TEXT NOT NULL,
    date TEXT NOT NULL,
    data JSON NOT NULL,
    fetched_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, tool, date)
);
CREATE INDEX IF NOT EXISTS idx_alt_daily_ticker ON alternative_data_daily(ticker);
CREATE INDEX IF NOT EXISTS idx_alt_daily_date ON alternative_data_daily(date);
CREATE INDEX IF NOT EXISTS idx_alt_daily_lookup ON alternative_data_daily(ticker, tool, date);

-- Model registry
CREATE TABLE IF NOT EXISTS model_registry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version TEXT NOT NULL UNIQUE,
    model_type TEXT NOT NULL,
    model_path TEXT NOT NULL,
    config JSON,
    metrics JSON,
    feature_names JSON,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_model_version ON model_registry(version);
CREATE INDEX IF NOT EXISTS idx_model_created ON model_registry(created_at);

-- Experiment tracking
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    config JSON NOT NULL,
    label_horizon INTEGER,
    label_threshold REAL,
    feature_tiers JSON,
    n_tickers INTEGER,
    n_samples INTEGER,
    -- Walk-Forward metrics (primary)
    wf_auc REAL,
    wf_accuracy REAL,
    wf_sharpe REAL,
    wf_max_drawdown REAL,
    wf_win_rate REAL,
    wf_n_folds INTEGER,
    -- PurgedKFold metrics (auxiliary)
    pkf_auc REAL,
    pkf_sharpe REAL,
    -- Model info
    model_version TEXT,
    -- Metadata
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    duration_seconds REAL,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_exp_auc ON experiments(wf_auc);
CREATE INDEX IF NOT EXISTS idx_exp_created ON experiments(created_at);

-- Prediction tracking & grading
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    prediction_date TEXT NOT NULL,       -- Date the prediction is FOR
    horizon INTEGER NOT NULL DEFAULT 1,  -- Prediction horizon in days
    direction INTEGER NOT NULL,          -- 1=up, 0=down
    confidence REAL NOT NULL,            -- Model confidence [0, 1]
    predicted_magnitude REAL,            -- Expected return magnitude
    model_version TEXT,
    -- Grading (filled in later by grade_predictions)
    actual_price_start REAL,             -- Price on prediction_date
    actual_price_end REAL,               -- Price on prediction_date + horizon
    actual_return REAL,                  -- Actual return over horizon
    is_correct INTEGER,                  -- 1=direction correct, 0=wrong
    graded_at TIMESTAMP,
    -- Metadata
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, prediction_date, model_version)
);
CREATE INDEX IF NOT EXISTS idx_pred_ticker ON predictions(ticker);
CREATE INDEX IF NOT EXISTS idx_pred_date ON predictions(prediction_date);
CREATE INDEX IF NOT EXISTS idx_pred_graded ON predictions(graded_at);

-- Gate evaluation results (7-gate pipeline)
CREATE TABLE IF NOT EXISTS gate_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    evaluation_date TEXT NOT NULL,
    gate1_score REAL, gate1_passed INTEGER,
    gate2_score REAL, gate2_passed INTEGER,
    gate3_score REAL, gate3_passed INTEGER,
    gate4_score REAL, gate4_passed INTEGER,
    gate5_score REAL, gate5_passed INTEGER,
    integrated_score REAL,
    final_verdict TEXT,
    allocation REAL,
    locked_until TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, evaluation_date)
);
CREATE INDEX IF NOT EXISTS idx_gate_ticker ON gate_evaluations(ticker);
CREATE INDEX IF NOT EXISTS idx_gate_date ON gate_evaluations(evaluation_date);

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_info (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


class QTPDatabase:
    """SQLite database manager for quant-trading-pipeline."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        """Initialize database schema with migration support."""
        with self._conn() as conn:
            conn.executescript(SCHEMA_SQL)

            # Migration: backfill alternative_data_daily from legacy table
            # if daily table is empty but legacy table has data
            daily_count = conn.execute("SELECT COUNT(*) FROM alternative_data_daily").fetchone()[0]
            if daily_count == 0:
                legacy_count = conn.execute("SELECT COUNT(*) FROM alternative_data").fetchone()[0]
                if legacy_count > 0:
                    conn.execute(
                        """INSERT OR IGNORE INTO alternative_data_daily
                           (ticker, tool, date, data, fetched_at)
                           SELECT ticker, tool, date(fetched_at), data, fetched_at
                           FROM alternative_data"""
                    )
                    logger.info(
                        "migrated_legacy_to_daily",
                        rows=legacy_count,
                    )

            conn.execute(
                "INSERT OR REPLACE INTO schema_info (key, value) VALUES (?, ?)",
                ("version", str(SCHEMA_VERSION)),
            )
        logger.debug("db_initialized", path=str(self.db_path))

    # =========================================================================
    # Alternative Data
    # =========================================================================

    def upsert_alternative(
        self,
        ticker: str,
        tool: str,
        data: dict,
        expires_hours: int = 24,
    ) -> None:
        """Insert or update alternative data for a ticker/tool."""
        now = datetime.now()
        expires = (now + timedelta(hours=expires_hours)) if expires_hours else None

        with self._conn() as conn:
            conn.execute(
                """INSERT INTO alternative_data (ticker, tool, data, fetched_at, expires_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(ticker, tool) DO UPDATE SET
                     data=excluded.data, fetched_at=excluded.fetched_at, expires_at=excluded.expires_at""",
                (
                    ticker,
                    tool,
                    json.dumps(data, default=str),
                    now.isoformat(),
                    expires.isoformat() if expires else None,
                ),
            )

    def get_alternative(self, ticker: str, tool: str) -> dict | None:
        """Get cached alternative data. Returns None if not found."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT data FROM alternative_data WHERE ticker=? AND tool=?",
                (ticker, tool),
            ).fetchone()
        if row:
            return json.loads(row["data"])
        return None

    def get_alternative_fresh(
        self,
        ticker: str,
        tool: str,
        max_age_hours: int = 24,
    ) -> dict | None:
        """Get alternative data only if fresh enough."""
        with self._conn() as conn:
            row = conn.execute(
                """SELECT data, fetched_at FROM alternative_data
                   WHERE ticker=? AND tool=?
                   AND fetched_at > datetime('now', ?)""",
                (ticker, tool, f"-{max_age_hours} hours"),
            ).fetchone()
        if row:
            return json.loads(row["data"])
        return None

    def list_stale_data(self, max_age_hours: int = 24) -> list[dict]:
        """List all alternative data entries older than max_age_hours."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT ticker, tool, fetched_at FROM alternative_data
                   WHERE fetched_at < datetime('now', ?)
                   ORDER BY fetched_at""",
                (f"-{max_age_hours} hours",),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_alternative_for_ticker(self, ticker: str) -> list[dict]:
        """List all alternative data for a ticker."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT tool, fetched_at FROM alternative_data WHERE ticker=? ORDER BY tool",
                (ticker,),
            ).fetchall()
        return [dict(r) for r in rows]

    def alternative_coverage(self) -> list[dict]:
        """Summary of alternative data coverage per ticker."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT ticker, COUNT(*) as n_tools,
                          MIN(fetched_at) as oldest, MAX(fetched_at) as newest
                   FROM alternative_data
                   GROUP BY ticker ORDER BY ticker""",
            ).fetchall()
        return [dict(r) for r in rows]

    # =========================================================================
    # Alternative Data — Daily Accumulation
    # =========================================================================

    def upsert_alternative_daily(
        self,
        ticker: str,
        tool: str,
        data: dict,
        date: str | None = None,
    ) -> None:
        """Save alternative data with a specific date for daily accumulation.

        Args:
            ticker: Ticker symbol (or '_market' for market-level data).
            tool: Tool name (e.g. 'earnings_trend').
            data: Tool response data as dict.
            date: ISO date string (YYYY-MM-DD). Defaults to today.
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        now = datetime.now()

        with self._conn() as conn:
            conn.execute(
                """INSERT INTO alternative_data_daily (ticker, tool, date, data, fetched_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(ticker, tool, date) DO UPDATE SET
                     data=excluded.data, fetched_at=excluded.fetched_at""",
                (
                    ticker,
                    tool,
                    date,
                    json.dumps(data, default=str),
                    now.isoformat(),
                ),
            )

        # Also update legacy snapshot table for backward compatibility
        self.upsert_alternative(ticker, tool, data)

    def get_alternative_history(
        self,
        ticker: str,
        tool: str,
        n_days: int = 30,
    ) -> list[dict]:
        """Get last N days of daily alternative data, ordered by date descending.

        Returns list of dicts with keys: date, data, fetched_at.
        """
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT date, data, fetched_at
                   FROM alternative_data_daily
                   WHERE ticker=? AND tool=?
                   ORDER BY date DESC
                   LIMIT ?""",
                (ticker, tool, n_days),
            ).fetchall()
        result = []
        for r in rows:
            entry = dict(r)
            entry["data"] = json.loads(entry["data"])
            result.append(entry)
        return result

    def get_alternative_as_of(
        self,
        ticker: str,
        tool: str,
        as_of_date: str,
    ) -> dict | None:
        """Get alternative data valid on a specific date.

        Returns the most recent record on or before as_of_date.
        """
        with self._conn() as conn:
            row = conn.execute(
                """SELECT data, date, fetched_at
                   FROM alternative_data_daily
                   WHERE ticker=? AND tool=? AND date <= ?
                   ORDER BY date DESC
                   LIMIT 1""",
                (ticker, tool, as_of_date),
            ).fetchone()
        if row:
            result = dict(row)
            result["data"] = json.loads(result["data"])
            return result
        return None

    # =========================================================================
    # Model Registry
    # =========================================================================

    def register_model(
        self,
        version: str,
        model_type: str,
        model_path: str,
        config: dict | None = None,
        metrics: dict | None = None,
        feature_names: list[str] | None = None,
        notes: str | None = None,
    ) -> None:
        """Register a trained model."""
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO model_registry
                   (version, model_type, model_path, config, metrics, feature_names, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    version,
                    model_type,
                    model_path,
                    json.dumps(config) if config else None,
                    json.dumps(metrics, default=str) if metrics else None,
                    json.dumps(feature_names) if feature_names else None,
                    notes,
                ),
            )

    def get_model(self, version: str) -> dict | None:
        """Get model metadata by version."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM model_registry WHERE version=?",
                (version,),
            ).fetchone()
        return dict(row) if row else None

    def list_models(self, limit: int = 20) -> list[dict]:
        """List models ordered by creation date (newest first)."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT version, model_type, metrics, created_at FROM model_registry ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def best_model(self, metric: str = "wf_auc_roc") -> dict | None:
        """Find the model with the best metric value."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT version, metrics, created_at FROM model_registry WHERE metrics IS NOT NULL ORDER BY created_at DESC",
            ).fetchall()

        best = None
        best_val = -float("inf")
        for row in rows:
            metrics = json.loads(row["metrics"])
            val = metrics.get(metric, -float("inf"))
            if isinstance(val, (int, float)) and val > best_val:
                best_val = val
                best = dict(row)
        return best

    # =========================================================================
    # Experiment Tracking
    # =========================================================================

    def log_experiment(
        self,
        config: dict,
        metrics: dict,
        model_version: str | None = None,
        name: str | None = None,
        duration_seconds: float | None = None,
        notes: str | None = None,
    ) -> int:
        """Log a pipeline experiment run. Returns experiment ID."""
        with self._conn() as conn:
            cursor = conn.execute(
                """INSERT INTO experiments
                   (name, config, label_horizon, label_threshold, feature_tiers,
                    n_tickers, n_samples,
                    wf_auc, wf_accuracy, wf_sharpe, wf_max_drawdown, wf_win_rate, wf_n_folds,
                    pkf_auc, pkf_sharpe,
                    model_version, duration_seconds, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    name,
                    json.dumps(config, default=str),
                    config.get("labels", {}).get("horizon"),
                    config.get("labels", {}).get("direction_threshold"),
                    json.dumps(config.get("features", {}).get("tiers")),
                    metrics.get("n_tickers"),
                    metrics.get("n_samples"),
                    metrics.get("wf_auc_roc"),
                    metrics.get("wf_accuracy"),
                    metrics.get("wf_sharpe"),
                    metrics.get("wf_max_drawdown"),
                    metrics.get("wf_win_rate"),
                    metrics.get("wf_n_folds"),
                    metrics.get("pkf_auc_roc"),
                    metrics.get("pkf_sharpe"),
                    model_version,
                    duration_seconds,
                    notes,
                ),
            )
            return cursor.lastrowid

    def list_experiments(self, limit: int = 20) -> list[dict]:
        """List experiments ordered by date (newest first)."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT id, name, label_horizon, label_threshold, feature_tiers,
                          wf_auc, wf_sharpe, wf_win_rate, model_version, created_at
                   FROM experiments ORDER BY created_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def best_experiments(self, metric: str = "wf_auc", limit: int = 5) -> list[dict]:
        """Find top experiments by a given metric."""
        valid_metrics = {"wf_auc", "wf_sharpe", "wf_accuracy", "wf_win_rate", "pkf_auc"}
        if metric not in valid_metrics:
            raise ValueError(f"Invalid metric: {metric}. Use one of {valid_metrics}")

        with self._conn() as conn:
            rows = conn.execute(
                f"""SELECT id, name, label_horizon, label_threshold, feature_tiers,
                           wf_auc, wf_sharpe, wf_win_rate, wf_n_folds,
                           model_version, created_at
                    FROM experiments
                    WHERE {metric} IS NOT NULL
                    ORDER BY {metric} DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def compare_experiments(self, exp_ids: list[int]) -> list[dict]:
        """Compare specific experiments side by side."""
        placeholders = ",".join("?" * len(exp_ids))
        with self._conn() as conn:
            rows = conn.execute(
                f"""SELECT * FROM experiments WHERE id IN ({placeholders})
                    ORDER BY wf_auc DESC""",
                exp_ids,
            ).fetchall()
        return [dict(r) for r in rows]

    # =========================================================================
    # Prediction Tracking
    # =========================================================================

    def save_prediction(
        self,
        ticker: str,
        prediction_date: str,
        direction: int,
        confidence: float,
        predicted_magnitude: float | None = None,
        model_version: str | None = None,
        horizon: int = 1,
    ) -> None:
        """Save a model prediction for later grading."""
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO predictions
                   (ticker, prediction_date, horizon, direction, confidence,
                    predicted_magnitude, model_version)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    ticker,
                    prediction_date,
                    horizon,
                    direction,
                    confidence,
                    predicted_magnitude,
                    model_version,
                ),
            )

    def save_predictions_batch(self, predictions: list[dict]) -> int:
        """Save multiple predictions at once. Returns count saved."""
        with self._conn() as conn:
            for p in predictions:
                conn.execute(
                    """INSERT OR REPLACE INTO predictions
                       (ticker, prediction_date, horizon, direction, confidence,
                        predicted_magnitude, model_version)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        p["ticker"],
                        p["prediction_date"],
                        p.get("horizon", 1),
                        p["direction"],
                        p["confidence"],
                        p.get("predicted_magnitude"),
                        p.get("model_version"),
                    ),
                )
        return len(predictions)

    def get_ungraded_predictions(self) -> list[dict]:
        """Get predictions that haven't been graded yet."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT id, ticker, prediction_date, horizon, direction,
                          confidence, predicted_magnitude, model_version
                   FROM predictions
                   WHERE graded_at IS NULL
                   ORDER BY prediction_date""",
            ).fetchall()
        return [dict(r) for r in rows]

    def grade_prediction(
        self,
        prediction_id: int,
        actual_price_start: float,
        actual_price_end: float,
    ) -> None:
        """Grade a prediction with actual price data."""
        actual_return = (actual_price_end - actual_price_start) / actual_price_start
        with self._conn() as conn:
            row = conn.execute(
                "SELECT direction FROM predictions WHERE id=?",
                (prediction_id,),
            ).fetchone()
            if not row:
                return

            predicted_up = row["direction"] == 1
            actual_up = actual_return > 0
            is_correct = 1 if (predicted_up == actual_up) else 0

            conn.execute(
                """UPDATE predictions SET
                     actual_price_start=?, actual_price_end=?,
                     actual_return=?, is_correct=?, graded_at=?
                   WHERE id=?""",
                (
                    actual_price_start,
                    actual_price_end,
                    actual_return,
                    is_correct,
                    datetime.now().isoformat(),
                    prediction_id,
                ),
            )

    def get_accuracy_summary(self, days: int | None = None) -> dict:
        """Get prediction accuracy summary.

        Args:
            days: If set, only look at predictions from the last N days.
        """
        where = "WHERE graded_at IS NOT NULL"
        params: list = []
        if days:
            where += " AND prediction_date > date('now', ?)"
            params.append(f"-{days} days")

        with self._conn() as conn:
            row = conn.execute(
                f"""SELECT
                      COUNT(*) as total,
                      SUM(is_correct) as correct,
                      AVG(is_correct) as accuracy,
                      AVG(actual_return) as avg_return,
                      AVG(CASE WHEN is_correct=1 THEN actual_return END) as avg_win,
                      AVG(CASE WHEN is_correct=0 THEN actual_return END) as avg_loss,
                      AVG(confidence) as avg_confidence
                    FROM predictions {where}""",
                params,
            ).fetchone()
        return dict(row) if row else {}

    def get_accuracy_by_confidence(self) -> list[dict]:
        """Get accuracy broken down by confidence bucket."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT
                      CASE
                        WHEN confidence >= 0.70 THEN '70-100%'
                        WHEN confidence >= 0.60 THEN '60-70%'
                        WHEN confidence >= 0.55 THEN '55-60%'
                        ELSE '50-55%'
                      END as bucket,
                      COUNT(*) as total,
                      SUM(is_correct) as correct,
                      ROUND(AVG(is_correct) * 100, 1) as accuracy_pct,
                      ROUND(AVG(actual_return) * 100, 3) as avg_return_pct
                    FROM predictions
                    WHERE graded_at IS NOT NULL
                    GROUP BY bucket
                    ORDER BY bucket DESC""",
            ).fetchall()
        return [dict(r) for r in rows]

    def get_accuracy_by_ticker(self) -> list[dict]:
        """Get accuracy broken down by ticker."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT
                      ticker,
                      COUNT(*) as total,
                      SUM(is_correct) as correct,
                      ROUND(AVG(is_correct) * 100, 1) as accuracy_pct,
                      ROUND(AVG(actual_return) * 100, 3) as avg_return_pct
                    FROM predictions
                    WHERE graded_at IS NOT NULL
                    GROUP BY ticker
                    ORDER BY accuracy_pct DESC""",
            ).fetchall()
        return [dict(r) for r in rows]

    def get_accuracy_trend(self, window_days: int = 7, n_windows: int = 4) -> list[dict]:
        """Get accuracy in rolling windows to detect degradation trend.

        Returns a list of dicts (newest window first), each with:
          - window: human-readable label like "-7d to -0d"
          - total: number of graded predictions in window
          - accuracy: fraction correct
        """
        results = []
        for i in range(n_windows):
            end_offset = i * window_days
            start_offset = (i + 1) * window_days
            with self._conn() as conn:
                row = conn.execute(
                    """SELECT COUNT(*) as total, SUM(is_correct) as correct,
                              AVG(is_correct) as accuracy
                       FROM predictions
                       WHERE graded_at IS NOT NULL
                       AND prediction_date BETWEEN date('now', ?) AND date('now', ?)""",
                    (f"-{start_offset} days", f"-{end_offset} days"),
                ).fetchone()
            if row and row["total"] and row["total"] > 0:
                results.append(
                    {
                        "window": f"-{start_offset}d to -{end_offset}d",
                        "total": row["total"],
                        "accuracy": row["accuracy"],
                    }
                )
        return results

    def get_recent_predictions(self, limit: int = 20) -> list[dict]:
        """Get recent predictions with grading results."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT ticker, prediction_date, horizon, direction, confidence,
                          predicted_magnitude, actual_return, is_correct, model_version
                   FROM predictions
                   ORDER BY prediction_date DESC, ticker
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # =========================================================================
    # Gate Evaluations
    # =========================================================================

    def save_gate_evaluation(
        self,
        ticker: str,
        evaluation_date: str,
        *,
        gate1_score: float | None = None,
        gate1_passed: bool | None = None,
        gate2_score: float | None = None,
        gate2_passed: bool | None = None,
        gate3_score: float | None = None,
        gate3_passed: bool | None = None,
        gate4_score: float | None = None,
        gate4_passed: bool | None = None,
        gate5_score: float | None = None,
        gate5_passed: bool | None = None,
        integrated_score: float | None = None,
        final_verdict: str | None = None,
        allocation: float | None = None,
        locked_until: str | None = None,
    ) -> None:
        """Save or update a gate evaluation result."""
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO gate_evaluations
                   (ticker, evaluation_date,
                    gate1_score, gate1_passed, gate2_score, gate2_passed,
                    gate3_score, gate3_passed, gate4_score, gate4_passed,
                    gate5_score, gate5_passed,
                    integrated_score, final_verdict, allocation, locked_until)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(ticker, evaluation_date) DO UPDATE SET
                     gate1_score=excluded.gate1_score,
                     gate1_passed=excluded.gate1_passed,
                     gate2_score=excluded.gate2_score,
                     gate2_passed=excluded.gate2_passed,
                     gate3_score=excluded.gate3_score,
                     gate3_passed=excluded.gate3_passed,
                     gate4_score=excluded.gate4_score,
                     gate4_passed=excluded.gate4_passed,
                     gate5_score=excluded.gate5_score,
                     gate5_passed=excluded.gate5_passed,
                     integrated_score=excluded.integrated_score,
                     final_verdict=excluded.final_verdict,
                     allocation=excluded.allocation,
                     locked_until=excluded.locked_until""",
                (
                    ticker,
                    evaluation_date,
                    gate1_score,
                    int(gate1_passed) if gate1_passed is not None else None,
                    gate2_score,
                    int(gate2_passed) if gate2_passed is not None else None,
                    gate3_score,
                    int(gate3_passed) if gate3_passed is not None else None,
                    gate4_score,
                    int(gate4_passed) if gate4_passed is not None else None,
                    gate5_score,
                    int(gate5_passed) if gate5_passed is not None else None,
                    integrated_score,
                    final_verdict,
                    allocation,
                    locked_until,
                ),
            )

    def get_cached_verdict(self, ticker: str) -> dict | None:
        """Get the most recent gate evaluation for *ticker*.

        Returns None if no evaluation exists or if the lock has expired.
        The returned dict includes all gate scores, final_verdict, allocation,
        and locked_until.
        """
        with self._conn() as conn:
            row = conn.execute(
                """SELECT * FROM gate_evaluations
                   WHERE ticker = ?
                   ORDER BY evaluation_date DESC
                   LIMIT 1""",
                (ticker,),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        # Check if still locked
        locked_until = result.get("locked_until")
        if locked_until:
            from datetime import date as _date

            try:
                lock_date = _date.fromisoformat(locked_until)
                if _date.today() <= lock_date:
                    result["is_locked"] = True
                else:
                    result["is_locked"] = False
            except (TypeError, ValueError):
                result["is_locked"] = False
        else:
            result["is_locked"] = False
        return result
