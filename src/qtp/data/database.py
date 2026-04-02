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

SCHEMA_VERSION = 1

SCHEMA_SQL = """
-- Alternative data from MCP tools (earnings_trend, analyst_actions, etc.)
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
        """Initialize database schema."""
        with self._conn() as conn:
            conn.executescript(SCHEMA_SQL)
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
