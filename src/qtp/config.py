"""Pipeline configuration with Pydantic validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    fetcher: str = "yfinance"
    storage_dir: str = "data"
    history_days: int = 756
    min_history_days: int = 504


class UniverseConfig(BaseModel):
    market: str = "us"
    tickers: list[str] = Field(default_factory=list)
    min_avg_daily_volume: int = 1_000_000


class FeaturesConfig(BaseModel):
    tiers: list[int] = Field(default_factory=lambda: [1, 2])
    max_features: int = 30
    lookback_padding_factor: float = 1.8
    selected: list[str] = Field(
        default_factory=list,
        description="Explicit feature subset from feature selection analysis. Empty = use all.",
    )
    excluded: list[str] = Field(
        default_factory=list,
        description="Features to exclude (e.g., zero-importance features). Applied after tier loading.",
    )


class LabelsConfig(BaseModel):
    horizon: int = 1
    direction_threshold: float = 0.0


class ModelConfig(BaseModel):
    type: str = "lgbm"
    ensemble_weights: dict[str, float] = Field(default_factory=lambda: {"lgbm": 0.6, "xgb": 0.4})
    tune: bool = True
    tune_n_trials: int = 100


class ValidationConfig(BaseModel):
    dev_cv: str = "purged_kfold"
    dev_cv_splits: int = 5
    dev_cv_purge_days: int = 5
    final_cv: str = "cpcv"
    walk_forward_train_days: int = 504
    walk_forward_test_days: int = 63
    walk_forward_step_days: int = 63
    walk_forward_max_train_days: int | None = None  # None=expanding, int=sliding window


class BacktestConfig(BaseModel):
    initial_capital: float = 10_000_000
    commission_pct: float = 0.001
    slippage_pct: float = 0.001
    confidence_threshold: float = 0.55
    magnitude_threshold: float = 0.002
    max_position_pct: float = 0.05


class ReportingConfig(BaseModel):
    benchmark_us: str = "SPY"
    benchmark_jp: str = "^N225"
    output_dir: str = "data/reports"


class PipelineConfig(BaseModel):
    data: DataConfig = DataConfig()
    universe: UniverseConfig = UniverseConfig()
    features: FeaturesConfig = FeaturesConfig()
    labels: LabelsConfig = LabelsConfig()
    model: ModelConfig = ModelConfig()
    validation: ValidationConfig = ValidationConfig()
    backtest: BacktestConfig = BacktestConfig()
    reporting: ReportingConfig = ReportingConfig()

    @classmethod
    def from_yaml(cls, path: Path) -> PipelineConfig:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        raw.pop("pipeline", None)
        return cls(**raw)

    @classmethod
    def from_yamls(cls, *paths: Path) -> PipelineConfig:
        merged: dict[str, Any] = {}
        for path in paths:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            data.pop("pipeline", None)
            merged = _deep_merge(merged, data)
        return cls(**merged)


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
