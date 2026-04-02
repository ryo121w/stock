"""SHAP-based feature selection."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import structlog

logger = structlog.get_logger()


class SHAPFeatureSelector:
    """Select top features by mean |SHAP value|."""

    def __init__(self, min_importance_pct: float = 0.01):
        self.min_importance_pct = min_importance_pct

    def select(
        self,
        model: Any,
        X: pl.DataFrame,
        max_features: int = 30,
    ) -> list[str]:
        import shap

        explainer = shap.TreeExplainer(model)
        X_pd = X.to_pandas()
        shap_values = explainer.shap_values(X_pd)

        # For binary classification, shap_values may be a list of 2 arrays
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        total = mean_abs_shap.sum()
        if total == 0:
            return list(X.columns)

        importance = dict(zip(X.columns, mean_abs_shap / total))
        # Filter by minimum importance
        filtered = {k: v for k, v in importance.items() if v >= self.min_importance_pct}
        # Sort and take top N
        sorted_features = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
        selected = [name for name, _ in sorted_features[:max_features]]

        logger.info("shap_feature_selection", total_features=len(X.columns),
                     selected=len(selected), top3=[s for s in selected[:3]])
        return selected
