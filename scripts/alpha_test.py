#!/usr/bin/env python3
"""Alpha Existence Test: Decile portfolio analysis with statistical significance.

For each feature:
1. Sort observations into deciles by feature value
2. Compute mean forward return for each decile
3. Calculate long-short spread (top decile - bottom decile)
4. Test statistical significance with t-test

Result: "Does this feature have predictive alpha, or is it noise?"
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats
import structlog

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from qtp.config import PipelineConfig
from qtp.data.fetchers.base import Market
from qtp.data.storage import ParquetStorage
from qtp.features.engine import FeatureEngine
from qtp.features.registry import FeatureRegistry

# Import feature definitions
import qtp.features.tier1_momentum  # noqa: F401
import qtp.features.tier2_volatility  # noqa: F401

logger = structlog.get_logger()


def load_dataset(config: PipelineConfig) -> pl.DataFrame:
    """Load complete feature + label dataset."""
    storage = ParquetStorage(project_root / config.data.storage_dir)
    engine = FeatureEngine(FeatureRegistry.instance(), storage)
    market = Market(config.universe.market)

    return engine.build_multi_ticker_dataset(
        tickers=config.universe.tickers,
        market=market,
        as_of=date.today(),
        tiers=config.features.tiers,
        horizon=config.labels.horizon,
    )


def decile_analysis(feature_values: np.ndarray, forward_returns: np.ndarray) -> dict:
    """Run decile portfolio analysis for a single feature.

    Returns dict with decile returns, spread, t-stat, and p-value.
    """
    # Remove NaN/Inf
    valid = np.isfinite(feature_values) & np.isfinite(forward_returns)
    feat = feature_values[valid]
    rets = forward_returns[valid]

    if len(feat) < 100:
        return {"error": "insufficient_data", "n_obs": len(feat)}

    # Assign deciles (0-9)
    try:
        decile_edges = np.percentile(feat, np.arange(10, 100, 10))
        deciles = np.digitize(feat, decile_edges)
    except Exception:
        return {"error": "decile_computation_failed"}

    # Mean return per decile
    decile_returns = {}
    for d in range(10):
        mask = deciles == d
        if mask.sum() > 0:
            decile_returns[d] = {
                "mean_return": float(np.mean(rets[mask])),
                "n_obs": int(mask.sum()),
                "std": float(np.std(rets[mask])),
            }

    # Long-short spread: top decile (9) vs bottom decile (0)
    top_mask = deciles == 9
    bottom_mask = deciles == 0

    if top_mask.sum() < 10 or bottom_mask.sum() < 10:
        return {
            "decile_returns": decile_returns,
            "error": "insufficient_extreme_decile_obs",
        }

    top_returns = rets[top_mask]
    bottom_returns = rets[bottom_mask]

    spread = float(np.mean(top_returns) - np.mean(bottom_returns))
    spread_annualized = spread * 252

    # Welch's t-test (unequal variance)
    t_stat, p_value = stats.ttest_ind(top_returns, bottom_returns, equal_var=False)

    # Monotonicity check: are decile returns roughly monotonic?
    means = [decile_returns.get(d, {}).get("mean_return", 0) for d in range(10)]
    rank_corr, _ = stats.spearmanr(range(10), means)

    return {
        "decile_returns": decile_returns,
        "spread_daily": spread,
        "spread_annualized": spread_annualized,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "significant_5pct": p_value < 0.05,
        "significant_1pct": p_value < 0.01,
        "rank_correlation": float(rank_corr),
        "monotonic": abs(rank_corr) > 0.7,
        "n_obs": int(valid.sum()),
    }


def format_results(results: dict[str, dict]) -> str:
    """Format decile analysis results as a table."""
    lines = [
        "",
        "=" * 90,
        "  ALPHA EXISTENCE TEST — Decile Portfolio Spreads",
        "=" * 90,
        "",
        f"  {'Feature':<30} {'Spread(ann)':>12} {'t-stat':>8} {'p-value':>10} {'Sig?':>5} {'Mono?':>6} {'RankCorr':>9}",
        f"  {'-'*30} {'-'*12} {'-'*8} {'-'*10} {'-'*5} {'-'*6} {'-'*9}",
    ]

    # Sort by absolute spread
    sorted_features = sorted(
        results.items(),
        key=lambda x: abs(x[1].get("spread_annualized", 0)),
        reverse=True,
    )

    alpha_count = 0
    for feat_name, res in sorted_features:
        if "error" in res:
            lines.append(f"  {feat_name:<30} {'ERROR':>12} ({res['error']})")
            continue

        sig = "***" if res["significant_1pct"] else ("**" if res["significant_5pct"] else "")
        mono = "Yes" if res["monotonic"] else "No"
        lines.append(
            f"  {feat_name:<30} {res['spread_annualized']:>+11.2%} "
            f"{res['t_stat']:>8.2f} {res['p_value']:>10.4f} {sig:>5} {mono:>6} "
            f"{res['rank_correlation']:>+8.3f}"
        )
        if res["significant_5pct"]:
            alpha_count += 1

    lines.extend([
        "",
        f"  Total features tested: {len(results)}",
        f"  Significant at 5%:     {alpha_count}",
        f"  Expected by chance:    {len(results) * 0.05:.1f}",
        "",
        "  Interpretation:",
        "  - If significant features ≈ expected by chance → no real alpha",
        "  - Look for features with BOTH significance AND monotonicity",
        "  - Rank correlation near ±1.0 = clean monotonic relationship",
        "  - Negative spread = feature predicts opposite direction",
        "",
        "=" * 90,
    ])
    return "\n".join(lines)


def main():
    config_path = project_root / "configs" / "default.yaml"
    if config_path.exists():
        config = PipelineConfig.from_yaml(config_path)
    else:
        config = PipelineConfig()

    print("Loading dataset...")
    dataset = load_dataset(config)

    label_cols = ["label_direction", "label_magnitude", "date", "ticker"]
    feature_cols = [c for c in dataset.columns if c not in label_cols]

    forward_returns = dataset["label_magnitude"].to_numpy()

    print(f"Dataset: {dataset.height} rows, {len(feature_cols)} features")
    print(f"Forward return stats: mean={np.mean(forward_returns):.4%}, "
          f"std={np.std(forward_returns):.4%}")
    print()

    print("Running decile analysis for each feature...")
    results = {}
    for feat_name in feature_cols:
        feat_values = dataset[feat_name].to_numpy().astype(float)
        results[feat_name] = decile_analysis(feat_values, forward_returns)
        status = "alpha" if results[feat_name].get("significant_5pct") else "noise"
        print(f"  {feat_name}: {status}")

    report = format_results(results)
    print(report)

    # Save report
    output_path = project_root / "data" / "reports" / "alpha_test.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"\nReport saved to {output_path}")


if __name__ == "__main__":
    main()
