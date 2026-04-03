"""Decorator-based feature registry."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Callable

import polars as pl


class FeatureTier(IntEnum):
    TIER1_MOMENTUM = 1
    TIER2_VOLATILITY = 2
    TIER3_FUNDAMENTAL = 3
    TIER4_MACRO = 4
    TIER5_ALTERNATIVE = 5
    TIER6_FUNDAMENTAL_TS = 6


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    tier: FeatureTier
    lookback_days: int
    compute_fn: Callable[[pl.DataFrame], pl.Series]
    description: str = ""
    dependencies: tuple[str, ...] = ()


class FeatureRegistry:
    _instance: FeatureRegistry | None = None

    def __init__(self) -> None:
        self._features: dict[str, FeatureDefinition] = {}

    @classmethod
    def instance(cls) -> FeatureRegistry:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    def register(
        self,
        name: str,
        tier: FeatureTier,
        lookback_days: int,
        description: str = "",
        dependencies: tuple[str, ...] = (),
    ) -> Callable:
        """Decorator to register a feature computation function."""

        def decorator(fn: Callable[[pl.DataFrame], pl.Series]) -> Callable:
            self._features[name] = FeatureDefinition(
                name=name,
                tier=tier,
                lookback_days=lookback_days,
                compute_fn=fn,
                description=description,
                dependencies=dependencies,
            )
            return fn

        return decorator

    def get(self, name: str) -> FeatureDefinition:
        return self._features[name]

    def by_tier(self, tier: FeatureTier) -> list[FeatureDefinition]:
        return [f for f in self._features.values() if f.tier == tier]

    def by_tiers(self, tiers: list[int]) -> list[FeatureDefinition]:
        return [f for f in self._features.values() if f.tier in tiers]

    def all_features(self) -> list[FeatureDefinition]:
        return list(self._features.values())

    def feature_names(self) -> list[str]:
        return list(self._features.keys())

    def max_lookback(self) -> int:
        if not self._features:
            return 0
        return max(f.lookback_days for f in self._features.values())
