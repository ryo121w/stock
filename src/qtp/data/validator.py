"""Data quality validation with anti-leakage checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import polars as pl
import structlog

logger = structlog.get_logger()


@dataclass
class ValidationResult:
    passed: bool
    issues: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        if self.passed:
            return "PASSED"
        return f"FAILED: {'; '.join(self.issues)}"


class DataValidator:
    """Validate OHLCV data quality and check for data leakage."""

    def validate_ohlcv(self, df: pl.DataFrame, as_of: date | None = None) -> ValidationResult:
        issues: list[str] = []

        if df.height == 0:
            issues.append("Empty DataFrame")
            return ValidationResult(passed=False, issues=issues)

        # 1. No future dates
        today = as_of or date.today()
        future_rows = df.filter(pl.col("date") > today).height
        if future_rows > 0:
            issues.append(f"CRITICAL: {future_rows} rows with future dates (data leakage)")

        # 2. Dates sorted ascending
        if not df["date"].is_sorted():
            issues.append("Dates not sorted ascending")

        # 3. No nulls in core columns
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                null_count = df[col].null_count()
                if null_count > 0:
                    issues.append(f"Column '{col}' has {null_count} nulls")

        # 4. Price sanity
        if df.filter(pl.col("close") <= 0).height > 0:
            issues.append("Non-positive close prices detected")
        if df.filter(pl.col("high") < pl.col("low")).height > 0:
            issues.append("High < Low detected")

        # 5. Volume sanity
        if df.filter(pl.col("volume") < 0).height > 0:
            issues.append("Negative volume detected")

        # 6. Duplicate dates
        dup_count = df.height - df["date"].n_unique()
        if dup_count > 0:
            issues.append(f"{dup_count} duplicate dates")

        if issues:
            logger.warning("validation_issues", issues=issues)

        return ValidationResult(passed=len(issues) == 0, issues=issues)

    def validate_no_lookahead(self, df: pl.DataFrame, as_of: date) -> ValidationResult:
        """Verify no data column contains information from after as_of."""
        issues: list[str] = []

        if "date" in df.columns:
            future = df.filter(pl.col("date") > as_of).height
            if future > 0:
                issues.append(f"CRITICAL: {future} rows after as_of date {as_of}")

        return ValidationResult(passed=len(issues) == 0, issues=issues)
