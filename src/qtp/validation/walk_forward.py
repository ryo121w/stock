"""Expanding Window Walk-Forward Cross-Validation for financial time series.

Unlike PurgedKFold which can leak future data, Walk-Forward strictly respects
temporal ordering: training always uses past data, testing on future data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WalkForwardSplit:
    """Metadata for a single walk-forward fold."""

    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int

    @property
    def train_size(self) -> int:
        return self.train_end - self.train_start

    @property
    def test_size(self) -> int:
        return self.test_end - self.test_start


class ExpandingWindowCV:
    """Expanding-window walk-forward cross-validation.

    The training window grows over time (expanding), while the test window
    remains a fixed size. A purge gap between train and test prevents leakage.

    Parameters
    ----------
    min_train_size : int
        Minimum number of samples in the initial training window.
    test_size : int
        Number of samples in each test window.
    step_size : int
        Number of samples to advance the test window each fold.
    purge_gap : int
        Number of samples to skip between train and test (prevents leakage
        from overlapping lookback windows).
    """

    def __init__(
        self,
        min_train_size: int = 504,
        test_size: int = 63,
        step_size: int = 63,
        purge_gap: int = 5,
        max_train_size: int | None = None,
    ):
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.step_size = step_size
        self.purge_gap = purge_gap
        self.max_train_size = max_train_size  # None = expanding, int = sliding window

    def split(self, X, y=None, groups=None):
        """Generate (train_idx, test_idx) tuples.

        When max_train_size is set, uses a sliding window (fixed-size train set)
        instead of expanding window. This prevents old market regimes from
        diluting the model's understanding of current conditions.

        Yields
        ------
        train_idx : np.ndarray
        test_idx : np.ndarray
        """
        n = len(X)

        # First test window starts after min_train_size + purge_gap
        test_start = self.min_train_size + self.purge_gap

        while test_start + self.test_size <= n:
            test_end = test_start + self.test_size
            train_end = test_start - self.purge_gap  # Leave gap before test

            # Sliding window: limit train start to keep window size <= max_train_size
            if self.max_train_size is not None:
                train_start = max(0, train_end - self.max_train_size)
            else:
                train_start = 0

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)

            if len(train_idx) >= self.min_train_size and len(test_idx) > 0:
                yield train_idx, test_idx

            test_start += self.step_size

    def get_splits(self, n_samples: int) -> list[WalkForwardSplit]:
        """Return metadata for all folds without actual indices."""
        splits = []
        fold = 0
        test_start = self.min_train_size + self.purge_gap

        while test_start + self.test_size <= n_samples:
            test_end = test_start + self.test_size
            train_end = test_start - self.purge_gap

            if self.max_train_size is not None:
                train_start = max(0, train_end - self.max_train_size)
            else:
                train_start = 0

            if (train_end - train_start) >= self.min_train_size:
                splits.append(
                    WalkForwardSplit(
                        fold=fold,
                        train_start=train_start,
                        train_end=train_end,
                        test_start=test_start,
                        test_end=test_end,
                    )
                )
                fold += 1

            test_start += self.step_size

        return splits

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        n = len(X) if X is not None else 0
        return len(self.get_splits(n))
