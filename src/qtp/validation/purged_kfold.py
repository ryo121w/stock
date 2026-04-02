"""Purged K-Fold Cross-Validation for financial time series."""

from __future__ import annotations

import numpy as np


class PurgedKFold:
    """K-Fold with purging (gap between train/test) to prevent leakage.

    Adjacent train/test samples are correlated due to overlapping lookback windows.
    The purge_gap removes samples within `purge_days` of the train/test boundary.
    """

    def __init__(self, n_splits: int = 5, purge_days: int = 5):
        self.n_splits = n_splits
        self.purge_days = purge_days

    def split(self, X, y=None, groups=None):
        n = len(X)
        fold_size = n // self.n_splits

        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = min((i + 1) * fold_size, n)

            # Purge zone: purge_days before and after test set
            purge_start = max(0, test_start - self.purge_days)
            purge_end = min(n, test_end + self.purge_days)

            train_mask = np.ones(n, dtype=bool)
            train_mask[purge_start:purge_end] = False

            train_idx = np.where(train_mask)[0]
            test_idx = np.arange(test_start, test_end)

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits
