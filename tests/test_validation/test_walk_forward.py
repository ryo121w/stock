"""Tests for Expanding Window Walk-Forward CV."""

from __future__ import annotations

import numpy as np

from qtp.validation.walk_forward import ExpandingWindowCV


class TestExpandingWindowCV:
    """Core behaviour of the expanding-window walk-forward splitter."""

    def test_basic_splits(self):
        """Generates correct number of expanding folds."""
        cv = ExpandingWindowCV(min_train_size=100, test_size=20, step_size=20, purge_gap=5)
        X = np.zeros((300, 3))
        splits = list(cv.split(X))

        assert len(splits) > 0
        # With 300 samples, min_train=100, test=20, step=20, gap=5:
        # first test starts at 105, last test ends <= 300
        assert len(splits) == 9

    def test_train_always_before_test(self):
        """Train indices must all be strictly before test indices (temporal ordering)."""
        cv = ExpandingWindowCV(min_train_size=50, test_size=10, step_size=10, purge_gap=3)
        X = np.zeros((200, 3))

        for train_idx, test_idx in cv.split(X):
            assert train_idx.max() < test_idx.min(), (
                f"Train max {train_idx.max()} >= test min {test_idx.min()}"
            )

    def test_purge_gap_respected(self):
        """Gap between train end and test start >= purge_gap."""
        purge = 5
        cv = ExpandingWindowCV(min_train_size=50, test_size=10, step_size=10, purge_gap=purge)
        X = np.zeros((200, 3))

        for train_idx, test_idx in cv.split(X):
            gap = test_idx.min() - train_idx.max() - 1
            assert gap >= purge, f"Gap {gap} < purge {purge}"

    def test_expanding_train_window(self):
        """Training window must grow with each fold."""
        cv = ExpandingWindowCV(min_train_size=50, test_size=10, step_size=10, purge_gap=3)
        X = np.zeros((200, 3))

        train_sizes = [len(tr) for tr, _ in cv.split(X)]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] > train_sizes[i - 1], (
                f"Fold {i} train ({train_sizes[i]}) not larger than fold {i - 1} ({train_sizes[i - 1]})"
            )

    def test_fixed_test_window(self):
        """Test window size must be constant across folds."""
        test_size = 15
        cv = ExpandingWindowCV(min_train_size=50, test_size=test_size, step_size=15, purge_gap=3)
        X = np.zeros((200, 3))

        for _, test_idx in cv.split(X):
            assert len(test_idx) == test_size

    def test_min_train_size_respected(self):
        """No fold should have fewer than min_train_size training samples."""
        min_train = 80
        cv = ExpandingWindowCV(min_train_size=min_train, test_size=10, step_size=10, purge_gap=3)
        X = np.zeros((200, 3))

        for train_idx, _ in cv.split(X):
            assert len(train_idx) >= min_train

    def test_no_overlap_between_train_and_test(self):
        """Train and test indices must not overlap."""
        cv = ExpandingWindowCV(min_train_size=50, test_size=10, step_size=10, purge_gap=3)
        X = np.zeros((200, 3))

        for train_idx, test_idx in cv.split(X):
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0, f"Overlapping indices: {overlap}"

    def test_insufficient_data_yields_no_splits(self):
        """If data is too short for even one fold, no splits are yielded."""
        cv = ExpandingWindowCV(min_train_size=100, test_size=20, step_size=20, purge_gap=5)
        X = np.zeros((110, 3))  # 100 + 5 + 20 = 125 needed, only 110

        splits = list(cv.split(X))
        assert len(splits) == 0

    def test_get_n_splits(self):
        """get_n_splits returns correct count."""
        cv = ExpandingWindowCV(min_train_size=100, test_size=20, step_size=20, purge_gap=5)
        X = np.zeros((300, 3))

        assert cv.get_n_splits(X) == len(list(cv.split(X)))

    def test_get_splits_metadata(self):
        """get_splits returns correct fold metadata."""
        cv = ExpandingWindowCV(min_train_size=50, test_size=10, step_size=10, purge_gap=3)
        splits_meta = cv.get_splits(150)

        assert len(splits_meta) > 0
        for s in splits_meta:
            assert s.train_start == 0
            assert s.train_end < s.test_start
            assert s.test_end - s.test_start == 10
            assert s.test_start - s.train_end >= 3  # purge gap
