"""
Tests for purged + embargoed walk-forward cross-validation.

No market data required — uses a synthetic DataFrame.
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Inline purged walk-forward (mirrors train_baselines.purged_walk_forward_splits)
# ---------------------------------------------------------------------------

def purged_walk_forward_splits(df: pd.DataFrame, n_folds: int, embargo_bars: int):
    df = df.sort_values("entry_time").reset_index(drop=True)
    n = len(df)
    fold_size = n // (n_folds + 1)
    bar_dur = pd.Timedelta(minutes=15)

    for fold in range(n_folds):
        test_start = (fold + 1) * fold_size
        test_end = test_start + fold_size
        test_idx = df.index[test_start:test_end]
        test_start_time = df.loc[test_idx[0], "entry_time"]
        test_end_time = df.loc[test_idx[-1], "touch_time"]

        embargo_cutoff = test_start_time - bar_dur * embargo_bars

        train_mask = (
            (df["touch_time"] < test_start_time) &
            (df["entry_time"] < embargo_cutoff)
        )
        train_idx = df.index[train_mask]

        if len(train_idx) < 30:
            continue

        yield train_idx, test_idx


def make_labeled_df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    entry_times = pd.date_range("2023-01-01", periods=n, freq="15min")
    touch_offsets = rng.integers(1, 96, n)
    touch_times = entry_times + pd.to_timedelta(touch_offsets * 15, unit="m")
    return pd.DataFrame({
        "entry_time": entry_times,
        "touch_time": touch_times,
        "y_binary": rng.integers(0, 2, n),
        "realized_r": rng.normal(0.3, 1.0, n),
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPurgedWalkForward:

    def test_produces_folds(self):
        df = make_labeled_df()
        folds = list(purged_walk_forward_splits(df, n_folds=5, embargo_bars=48))
        assert len(folds) >= 3  # some folds may be skipped if too small

    def test_no_test_in_train(self):
        """Train indices must never overlap test indices."""
        df = make_labeled_df(500)
        for train_idx, test_idx in purged_walk_forward_splits(df, 5, 48):
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0, f"Leakage: {len(overlap)} overlapping indices"

    def test_train_ends_before_test(self):
        """All train touch_times must be strictly before test entry_times."""
        df = make_labeled_df(500)
        for train_idx, test_idx in purged_walk_forward_splits(df, 5, 48):
            max_train_touch = df.loc[train_idx, "touch_time"].max()
            min_test_entry = df.loc[test_idx, "entry_time"].min()
            assert max_train_touch < min_test_entry, (
                f"Train touch bleeds into test: {max_train_touch} >= {min_test_entry}"
            )

    def test_embargo_respected(self):
        """Train entry_times must be at least embargo_bars * 15min before test start."""
        df = make_labeled_df(500)
        embargo_bars = 48
        bar_dur = pd.Timedelta(minutes=15)
        for train_idx, test_idx in purged_walk_forward_splits(df, 5, embargo_bars):
            min_test_entry = df.loc[test_idx, "entry_time"].min()
            max_train_entry = df.loc[train_idx, "entry_time"].max()
            gap = min_test_entry - max_train_entry
            assert gap >= bar_dur * embargo_bars, (
                f"Embargo violated: gap={gap} < {bar_dur * embargo_bars}"
            )

    def test_folds_are_walk_forward(self):
        """Test windows must be in chronological order across folds."""
        df = make_labeled_df(600)
        test_starts = []
        for _, test_idx in purged_walk_forward_splits(df, 5, 48):
            test_starts.append(df.loc[test_idx, "entry_time"].min())
        for i in range(1, len(test_starts)):
            assert test_starts[i] > test_starts[i - 1]
