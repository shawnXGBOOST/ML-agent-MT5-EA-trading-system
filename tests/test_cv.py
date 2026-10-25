"""
Tests for the purged + embargoed walk-forward splitter.

Calls the real `purged_walk_forward_splits` on a synthetic label table.
"""

import numpy as np
import pandas as pd

from mltrader.models.train_baselines import purged_walk_forward_splits

BAR = pd.Timedelta(minutes=15)


def make_labels(n=600, max_hold_bars=20, seed=0):
    """Events one bar apart, each held 1..max_hold_bars bars. Already time-sorted,
    so row positions match the indices the splitter returns."""
    rng = np.random.default_rng(seed)
    entry = pd.date_range("2024-01-01", periods=n, freq="15min")
    hold = rng.integers(1, max_hold_bars + 1, n)
    return pd.DataFrame({
        "entry_time": entry,
        "touch_time": entry + pd.to_timedelta(hold * 15, unit="m"),
    })


def folds(df, n_folds=5, embargo=8):
    return list(purged_walk_forward_splits(df, n_folds, embargo))


class TestSplitStructure:

    def test_yields_folds(self):
        assert len(folds(make_labels())) >= 3

    def test_train_and_test_disjoint(self):
        for train, test in folds(make_labels()):
            assert set(train).isdisjoint(set(test))

    def test_walk_forward_not_kfold(self):
        """Every training row precedes every test row in time."""
        df = make_labels()
        for train, test in folds(df):
            assert df.loc[train, "entry_time"].max() < df.loc[test, "entry_time"].min()

    def test_test_blocks_move_forward(self):
        df = make_labels()
        starts = [df.loc[test, "entry_time"].min() for _, test in folds(df)]
        assert starts == sorted(starts)
        assert len(set(starts)) == len(starts)


class TestLeakageProtection:

    def test_purge_no_label_window_overlaps_test(self):
        """No training label may still be 'open' when the test fold begins."""
        df = make_labels()
        for train, test in folds(df):
            test_start = df.loc[test, "entry_time"].min()
            assert df.loc[train, "touch_time"].max() < test_start

    def test_embargo_gap_respected(self):
        df = make_labels()
        embargo = 12
        for train, test in folds(df, embargo=embargo):
            test_start = df.loc[test, "entry_time"].min()
            assert df.loc[train, "entry_time"].max() < test_start - embargo * BAR

    def test_purge_removes_straddling_label(self):
        """A row just before the boundary whose outcome lands inside the test
        window must be dropped; its short-lived neighbour must be kept."""
        df = make_labels(n=600, max_hold_bars=2)
        boundary = int(np.linspace(0, len(df), 6).astype(int)[1])
        test_start = df.loc[boundary, "entry_time"]
        df.loc[boundary - 1, "touch_time"] = test_start + 10 * BAR   # straddles
        train, _ = folds(df, embargo=0)[0]
        assert (boundary - 1) not in set(train)
        assert (boundary - 3) in set(train)

    def test_more_embargo_never_adds_training_rows(self):
        df = make_labels()
        small = folds(df, embargo=2)
        large = folds(df, embargo=40)
        for (tr_small, _), (tr_large, _) in zip(small, large):
            assert len(tr_large) <= len(tr_small)

    def test_tiny_dataset_yields_nothing(self):
        assert folds(make_labels(n=40)) == []
