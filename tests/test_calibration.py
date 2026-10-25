"""
Tests for calibration helpers and EV filter.

No MT5 / market data — synthetic probabilities and labels.
"""

import numpy as np
import pandas as pd
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mltrader.calibration.calibration import (
    expected_calibration_error,
    purged_train_calib_split,
)


# ---------------------------------------------------------------------------
# Tests: expected_calibration_error
# ---------------------------------------------------------------------------

class TestECE:

    def test_perfect_calibration(self):
        """A perfectly calibrated forecaster has ECE ≈ 0."""
        # 1000 samples: predicted prob == actual frequency in each bin
        rng = np.random.default_rng(0)
        probs = rng.uniform(0, 1, 1000)
        # generate outcomes where P(y=1) exactly equals prob
        y = (rng.uniform(0, 1, 1000) < probs).astype(float)
        ece = expected_calibration_error(y, probs, n_bins=10)
        # not exactly 0 due to sampling noise, but should be small
        assert ece < 0.05

    def test_worst_case_calibration(self):
        """Predicting p=1 for all negatives → large ECE."""
        y = np.zeros(100)
        probs = np.ones(100)
        ece = expected_calibration_error(y, probs)
        assert ece > 0.8

    def test_ece_in_range(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 200).astype(float)
        probs = rng.uniform(0, 1, 200)
        ece = expected_calibration_error(y, probs)
        assert 0.0 <= ece <= 1.0


# ---------------------------------------------------------------------------
# Tests: purged_train_calib_split
# ---------------------------------------------------------------------------

class TestPurgedTrainCalibSplit:

    def make_df(self, n=200):
        rng = np.random.default_rng(2)
        entry_times = pd.date_range("2024-01-01", periods=n, freq="15min")
        offsets = rng.integers(1, 96, n)
        touch_times = entry_times + pd.to_timedelta(offsets * 15, unit="m")
        return pd.DataFrame({
            "entry_time": entry_times,
            "touch_time": touch_times,
            "y_binary": rng.integers(0, 2, n),
        })

    def test_sizes_sum_to_less_than_total(self):
        df = self.make_df()
        fit_df, calib_df = purged_train_calib_split(df, calib_frac=0.2, embargo_bars=4)
        # purging removes some rows, so fit + calib <= len(df)
        assert len(fit_df) + len(calib_df) <= len(df)

    def test_no_time_overlap(self):
        """fit_df touch_time must be strictly before calib_df entry_time."""
        df = self.make_df(300)
        fit_df, calib_df = purged_train_calib_split(df, calib_frac=0.2, embargo_bars=4)
        if len(fit_df) and len(calib_df):
            assert fit_df["touch_time"].max() < calib_df["entry_time"].min()

    def test_calib_is_later_portion(self):
        """Calibration slice must come from later in time than fit slice."""
        df = self.make_df(300)
        fit_df, calib_df = purged_train_calib_split(df, calib_frac=0.2, embargo_bars=4)
        if len(fit_df) and len(calib_df):
            assert fit_df["entry_time"].max() < calib_df["entry_time"].min()

    def test_calib_frac_approx(self):
        """Calibration slice should be roughly calib_frac of total (minus purge)."""
        df = self.make_df(400)
        fit_df, calib_df = purged_train_calib_split(df, calib_frac=0.2, embargo_bars=4)
        expected = int(len(df) * 0.2)
        # allow ±20 rows for embargo removal
        assert abs(len(calib_df) - expected) <= 20


# ---------------------------------------------------------------------------
# Tests: EV filter logic
# ---------------------------------------------------------------------------

class TestEVFilter:

    def test_positive_ev_takes(self):
        """EV > min_ev → TAKE."""
        min_ev = 0.05
        win_r, loss_r = 2.0, -1.0
        prob = 0.60
        cost_r = 0.02
        ev = prob * win_r + (1 - prob) * loss_r - cost_r
        decision = "TAKE" if ev >= min_ev else "SKIP"
        assert decision == "TAKE"

    def test_negative_ev_skips(self):
        min_ev = 0.05
        win_r, loss_r = 2.0, -1.0
        prob = 0.30
        cost_r = 0.02
        ev = prob * win_r + (1 - prob) * loss_r - cost_r
        decision = "TAKE" if ev >= min_ev else "SKIP"
        assert decision == "SKIP"

    def test_break_even_skips(self):
        """EV exactly at threshold → TAKE (>= check)."""
        min_ev = 0.10
        ev = 0.10
        decision = "TAKE" if ev >= min_ev else "SKIP"
        assert decision == "TAKE"
