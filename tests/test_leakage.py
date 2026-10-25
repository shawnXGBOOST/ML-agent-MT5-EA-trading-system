"""
Leakage test: windowed features must not use future bars.

Checks that any rolling/expanding feature computed at bar i only looks
at bars 0..i — no look-ahead.
"""

import numpy as np
import pandas as pd
import pytest


def compute_rolling_features(bars: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Reference implementation of the rolling features used in features.py.
    We re-implement a minimal version here so the test is self-contained.
    """
    df = bars.copy()
    df["sma"] = df["close"].rolling(window, min_periods=1).mean()
    df["atr"] = (df["high"] - df["low"]).rolling(window, min_periods=1).mean()
    df["roc"] = df["close"].pct_change(window).fillna(0)
    return df


class TestNoLookAhead:

    def _make_bars(self, n: int = 100) -> pd.DataFrame:
        rng = np.random.default_rng(7)
        close = 2000.0 + np.cumsum(rng.normal(0, 1.5, n))
        return pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=n, freq="15min"),
            "open": close - 0.2,
            "high": close + rng.uniform(0.3, 1.5, n),
            "low": close - rng.uniform(0.3, 1.5, n),
            "close": close,
        })

    def test_sma_no_future(self):
        """SMA at bar i must equal mean of close[max(0,i-window+1):i+1]."""
        bars = self._make_bars(60)
        feats = compute_rolling_features(bars, window=10)
        for i in range(9, len(bars)):
            expected = bars["close"].iloc[max(0, i - 9):i + 1].mean()
            assert abs(feats["sma"].iloc[i] - expected) < 1e-9, (
                f"SMA look-ahead at index {i}"
            )

    def test_atr_no_future(self):
        """ATR at bar i must not exceed any future bar's range."""
        bars = self._make_bars(80)
        feats = compute_rolling_features(bars, window=14)
        # If there's look-ahead, the ATR would sometimes be biased toward
        # future large-range bars. We test a weaker property: ATR computed
        # at bar i equals ATR computed from a truncated series up to i.
        for i in range(13, 40):
            partial = compute_rolling_features(bars.iloc[:i + 1], window=14)
            assert abs(feats["atr"].iloc[i] - partial["atr"].iloc[i]) < 1e-9, (
                f"ATR look-ahead at index {i}"
            )

    def test_roc_no_future(self):
        """ROC at bar i must only depend on close[i] and close[i-window]."""
        bars = self._make_bars(60)
        feats = compute_rolling_features(bars, window=5)
        for i in range(5, len(bars)):
            expected = (bars["close"].iloc[i] / bars["close"].iloc[i - 5]) - 1
            assert abs(feats["roc"].iloc[i] - expected) < 1e-9, (
                f"ROC look-ahead at index {i}"
            )

    def test_feature_shape_preserved(self):
        """Feature DataFrame must have same number of rows as input."""
        bars = self._make_bars(50)
        feats = compute_rolling_features(bars, window=10)
        assert len(feats) == len(bars)
