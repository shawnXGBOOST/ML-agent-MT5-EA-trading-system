"""
Leakage test for windowed features.

Principle: a feature computed at bar t may only use bars <= t. So computing
features on the full series and on a series truncated at bar k must agree on
every bar before k. If a feature peeks at the future, the two diverge.

Runs the real feature functions from mltrader.features.features, plus a
deliberately leaky control to prove the check actually catches look-ahead.
"""

import numpy as np
import pandas as pd
import pytest

from mltrader.features.features import (
    build_core_features,
    ma_slope,
    roc,
    rsi,
)

N = 400
CUT = 250


def make_bars(n=N, seed=3):
    rng = np.random.default_rng(seed)
    close = 2000 + np.cumsum(rng.normal(0, 2, n))
    high = close + rng.uniform(0.2, 2, n)
    low = close - rng.uniform(0.2, 2, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    hour = idx.hour
    return pd.DataFrame({
        "datetime": idx,
        "open": close + rng.normal(0, 0.3, n),
        "high": high, "low": low, "close": close,
        "atr": pd.Series(high - low).rolling(14, min_periods=1).mean(),
        "swing_high": rng.random(n) < 0.08,
        "swing_low": rng.random(n) < 0.08,
        "session_london": (hour >= 8) & (hour < 16),
        "session_ny": (hour >= 13) & (hour < 21),
    })


def assert_same_on_prefix(full: pd.Series, truncated: pd.Series, name: str):
    pd.testing.assert_series_equal(
        full.iloc[:CUT].reset_index(drop=True),
        truncated.iloc[:CUT].reset_index(drop=True),
        check_names=False, rtol=1e-9, atol=1e-12,
        obj=f"{name} (full vs truncated)",
    )


class TestSingleIndicators:

    @pytest.mark.parametrize("fn", [rsi, roc, ma_slope])
    def test_indicator_does_not_use_future_bars(self, fn):
        close = make_bars()["close"]
        assert_same_on_prefix(fn(close), fn(close.iloc[:CUT]), fn.__name__)


class TestCoreFeatureBuilder:

    def test_no_feature_column_uses_future_bars(self):
        bars = make_bars()
        full = build_core_features(bars)
        trunc = build_core_features(bars.iloc[:CUT].copy())
        new_cols = [c for c in full.columns if c not in bars.columns]
        assert len(new_cols) >= 10
        for col in new_cols:
            assert_same_on_prefix(full[col], trunc[col], col)

    def test_adds_expected_model_inputs(self):
        full = build_core_features(make_bars())
        for col in ("return_1", "return_48", "rsi_14", "roc_10", "ma_slope_20",
                    "realized_vol_20", "dist_to_last_swing_high_atr",
                    "bars_since_session_london_open"):
            assert col in full.columns


class TestLeakDetectorWorks:

    def test_check_catches_a_deliberately_leaky_feature(self):
        """Control: a feature that looks one bar ahead must FAIL the prefix check,
        otherwise the tests above would pass vacuously."""
        close = make_bars()["close"]

        def leaky(s):
            return s.shift(-1)

        with pytest.raises(AssertionError):
            assert_same_on_prefix(leaky(close), leaky(close.iloc[:CUT]), "leaky")
