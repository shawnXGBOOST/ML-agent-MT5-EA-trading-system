"""
Tests for the production triple-barrier labeler (staged exit with stop-lock).

Calls the real `compute_staged_labels` on tiny hand-built bar series where
the correct outcome can be worked out by hand. No market data needed.

Geometry used throughout (long, ATR = 1, entry = 100):
    initial stop   = 99.0
    full TP        = 102.0   (tp_atr_mult = 2)
    activation     = 100.8   (40% of the way to TP)
    lock price     = 100.8   (lock_fraction = 1.0)
"""

import numpy as np
import pandas as pd
import pytest

from mltrader.labeling.triple_barrier import compute_staged_labels

BASE = 100.0


def make_bars(n=10, bullish=True):
    """Flat bars at 100 with ATR 1; a BOS event on bar 0 (entry on bar 1 open)."""
    df = pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=n, freq="15min"),
        "open": BASE, "high": BASE + 0.5, "low": BASE - 0.5, "close": BASE,
        "atr": 1.0,
        "bos_bullish": False, "bos_bearish": False,
    })
    df.loc[0, "bos_bullish" if bullish else "bos_bearish"] = True
    return df


def label(df, **kw):
    out = compute_staged_labels(df, **kw)
    assert len(out) == 1
    return out.iloc[0]


class TestLongOutcomes:

    def test_full_tp(self):
        df = make_bars()
        df.loc[1, ["high", "low"]] = [102.5, 99.9]
        row = label(df)
        assert row["label"] == "FULL_TP"
        assert row["realized_r"] == pytest.approx(2.0)
        assert row["y_binary"] == 1

    def test_initial_stop(self):
        df = make_bars()
        df.loc[1, ["high", "low"]] = [100.5, 98.9]
        row = label(df)
        assert row["label"] == "INITIAL_STOP"
        assert row["realized_r"] == pytest.approx(-1.0)
        assert row["y_binary"] == 0

    def test_locked_stop_after_activation(self):
        """Bar 1 reaches activation (100.8) without hitting TP/stop; bar 2 falls
        back through the locked stop -> small win of +0.8R, not a loss."""
        df = make_bars()
        df.loc[1, ["high", "low"]] = [100.9, 99.5]
        df.loc[2, ["high", "low"]] = [100.9, 100.5]
        row = label(df)
        assert row["label"] == "LOCKED_STOP"
        assert row["realized_r"] == pytest.approx(0.8)
        assert row["y_binary"] == 1

    def test_stop_before_activation_is_a_loss(self):
        """Dropping through the stop without ever reaching activation stays -1R."""
        df = make_bars()
        df.loc[1, ["high", "low"]] = [100.5, 99.5]
        df.loc[2, ["high", "low"]] = [100.5, 98.5]
        row = label(df)
        assert row["label"] == "INITIAL_STOP"
        assert row["realized_r"] == pytest.approx(-1.0)

    def test_timeout_uses_last_close(self):
        df = make_bars()
        df.loc[4, "close"] = 100.2          # time barrier = entry_idx(1) + max_bars(3)
        row = label(df, max_bars=3)
        assert row["label"] == "TIMEOUT"
        assert row["realized_r"] == pytest.approx(0.2)
        assert row["bars_to_touch"] == 3


class TestShortOutcomes:

    def test_full_tp_short(self):
        df = make_bars(bullish=False)
        df.loc[1, ["high", "low"]] = [100.1, 97.5]
        row = label(df)
        assert row["direction"] == "short"
        assert row["label"] == "FULL_TP"
        assert row["realized_r"] == pytest.approx(2.0)

    def test_initial_stop_short(self):
        df = make_bars(bullish=False)
        df.loc[1, ["high", "low"]] = [101.1, 99.5]
        row = label(df)
        assert row["label"] == "INITIAL_STOP"
        assert row["realized_r"] == pytest.approx(-1.0)

    def test_locked_stop_short(self):
        df = make_bars(bullish=False)
        df.loc[1, ["high", "low"]] = [100.5, 99.1]     # reaches activation 99.2
        df.loc[2, ["high", "low"]] = [99.5, 99.1]      # high 99.5 >= lock 99.2
        row = label(df)
        assert row["label"] == "LOCKED_STOP"
        assert row["realized_r"] == pytest.approx(0.8)


class TestSchemaAndGuards:

    def test_output_invariants(self):
        df = make_bars(n=40)
        df.loc[[0, 10, 20], "bos_bullish"] = True
        out = compute_staged_labels(df)
        assert len(out) == 3
        assert (out["touch_time"] >= out["entry_time"]).all()
        assert (out["y_binary"] == (out["realized_r"] > 0).astype(int)).all()
        assert set(out["label"]) <= {"FULL_TP", "INITIAL_STOP", "LOCKED_STOP", "TIMEOUT"}

    def test_price_levels_recorded(self):
        row = label(make_bars())
        assert row["entry_price"] == pytest.approx(100.0)
        assert row["sl_price"] == pytest.approx(99.0)
        assert row["tp_price"] == pytest.approx(102.0)
        assert row["activation_price"] == pytest.approx(100.8)

    def test_event_with_invalid_atr_is_skipped(self):
        df = make_bars()
        df.loc[0, "atr"] = np.nan
        assert len(compute_staged_labels(df)) == 0

    def test_last_bar_event_has_no_entry(self):
        df = make_bars(n=5)
        df["bos_bullish"] = False
        df.loc[4, "bos_bullish"] = True
        assert len(compute_staged_labels(df)) == 0

    def test_lock_fraction_above_one_rejected(self):
        with pytest.raises(ValueError):
            compute_staged_labels(make_bars(), lock_fraction=1.5)

    def test_missing_columns_rejected(self):
        with pytest.raises(ValueError):
            compute_staged_labels(make_bars().drop(columns=["atr"]))
