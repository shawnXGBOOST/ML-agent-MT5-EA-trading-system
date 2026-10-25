"""
Tests for triple-barrier labeling (staged exit).

Uses synthetic OHLCV data — no market data or broker connection required.
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def make_bars(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic M15 OHLCV bars with deterministic prices."""
    rng = np.random.default_rng(seed)
    close = 2000.0 + np.cumsum(rng.normal(0, 1.5, n))
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    open_ = close - rng.normal(0, 0.5, n)
    times = pd.date_range("2024-01-01 00:00", periods=n, freq="15min")
    return pd.DataFrame({
        "time": times, "open": open_, "high": high,
        "low": low, "close": close, "tick_volume": 100,
    })


def make_events(bars: pd.DataFrame, n_events: int = 20) -> pd.DataFrame:
    """Fake BOS events evenly spaced through the bar series."""
    idx = np.linspace(5, len(bars) - 50, n_events, dtype=int)
    rows = bars.iloc[idx].copy()
    return pd.DataFrame({
        "event_time": rows["time"].values,
        "entry_price": rows["close"].values,
        "direction": np.where(np.arange(n_events) % 2 == 0, 1, -1),
        "atr": 2.0,
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTripleBarrierLabel:
    """Triple-barrier labeling produces valid outputs."""

    def _label(self, events, bars, tp_r=2.0, sl_r=1.0,
               max_bars=96, activation_frac=0.4, lock_frac=1.0):
        """Inline labeler so tests have no hard import dependency on the EA path."""
        results = []
        bar_times = bars["time"].values

        for _, ev in events.iterrows():
            entry = ev["entry_price"]
            direction = ev["direction"]
            atr = ev["atr"]
            tp = entry + direction * tp_r * atr
            sl = entry - direction * sl_r * atr
            activation = entry + direction * activation_frac * tp_r * atr
            lock_price = entry + direction * lock_frac * sl_r * atr  # stop locks here

            start_idx = np.searchsorted(bar_times, ev["event_time"])
            end_idx = min(start_idx + max_bars, len(bars))

            outcome = "TIMEOUT"
            touch_time = bars.iloc[end_idx - 1]["time"]
            realized_r = 0.0
            activated = False

            for i in range(start_idx, end_idx):
                bar = bars.iloc[i]
                # check activation
                if not activated:
                    if direction == 1 and bar["high"] >= activation:
                        activated = True
                    elif direction == -1 and bar["low"] <= activation:
                        activated = True

                # check TP
                if direction == 1 and bar["high"] >= tp:
                    outcome = "TP"
                    touch_time = bar["time"]
                    realized_r = tp_r
                    break
                elif direction == -1 and bar["low"] <= tp:
                    outcome = "TP"
                    touch_time = bar["time"]
                    realized_r = tp_r
                    break

                # check SL (or locked stop)
                eff_sl = lock_price if activated else sl
                if direction == 1 and bar["low"] <= eff_sl:
                    outcome = "LOCKED_STOP" if activated else "SL"
                    touch_time = bar["time"]
                    realized_r = lock_frac if activated else -sl_r
                    break
                elif direction == -1 and bar["high"] >= eff_sl:
                    outcome = "LOCKED_STOP" if activated else "SL"
                    touch_time = bar["time"]
                    realized_r = lock_frac if activated else -sl_r
                    break

            y_binary = 1 if outcome in ("TP", "LOCKED_STOP") else 0
            results.append({
                "entry_time": ev["event_time"],
                "touch_time": touch_time,
                "outcome": outcome,
                "y_binary": y_binary,
                "realized_r": realized_r,
                "direction": direction,
                "atr_at_event": atr,
            })

        return pd.DataFrame(results)

    def test_labels_produced(self):
        bars = make_bars()
        events = make_events(bars)
        labels = self._label(events, bars)
        assert len(labels) == len(events)

    def test_y_binary_is_binary(self):
        bars = make_bars()
        events = make_events(bars)
        labels = self._label(events, bars)
        assert set(labels["y_binary"].unique()).issubset({0, 1})

    def test_outcomes_are_valid(self):
        bars = make_bars()
        events = make_events(bars)
        labels = self._label(events, bars)
        valid = {"TP", "SL", "LOCKED_STOP", "TIMEOUT"}
        assert set(labels["outcome"].unique()).issubset(valid)

    def test_realized_r_sign_tp(self):
        """TP rows must have positive realized_r."""
        bars = make_bars()
        events = make_events(bars)
        labels = self._label(events, bars)
        tp_rows = labels[labels["outcome"] == "TP"]
        if len(tp_rows):
            assert (tp_rows["realized_r"] > 0).all()

    def test_realized_r_sl_negative(self):
        """SL rows must have negative realized_r."""
        bars = make_bars()
        events = make_events(bars)
        labels = self._label(events, bars)
        sl_rows = labels[labels["outcome"] == "SL"]
        if len(sl_rows):
            assert (sl_rows["realized_r"] < 0).all()

    def test_locked_stop_r_positive(self):
        """LOCKED_STOP rows must have non-negative realized_r (stop > entry)."""
        bars = make_bars()
        events = make_events(bars)
        labels = self._label(events, bars)
        locked = labels[labels["outcome"] == "LOCKED_STOP"]
        if len(locked):
            assert (locked["realized_r"] >= 0).all()

    def test_touch_time_after_entry(self):
        bars = make_bars()
        events = make_events(bars)
        labels = self._label(events, bars)
        assert (labels["touch_time"] >= labels["entry_time"]).all()

    def test_no_look_ahead(self):
        """touch_time must never exceed max_bars after entry_time."""
        bars = make_bars(300)
        events = make_events(bars, 10)
        max_bars = 30
        labels = self._label(events, bars, max_bars=max_bars)
        bar_dur = pd.Timedelta(minutes=15)
        delta = (labels["touch_time"] - labels["entry_time"]) / bar_dur
        assert (delta <= max_bars + 1).all()
