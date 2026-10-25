"""
XAUUSD/NAS100 ML+Agent+EA - Phase 2 v2: Fill-Consistent Triple-Barrier Labeling
==================================================================================

FIXES A REAL METHODOLOGY MISMATCH: the original triple_barrier.py assumed
entry_price = NEXT bar's open (a reasonable simplification for pure research,
avoiding lookahead). But live_runner.py and realistic_backtest.py actually
place a PENDING LIMIT ORDER at the event bar's own close, which only fills
if price reaches that level within an expiry window. That's a different
price reference - the model was validated against one convention while
live/backtest execution uses another, which likely explains why realistic
backtest win rates came in meaningfully below the Phase 5 holdout numbers.

This script re-labels using the EXACT same fill logic as
realistic_backtest.py's simulate_fill_and_outcome():
  1. Entry price = the event bar's own close (a pending limit order reference)
  2. Order "fills" only if price reaches that level within expiry_bars
     (default 1 bar = ~15 min, matching live_runner's 10-min expiry)
  3. If filled, walk forward from the FILL bar (not the event bar) for the
     triple-barrier TP/SL/timeout outcome
  4. If never filled within the expiry window, labeled "UNFILLED" - a new
     category the original script didn't have, matching exactly what
     realistic_backtest.py already tracks separately

Output schema matches the original triple_barrier.py exactly (same column
names) so features.py and everything downstream works unchanged - just
point it at this file's output instead.

Usage:
    python triple_barrier_v2.py --events phase1_output/events/M15_events.csv \\
        --out phase1_output/labels_v2 --tp-atr 2.0 --sl-atr 1.0 --max-bars 48 \\
        --expiry-minutes 10
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compute_fill_consistent_labels(
    df: pd.DataFrame,
    tp_atr_mult: float = 2.0,
    sl_atr_mult: float = 1.0,
    max_bars: int = 48,
    expiry_bars: int = 1,
    ambiguous_resolution: str = "conservative",
) -> pd.DataFrame:
    """
    df must contain: datetime, open, high, low, close, atr, bos_bullish, bos_bearish
    Matches realistic_backtest.py's simulate_fill_and_outcome exactly, but
    walks the whole historical events file like the original triple_barrier.py did.
    """
    required = {"datetime", "open", "high", "low", "close", "atr", "bos_bullish", "bos_bearish"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input events file missing required columns: {missing}")

    df = df.reset_index(drop=True)
    n = len(df)
    records = []

    event_idxs = df.index[(df["bos_bullish"]) | (df["bos_bearish"])].tolist()

    for i in event_idxs:
        direction = "long" if df.at[i, "bos_bullish"] else "short"

        atr_at_event = df.at[i, "atr"]
        if pd.isna(atr_at_event) or atr_at_event <= 0:
            continue

        # Entry reference is THIS bar's close (the pending limit order price),
        # matching live_runner.py exactly - NOT the next bar's open.
        entry_price = df.at[i, "close"]
        entry_time = df.at[i, "datetime"]

        if direction == "long":
            tp_price = entry_price + tp_atr_mult * atr_at_event
            sl_price = entry_price - sl_atr_mult * atr_at_event
        else:
            tp_price = entry_price - tp_atr_mult * atr_at_event
            sl_price = entry_price + sl_atr_mult * atr_at_event

        # Fill-window check: order placed at bar i+1, must fill within expiry_bars
        order_placed_idx = i + 1
        if order_placed_idx >= n:
            continue
        fill_window_end = min(order_placed_idx + expiry_bars, n - 1)

        fill_idx = None
        for j in range(order_placed_idx, fill_window_end + 1):
            row = df.iloc[j]
            if direction == "long":
                if row["low"] <= entry_price:
                    fill_idx = j
                    break
            else:
                if row["high"] >= entry_price:
                    fill_idx = j
                    break

        if fill_idx is None:
            records.append({
                "event_idx": i, "event_time": df.at[i, "datetime"], "direction": direction,
                "entry_time": entry_time, "entry_price": entry_price, "atr_at_event": atr_at_event,
                "tp_price": tp_price, "sl_price": sl_price, "label": "UNFILLED",
                "y_binary": 0, "touch_time": None, "bars_to_touch": None,
                "mfe_in_atr": None, "mae_in_atr": None, "realized_return": None,
            })
            continue

        time_barrier_idx = min(fill_idx + max_bars, n - 1)
        if time_barrier_idx <= fill_idx:
            continue

        window = df.iloc[fill_idx:time_barrier_idx + 1]
        label, touch_idx = None, None
        for j, row in window.iterrows():
            hit_tp = row["high"] >= tp_price if direction == "long" else row["low"] <= tp_price
            hit_sl = row["low"] <= sl_price if direction == "long" else row["high"] >= sl_price
            if hit_tp and hit_sl:
                label = "SL" if ambiguous_resolution == "conservative" else "TP"
                touch_idx = j
                break
            elif hit_tp:
                label, touch_idx = "TP", j
                break
            elif hit_sl:
                label, touch_idx = "SL", j
                break
        if label is None:
            label, touch_idx = "TIMEOUT", time_barrier_idx

        path = df.iloc[fill_idx:touch_idx + 1]
        if direction == "long":
            mfe = (path["high"].max() - entry_price) / atr_at_event
            mae = (entry_price - path["low"].min()) / atr_at_event
            realized_return = (df.at[touch_idx, "close"] - entry_price) / entry_price
        else:
            mfe = (entry_price - path["low"].min()) / atr_at_event
            mae = (path["high"].max() - entry_price) / atr_at_event
            realized_return = (entry_price - df.at[touch_idx, "close"]) / entry_price

        records.append({
            "event_idx": i, "event_time": df.at[i, "datetime"], "direction": direction,
            "entry_time": df.at[fill_idx, "datetime"], "entry_price": entry_price,
            "atr_at_event": atr_at_event, "tp_price": tp_price, "sl_price": sl_price,
            "label": label, "y_binary": 1 if label == "TP" else 0,
            "touch_time": df.at[touch_idx, "datetime"], "bars_to_touch": touch_idx - fill_idx,
            "mfe_in_atr": mfe, "mae_in_atr": mae, "realized_return": realized_return,
        })

    return pd.DataFrame(records)


def run(events_path: str, out_dir: str, tp_atr: float, sl_atr: float, max_bars: int, expiry_minutes: int):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading events from {events_path}...")
    df = pd.read_csv(events_path, parse_dates=["datetime"])

    expiry_bars = max(1, round(expiry_minutes / 15))
    print(f"Running FILL-CONSISTENT triple-barrier labeling: TP={tp_atr}x ATR, SL={sl_atr}x ATR, "
          f"time barrier={max_bars} bars, expiry={expiry_minutes}min ({expiry_bars} bars)...")

    labels = compute_fill_consistent_labels(
        df, tp_atr_mult=tp_atr, sl_atr_mult=sl_atr, max_bars=max_bars, expiry_bars=expiry_bars
    )

    if labels.empty:
        print("WARNING: zero labeled events produced.")
    else:
        counts = labels["label"].value_counts()
        total = len(labels)
        print(f"\nLabeled {total} events:")
        for lbl, cnt in counts.items():
            print(f"  {lbl}: {cnt} ({cnt/total:.1%})")

        filled = labels[labels["label"] != "UNFILLED"]
        if not filled.empty:
            print(f"\nOf {len(filled)} FILLED events:")
            filled_counts = filled["label"].value_counts()
            for lbl, cnt in filled_counts.items():
                print(f"  {lbl}: {cnt} ({cnt/len(filled):.1%})")

    out_path = out / "M15_triple_barrier_labels_v2.csv"
    labels.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path.resolve()}")
    print("\nNext: features.py, pointed at this new labels file, then re-run "
          "train_baselines.py -> calibrate_and_ev.py -> regime.py -> final_holdout_eval.py "
          "-> train_final_model.py, same as before. This re-validates the WHOLE pipeline "
          "on labels that actually match what live/backtest execution does.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 v2: fill-consistent triple-barrier labeling")
    parser.add_argument("--events", required=True)
    parser.add_argument("--out", default="./phase1_output/labels_v2")
    parser.add_argument("--tp-atr", type=float, default=2.0)
    parser.add_argument("--sl-atr", type=float, default=1.0)
    parser.add_argument("--max-bars", type=int, default=48)
    parser.add_argument("--expiry-minutes", type=int, default=10)
    args = parser.parse_args()

    run(args.events, args.out, args.tp_atr, args.sl_atr, args.max_bars, args.expiry_minutes)
