"""
Step 1 of rebuilding the pipeline around the staged-exit scheme.

Regenerates labels using the VALIDATED activation/lock exit rule (default:
activate at 40% of the way to the old 2xATR target, lock in exactly what's
been covered, full position stays open toward the original TP) instead of
the fixed TP/SL rule the whole project was built on until now.

Output schema matches the original triple_barrier.py exactly (same column
names) PLUS new fields (activation_price, lock_price) so features.py and
everything downstream works unchanged - just point it at this file's output.

IMPORTANT CHANGE IN MEANING: y_binary now means "was this trade profitable
in ANY form" (reached activation OR full TP), not strictly "reached full
TP". realized_r is now the ACTUAL R-multiple achieved (variable), not a
fixed +2/-1.

Usage:
    python triple_barrier_staged.py --events phase1_output/events/M15_events.csv \\
        --out phase1_output/labels_staged --tp-atr 2.0 --sl-atr 1.0 --max-bars 48 \\
        --activation-fraction 0.4 --lock-fraction 1.0
"""

import argparse
from pathlib import Path

import pandas as pd


def compute_staged_labels(df, tp_atr_mult=2.0, sl_atr_mult=1.0, max_bars=48,
                           activation_fraction=0.4, lock_fraction=1.0):
    required = {"datetime", "open", "high", "low", "close", "atr", "bos_bullish", "bos_bearish"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input events file missing required columns: {missing}")

    if lock_fraction > 1.0:
        raise ValueError(f"lock_fraction must be <= 1.0 (got {lock_fraction}) - a value "
                          f"above 1.0 places the stop ahead of price actually reached, "
                          f"which is invalid.")

    df = df.reset_index(drop=True)
    n = len(df)
    records = []

    event_idxs = df.index[(df["bos_bullish"]) | (df["bos_bearish"])].tolist()

    for i in event_idxs:
        direction = "long" if df.at[i, "bos_bullish"] else "short"
        entry_idx = i + 1
        if entry_idx >= n:
            continue

        atr_at_event = df.at[i, "atr"]
        if pd.isna(atr_at_event) or atr_at_event <= 0:
            continue

        entry_price = df.at[entry_idx, "open"]
        entry_time = df.at[entry_idx, "datetime"]

        if direction == "long":
            initial_sl = entry_price - sl_atr_mult * atr_at_event
            tp_price = entry_price + tp_atr_mult * atr_at_event
            activation_price = entry_price + activation_fraction * tp_atr_mult * atr_at_event
        else:
            initial_sl = entry_price + sl_atr_mult * atr_at_event
            tp_price = entry_price - tp_atr_mult * atr_at_event
            activation_price = entry_price - activation_fraction * tp_atr_mult * atr_at_event

        current_stop = initial_sl
        activated = False
        lock_price = None

        time_barrier_idx = min(entry_idx + max_bars, n - 1)
        if time_barrier_idx <= entry_idx:
            continue

        label, touch_idx, realized_r = None, None, None
        running_mfe, running_mae = 0.0, 0.0

        for j in range(entry_idx, time_barrier_idx + 1):
            row = df.iloc[j]
            high, low = row["high"], row["low"]

            # Track max favorable/adverse excursion over the whole path, same
            # definition as the original triple_barrier.py, so features.py
            # (which expects these columns) works unchanged.
            if direction == "long":
                running_mfe = max(running_mfe, (high - entry_price) / atr_at_event)
                running_mae = max(running_mae, (entry_price - low) / atr_at_event)
            else:
                running_mfe = max(running_mfe, (entry_price - low) / atr_at_event)
                running_mae = max(running_mae, (high - entry_price) / atr_at_event)

            if direction == "long":
                hit_stop = low <= current_stop
                hit_tp = high >= tp_price
                reaches_activation = (not activated) and high >= activation_price
            else:
                hit_stop = high >= current_stop
                hit_tp = low <= tp_price
                reaches_activation = (not activated) and low <= activation_price

            if hit_stop:
                stop_move = (current_stop - entry_price) if direction == "long" else (entry_price - current_stop)
                realized_r = stop_move / (sl_atr_mult * atr_at_event)
                label = "LOCKED_STOP" if activated else "INITIAL_STOP"
                touch_idx = j
                break

            if hit_tp:
                realized_r = tp_atr_mult / sl_atr_mult
                label = "FULL_TP"
                touch_idx = j
                break

            if reaches_activation:
                activated = True
                locked_distance = activation_fraction * tp_atr_mult * atr_at_event * lock_fraction
                lock_price = (entry_price + locked_distance if direction == "long"
                             else entry_price - locked_distance)
                current_stop = lock_price

        if label is None:
            touch_idx = time_barrier_idx
            last_close = df.at[touch_idx, "close"]
            move = (last_close - entry_price) if direction == "long" else (entry_price - last_close)
            realized_r = move / (sl_atr_mult * atr_at_event)
            label = "TIMEOUT"

        records.append({
            "event_idx": i, "event_time": df.at[i, "datetime"], "direction": direction,
            "entry_time": entry_time, "entry_price": entry_price, "atr_at_event": atr_at_event,
            "tp_price": tp_price, "sl_price": initial_sl,
            "activation_price": activation_price, "lock_price": lock_price,
            "label": label,
            "y_binary": 1 if realized_r > 0 else 0,
            "touch_time": df.at[touch_idx, "datetime"], "bars_to_touch": touch_idx - entry_idx,
            "realized_r": realized_r,
            "mfe_in_atr": running_mfe, "mae_in_atr": running_mae,
            "realized_return": ((df.at[touch_idx, "close"] - entry_price) / entry_price if direction == "long"
                                else (entry_price - df.at[touch_idx, "close"]) / entry_price),
        })

    return pd.DataFrame(records)


def run(events_path, out_dir, tp_atr, sl_atr, max_bars, activation_fraction, lock_fraction):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading events from {events_path}...")
    df = pd.read_csv(events_path, parse_dates=["datetime"])

    print(f"Running STAGED EXIT labeling: TP={tp_atr}x ATR, SL={sl_atr}x ATR, "
          f"activation={activation_fraction} (of TP distance), lock={lock_fraction}, "
          f"time barrier={max_bars} bars...")

    labels = compute_staged_labels(df, tp_atr, sl_atr, max_bars, activation_fraction, lock_fraction)

    if labels.empty:
        print("WARNING: zero labeled events produced.")
    else:
        counts = labels["label"].value_counts()
        total = len(labels)
        print(f"\nLabeled {total} events:")
        for lbl, cnt in counts.items():
            print(f"  {lbl}: {cnt} ({cnt/total:.1%})")

        print(f"\nOverall win rate (y_binary=1, i.e. realized_r > 0): {labels['y_binary'].mean():.1%}")
        print(f"Avg realized R: {labels['realized_r'].mean():+.4f}")
        print(f"Avg bars to touch: {labels['bars_to_touch'].mean():.1f}")
        print(f"\nSanity check: this should roughly match stop_lock_test.py's result for "
              f"activation={activation_fraction}, lock={lock_fraction} (small differences from "
              f"edge-case handling are expected, not a red flag unless the gap is large).")

    out_path = out / "M15_staged_labels.csv"
    labels.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path.resolve()}")
    print("\nNext step: features.py, pointed at THIS labels file instead of the old one.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 1: staged-exit triple-barrier labeling")
    parser.add_argument("--events", required=True)
    parser.add_argument("--out", default="./phase1_output/labels_staged")
    parser.add_argument("--tp-atr", type=float, default=2.0)
    parser.add_argument("--sl-atr", type=float, default=1.0)
    parser.add_argument("--max-bars", type=int, default=48)
    parser.add_argument("--activation-fraction", type=float, default=0.4)
    parser.add_argument("--lock-fraction", type=float, default=1.0)
    args = parser.parse_args()

    run(args.events, args.out, args.tp_atr, args.sl_atr, args.max_bars,
        args.activation_fraction, args.lock_fraction)
