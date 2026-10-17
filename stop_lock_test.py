"""
Single Stop-Lock Exit Test
==============================

Different from earlier exit tests: keeps the FULL position size, but once
price reaches `activation_fraction` of the way to TP (e.g. 2/3 or 3/4),
jumps the stop ONCE to lock in `lock_fraction` of the distance already
covered. The position stays whole and can still reach full TP.

Sweeps both activation and lock fractions to show the shape of the
relationship, not one cherry-picked combination.

Usage:
    python stop_lock_test.py --events phase1_output/events/M15_events.csv \\
        --out phase1_output/stop_lock_test --max-bars 48
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def simulate_stop_lock(df, entry_idx, direction, entry_price, atr, sl_mult, tp_mult,
                        activation_fraction, lock_fraction, max_bars):
    if direction == "long":
        initial_sl = entry_price - sl_mult * atr
        tp_price = entry_price + tp_mult * atr
    else:
        initial_sl = entry_price + sl_mult * atr
        tp_price = entry_price - tp_mult * atr

    tp_distance = tp_mult * atr
    activation_price = (entry_price + activation_fraction * tp_distance if direction == "long"
                        else entry_price - activation_fraction * tp_distance)

    current_stop = initial_sl
    activated = False
    end_idx = min(entry_idx + max_bars, len(df) - 1)

    for j in range(entry_idx, end_idx + 1):
        row = df.iloc[j]
        high, low = row["high"], row["low"]

        if direction == "long":
            hit_stop = low <= current_stop
            hit_tp = high >= tp_price
            reaches_activation = (not activated) and high >= activation_price
        else:
            hit_stop = high >= current_stop
            hit_tp = low <= tp_price
            reaches_activation = (not activated) and low <= activation_price

        if hit_stop:
            stop_r = (current_stop - entry_price) if direction == "long" else (entry_price - current_stop)
            return stop_r / (sl_mult * atr)

        if hit_tp:
            return tp_mult / sl_mult

        if reaches_activation:
            activated = True
            locked_distance = activation_fraction * tp_distance * lock_fraction
            current_stop = (entry_price + locked_distance if direction == "long"
                           else entry_price - locked_distance)

    last_close = df.iloc[end_idx]["close"]
    move = (last_close - entry_price) if direction == "long" else (entry_price - last_close)
    return move / (sl_mult * atr)


def simulate_baseline(df, entry_idx, direction, entry_price, atr, sl_mult, tp_mult, max_bars):
    if direction == "long":
        tp_price = entry_price + tp_mult * atr
        sl_price = entry_price - sl_mult * atr
    else:
        tp_price = entry_price - tp_mult * atr
        sl_price = entry_price + sl_mult * atr

    end_idx = min(entry_idx + max_bars, len(df) - 1)
    for j in range(entry_idx, end_idx + 1):
        row = df.iloc[j]
        hit_tp = row["high"] >= tp_price if direction == "long" else row["low"] <= tp_price
        hit_sl = row["low"] <= sl_price if direction == "long" else row["high"] >= sl_price
        if hit_sl:
            return -1.0
        elif hit_tp:
            return tp_mult / sl_mult
    last_close = df.iloc[end_idx]["close"]
    move = (last_close - entry_price) if direction == "long" else (entry_price - last_close)
    return move / (sl_mult * atr)


def run(events_path, out_dir, max_bars, sl_mult, tp_mult, activation_fractions, lock_fractions):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading events from {events_path}...")
    df = pd.read_csv(events_path, parse_dates=["datetime"]).reset_index(drop=True)
    n = len(df)
    event_idxs = df.index[(df["bos_bullish"]) | (df["bos_bearish"])].tolist()
    print(f"Found {len(event_idxs)} BOS events.\n")

    entries = []
    for i in event_idxs:
        direction = "long" if df.at[i, "bos_bullish"] else "short"
        entry_idx = i + 1
        if entry_idx >= n:
            continue
        atr = df.at[i, "atr"]
        if pd.isna(atr) or atr <= 0:
            continue
        entries.append((entry_idx, direction, df.at[entry_idx, "open"], atr))

    baseline_r = np.array([simulate_baseline(df, ei, d, ep, a, sl_mult, tp_mult, max_bars)
                            for ei, d, ep, a in entries])
    print(f"BASELINE: avgR={baseline_r.mean():+.4f}  std={baseline_r.std():.4f}  "
          f"win={(baseline_r>0).mean():.1%}  sharpe={baseline_r.mean()/baseline_r.std():.4f}\n")

    print("=" * 90)
    print(f"{'activation':>10}{'lock':>8}{'avgR':>10}{'vs base':>10}{'std':>9}{'win%':>8}{'sharpe':>9}")
    print("=" * 90)

    results = []
    for act in activation_fractions:
        for lock in lock_fractions:
            r_arr = np.array([simulate_stop_lock(df, ei, d, ep, a, sl_mult, tp_mult, act, lock, max_bars)
                              for ei, d, ep, a in entries])
            diff = r_arr.mean() - baseline_r.mean()
            sharpe = r_arr.mean() / r_arr.std() if r_arr.std() > 0 else 0
            results.append({"activation_fraction": act, "lock_fraction": lock,
                           "avg_r": r_arr.mean(), "vs_baseline": diff, "std_r": r_arr.std(),
                           "win_rate": (r_arr > 0).mean(), "sharpe_like": sharpe})
            print(f"{act:>10.2f}{lock:>8.2f}{r_arr.mean():>+10.4f}{diff:>+10.4f}"
                  f"{r_arr.std():>9.4f}{(r_arr>0).mean():>8.1%}{sharpe:>9.4f}")

    rdf = pd.DataFrame(results)
    best = rdf.loc[rdf["avg_r"].idxmax()]

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"Best combo: activation={best['activation_fraction']:.2f}, lock={best['lock_fraction']:.2f}  "
          f"avgR={best['avg_r']:+.4f} ({best['vs_baseline']:+.4f} vs baseline)")

    if best["vs_baseline"] > 0.01:
        print("  A real improvement over baseline exists in this sweep. Check that it's not")
        print("  a single spiky point - look for NEIGHBORING combos also improving, which")
        print("  would make this more trustworthy than an isolated peak.")
    else:
        print("  No combination in this sweep beats the baseline on raw expectancy. Same")
        print("  pattern as the earlier exit-scheme tests: once TP/SL geometry is already")
        print("  validated, additional stop management tends to cost more than it protects.")

    rdf.to_csv(out / "stop_lock_sweep.csv", index=False)
    print(f"\nSaved to: {out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single stop-lock exit sweep")
    parser.add_argument("--events", required=True)
    parser.add_argument("--out", default="./stop_lock_test")
    parser.add_argument("--max-bars", type=int, default=48)
    parser.add_argument("--sl-mult", type=float, default=1.0)
    parser.add_argument("--tp-mult", type=float, default=2.0)
    parser.add_argument("--activation-fractions", type=str, default="0.5,0.67,0.75,0.85")
    parser.add_argument("--lock-fractions", type=str, default="0.25,0.5,0.75,1.0",
                         help="Fraction of distance-covered-at-activation to lock in. "
                              "MUST be <= 1.0 — a value above 1.0 places the new stop "
                              "BEYOND where price has actually reached, which breaks the "
                              "stop-hit comparison (it starts triggering on essentially "
                              "the very next bar regardless of real price action, since "
                              "price hasn't gotten there yet). This produced an invalid, "
                              "inflated result in an earlier run before this check existed.")
    args = parser.parse_args()

    act_list = [float(x) for x in args.activation_fractions.split(",")]
    lock_list = [float(x) for x in args.lock_fractions.split(",")]

    invalid_locks = [l for l in lock_list if l > 1.0]
    if invalid_locks:
        print(f"ERROR: lock_fractions {invalid_locks} exceed 1.0 — these place the stop "
              f"ahead of price and produce invalid results (see script docstring). "
              f"Refusing to run. Use values in (0, 1.0].")
        sys.exit(1)

    run(args.events, args.out, args.max_bars, args.sl_mult, args.tp_mult, act_list, lock_list)
