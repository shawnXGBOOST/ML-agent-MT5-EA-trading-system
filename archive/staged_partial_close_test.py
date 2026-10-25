"""
Staged Partial-Close Exit Test
==================================

Tests a more sophisticated exit scheme: close 1/3 of the position at an
intermediate profit level, move stop to breakeven; close another 1/3
further along, move stop to lock in the first level's profit; let the
final 1/3 ride to full TP (or get stopped at the locked-in level).

IMPORTANT: this is a DIAGNOSTIC. If it looks meaningfully better, it
requires FULL re-validation (new labels, retrain, recalibrate, fresh OOS
test) before touching anything live.

Usage:
    python staged_partial_close_test.py --events phase1_output/events/M15_events.csv \\
        --out phase1_output/staged_exit_test --max-bars 48
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def simulate_staged_exit(df: pd.DataFrame, entry_idx: int, direction: str, entry_price: float,
                          atr: float, sl_mult: float, stage1_mult: float, stage2_mult: float,
                          full_tp_mult: float, max_bars: int) -> float:
    if direction == "long":
        initial_sl = entry_price - sl_mult * atr
        stage1_price = entry_price + stage1_mult * atr
        stage2_price = entry_price + stage2_mult * atr
        full_tp_price = entry_price + full_tp_mult * atr
    else:
        initial_sl = entry_price + sl_mult * atr
        stage1_price = entry_price - stage1_mult * atr
        stage2_price = entry_price - stage2_mult * atr
        full_tp_price = entry_price - full_tp_mult * atr

    current_stop = initial_sl
    stage = 0
    realized_r = 0.0
    remaining_fraction = 1.0

    end_idx = min(entry_idx + max_bars, len(df) - 1)

    for j in range(entry_idx, end_idx + 1):
        row = df.iloc[j]
        high, low = row["high"], row["low"]

        if direction == "long":
            hit_stop = low <= current_stop
            hit_stage1 = stage == 0 and high >= stage1_price
            hit_stage2 = stage == 1 and high >= stage2_price
            hit_full_tp = stage == 2 and high >= full_tp_price
        else:
            hit_stop = high >= current_stop
            hit_stage1 = stage == 0 and low <= stage1_price
            hit_stage2 = stage == 1 and low <= stage2_price
            hit_full_tp = stage == 2 and low <= full_tp_price

        if hit_stop:
            stop_r = (current_stop - entry_price) if direction == "long" else (entry_price - current_stop)
            realized_r += remaining_fraction * (stop_r / (sl_mult * atr))
            return realized_r

        if stage == 0 and hit_stage1:
            realized_r += (1 / 3) * (stage1_mult / sl_mult)
            remaining_fraction -= 1 / 3
            current_stop = entry_price
            stage = 1
            continue

        if stage == 1 and hit_stage2:
            realized_r += (1 / 3) * (stage2_mult / sl_mult)
            remaining_fraction -= 1 / 3
            current_stop = stage1_price
            stage = 2
            continue

        if stage == 2 and hit_full_tp:
            realized_r += remaining_fraction * (full_tp_mult / sl_mult)
            return realized_r

    last_close = df.iloc[end_idx]["close"]
    remaining_move = (last_close - entry_price) if direction == "long" else (entry_price - last_close)
    realized_r += remaining_fraction * (remaining_move / (sl_mult * atr))
    return realized_r


def simulate_baseline(df: pd.DataFrame, entry_idx: int, direction: str, entry_price: float,
                       atr: float, sl_mult: float, tp_mult: float, max_bars: int) -> float:
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


def run(events_path: str, out_dir: str, max_bars: int, sl_mult: float, tp_mult: float,
        stage1_mult: float, stage2_mult: float):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading events from {events_path}...")
    df = pd.read_csv(events_path, parse_dates=["datetime"]).reset_index(drop=True)
    n = len(df)
    event_idxs = df.index[(df["bos_bullish"]) | (df["bos_bearish"])].tolist()
    print(f"Found {len(event_idxs)} BOS events.")
    print(f"Staged scheme: 1/3 @ {stage1_mult}x ATR (-> breakeven), "
          f"1/3 @ {stage2_mult}x ATR (-> lock stage1), 1/3 @ {tp_mult}x ATR (full TP)\n")

    baseline_r, staged_r = [], []

    for i in event_idxs:
        direction = "long" if df.at[i, "bos_bullish"] else "short"
        entry_idx = i + 1
        if entry_idx >= n:
            continue
        atr = df.at[i, "atr"]
        if pd.isna(atr) or atr <= 0:
            continue
        entry_price = df.at[entry_idx, "open"]

        baseline_r.append(simulate_baseline(df, entry_idx, direction, entry_price, atr,
                                             sl_mult, tp_mult, max_bars))
        staged_r.append(simulate_staged_exit(df, entry_idx, direction, entry_price, atr,
                                              sl_mult, stage1_mult, stage2_mult, tp_mult, max_bars))

    baseline_arr = np.array(baseline_r)
    staged_arr = np.array(staged_r)

    print("=" * 60)
    print("COMPARISON (same events, same entries)")
    print("=" * 60)
    for name, arr in [("BASELINE (fixed TP)", baseline_arr), ("STAGED PARTIAL CLOSE", staged_arr)]:
        print(f"\n{name}")
        print(f"  N trades: {len(arr)}")
        print(f"  Avg R: {arr.mean():+.4f}")
        print(f"  Std dev: {arr.std():.4f}")
        print(f"  Win rate (R>0): {(arr > 0).mean():.1%}")
        print(f"  Sharpe-like (avg/std): {arr.mean()/arr.std():.4f}")

    print("\n" + "-" * 60)
    diff = staged_arr.mean() - baseline_arr.mean()
    print(f"Avg R difference (staged - baseline): {diff:+.4f}")
    sharpe_diff = (staged_arr.mean()/staged_arr.std()) - (baseline_arr.mean()/baseline_arr.std())
    print(f"Sharpe-like difference: {sharpe_diff:+.4f}")

    print("\nVERDICT:")
    if diff > 0.01 and sharpe_diff > 0:
        print("  Staged partial close wins on BOTH raw return and risk-adjusted return -")
        print("  a genuinely rare, strong result. Worth pursuing with full re-validation.")
    elif sharpe_diff > 0.02:
        print("  Staged partial close trades some raw R for meaningfully better risk-adjusted")
        print("  return (smoother equity curve). Legitimate choice if you value consistency")
        print("  over maximum expectancy.")
    else:
        print("  No clear win over the current baseline on either dimension.")

    pd.DataFrame({"baseline_r": baseline_r, "staged_r": staged_r}).to_csv(
        out / "staged_exit_comparison.csv", index=False)
    print(f"\nSaved to: {out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Staged partial-close exit test")
    parser.add_argument("--events", required=True)
    parser.add_argument("--out", default="./staged_exit_test")
    parser.add_argument("--max-bars", type=int, default=48)
    parser.add_argument("--sl-mult", type=float, default=1.0)
    parser.add_argument("--tp-mult", type=float, default=2.0)
    parser.add_argument("--stage1-mult", type=float, default=1.0)
    parser.add_argument("--stage2-mult", type=float, default=1.5)
    args = parser.parse_args()

    run(args.events, args.out, args.max_bars, args.sl_mult, args.tp_mult,
        args.stage1_mult, args.stage2_mult)
