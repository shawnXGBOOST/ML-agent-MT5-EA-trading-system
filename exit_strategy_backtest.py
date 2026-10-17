"""
XAUUSD ML+Agent+EA — Exit Strategy Comparison
=================================================

Tests whether a smarter exit scheme captures more of the edge already found
in Phase 2's triple-barrier labeling. The clue that motivated this: average
MFE (max favorable excursion) was 1.66x ATR, but the fixed TP target is
2x ATR — meaning many trades get CLOSE to target then reverse before
hitting it. That's money potentially being left on the table.

Three variants compared, all on the SAME BOS events for a fair comparison:

  A) BASELINE (current) — fixed TP=2xATR, fixed SL=1xATR, no changes during
     the trade. This is exactly what Phase 2 already validated.

  B) PARTIAL PROFIT — take 50% off at 1.2xATR, move stop on the remainder to
     breakeven, let the rest ride to the original 2xATR TP (or get stopped
     at breakeven).

  C) TRAILING STOP — no fixed TP. Once price moves 1xATR in favor, the stop
     trails at a fixed 1xATR distance behind the best price seen. Lets
     winners run further than 2xATR when the market allows it, at the cost
     of giving back more profit on trades that partially work then reverse.

Usage:
    python exit_strategy_backtest.py --events phase1_output/events/M15_events.csv \\
        --out phase1_output/exit_comparison
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def simulate_baseline(df: pd.DataFrame, entry_idx: int, direction: str, entry_price: float,
                       atr: float, tp_mult: float, sl_mult: float, max_bars: int) -> dict:
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
        if hit_tp and hit_sl:
            return {"realized_r": -sl_mult / sl_mult, "label": "SL"}  # conservative on ambiguous bar
        elif hit_tp:
            return {"realized_r": tp_mult / sl_mult, "label": "TP"}
        elif hit_sl:
            return {"realized_r": -1.0, "label": "SL"}
    return {"realized_r": 0.0, "label": "TIMEOUT"}  # simplified: flat at timeout for comparison purposes


def simulate_partial_profit(df: pd.DataFrame, entry_idx: int, direction: str, entry_price: float,
                             atr: float, partial_mult: float, full_tp_mult: float, sl_mult: float,
                             max_bars: int) -> dict:
    """50% off at partial_mult*ATR (move stop to breakeven for the rest),
    remaining 50% rides to full_tp_mult*ATR or breakeven stop."""
    if direction == "long":
        partial_price = entry_price + partial_mult * atr
        full_tp_price = entry_price + full_tp_mult * atr
        initial_sl = entry_price - sl_mult * atr
    else:
        partial_price = entry_price - partial_mult * atr
        full_tp_price = entry_price - full_tp_mult * atr
        initial_sl = entry_price + sl_mult * atr

    end_idx = min(entry_idx + max_bars, len(df) - 1)
    partial_taken = False
    current_sl = initial_sl

    for j in range(entry_idx, end_idx + 1):
        row = df.iloc[j]
        hit_initial_sl = (row["low"] <= current_sl) if direction == "long" else (row["high"] >= current_sl)
        hit_partial = (row["high"] >= partial_price) if direction == "long" else (row["low"] <= partial_price)
        hit_full_tp = (row["high"] >= full_tp_price) if direction == "long" else (row["low"] <= full_tp_price)

        if not partial_taken:
            if hit_initial_sl and hit_partial:
                # ambiguous same-bar — conservative: assume SL first
                return {"realized_r": -1.0, "label": "SL_before_partial"}
            if hit_initial_sl:
                return {"realized_r": -1.0, "label": "SL_before_partial"}
            if hit_partial:
                partial_taken = True
                current_sl = entry_price  # move to breakeven
                # continue the loop from the SAME bar to check if full TP also
                # hit later bars — but not within this same bar (simplification:
                # a partial + full-TP-in-one-bar is treated as partial only this bar)
                continue
        else:
            if hit_initial_sl:  # current_sl is now breakeven, variable name kept for simplicity
                pass
            hit_breakeven = (row["low"] <= current_sl) if direction == "long" else (row["high"] >= current_sl)
            if hit_breakeven and hit_full_tp:
                return {"realized_r": 0.5 * (partial_mult / sl_mult) + 0.5 * 0.0, "label": "partial_then_breakeven"}
            if hit_breakeven:
                return {"realized_r": 0.5 * (partial_mult / sl_mult) + 0.5 * 0.0, "label": "partial_then_breakeven"}
            if hit_full_tp:
                return {"realized_r": 0.5 * (partial_mult / sl_mult) + 0.5 * (full_tp_mult / sl_mult),
                        "label": "partial_then_full_tp"}

    if partial_taken:
        # timed out after partial — mark remaining half flat at last close
        last_close = df.iloc[end_idx]["close"]
        remaining_r = (last_close - entry_price) / atr if direction == "long" else (entry_price - last_close) / atr
        return {"realized_r": 0.5 * (partial_mult / sl_mult) + 0.5 * (remaining_r / sl_mult), "label": "partial_then_timeout"}
    return {"realized_r": 0.0, "label": "TIMEOUT_no_partial"}


def simulate_trailing_stop(df: pd.DataFrame, entry_idx: int, direction: str, entry_price: float,
                            atr: float, activation_mult: float, trail_mult: float, sl_mult: float,
                            max_bars: int) -> dict:
    """No fixed TP. Initial SL at sl_mult*ATR. Once price moves activation_mult*ATR
    in favor, the stop trails at trail_mult*ATR behind the best price seen."""
    if direction == "long":
        initial_sl = entry_price - sl_mult * atr
    else:
        initial_sl = entry_price + sl_mult * atr

    end_idx = min(entry_idx + max_bars, len(df) - 1)
    current_sl = initial_sl
    best_price = entry_price
    activated = False

    for j in range(entry_idx, end_idx + 1):
        row = df.iloc[j]
        if direction == "long":
            best_price = max(best_price, row["high"])
            if not activated and (best_price - entry_price) >= activation_mult * atr:
                activated = True
            if activated:
                current_sl = max(current_sl, best_price - trail_mult * atr)
            if row["low"] <= current_sl:
                realized_r = (current_sl - entry_price) / atr / sl_mult
                return {"realized_r": realized_r, "label": "trailed_stop"}
        else:
            best_price = min(best_price, row["low"])
            if not activated and (entry_price - best_price) >= activation_mult * atr:
                activated = True
            if activated:
                current_sl = min(current_sl, best_price + trail_mult * atr)
            if row["high"] >= current_sl:
                realized_r = (entry_price - current_sl) / atr / sl_mult
                return {"realized_r": realized_r, "label": "trailed_stop"}

    last_close = df.iloc[end_idx]["close"]
    realized_r = ((last_close - entry_price) if direction == "long" else (entry_price - last_close)) / atr / sl_mult
    return {"realized_r": realized_r, "label": "TIMEOUT"}


def run(events_path: str, out_dir: str, max_bars: int):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading events from {events_path}...")
    df = pd.read_csv(events_path, parse_dates=["datetime"]).reset_index(drop=True)
    n = len(df)

    event_idxs = df.index[(df["bos_bullish"]) | (df["bos_bearish"])].tolist()
    print(f"Found {len(event_idxs)} BOS events. Simulating 3 exit variants on each...")

    results = {"baseline": [], "partial_profit": [], "trailing_stop": []}

    for i in event_idxs:
        direction = "long" if df.at[i, "bos_bullish"] else "short"
        entry_idx = i + 1
        if entry_idx >= n:
            continue
        atr = df.at[i, "atr"]
        if pd.isna(atr) or atr <= 0:
            continue
        entry_price = df.at[entry_idx, "open"]

        r_baseline = simulate_baseline(df, entry_idx, direction, entry_price, atr,
                                        tp_mult=2.0, sl_mult=1.0, max_bars=max_bars)
        r_partial = simulate_partial_profit(df, entry_idx, direction, entry_price, atr,
                                             partial_mult=1.2, full_tp_mult=2.0, sl_mult=1.0, max_bars=max_bars)
        r_trailing = simulate_trailing_stop(df, entry_idx, direction, entry_price, atr,
                                             activation_mult=1.0, trail_mult=1.0, sl_mult=1.0, max_bars=max_bars)

        results["baseline"].append(r_baseline["realized_r"])
        results["partial_profit"].append(r_partial["realized_r"])
        results["trailing_stop"].append(r_trailing["realized_r"])

    print("\n" + "=" * 60)
    print("EXIT STRATEGY COMPARISON (same BOS events, same entries)")
    print("=" * 60)
    summary = {}
    for name, r_list in results.items():
        r_arr = np.array(r_list)
        summary[name] = {
            "n": len(r_arr), "avg_r": r_arr.mean(), "total_r": r_arr.sum(),
            "win_rate": (r_arr > 0).mean(), "std_r": r_arr.std(),
        }
        print(f"\n{name.upper()}")
        print(f"  N trades: {len(r_arr)}")
        print(f"  Avg R per trade: {r_arr.mean():+.4f}")
        print(f"  Total R (sum): {r_arr.sum():+.1f}")
        print(f"  Win rate (R>0): {(r_arr > 0).mean():.1%}")
        print(f"  Std dev of R: {r_arr.std():.3f}")

    print("\n" + "-" * 60)
    best = max(summary, key=lambda k: summary[k]["avg_r"])
    print(f"Highest avg R per trade: {best} ({summary[best]['avg_r']:+.4f}R)")
    print("\nNOTE: this compares RAW exit mechanics only — it does NOT yet apply "
          "your ML/EV filter or agent risk gates. This tells you whether a smarter "
          "exit is worth pursuing at all; if one wins clearly here, the next step "
          "is re-running the FULL pipeline (features -> ML -> calibration -> EV -> "
          "agent) with that exit scheme instead of the fixed 2:1 baseline.")

    pd.DataFrame(results).to_csv(out / "exit_strategy_comparison.csv", index=False)
    print(f"\nSaved per-trade results to: {(out / 'exit_strategy_comparison.csv').resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare exit strategies on historical BOS events")
    parser.add_argument("--events", required=True, help="Path to M15_events.csv from Phase 1")
    parser.add_argument("--out", default="./phase1_output/exit_comparison")
    parser.add_argument("--max-bars", type=int, default=48)
    args = parser.parse_args()

    run(args.events, args.out, args.max_bars)
