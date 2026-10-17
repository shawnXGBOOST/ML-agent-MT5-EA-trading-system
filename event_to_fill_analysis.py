"""
Event -> Fill Comparison
============================

ChatGPT's Finding 7: the label is defined from the event bar's price, but
the actual pending order doesn't fill until ~15 minutes later. This checks
whether that gap represents a meaningfully DIFFERENT market state - i.e.
whether "waiting for the pullback to fill" is quietly selecting adverse
setups that look fine at signal time but have already started reversing by
the time the order actually fills.

Usage:
    python event_to_fill_analysis.py \\
        --events phase1_output_nas100/events/M15_events.csv \\
        --trades phase1_output_nas100/oos_test_backtest/backtest_trades.csv \\
        --out phase1_output_nas100/event_fill_analysis
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def run(events_path: str, trades_path: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(events_path, parse_dates=["datetime"]).reset_index(drop=True)
    date_cols = [c for c in ["event_time", "fill_time", "touch_time"] if c in pd.read_csv(trades_path, nrows=0).columns]
    trades = pd.read_csv(trades_path, parse_dates=date_cols)

    print(f"Loaded {len(events)} M15 bars, {len(trades)} trades to analyze")

    events_by_time = events.set_index("datetime")

    rows = []
    for _, tr in trades.iterrows():
        event_time = tr["event_time"]
        fill_time = tr["fill_time"]
        direction = tr["direction"]

        if event_time not in events_by_time.index or fill_time not in events_by_time.index:
            continue

        event_idx = events.index[events["datetime"] == event_time][0]
        fill_idx = events.index[events["datetime"] == fill_time][0]

        event_price = events.at[event_idx, "close"]
        actual_fill_price = tr.get("entry_price", events.at[fill_idx, "close"])

        atr_at_event = events.at[event_idx, "atr"]
        if pd.isna(atr_at_event) or atr_at_event <= 0:
            continue

        raw_move = actual_fill_price - event_price
        signed_move_atr = (raw_move if direction == "long" else -raw_move) / atr_at_event

        path = events.iloc[event_idx:fill_idx + 1] if fill_idx >= event_idx else events.iloc[event_idx:event_idx+1]
        if direction == "long":
            mfe_during_wait = (path["high"].max() - event_price) / atr_at_event
            mae_during_wait = (event_price - path["low"].min()) / atr_at_event
        else:
            mfe_during_wait = (event_price - path["low"].min()) / atr_at_event
            mae_during_wait = (path["high"].max() - event_price) / atr_at_event

        rows.append({
            "event_time": event_time, "fill_time": fill_time, "direction": direction,
            "event_price": event_price, "fill_price": actual_fill_price,
            "bars_waited": fill_idx - event_idx,
            "signed_move_during_wait_atr": signed_move_atr,
            "mfe_during_wait_atr": mfe_during_wait,
            "mae_during_wait_atr": mae_during_wait,
            "won": tr.get("won", None),
        })

    result = pd.DataFrame(rows)
    if result.empty:
        print("Could not match any trades to their event bars - check that event_time/fill_time "
              "in the trades file exactly match timestamps in the events file.")
        return

    print(f"\nMatched {len(result)}/{len(trades)} trades to their event->fill path")
    print(f"\nAvg bars waited between event and fill: {result['bars_waited'].mean():.2f}")

    print("\n" + "=" * 70)
    print("PRICE MOVEMENT DURING THE EVENT->FILL WAIT")
    print("=" * 70)
    print(f"Mean signed move (positive=favorable, negative=adverse to the trade direction): "
          f"{result['signed_move_during_wait_atr'].mean():+.4f} ATR")
    print("  (A negative mean here would mean price systematically already moved AGAINST")
    print("   the trade direction before the order even filled.)")
    print(f"Std dev: {result['signed_move_during_wait_atr'].std():.4f}")
    print(f"Median: {result['signed_move_during_wait_atr'].median():+.4f}")

    if "won" in result.columns and result["won"].notna().any():
        print("\n" + "=" * 70)
        print("DOES THE EVENT->FILL MOVE PREDICT THE EVENTUAL OUTCOME?")
        print("=" * 70)
        won_moves = result[result["won"] == True]["signed_move_during_wait_atr"]
        lost_moves = result[result["won"] == False]["signed_move_during_wait_atr"]
        print(f"Winning trades - avg move during wait: {won_moves.mean():+.4f} ATR (n={len(won_moves)})")
        print(f"Losing trades  - avg move during wait: {lost_moves.mean():+.4f} ATR (n={len(lost_moves)})")
        gap = won_moves.mean() - lost_moves.mean()
        print(f"\nGap: {gap:+.4f} ATR")
        if abs(gap) > 0.1:
            print("-> There IS a meaningful relationship between what happens while waiting")
            print("   to fill and the eventual outcome. This supports Finding 7.")
        else:
            print("-> No meaningful relationship - argues AGAINST Finding 7 as the main cause.")

    result.to_csv(out / "event_to_fill.csv", index=False)
    print(f"\nSaved to: {out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Event-to-fill market state comparison")
    parser.add_argument("--events", required=True)
    parser.add_argument("--trades", required=True)
    parser.add_argument("--out", default="./event_fill_analysis")
    args = parser.parse_args()

    run(args.events, args.trades, args.out)
