"""
Regime Trade Inspector
==========================

Before excluding a regime outright, look at what's actually inside it.
Checks for: direction bias, time clustering (day/hour), confidence level,
session split, and whether losses cluster in a specific sub-window rather
than being spread evenly - any of which would suggest a narrower, smarter
fix than a blanket exclusion.

Usage:
    python inspect_regime_trades.py \\
        --outcomes phase1_output/live_tracking/live_outcomes.csv \\
        --regime trending_normal_vol --symbol XAUUSD
"""

import argparse

import pandas as pd


def run(outcomes_path: str, regime: str, symbol: str):
    df = pd.read_csv(outcomes_path, parse_dates=["open_time", "close_time"])
    if "symbol" in df.columns:
        df = df[df["symbol"] == symbol]
    sub = df[df["regime"] == regime].sort_values("open_time")

    print(f"{symbol} / {regime}: {len(sub)} trades\n")
    if sub.empty:
        print("No trades found for this regime/symbol combination.")
        return

    cols = [c for c in ["open_time", "direction", "calibrated_prob", "ev_r", "confidence_tier",
                        "session_london", "session_ny", "session_asia", "won", "realized_r",
                        "entry_price", "sl_price", "tp_price"] if c in sub.columns]
    print("=" * 100)
    print("PER-TRADE DETAIL")
    print("=" * 100)
    print(sub[cols].to_string(index=False))

    print("\n" + "=" * 70)
    print("PATTERN CHECKS")
    print("=" * 70)

    print("\nDirection split:")
    print(sub.groupby("direction").agg(n=("won", "count"), win_rate=("won", "mean")).to_string())

    if "confidence_tier" in sub.columns:
        print("\nConfidence tier split:")
        print(sub.groupby("confidence_tier").agg(n=("won", "count"), win_rate=("won", "mean")).to_string())

    print("\nTime clustering (day of week):")
    sub["day_of_week"] = sub["open_time"].dt.day_name()
    print(sub.groupby("day_of_week").agg(n=("won", "count"), win_rate=("won", "mean")).to_string())

    print("\nTime clustering (hour of day, broker time):")
    sub["hour"] = sub["open_time"].dt.hour
    print(sub.groupby("hour").agg(n=("won", "count"), win_rate=("won", "mean")).to_string())

    if {"session_london", "session_ny", "session_asia"}.issubset(sub.columns):
        print("\nSession split:")
        for sess in ["session_london", "session_ny", "session_asia"]:
            g = sub[sub[sess] == True]
            if len(g) > 0:
                print(f"  {sess}: n={len(g)}  win_rate={g['won'].mean():.1%}")

    print(f"\nAvg calibrated_prob (this regime): {sub['calibrated_prob'].mean():.3f}")
    if sub["won"].any():
        print(f"Avg calibrated_prob (won trades):  {sub[sub['won']==True]['calibrated_prob'].mean():.3f}")
    else:
        print("Avg calibrated_prob (won trades):  (no wins)")
    if (~sub["won"]).any():
        print(f"Avg calibrated_prob (lost trades): {sub[sub['won']==False]['calibrated_prob'].mean():.3f}")
    else:
        print("Avg calibrated_prob (lost trades): (no losses)")

    print("\n" + "=" * 70)
    print("VERDICT GUIDE")
    print("=" * 70)
    print("- If losses are spread evenly across direction/time/session -> genuinely this")
    print("  regime, not a narrower sub-pattern. Blanket exclusion is the right call.")
    print("- If losses cluster heavily in ONE direction, ONE session, or ONE tight time")
    print("  window -> a narrower, smarter filter could preserve the good trades in this")
    print("  regime while cutting the bad ones.")
    print("- If losing trades have notably LOWER calibrated_prob than winners -> the EV")
    print("  filter is already doing its job correctly; the regime itself is just weak")
    print("  right now, not a calibration failure.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a specific regime's live trades")
    parser.add_argument("--outcomes", required=True)
    parser.add_argument("--regime", required=True)
    parser.add_argument("--symbol", required=True)
    args = parser.parse_args()

    run(args.outcomes, args.regime, args.symbol)
