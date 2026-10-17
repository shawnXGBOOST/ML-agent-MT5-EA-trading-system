"""
Target Geometry Stability Check (ChatGPT's Finding E)
==========================================================

Tests a genuinely different hypothesis: maybe NAS100's directional
predictability never broke - maybe the 2xATR/1xATR barrier geometry itself
stopped fitting how NAS100 actually moves recently. If recent price action
shifted from "fast continuation" toward "deeper retracement then
continuation", a 1xATR stop gets hit before the real move happens - the
label would count a directionally-correct trade as a loss.

Checkable directly from data already in the feature dataset (mfe_in_atr,
mae_in_atr, bars_to_touch) - no retraining needed.

Usage:
    python target_geometry_check.py \\
        --dataset phase1_output_nas100/features/M15_feature_dataset.csv \\
        --split-date 2026-03-11 --out phase1_output_nas100/target_geometry
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def summarize(df: pd.DataFrame, label: str):
    print(f"\n{label} (n={len(df)}):")
    print(f"  MFE (ATR): mean={df['mfe_in_atr'].mean():.3f}  median={df['mfe_in_atr'].median():.3f}  "
          f"p75={df['mfe_in_atr'].quantile(0.75):.3f}  p90={df['mfe_in_atr'].quantile(0.90):.3f}")
    print(f"  MAE (ATR): mean={df['mae_in_atr'].mean():.3f}  median={df['mae_in_atr'].median():.3f}  "
          f"p75={df['mae_in_atr'].quantile(0.75):.3f}  p90={df['mae_in_atr'].quantile(0.90):.3f}")
    print(f"  Bars to touch: mean={df['bars_to_touch'].mean():.1f}  median={df['bars_to_touch'].median():.1f}")

    stopped_but_had_room = df[(df["mae_in_atr"].between(0.9, 1.3)) & (df["mfe_in_atr"] > 2.0)]
    pct = len(stopped_but_had_room) / max(len(df), 1)
    print(f"  Trades stopped near 1xATR (0.9-1.3) that ALSO showed MFE>2.0 "
          f"(directionally correct, geometry too tight): {len(stopped_but_had_room)} ({pct:.1%})")
    return {
        "n": len(df), "mfe_mean": df["mfe_in_atr"].mean(), "mae_mean": df["mae_in_atr"].mean(),
        "mfe_median": df["mfe_in_atr"].median(), "mae_median": df["mae_in_atr"].median(),
        "stopped_but_had_room_pct": pct,
    }


def run(dataset_path: str, split_date: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_path, parse_dates=["event_time", "entry_time", "touch_time"])
    df = df.sort_values("entry_time").reset_index(drop=True)
    df = df.dropna(subset=["mfe_in_atr", "mae_in_atr"])

    split = pd.Timestamp(split_date)
    train = df[df["entry_time"] < split]
    oos = df[df["entry_time"] >= split]

    print(f"Dataset: {len(df)} events with resolved MFE/MAE")
    print(f"Split at {split_date}: train={len(train)}, oos={len(oos)}")

    train_stats = summarize(train, "TRAIN period")
    oos_stats = summarize(oos, "OOS period (2026)")

    print("\n" + "=" * 70)
    print("STATISTICAL TESTS: did the MFE/MAE distributions actually shift?")
    print("=" * 70)
    mfe_ks = stats.ks_2samp(train["mfe_in_atr"], oos["mfe_in_atr"])
    mae_ks = stats.ks_2samp(train["mae_in_atr"], oos["mae_in_atr"])
    print(f"MFE distribution - KS p-value: {mfe_ks.pvalue:.4f} "
          f"({'SIGNIFICANT shift' if mfe_ks.pvalue < 0.01 else 'no significant shift'})")
    print(f"MAE distribution - KS p-value: {mae_ks.pvalue:.4f} "
          f"({'SIGNIFICANT shift' if mae_ks.pvalue < 0.01 else 'no significant shift'})")

    print("\n" + "=" * 70)
    print("THE KEY COMPARISON")
    print("=" * 70)
    room_change = oos_stats["stopped_but_had_room_pct"] - train_stats["stopped_but_had_room_pct"]
    print(f"'Stopped near 1xATR but had >2xATR of room' rate: "
          f"{train_stats['stopped_but_had_room_pct']:.1%} (train) -> "
          f"{oos_stats['stopped_but_had_room_pct']:.1%} (OOS), change: {room_change:+.1%}")

    print("\nVERDICT:")
    if room_change > 0.03 and mae_ks.pvalue < 0.05:
        print("  Evidence SUPPORTS the target-geometry hypothesis: trades are more often")
        print("  getting stopped out right before a real move happens than they used to.")
        print("  Worth testing wider stops (e.g. 1.3-1.5x ATR) as a DIAGNOSTIC, not a tuned")
        print("  optimization, to see if win rate recovers with more breathing room.")
    else:
        print("  No strong evidence for a target-geometry shift. This argues AGAINST 'the")
        print("  stops are wrong' as the explanation, and points back toward either a")
        print("  genuinely smaller real edge, or the untested session/event-representation")
        print("  hypotheses.")

    pd.DataFrame([{"period": "train", **train_stats}, {"period": "oos", **oos_stats}]).to_csv(
        out / "target_geometry_summary.csv", index=False)
    print(f"\nSaved to: {out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Target geometry stability check")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split-date", required=True)
    parser.add_argument("--out", default="./target_geometry")
    args = parser.parse_args()

    run(args.dataset, args.split_date, args.out)
