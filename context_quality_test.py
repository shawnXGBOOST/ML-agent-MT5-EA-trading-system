"""
Context Quality Stratification (tests the core hypothesis BEFORE building
any new features)
=====================================================================================

ChatGPT's hypothesis: "All BOS = ~45%, but high-quality contextual BOS
= 55-60%, while low-quality contextual BOS = 30-35%." If true, a real
"context quality" axis should already produce visible dispersion using
features we ALREADY have - HTF alignment, proximity to swing levels,
volatility regime - without writing a single new feature.

Cheapest possible test of the whole proposed direction before investing in
building order-block/FVG/SNR detectors.

Quality score built ONLY from features already in the dataset:
  +1 if H1 structure aligns with the M15 BOS direction
  +1 if H4 structure aligns with the M15 BOS direction
  +1 if the BOS occurs during a volatility expansion (top-tercile ATR percentile)
  +1 if price is NOT already extended from the last swing level in that
     direction (a crude proxy for "room to move")

Usage:
    python context_quality_test.py \\
        --dataset phase1_output_nas100/features/M15_feature_dataset.csv \\
        --out phase1_output_nas100/context_quality
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_quality_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    score = pd.Series(0, index=df.index)
    score += df["h1_aligned_with_m15"].fillna(0).astype(int)
    score += df["h4_aligned_with_m15"].fillna(0).astype(int)
    score += (df["atr_percentile"] >= 0.66).astype(int)

    room = np.where(
        df["direction"] == "long",
        (-df["dist_to_last_swing_high_atr"] > 1.0).astype(int),
        (df["dist_to_last_swing_low_atr"] > 1.0).astype(int),
    )
    score += room

    df["context_quality_score"] = score
    return df


def run(dataset_path: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_path, parse_dates=["event_time", "entry_time", "touch_time"])
    required = ["h1_aligned_with_m15", "h4_aligned_with_m15", "atr_percentile",
                "dist_to_last_swing_high_atr", "dist_to_last_swing_low_atr", "direction", "y_binary"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    df = build_quality_score(df)

    print(f"Loaded {len(df)} events. Overall TP rate: {df['y_binary'].mean():.1%}\n")

    print("=" * 70)
    print("TP RATE BY CONTEXT QUALITY SCORE (0-4, higher = more factors aligned)")
    print("=" * 70)
    summary = df.groupby("context_quality_score").agg(
        n=("y_binary", "count"), tp_rate=("y_binary", "mean")
    )
    print(summary.to_string())

    low = df[df["context_quality_score"] <= 1]
    high = df[df["context_quality_score"] >= 3]
    print(f"\nLOW quality (score 0-1):  n={len(low)}   TP rate={low['y_binary'].mean():.1%}")
    print(f"HIGH quality (score 3-4): n={len(high)}   TP rate={high['y_binary'].mean():.1%}")
    spread = high["y_binary"].mean() - low["y_binary"].mean()
    print(f"\nSpread (high - low): {spread:+.1%}")

    for name, g in [("LOW", low), ("HIGH", high), ("ALL", df)]:
        wr = g["y_binary"].mean()
        ev = wr * 2.0 - (1 - wr) * 1.0
        print(f"  {name:<5} raw expectancy at 2:1 R:R: {ev:+.3f}R (n={len(g)})")

    print("\nVERDICT:")
    if spread > 0.10:
        print("  A REAL quality axis already exists in current features. Supports")
        print("  investing further in dedicated context/structure features.")
    elif spread > 0.05:
        print("  A modest quality axis exists - some signal, but not dramatic. Worth a")
        print("  cautious next step (one well-defined feature at a time, strict OOS).")
    else:
        print("  Little to no dispersion from these proxies. Doesn't rule out that")
        print("  better-defined context features could find something, but the effect")
        print("  isn't showing up cheaply, and a large speculative feature-engineering")
        print("  effort carries real overfitting risk given the modest baseline edge.")

    df.to_csv(out / "context_quality_scored.csv", index=False)
    print(f"\nSaved to: {out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Context quality stratification test")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", default="./context_quality")
    args = parser.parse_args()

    run(args.dataset, args.out)
