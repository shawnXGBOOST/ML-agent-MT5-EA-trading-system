"""
Calibration-by-Direction Diagnostic
=======================================

Checks whether the model's calibrated probability is equally trustworthy for
longs vs shorts, and whether that calibration held up in the untouched 2026
OOS window vs the full historical CV period. Per the "second opinion" logic:
unconditional win rates (long 45% / short 46%, say) can look fine even while
calibration is systematically off in one direction - and since the EV filter
operates on calibrated probability, a direction-specific calibration bug
would explain selective underperformance without showing up in a simple
win-rate split.

Usage:
    python calibration_by_direction.py \\
        --reference phase1_output_nas100/ev/M15_calibrated_ev_dataset.csv \\
        --test phase1_output_nas100/oos_test_backtest/backtest_trades.csv \\
        --out phase1_output_nas100/calibration_diagnostic
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def calibration_table(df: pd.DataFrame, prob_col: str, outcome_col: str, bins=None) -> pd.DataFrame:
    if bins is None:
        bins = np.arange(0.30, 0.85, 0.05)
    df = df.copy()
    df["prob_bucket"] = pd.cut(df[prob_col], bins=bins)
    table = df.groupby("prob_bucket", observed=True).agg(
        n=(outcome_col, "count"),
        avg_predicted=(prob_col, "mean"),
        actual_rate=(outcome_col, "mean"),
    )
    table["gap"] = table["avg_predicted"] - table["actual_rate"]
    return table


def run(reference_path: str, test_path: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ref = pd.read_csv(reference_path)
    test = pd.read_csv(test_path)

    ref_outcome_col = "y_binary" if "y_binary" in ref.columns else "won"
    test_outcome_col = "y_binary" if "y_binary" in test.columns else "won"
    if test_outcome_col == "won":
        test[test_outcome_col] = test[test_outcome_col].astype(int)

    if "decision" in ref.columns:
        ref = ref[ref["decision"] == "TAKE"]

    print("=" * 70)
    print("REFERENCE (full history, cross-validated) - calibration by direction")
    print("=" * 70)
    for direction, g in ref.groupby("direction"):
        print(f"\n{direction.upper()} (n={len(g)}):")
        tbl = calibration_table(g, "calibrated_prob", ref_outcome_col)
        print(tbl.to_string())
        weighted_gap = (tbl["gap"] * tbl["n"]).sum() / tbl["n"].sum()
        print(f"  Weighted avg calibration gap (predicted - actual): {weighted_gap:+.3f}")

    print("\n" + "=" * 70)
    print("TEST (2026 OOS window) - calibration by direction")
    print("=" * 70)
    for direction, g in test.groupby("direction"):
        if len(g) < 15:
            print(f"\n{direction.upper()} (n={len(g)}): too few trades for a reliable calibration table")
            continue
        print(f"\n{direction.upper()} (n={len(g)}):")
        tbl = calibration_table(g, "calibrated_prob", test_outcome_col)
        print(tbl.to_string())
        weighted_gap = (tbl["gap"] * tbl["n"]).sum() / tbl["n"].sum()
        print(f"  Weighted avg calibration gap (predicted - actual): {weighted_gap:+.3f}")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("Positive gap = model was OVERCONFIDENT (predicted higher than actual) for that bucket/direction.")
    print("Negative gap = model was UNDERCONFIDENT.")
    print("Compare the REFERENCE gaps (expected to be small, since calibration was fit on this pool)")
    print("against the TEST gaps (2026 OOS). If one direction shows a much larger positive gap in")
    print("TEST than in REFERENCE, that direction's calibration broke down specifically in 2026 -")
    print("meaning the EV filter is letting through trades based on an overconfident probability")
    print("for that direction, which would explain selective underperformance.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibration-by-direction diagnostic")
    parser.add_argument("--reference", required=True,
                         help="Full-history calibrated EV dataset (from calibrate_and_ev.py)")
    parser.add_argument("--test", required=True,
                         help="OOS backtest trades CSV (from realistic_backtest.py)")
    parser.add_argument("--out", default="./calibration_diagnostic")
    args = parser.parse_args()

    run(args.reference, args.test, args.out)
