"""
Calibration Resolution Diagnostic
=====================================

Checks ChatGPT's specific finding: median and 75th percentile of calibrated
probabilities in the filled-trade sample were IDENTICAL (0.437136), which
suggests isotonic regression may be collapsing a range of different raw
XGBoost scores into one flat output - a real resolution problem, not
just an artifact of small sample size.

Isotonic regression fits a step function by construction, so SOME flat
regions are normal. The question is whether the flat region your actual
trades landed in is unusually wide (many different raw scores producing
the exact same calibrated output) - if so, the model may be less granular
than it appears, and a lot of "different confidence" trades are actually
being treated identically.

Usage:
    python calibration_resolution_check.py \\
        --calibrator phase1_output_nas100/oos_test_model/calibrator.joblib \\
        --raw-probs-source phase1_output_nas100/oos_test_backtest/backtest_trades.csv \\
        --out phase1_output_nas100/calibration_resolution
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def run(calibrator_path: str, trades_path: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    calibrator = joblib.load(calibrator_path)
    trades = pd.read_csv(trades_path)

    print("=" * 70)
    print("PART 1 - Calibrator's own step structure")
    print("=" * 70)
    grid = np.linspace(0.0, 1.0, 2001)
    calibrated_grid = calibrator.predict(grid)

    unique_outputs = np.unique(calibrated_grid)
    print(f"Probed {len(grid)} raw-probability points -> {len(unique_outputs)} distinct calibrated outputs")
    print("(Isotonic regression is a step function by construction - some flatness is normal.")
    print(" The question is whether any single step is unusually WIDE.)\n")

    step_widths = []
    current_val = calibrated_grid[0]
    step_start = grid[0]
    for i in range(1, len(grid)):
        if calibrated_grid[i] != current_val:
            step_widths.append({"calibrated_value": current_val, "raw_range_start": step_start,
                                 "raw_range_end": grid[i-1], "width": grid[i-1] - step_start})
            current_val = calibrated_grid[i]
            step_start = grid[i]
    step_widths.append({"calibrated_value": current_val, "raw_range_start": step_start,
                         "raw_range_end": grid[-1], "width": grid[-1] - step_start})

    steps_df = pd.DataFrame(step_widths)
    steps_df = steps_df.sort_values("width", ascending=False)
    print("Top 10 WIDEST flat regions in the calibration curve (raw probability -> same output):")
    print(steps_df.head(10).to_string(index=False))

    target = 0.437136
    near_target = steps_df[(steps_df["calibrated_value"] - target).abs() < 0.01]
    if not near_target.empty:
        print(f"\nStep(s) near the flagged value {target}:")
        print(near_target.to_string(index=False))
        widest_near_target = near_target["width"].max()
        print(f"\nWidest matching step covers a raw-probability RANGE of {widest_near_target:.4f} "
              f"({widest_near_target*100:.1f}% of the full 0-1 range mapped to one output)")
        if widest_near_target > 0.05:
            print("-> This IS unusually wide. A meaningful range of different raw model outputs "
                  "collapse to the same calibrated probability here.")
        else:
            print("-> Not unusually wide - normal isotonic step size for this calibration sample.")
    else:
        print(f"\nNo step found near {target} - the flagged value may be specific to a different "
              f"model run than the one this calibrator file represents. Re-check which "
              f"calibrator.joblib corresponds to the trades file you're comparing against.")

    print("\n" + "=" * 70)
    print("PART 2 - Your actual trades: raw probability spread within each")
    print("         calibrated-probability bucket")
    print("=" * 70)
    if "calibrated_prob" not in trades.columns:
        print("Trades file has no 'calibrated_prob' column - cannot complete Part 2.")
        return

    trades = trades.copy()
    trades["calibrated_prob_rounded"] = trades["calibrated_prob"].round(4)
    grouped = trades.groupby("calibrated_prob_rounded").size().sort_values(ascending=False)
    print("\nMost common exact calibrated_prob values among your actual trades:")
    print(grouped.head(10))

    top_value = grouped.index[0]
    top_count = grouped.iloc[0]
    print(f"\nMost common single value: {top_value} appears in {top_count}/{len(trades)} "
          f"trades ({top_count/len(trades):.1%})")
    if top_count / len(trades) > 0.15:
        print("-> A meaningfully large share of your trades share the EXACT SAME calibrated")
        print("   probability. Combined with Part 1, this confirms calibration resolution")
        print("   is coarse in the region your trades actually land in - the model may be")
        print("   distinguishing confidence levels less finely than the raw score suggests.")
    else:
        print("-> Not concentrated enough to indicate a real resolution problem.")

    steps_df.to_csv(out / "calibration_steps.csv", index=False)
    print(f"\nSaved to: {out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibration resolution diagnostic")
    parser.add_argument("--calibrator", required=True)
    parser.add_argument("--raw-probs-source", required=True, dest="trades_path",
                         help="CSV with a calibrated_prob column (e.g. backtest_trades.csv)")
    parser.add_argument("--out", default="./calibration_resolution")
    args = parser.parse_args()

    run(args.calibrator, args.trades_path, args.out)
