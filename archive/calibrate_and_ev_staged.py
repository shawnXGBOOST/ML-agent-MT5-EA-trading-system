"""
Step 4: Calibration + EV Engine for the Staged Exit Scheme
================================================================

Adapted from calibrate_and_ev.py. The original script assumed a FIXED
reward of +2R for every win - correct for the old binary TP/SL labeling,
but WRONG here: under the staged exit, a "win" (y_binary=1) is either a
LOCKED_STOP (~0.8R) or a FULL_TP (2.0R), and LOCKED_STOP is the large
majority of wins. Using a fixed +2R would badly overstate EV.

This version computes the EMPIRICAL average realized_r separately for
winners and losers, from each fold's TRAINING data only (same purging
discipline as everywhere else), instead of a fixed assumption.

Usage:
    python calibrate_and_ev_staged.py --dataset phase1_output/features_staged/M15_feature_dataset.csv \\
        --out phase1_output/ev_staged --n-folds 5 --embargo-bars 48 \\
        --spread-price 0.30 --commission-price 0.05 --min-ev 0.05
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

import xgboost as xgb

from train_baselines import FEATURE_COLS, purged_walk_forward_splits
from calibrate_and_ev import expected_calibration_error, purged_train_calib_split


def run(dataset_path, out_dir, n_folds, embargo_bars, spread_price, commission_price, min_ev):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path, parse_dates=["event_time", "entry_time", "touch_time"])

    if "realized_r" not in df.columns:
        raise ValueError("This dataset doesn't have 'realized_r' - are you sure this is a "
                          "STAGED-exit feature dataset (from triple_barrier_staged.py + "
                          "features.py), not the original fixed-TP dataset?")

    print(f"  {len(df)} labeled events")
    print(f"  Base rate (y_binary=1): {df['y_binary'].mean():.1%}")

    all_results = []
    fold_num = 0

    for train_idx, test_idx in purged_walk_forward_splits(df, n_folds, embargo_bars):
        fold_num += 1
        train_df = df.loc[train_idx]
        test_df = df.loc[test_idx].copy()

        fit_df, calib_df = purged_train_calib_split(train_df, calib_frac=0.2, embargo_bars=embargo_bars)
        if len(fit_df) < 50 or len(calib_df) < 30:
            print(f"Fold {fold_num}: skipped (not enough data for fit/calibration split)")
            continue

        win_r = fit_df.loc[fit_df["y_binary"] == 1, "realized_r"].mean()
        loss_r = fit_df.loc[fit_df["y_binary"] == 0, "realized_r"].mean()

        model = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=42, n_jobs=-1,
        )
        model.fit(fit_df[FEATURE_COLS], fit_df["y_binary"])

        raw_calib_prob = model.predict_proba(calib_df[FEATURE_COLS])[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_calib_prob, calib_df["y_binary"].values)

        raw_test_prob = model.predict_proba(test_df[FEATURE_COLS])[:, 1]
        calibrated_test_prob = calibrator.predict(raw_test_prob)

        y_test = test_df["y_binary"].values
        brier_before = brier_score_loss(y_test, raw_test_prob)
        brier_after = brier_score_loss(y_test, calibrated_test_prob)
        ece_before = expected_calibration_error(y_test, raw_test_prob)
        ece_after = expected_calibration_error(y_test, calibrated_test_prob)

        test_df["raw_prob"] = raw_test_prob
        test_df["calibrated_prob"] = calibrated_test_prob
        test_df["fold_win_r"] = win_r
        test_df["fold_loss_r"] = loss_r

        sl_atr_mult = 1.0
        sl_distance_price = sl_atr_mult * test_df["atr_at_event"]
        test_df["cost_r"] = (spread_price + commission_price) / sl_distance_price

        test_df["ev_r"] = (
            test_df["calibrated_prob"] * win_r
            + (1 - test_df["calibrated_prob"]) * loss_r
            - test_df["cost_r"]
        )
        test_df["fold"] = fold_num
        all_results.append(test_df)

        print(f"\nFold {fold_num}: n={len(test_df)}  "
              f"AUC(raw)={roc_auc_score(y_test, raw_test_prob):.3f}  "
              f"Brier {brier_before:.3f} -> {brier_after:.3f}  "
              f"ECE {ece_before:.3f} -> {ece_after:.3f}")
        print(f"  Empirical win_r={win_r:+.3f}  loss_r={loss_r:+.3f}  "
              f"(from this fold's TRAINING data)")

    full = pd.concat(all_results, ignore_index=True)

    full["decision"] = np.where(full["ev_r"] >= min_ev, "TAKE", "SKIP")
    taken = full[full["decision"] == "TAKE"]
    skipped = full[full["decision"] == "SKIP"]

    print("\n" + "=" * 60)
    print("EV FILTER RESULTS (empirical reward/risk, cost-inclusive)")
    print("=" * 60)
    print(f"Total events: {len(full)}")
    print(f"  TAKE: {len(taken)} ({len(taken)/len(full):.1%})  "
          f"actual win rate: {taken['y_binary'].mean():.1%}  "
          f"avg realized_r: {taken['realized_r'].mean():+.3f}")
    print(f"  SKIP: {len(skipped)} ({len(skipped)/len(full):.1%})  "
          f"actual win rate: {skipped['y_binary'].mean():.1%}  "
          f"avg realized_r: {skipped['realized_r'].mean():+.3f}")

    print(f"\nExpectancy - ALL events (actual realized_r): {full['realized_r'].mean():+.3f}R per trade")
    print(f"Expectancy - TAKE-only (actual realized_r): {taken['realized_r'].mean():+.3f}R per trade")
    print(f"(cost-exclusive above - subtract avg cost_r ({full['cost_r'].mean():.3f}R): "
          f"{taken['realized_r'].mean() - taken['cost_r'].mean():+.3f}R net)")

    out_path = out / "M15_calibrated_ev_dataset_staged.csv"
    full.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path.resolve()}")
    print("\nNext: regime.py (unchanged) on this file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 4: calibration + EV for staged exit scheme")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", default="./phase1_output/ev_staged")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--embargo-bars", type=int, default=48)
    parser.add_argument("--spread-price", type=float, default=0.30)
    parser.add_argument("--commission-price", type=float, default=0.05)
    parser.add_argument("--min-ev", type=float, default=0.05)
    args = parser.parse_args()

    run(args.dataset, args.out, args.n_folds, args.embargo_bars,
        args.spread_price, args.commission_price, args.min_ev)
