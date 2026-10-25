"""
XAUUSD ML+Agent+EA — Phase 7c: Train + Persist Final Production Model
=========================================================================

Everything so far retrained a fresh model every script run — fine for
research/validation, not workable for a live loop running every 15 minutes.

This trains ONE final model + calibrator on ALL your available historical
data (now that Phase 5 already validated the approach on an untouched
holdout) and saves them to disk. The live runner loads these instead of
retraining.

Usage:
    python train_final_model.py --dataset phase1_output/features/M15_feature_dataset.csv \
        --out phase1_output/production_model --embargo-bars 48
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from mltrader.calibration.calibration import purged_train_calib_split
from mltrader.calibration.calibrators import PlattCalibrator
from mltrader.models.train_baselines import FEATURE_COLS


def run(dataset_path: str, out_dir: str, embargo_bars: int, exclude_recent_months: int = 0,
        start_date: str = None, calibration_method: str = "isotonic",
        activation_fraction: float = 0.4, lock_fraction: float = 1.0):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading full dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path, parse_dates=["event_time", "entry_time", "touch_time"])
    df = df.sort_values("entry_time").reset_index(drop=True)
    print(f"  {len(df)} events, {df['entry_time'].min()} -> {df['entry_time'].max()}")

    if start_date:
        start_dt = pd.Timestamp(start_date)
        n_before = len(df)
        df = df[df["entry_time"] >= start_dt].reset_index(drop=True)
        print(f"  Restricting training window to start at {start_date}: {n_before} -> {len(df)} events "
              f"(for temporal model-age comparison — train on different historical slices, "
              f"test all variants against the SAME untouched OOS window)")

    if exclude_recent_months > 0:
        cutoff = df["entry_time"].max() - pd.DateOffset(months=exclude_recent_months)
        n_before = len(df)
        df = df[df["entry_time"] < cutoff].reset_index(drop=True)
        print(f"  EXCLUDING most recent {exclude_recent_months} months (cutoff={cutoff}) for a genuine "
              f"out-of-sample test: {n_before} -> {len(df)} events retained for training.")
        print(f"  This model should ONLY be backtested against the excluded {exclude_recent_months}-month "
              f"window — using it on data it was trained on defeats the purpose of this flag.")

    fit_df, calib_df = purged_train_calib_split(df, calib_frac=0.15, embargo_bars=embargo_bars)
    print(f"Fit set: {len(fit_df)}  |  Calibration set: {len(calib_df)}")

    print("\nTraining final XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        random_state=42, n_jobs=-1,
    )
    model.fit(fit_df[FEATURE_COLS], fit_df["y_binary"])

    raw_calib_prob = model.predict_proba(calib_df[FEATURE_COLS])[:, 1]
    if calibration_method == "platt":
        calibrator = PlattCalibrator()
    else:
        calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_calib_prob, calib_df["y_binary"].values)
    print(f"Calibration method: {calibration_method}")

    # Sanity-check on the calibration set itself (not a real OOS test — Phase 5
    # already did that — this is just confirming the saved artifacts work)
    calibrated_calib_prob = calibrator.predict(raw_calib_prob)
    auc = roc_auc_score(calib_df["y_binary"], raw_calib_prob)
    brier = brier_score_loss(calib_df["y_binary"], calibrated_calib_prob)
    print(f"\nSanity check on calibration slice: AUC={auc:.3f}, Brier={brier:.3f}")
    print("(This is expected to look good since it's near-training data — "
          "the real validation was Phase 5's untouched holdout, already done.)")

    if calibration_method == "platt":
        grid = np.linspace(raw_calib_prob.min(), raw_calib_prob.max(), 500)
        grid_out = calibrator.predict(grid)
        is_monotonic = np.all(np.diff(grid_out) >= -1e-9)
        print(f"Platt calibration monotonicity check: {'PASS' if is_monotonic else 'FAIL'}")
        if not is_monotonic:
            print("  WARNING: the fitted sigmoid is NOT monotonic across the observed raw-score "
                  "range — this usually means the calibration slice is too small or too noisy for "
                  "even a simple logistic fit to find a stable direction. Consider a larger "
                  "calibration set, or check whether the underlying model has real signal here "
                  "before trusting this calibrator.")

    model_path = out / "xgb_model.joblib"
    calibrator_path = out / "calibrator.joblib"
    joblib.dump(model, model_path)
    joblib.dump(calibrator, calibrator_path)

    # Detect staged-exit scheme (has 'realized_r') vs original fixed TP/SL scheme
    # and save the correct reward/risk basis for downstream EV calculations —
    # a fixed +2R/-1R assumption is WRONG for the staged scheme (see
    # calibrate_and_ev_staged.py for the full explanation).
    is_staged = "realized_r" in fit_df.columns
    if is_staged:
        empirical_win_r = float(fit_df.loc[fit_df["y_binary"] == 1, "realized_r"].mean())
        empirical_loss_r = float(fit_df.loc[fit_df["y_binary"] == 0, "realized_r"].mean())
        print(f"Staged-exit scheme detected. Empirical win_r={empirical_win_r:+.3f}, "
              f"loss_r={empirical_loss_r:+.3f} (saved to model_meta.json for live use).")
    else:
        empirical_win_r, empirical_loss_r = 2.0, -1.0  # original fixed-TP scheme

    meta = {
        "feature_cols": FEATURE_COLS,
        "trained_on_events": len(fit_df),
        "calibrated_on_events": len(calib_df),
        "data_range_start": str(df["entry_time"].min()),
        "data_range_end": str(df["entry_time"].max()),
        "tp_atr_mult": 2.0,
        "sl_atr_mult": 1.0,
        "max_bars_time_barrier": 48,
        "excluded_recent_months": exclude_recent_months,
        "calibration_method": calibration_method,
        "is_staged_exit": is_staged,
        "empirical_win_r": empirical_win_r,
        "empirical_loss_r": empirical_loss_r,
        "activation_fraction": activation_fraction if is_staged else None,
        "lock_fraction": lock_fraction if is_staged else None,
    }
    with open(out / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\nSaved:")
    print(f"  {model_path.resolve()}")
    print(f"  {calibrator_path.resolve()}")
    print(f"  {(out / 'model_meta.json').resolve()}")
    print("\nThe live runner will load these three files — don't move/rename them "
          "without updating live_runner.py's --model-dir argument.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 7c: train + persist final model")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", default="./phase1_output/production_model")
    parser.add_argument("--embargo-bars", type=int, default=48)
    parser.add_argument("--exclude-recent-months", type=int, default=0,
                         help="Excludes the most recent N months from training, for a genuine "
                              "out-of-sample test. The resulting model should ONLY be backtested "
                              "against that excluded window.")
    parser.add_argument("--start-date", type=str, default=None,
                         help="Restrict training to events on/after this date (YYYY-MM-DD). "
                              "Use with --exclude-recent-months to build temporal model-age "
                              "comparisons — e.g. --start-date 2022-01-01 trains only on 2022+ data.")
    parser.add_argument("--calibration-method", choices=["isotonic", "platt"], default="isotonic",
                         help="isotonic: more flexible, needs more calibration data, can produce "
                              "flat steps with small samples. platt: smooth sigmoid, cannot produce "
                              "flat steps, more robust with smaller calibration sets. Use 'platt' "
                              "if calibration_resolution_check.py found a wide flat step.")
    parser.add_argument("--activation-fraction", type=float, default=0.4,
                         help="Only recorded in meta.json for staged-exit datasets (has 'realized_r') "
                              "— must match what triple_barrier_staged.py actually used to generate "
                              "the labels this model was trained on. Validated value: 0.4.")
    parser.add_argument("--lock-fraction", type=float, default=1.0,
                         help="Same as above — must match the labeling run. Validated value: 1.0.")
    args = parser.parse_args()

    run(args.dataset, args.out, args.embargo_bars, args.exclude_recent_months, args.start_date,
        args.calibration_method, args.activation_fraction, args.lock_fraction)
