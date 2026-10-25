"""
XAUUSD ML+Agent+EA — Phase 4a: Calibration + Expected-Value Engine
=====================================================================

Two things happen here:

1. CALIBRATION: Raw XGBoost probabilities are often overconfident/miscalibrated
   (a "0.70" prediction doesn't necessarily win 70% of the time). We fit an
   isotonic regression calibrator per fold, using a purged/embargoed slice
   carved out of each fold's training data (never touching the test fold),
   then check calibration quality (Brier score, ECE) before vs after.

2. EXPECTED VALUE: Converts calibrated P(TP) into an expected-value estimate
   in R-multiples (R = the SL distance, i.e. your risk unit), including a
   cost adjustment for spread + commission. Only trades clearing a minimum
   EV threshold get a "take" decision — this is the actual meta-labeling
   filter your agent will use in Phase 6.

   EV_R = P(TP) * reward_R  -  P(not-TP) * risk_R  -  cost_R

   where reward_R = tp_atr_mult / sl_atr_mult (2.0 with your Phase 2 settings),
   risk_R = 1.0 by definition, and cost_R = (spread + commission) / SL distance.

   NOTE: P(not-TP) lumps SL and TIMEOUT together as risk_R loss. TIMEOUT is
   only ~1.3% of events and its realized outcome varies (not always a full
   loss), so this slightly *understates* true EV — a deliberately conservative
   simplification rather than an optimistic one.

Usage:
    python calibrate_and_ev.py --dataset phase1_output/features/M15_feature_dataset.csv \
        --out phase1_output/ev --n-folds 5 --embargo-bars 48 \
        --spread-price 0.30 --commission-price 0.05 --min-ev 0.05
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

import xgboost as xgb

from train_baselines import FEATURE_COLS, purged_walk_forward_splits


def expected_calibration_error(y_true, y_prob, n_bins=10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(y_prob, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        bin_conf = y_prob[mask].mean()
        bin_acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return ece


def purged_train_calib_split(train_df: pd.DataFrame, calib_frac: float, embargo_bars: int):
    """Carve a calibration slice out of a fold's training data, using the same
    purge+embargo logic as the outer split so the calibrator is never fit on
    information that overlaps its own held-out labels."""
    train_df = train_df.sort_values("entry_time").reset_index(drop=True)
    n = len(train_df)
    split_idx = int(n * (1 - calib_frac))

    calib_idx = train_df.index[split_idx:]
    calib_start_time = train_df.loc[calib_idx, "entry_time"].min()

    candidate_fit_idx = train_df.index[:split_idx]
    overlaps = train_df.loc[candidate_fit_idx, "touch_time"] >= calib_start_time
    purged_fit_idx = candidate_fit_idx[~overlaps]

    bar_duration = pd.Timedelta(minutes=15)
    embargo_cutoff = calib_start_time - bar_duration * embargo_bars
    fit_idx = purged_fit_idx[train_df.loc[purged_fit_idx, "entry_time"] < embargo_cutoff]

    return train_df.loc[fit_idx], train_df.loc[calib_idx]


def run(
    dataset_path: str,
    out_dir: str,
    n_folds: int,
    embargo_bars: int,
    spread_price: float,
    commission_price: float,
    min_ev: float,
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path, parse_dates=["event_time", "entry_time", "touch_time"])

    reward_R = 2.0  # matches --tp-atr 2.0 / --sl-atr 1.0 used in Phase 2
    risk_R = 1.0

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

        test_df["raw_prob"] = raw_test_prob
        test_df["calibrated_prob"] = calibrated_test_prob

        # Real per-trade cost in R-multiples: SL distance = sl_atr_mult * ATR (price units).
        # sl_atr_mult=1.0 matches the Phase 2 default you confirmed (TP=2xATR, SL=1xATR).
        sl_atr_mult = 1.0
        sl_distance_price = sl_atr_mult * test_df["atr_at_event"]
        test_df["cost_r"] = (spread_price + commission_price) / sl_distance_price

        test_df["ev_r"] = (
            test_df["calibrated_prob"] * reward_R
            - (1 - test_df["calibrated_prob"]) * risk_R
            - test_df["cost_r"]
        )
        test_df["fold"] = fold_num
        all_results.append(test_df)

        print(
            f"\nFold {fold_num}: n={len(test_df)}  "
            f"AUC(raw)={roc_auc_score(y_test, raw_test_prob):.3f}  "
            f"Brier {brier_before:.3f} -> {brier_after:.3f}  "
            f"ECE {ece_before:.3f} -> {ece_after:.3f}"
        )

    full = pd.concat(all_results, ignore_index=True)

    print(f"\nCost model: spread={spread_price} + commission={commission_price} price-units per trade, "
          f"converted to R-multiples per-trade using each event's actual ATR at entry "
          f"(SL distance = 1.0 x ATR). EV below is cost-INCLUSIVE.")

    full["decision"] = np.where(full["ev_r"] >= min_ev, "TAKE", "SKIP")

    taken = full[full["decision"] == "TAKE"]
    skipped = full[full["decision"] == "SKIP"]

    print("\n" + "=" * 60)
    print("EV FILTER RESULTS (cost-inclusive)")
    print("=" * 60)
    print(f"Total events: {len(full)}")
    print(f"  TAKE: {len(taken)} ({len(taken)/len(full):.1%})  "
          f"actual TP rate: {taken['y_binary'].mean():.1%}  "
          f"avg calibrated_prob: {taken['calibrated_prob'].mean():.3f}")
    print(f"  SKIP: {len(skipped)} ({len(skipped)/len(full):.1%})  "
          f"actual TP rate: {skipped['y_binary'].mean():.1%}  "
          f"avg calibrated_prob: {skipped['calibrated_prob'].mean():.3f}")

    baseline_expectancy_R = full["y_binary"].mean() * reward_R - (1 - full["y_binary"].mean()) * risk_R
    if len(taken) > 0:
        taken_expectancy_R = taken["y_binary"].mean() * reward_R - (1 - taken["y_binary"].mean()) * risk_R
        avg_cost_r = full["cost_r"].mean()
        print(f"\nExpectancy — ALL events (excl. costs): {baseline_expectancy_R:+.3f}R per trade")
        print(f"Expectancy — TAKE-only (excl. costs):  {taken_expectancy_R:+.3f}R per trade")
        print(f"Avg cost per trade: {avg_cost_r:.3f}R  ->  "
              f"TAKE-only net of costs: {taken_expectancy_R - avg_cost_r:+.3f}R per trade")

    out_path = out / "M15_calibrated_ev_dataset.csv"
    full.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path.resolve()}")
    print("\nNext: regime detection — check whether this edge concentrates in "
          "specific volatility/trend regimes, which feeds directly into the agent's "
          "strategy-switching logic in Phase 6.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4a: calibration + EV engine")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", default="./phase1_output/ev")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--embargo-bars", type=int, default=48)
    parser.add_argument("--spread-price", type=float, default=0.30,
                         help="Typical spread in price units (e.g. 0.30 = $0.30 on gold)")
    parser.add_argument("--commission-price", type=float, default=0.05,
                         help="Round-turn commission converted to price-unit equivalent")
    parser.add_argument("--min-ev", type=float, default=0.05,
                         help="Minimum EV (in R-multiples) to mark a trade as TAKE")
    args = parser.parse_args()

    run(
        args.dataset, args.out, args.n_folds, args.embargo_bars,
        args.spread_price, args.commission_price, args.min_ev,
    )
