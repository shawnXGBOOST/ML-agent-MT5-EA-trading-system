"""
Step 5: Final Untouched Holdout for the Staged Exit Scheme
================================================================

Same discipline as final_holdout_eval.py (reserve the most recent N months,
train only on everything before, evaluate ONCE on the untouched period) -
but uses EMPIRICAL win/loss reward from the pre-cutoff training pool instead
of the old fixed +2R/-1R assumption.

Run this ONLY after you're done tuning on the pre-cutoff data. This number
gets seen once - if it disappoints, don't re-tune and re-run.

Usage:
    python final_holdout_eval_staged.py --dataset phase1_output/features_staged/M15_feature_dataset.csv \\
        --regime-events phase1_output/events/M15_events.csv \\
        --out phase1_output/holdout_staged --holdout-months 6 \\
        --spread-price 0.30 --commission-price 0.05 --min-ev 0.05 --embargo-bars 48
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

import xgboost as xgb

from train_baselines import FEATURE_COLS
from calibrate_and_ev import expected_calibration_error, purged_train_calib_split
from regime import compute_adx, bucket_trend, bucket_volatility


def run(dataset_path, regime_events_path, out_dir, holdout_months, embargo_bars,
        spread_price, commission_price, min_ev):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_path, parse_dates=["event_time", "entry_time", "touch_time"])
    df = df.sort_values("entry_time").reset_index(drop=True)

    if "realized_r" not in df.columns:
        raise ValueError("This dataset doesn't have 'realized_r' - are you sure this is a "
                          "STAGED-exit feature dataset, not the original fixed-TP dataset?")

    cutoff = df["entry_time"].max() - pd.DateOffset(months=holdout_months)
    print(f"Data range: {df['entry_time'].min()} -> {df['entry_time'].max()}")
    print(f"Holdout cutoff: {cutoff}  (most recent {holdout_months} months reserved, untouched)")

    train_pool = df[df["entry_time"] < cutoff]
    holdout = df[df["entry_time"] >= cutoff].copy()

    embargo_delta = pd.Timedelta(minutes=15) * embargo_bars
    train_pool = train_pool[train_pool["touch_time"] < (cutoff - embargo_delta)]

    print(f"Train pool: {len(train_pool)} events  |  Holdout: {len(holdout)} events")
    if len(holdout) < 100:
        print("WARNING: holdout set is small - consider a longer --holdout-months window.")

    fit_df, calib_df = purged_train_calib_split(train_pool, calib_frac=0.15, embargo_bars=embargo_bars)
    print(f"Fit set: {len(fit_df)}  |  Calibration set: {len(calib_df)}")

    win_r = fit_df.loc[fit_df["y_binary"] == 1, "realized_r"].mean()
    loss_r = fit_df.loc[fit_df["y_binary"] == 0, "realized_r"].mean()
    print(f"Empirical win_r={win_r:+.3f}  loss_r={loss_r:+.3f}  (from pre-cutoff training data only)")

    print("\nTraining final model on all pre-cutoff data...")
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        random_state=42, n_jobs=-1,
    )
    model.fit(fit_df[FEATURE_COLS], fit_df["y_binary"])

    raw_calib_prob = model.predict_proba(calib_df[FEATURE_COLS])[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_calib_prob, calib_df["y_binary"].values)

    print("Evaluating ONCE on the untouched holdout...\n")
    raw_holdout_prob = model.predict_proba(holdout[FEATURE_COLS])[:, 1]
    calibrated_holdout_prob = calibrator.predict(raw_holdout_prob)
    y_holdout = holdout["y_binary"].values

    auc = roc_auc_score(y_holdout, raw_holdout_prob)
    brier = brier_score_loss(y_holdout, calibrated_holdout_prob)
    ece = expected_calibration_error(y_holdout, calibrated_holdout_prob)

    print("=" * 60)
    print("FINAL HOLDOUT RESULT (seen once - do not re-tune against this)")
    print("=" * 60)
    print(f"AUC: {auc:.3f}")
    print(f"Calibrated Brier: {brier:.3f}")
    print(f"Calibrated ECE: {ece:.3f}")

    holdout["calibrated_prob"] = calibrated_holdout_prob
    sl_atr_mult = 1.0
    holdout["cost_r"] = (spread_price + commission_price) / (sl_atr_mult * holdout["atr_at_event"])
    holdout["ev_r"] = (holdout["calibrated_prob"] * win_r
                       + (1 - holdout["calibrated_prob"]) * loss_r
                       - holdout["cost_r"])
    holdout["decision"] = np.where(holdout["ev_r"] >= min_ev, "TAKE", "SKIP")

    taken = holdout[holdout["decision"] == "TAKE"]
    print(f"\nTAKE: {len(taken)}/{len(holdout)} ({len(taken)/len(holdout):.1%})")
    if len(taken) > 0:
        tp_rate = taken["y_binary"].mean()
        expectancy_excl = taken["realized_r"].mean()
        expectancy_incl = expectancy_excl - taken["cost_r"].mean()
        print(f"TAKE actual win rate (y_binary=1): {tp_rate:.1%}")
        print(f"TAKE expectancy (actual realized_r, excl cost): {expectancy_excl:+.3f}R")
        print(f"TAKE expectancy (incl cost): {expectancy_incl:+.3f}R  <-- the real number")

    print("\nRegime breakdown on holdout:")
    m15 = pd.read_csv(regime_events_path, parse_dates=["datetime"])
    m15["adx_14"] = compute_adx(m15)
    m15["trend_regime"] = bucket_trend(m15["adx_14"])
    m15["vol_regime"] = bucket_volatility(m15["atr_percentile"])
    m15 = m15.reset_index().rename(columns={"index": "event_idx"})
    regime_lookup = m15[["event_idx", "trend_regime", "vol_regime"]]
    holdout_r = holdout.merge(regime_lookup, on="event_idx", how="left")
    holdout_r["regime"] = holdout_r["trend_regime"].astype(str) + "_" + holdout_r["vol_regime"].astype(str)

    taken_r = holdout_r[holdout_r["decision"] == "TAKE"]
    for regime, g in taken_r.groupby("regime"):
        if len(g) < 15:
            continue
        tp = g["y_binary"].mean()
        ev_incl = g["realized_r"].mean() - g["cost_r"].mean()
        print(f"  {regime:30s} n={len(g):4d}  TP%={tp:.1%}  EV(incl cost)={ev_incl:+.3f}R")

    out_path = out / "final_holdout_results_staged.csv"
    holdout_r.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path.resolve()}")

    print("\n" + "=" * 60)
    print("VERDICT GUIDE")
    print("=" * 60)
    print("Compare this holdout expectancy against the walk-forward CV result from Step 4. "
          "Close agreement, same sign -> the edge generalizes, proceed with reasonable "
          "confidence. Much worse or negative -> the CV result was optimistic, do not "
          "deploy this as-is.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 5: final holdout for staged exit scheme")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--regime-events", required=True)
    parser.add_argument("--out", default="./phase1_output/holdout_staged")
    parser.add_argument("--holdout-months", type=int, default=6)
    parser.add_argument("--embargo-bars", type=int, default=48)
    parser.add_argument("--spread-price", type=float, default=0.30)
    parser.add_argument("--commission-price", type=float, default=0.05)
    parser.add_argument("--min-ev", type=float, default=0.05)
    args = parser.parse_args()

    run(args.dataset, args.regime_events, args.out, args.holdout_months, args.embargo_bars,
        args.spread_price, args.commission_price, args.min_ev)
