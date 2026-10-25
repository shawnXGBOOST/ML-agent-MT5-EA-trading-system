"""
Calibration + Expected-Value Engine
=====================================

Two variants in one module:

``run_fixed`` (original)
    Assumes a fixed +2R reward for every win.  Used during early development.
    Kept for reference; the live pipeline uses ``run_staged``.

``run_staged`` (production)
    Computes *empirical* average realized_r for winners/losers from each
    fold's training data, which is correct for the staged-exit scheme where
    a win can be either a LOCKED_STOP (~0.8R) or a FULL_TP (2.0R).

Shared helpers
    ``expected_calibration_error``, ``purged_train_calib_split``

Usage (staged — what the live pipeline calls):
    python -m mltrader.calibration.calibration \\
        --dataset phase1_output/features_staged/M15_feature_dataset.csv \\
        --out phase1_output/ev_staged --n-folds 5 --embargo-bars 48 \\
        --spread-price 0.30 --commission-price 0.05 --min-ev 0.05
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from mltrader.models.train_baselines import FEATURE_COLS, purged_walk_forward_splits

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(y_prob[mask].mean() - y_true[mask].mean())
    return ece


def purged_train_calib_split(train_df: pd.DataFrame, calib_frac: float, embargo_bars: int):
    """Carve a calibration slice out of a fold's training data with the same
    purge+embargo logic as the outer split so the calibrator is never fit on
    information that overlaps its held-out labels."""
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


# ---------------------------------------------------------------------------
# Fixed-reward variant (original, kept for reference)
# ---------------------------------------------------------------------------

def run_fixed(
    dataset_path: str,
    out_dir: str,
    n_folds: int,
    embargo_bars: int,
    spread_price: float,
    commission_price: float,
    min_ev: float,
):
    """Original calibration + EV with a fixed +2R reward assumption."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path, parse_dates=["event_time", "entry_time", "touch_time"])

    reward_R = 2.0
    risk_R = 1.0
    all_results = []
    fold_num = 0

    for train_idx, test_idx in purged_walk_forward_splits(df, n_folds, embargo_bars):
        fold_num += 1
        train_df = df.loc[train_idx]
        test_df = df.loc[test_idx].copy()

        fit_df, calib_df = purged_train_calib_split(train_df, calib_frac=0.2, embargo_bars=embargo_bars)
        if len(fit_df) < 50 or len(calib_df) < 30:
            print(f"Fold {fold_num}: skipped (not enough data)")
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

        sl_distance_price = 1.0 * test_df["atr_at_event"]
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
    full["decision"] = np.where(full["ev_r"] >= min_ev, "TAKE", "SKIP")
    taken = full[full["decision"] == "TAKE"]

    print(f"\nTAKE: {len(taken)} ({len(taken)/len(full):.1%})  "
          f"actual TP rate: {taken['y_binary'].mean():.1%}")

    out_path = out / "M15_calibrated_ev_dataset.csv"
    full.to_csv(out_path, index=False)
    print(f"Saved to: {out_path.resolve()}")


# ---------------------------------------------------------------------------
# Staged-exit variant (production)
# ---------------------------------------------------------------------------

def run_staged(
    dataset_path: str,
    out_dir: str,
    n_folds: int,
    embargo_bars: int,
    spread_price: float,
    commission_price: float,
    min_ev: float,
):
    """Calibration + EV using empirical win/loss R from each fold's training
    data.  Correct for the staged-exit scheme (LOCKED_STOP ~0.8R or
    FULL_TP 2.0R rather than a fixed +2R)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path, parse_dates=["event_time", "entry_time", "touch_time"])

    if "realized_r" not in df.columns:
        raise ValueError(
            "Column 'realized_r' not found.  Pass a staged-exit feature dataset "
            "(produced by triple_barrier.py + features.py with --activation-fraction)."
        )

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
            print(f"Fold {fold_num}: skipped (not enough data)")
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

        sl_distance_price = 1.0 * test_df["atr_at_event"]
        test_df["cost_r"] = (spread_price + commission_price) / sl_distance_price
        test_df["ev_r"] = (
            test_df["calibrated_prob"] * win_r
            + (1 - test_df["calibrated_prob"]) * loss_r
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

    out_path = out / "M15_calibrated_ev_dataset_staged.csv"
    full.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path.resolve()}")


# ---------------------------------------------------------------------------
# CLI entry point (staged by default)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibration + EV engine (staged exit)")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", default="./phase1_output/ev_staged")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--embargo-bars", type=int, default=48)
    parser.add_argument("--spread-price", type=float, default=0.30)
    parser.add_argument("--commission-price", type=float, default=0.05)
    parser.add_argument("--min-ev", type=float, default=0.05)
    parser.add_argument("--variant", choices=["staged", "fixed"], default="staged")
    args = parser.parse_args()

    fn = run_staged if args.variant == "staged" else run_fixed
    fn(args.dataset, args.out, args.n_folds, args.embargo_bars,
       args.spread_price, args.commission_price, args.min_ev)
