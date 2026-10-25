"""
XAUUSD ML+Agent+EA — Phase 3b: Baseline ML Models with Purged CV
===================================================================

Trains Logistic Regression -> Random Forest -> XGBoost -> LightGBM on the
feature dataset from Phase 3a, predicting y_binary (TP=1 vs SL/TIMEOUT=0).

Critically, this uses PURGED + EMBARGOED walk-forward validation, not plain
k-fold. Why this matters: each label's outcome depends on a window of future
bars (entry_time -> touch_time). If we used random k-fold, some training
samples' label windows would overlap the test fold's time range, leaking
future information into training. Purging removes those overlapping samples;
the embargo adds a buffer after each test fold before training resumes.

This mirrors the blueprint's validation hierarchy:
  purged/embargoed splits -> (CPCV later) -> walk-forward -> untouched OOS -> paper trading

Usage:
    python train_baselines.py --dataset phase1_output/features/M15_feature_dataset.csv \
        --out phase1_output/models --n-folds 5 --embargo-bars 48
"""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "return_1", "return_4", "return_16", "return_48",
    "candle_range", "candle_body_ratio",
    "rsi_14", "roc_10", "ma_slope_20", "realized_vol_20",
    "atr_percentile", "dist_to_last_swing_high_atr", "dist_to_last_swing_low_atr",
    "dist_to_prev_day_high_atr", "dist_to_prev_day_low_atr",
    "bars_since_session_london_open", "bars_since_session_ny_open",
    "session_london", "session_ny", "session_london_ny_overlap", "session_asia",
    "h1_bias", "h1_atr_percentile", "h4_bias", "h4_atr_percentile",
    "h1_aligned_with_m15", "h4_aligned_with_m15",
]


# ---------------------------------------------------------------------------
# Purged + embargoed walk-forward split
# ---------------------------------------------------------------------------

def purged_walk_forward_splits(df: pd.DataFrame, n_folds: int, embargo_bars: int):
    """
    Yields (train_idx, test_idx) pairs. Each fold's test set is a contiguous,
    chronologically later block. Training uses ONLY prior data (walk-forward,
    not k-fold shuffling), with:
      - purging: any training sample whose [entry_time, touch_time] window
        overlaps the test fold's time range is removed
      - embargo: an additional buffer of `embargo_bars` M15 bars is excluded
        from training immediately before the test fold starts, since those
        samples' outcomes may still be resolving right as the test period begins
    """
    df = df.sort_values("entry_time").reset_index(drop=True)
    n = len(df)
    fold_edges = np.linspace(0, n, n_folds + 1).astype(int)

    bar_duration = pd.Timedelta(minutes=15)
    embargo_delta = bar_duration * embargo_bars

    for f in range(1, n_folds):  # first fold has no prior data to train on
        test_start_idx = fold_edges[f]
        test_end_idx = fold_edges[f + 1] if f + 1 < len(fold_edges) else n

        test_idx = df.index[test_start_idx:test_end_idx]
        test_start_time = df.loc[test_idx, "entry_time"].min()
        test_end_time = df.loc[test_idx, "touch_time"].max()

        candidate_train_idx = df.index[:test_start_idx]

        # Purge: drop training rows whose label window overlaps the test window
        overlaps = (
            (df.loc[candidate_train_idx, "touch_time"] >= test_start_time)
        )
        purged_train_idx = candidate_train_idx[~overlaps]

        # Embargo: also drop training rows whose entry falls within embargo_bars
        # of the test start (extra safety buffer)
        embargo_cutoff = test_start_time - embargo_delta
        embargoed_train_idx = purged_train_idx[df.loc[purged_train_idx, "entry_time"] < embargo_cutoff]

        if len(embargoed_train_idx) < 50 or len(test_idx) < 20:
            continue  # not enough data in this fold to be meaningful

        yield embargoed_train_idx, test_idx


# ---------------------------------------------------------------------------
# Model training + evaluation
# ---------------------------------------------------------------------------

def get_models():
    return {
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=20,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=42, n_jobs=-1,
        ),
        "lightgbm": lgb.LGBMClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            n_jobs=-1, verbose=-1,
        ),
    }


def evaluate_predictions(y_true, y_prob) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else np.nan,
        "pr_auc": average_precision_score(y_true, y_prob) if len(set(y_true)) > 1 else np.nan,
        "log_loss": log_loss(y_true, y_prob, labels=[0, 1]),
        "brier": brier_score_loss(y_true, y_prob),
        "accuracy": (y_pred == y_true).mean(),
        "base_rate": y_true.mean(),
        "n": len(y_true),
    }


def run(dataset_path: str, out_dir: str, n_folds: int, embargo_bars: int):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path, parse_dates=["event_time", "entry_time", "touch_time"])
    print(f"  {len(df)} labeled events")
    print(f"  Base rate (TP fraction): {df['y_binary'].mean():.1%}")

    models = get_models()
    results = {name: [] for name in models}

    fold_num = 0
    for train_idx, test_idx in purged_walk_forward_splits(df, n_folds, embargo_bars):
        fold_num += 1
        train_df = df.loc[train_idx]
        test_df = df.loc[test_idx]

        print(f"\nFold {fold_num}: train={len(train_df)} rows "
              f"({train_df['entry_time'].min()} -> {train_df['entry_time'].max()}), "
              f"test={len(test_df)} rows "
              f"({test_df['entry_time'].min()} -> {test_df['entry_time'].max()})")

        X_train, y_train = train_df[FEATURE_COLS], train_df["y_binary"]
        X_test, y_test = test_df[FEATURE_COLS], test_df["y_binary"]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        for name, model in models.items():
            if name == "logreg":
                model.fit(X_train_scaled, y_train)
                prob = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                prob = model.predict_proba(X_test)[:, 1]

            metrics = evaluate_predictions(y_test.values, prob)
            metrics["fold"] = fold_num
            results[name].append(metrics)
            print(f"  {name:15s} AUC={metrics['auc']:.3f}  PR-AUC={metrics['pr_auc']:.3f}  "
                  f"LogLoss={metrics['log_loss']:.3f}  Brier={metrics['brier']:.3f}")

    print("\n" + "=" * 60)
    print("SUMMARY (mean across folds)")
    print("=" * 60)
    summary = {}
    for name, fold_results in results.items():
        if not fold_results:
            continue
        rdf = pd.DataFrame(fold_results)
        mean_row = rdf[["auc", "pr_auc", "log_loss", "brier", "accuracy"]].mean()
        summary[name] = mean_row.to_dict()
        print(f"{name:15s} AUC={mean_row['auc']:.3f}  PR-AUC={mean_row['pr_auc']:.3f}  "
              f"LogLoss={mean_row['log_loss']:.3f}  Brier={mean_row['brier']:.3f}  "
              f"Acc={mean_row['accuracy']:.3f}")

    print("\nInterpretation guide:")
    print("  AUC ~0.50-0.53 -> weak/no edge (common, be skeptical of anything higher)")
    print("  AUC ~0.55-0.60 -> modest, potentially real edge worth pursuing further")
    print("  AUC >0.65 on 6+ years OOS -> investigate for leakage before celebrating")

    with open(out / "baseline_results.json", "w") as f:
        json.dump(
            {name: [{k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in r.items()} for r in res]
             for name, res in results.items()},
            f, indent=2, default=str,
        )
    print(f"\nFull per-fold results saved to: {(out / 'baseline_results.json').resolve()}")
    print("\nNext (Phase 4): meta-labeling, probability calibration, regime detection, "
          "and the expected-value decision engine on top of whichever baseline wins here.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3b: baseline ML training with purged CV")
    parser.add_argument("--dataset", required=True, help="Path to M15_feature_dataset.csv")
    parser.add_argument("--out", default="./phase1_output/models")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--embargo-bars", type=int, default=48, help="Embargo buffer, in M15 bars")
    args = parser.parse_args()

    run(args.dataset, args.out, args.n_folds, args.embargo_bars)
