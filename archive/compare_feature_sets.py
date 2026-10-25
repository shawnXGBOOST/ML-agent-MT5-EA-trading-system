"""
Feature Set A/B Comparison
==============================

Trains the SAME model with identical purged/embargoed walk-forward CV on two
feature sets, so the only difference is the features themselves:

  A) BASELINE   — the original 27 features
  B) +SESSION   — baseline plus the 15 index-specific US-session features

This directly answers: do session-relative features actually add predictive
information for NAS100, or was that hypothesis wrong?

Both outcomes are useful. If session features don't improve AUC, we've ruled
out "missing session structure" as the explanation for NAS100's gap and can
stop pursuing it — which is worth knowing before investing more in that
direction.

Usage:
    python compare_feature_sets.py \\
        --dataset phase1_output_nas100/features_session/M15_feature_dataset_session.csv \\
        --out phase1_output_nas100/feature_comparison --n-folds 5 --embargo-bars 48
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

import xgboost as xgb

from train_baselines import purged_walk_forward_splits
from features_session import SESSION_FEATURE_COLS


BASE_FEATURE_COLS = [
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


def make_model():
    return xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        random_state=42, n_jobs=-1,
    )


def run(dataset_path: str, out_dir: str, n_folds: int, embargo_bars: int):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_path, parse_dates=["event_time", "entry_time", "touch_time"])
    print(f"Loaded {len(df)} events, {df['entry_time'].min()} -> {df['entry_time'].max()}")

    missing_session = [c for c in SESSION_FEATURE_COLS if c not in df.columns]
    if missing_session:
        raise ValueError(f"Dataset is missing session features: {missing_session}. "
                          f"Did you run features_session.py to build this dataset?")

    feature_sets = {
        "baseline": BASE_FEATURE_COLS,
        "with_session": BASE_FEATURE_COLS + SESSION_FEATURE_COLS,
    }

    results = {name: [] for name in feature_sets}
    importances = {}

    fold_num = 0
    for train_idx, test_idx in purged_walk_forward_splits(df, n_folds, embargo_bars):
        fold_num += 1
        train_df, test_df = df.loc[train_idx], df.loc[test_idx]
        print(f"\nFold {fold_num}: train={len(train_df)}, test={len(test_df)} "
              f"({test_df['entry_time'].min().date()} -> {test_df['entry_time'].max().date()})")

        for name, cols in feature_sets.items():
            model = make_model()
            model.fit(train_df[cols], train_df["y_binary"])
            prob = model.predict_proba(test_df[cols])[:, 1]
            y = test_df["y_binary"].values

            auc = roc_auc_score(y, prob) if len(set(y)) > 1 else np.nan
            pr = average_precision_score(y, prob) if len(set(y)) > 1 else np.nan
            ll = log_loss(y, prob, labels=[0, 1])
            results[name].append({"fold": fold_num, "auc": auc, "pr_auc": pr, "log_loss": ll, "n": len(y)})
            print(f"  {name:<14} AUC={auc:.4f}  PR-AUC={pr:.4f}  LogLoss={ll:.4f}")

            if name == "with_session":
                imp = pd.Series(model.feature_importances_, index=cols)
                importances[fold_num] = imp

    print("\n" + "=" * 70)
    print("SUMMARY — mean across folds")
    print("=" * 70)
    summary = {}
    for name, res in results.items():
        rdf = pd.DataFrame(res)
        summary[name] = rdf[["auc", "pr_auc", "log_loss"]].mean().to_dict()
        print(f"{name:<14} AUC={summary[name]['auc']:.4f}  "
              f"PR-AUC={summary[name]['pr_auc']:.4f}  LogLoss={summary[name]['log_loss']:.4f}")

    delta_auc = summary["with_session"]["auc"] - summary["baseline"]["auc"]
    print(f"\nAUC change from adding session features: {delta_auc:+.4f}")

    # Per-fold consistency matters more than the average — a gain that only
    # shows up in one fold is noise, not signal.
    base_by_fold = {r["fold"]: r["auc"] for r in results["baseline"]}
    sess_by_fold = {r["fold"]: r["auc"] for r in results["with_session"]}
    wins = sum(1 for f in base_by_fold if sess_by_fold[f] > base_by_fold[f])
    print(f"Session features improved AUC in {wins}/{len(base_by_fold)} folds")

    print("\nVERDICT:")
    if delta_auc > 0.01 and wins >= len(base_by_fold) - 1:
        print("  Session features give a consistent, meaningful improvement.")
        print("  Worth rebuilding the NAS100 pipeline around this feature set.")
    elif delta_auc > 0.005:
        print("  Marginal improvement — real but small. Probably not worth the added")
        print("  complexity on its own, but keep them if you rebuild anyway.")
    else:
        print("  No meaningful improvement. This RULES OUT 'missing session structure'")
        print("  as the explanation for NAS100's gap — useful negative result. The")
        print("  problem lies elsewhere (event definition, or genuinely lower")
        print("  predictability in this instrument's recent regime mix).")

    if importances:
        avg_imp = pd.concat(importances.values(), axis=1).mean(axis=1).sort_values(ascending=False)
        print("\nTop 15 features by importance (session-augmented model):")
        for feat, val in avg_imp.head(15).items():
            tag = "  <-- NEW session feature" if feat in SESSION_FEATURE_COLS else ""
            print(f"  {feat:<42} {val:.4f}{tag}")

        session_imp_share = avg_imp[avg_imp.index.isin(SESSION_FEATURE_COLS)].sum()
        print(f"\nSession features account for {session_imp_share:.1%} of total feature importance")
        avg_imp.to_csv(out / "feature_importance.csv")

    with open(out / "comparison_results.json", "w") as f:
        json.dump({"summary": summary, "per_fold": results, "delta_auc": delta_auc}, f, indent=2, default=str)
    print(f"\nSaved to: {out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare baseline vs session-augmented feature sets")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", default="./feature_comparison")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--embargo-bars", type=int, default=48)
    args = parser.parse_args()

    run(args.dataset, args.out, args.n_folds, args.embargo_bars)
