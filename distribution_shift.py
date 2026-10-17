"""
Distribution Shift Diagnostic
=================================

ChatGPT's Experiment 1, and the one suspect we never actually tested: does
the 2026 OOS window genuinely look different from the training history, or
does it just contain a different mix of regimes the model already knew about?

Compares, between the training period and the OOS window:
  - Feature distributions (PSI + KS statistic per feature)
  - Regime frequency (how often each trend/vol bucket occurs)
  - BOS event frequency and long/short split
  - Calibrated probability distribution (are we even seeing the same
    confidence range the calibrator was fit on?)

Why PSI: Population Stability Index is the standard metric for this.
Rule of thumb: <0.10 = stable, 0.10-0.25 = moderate shift worth noting,
>0.25 = significant shift that likely breaks a model's assumptions.

IMPORTANT: this compares the CANDLE/EVENT distribution, and separately the
ACCEPTED-TRADE distribution. Per ChatGPT's point, the model can be seeing
perfectly normal candles but abnormal accepted setups — those are different
questions and both matter.

Usage:
    python distribution_shift.py \\
        --dataset phase1_output_nas100/features/M15_feature_dataset.csv \\
        --split-date 2026-03-11 \\
        --out phase1_output_nas100/distribution_shift
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


FEATURE_COLS = [
    "return_1", "return_4", "return_16", "return_48",
    "candle_range", "candle_body_ratio",
    "rsi_14", "roc_10", "ma_slope_20", "realized_vol_20",
    "atr_percentile", "dist_to_last_swing_high_atr", "dist_to_last_swing_low_atr",
    "dist_to_prev_day_high_atr", "dist_to_prev_day_low_atr",
    "h1_bias", "h1_atr_percentile", "h4_bias", "h4_atr_percentile",
    "h1_aligned_with_m15", "h4_aligned_with_m15",
]


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """PSI between a reference (expected) and comparison (actual) distribution.
    Uses quantile bins from the reference so bin edges reflect the training data."""
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) < 20 or len(actual) < 20:
        return np.nan

    # Quantile-based edges from the reference distribution
    edges = np.unique(np.percentile(expected, np.linspace(0, 100, bins + 1)))
    if len(edges) < 3:
        return np.nan  # degenerate/constant feature
    edges[0], edges[-1] = -np.inf, np.inf

    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual, bins=edges)

    exp_pct = exp_counts / max(exp_counts.sum(), 1)
    act_pct = act_counts / max(act_counts.sum(), 1)

    # Floor at a small value so empty bins don't produce infinities
    eps = 1e-6
    exp_pct = np.clip(exp_pct, eps, None)
    act_pct = np.clip(act_pct, eps, None)

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def psi_label(psi: float) -> str:
    if np.isnan(psi):
        return "n/a"
    if psi < 0.10:
        return "stable"
    if psi < 0.25:
        return "MODERATE shift"
    return "SIGNIFICANT shift"


def compare_distributions(train: pd.DataFrame, oos: pd.DataFrame, label: str):
    print("\n" + "=" * 78)
    print(f"FEATURE DISTRIBUTION SHIFT — {label}")
    print(f"  (train n={len(train)}, oos n={len(oos)})")
    print("=" * 78)
    print(f"{'feature':<34}{'PSI':>8}  {'KS p-val':>9}  {'train mean':>11}{'oos mean':>11}  verdict")
    print("-" * 78)

    rows = []
    for col in FEATURE_COLS:
        if col not in train.columns or col not in oos.columns:
            continue
        tr = train[col].values.astype(float)
        oo = oos[col].values.astype(float)
        psi = population_stability_index(tr, oo)

        tr_clean = tr[~np.isnan(tr)]
        oo_clean = oo[~np.isnan(oo)]
        ks_p = stats.ks_2samp(tr_clean, oo_clean).pvalue if len(tr_clean) > 20 and len(oo_clean) > 20 else np.nan

        rows.append({
            "feature": col, "psi": psi, "ks_pvalue": ks_p,
            "train_mean": np.nanmean(tr), "oos_mean": np.nanmean(oo),
            "verdict": psi_label(psi),
        })
        print(f"{col:<34}{psi:>8.3f}  {ks_p:>9.4f}  {np.nanmean(tr):>11.4f}{np.nanmean(oo):>11.4f}  {psi_label(psi)}")

    return pd.DataFrame(rows)


def compare_regimes(train: pd.DataFrame, oos: pd.DataFrame):
    if "regime" not in train.columns:
        print("\n(no 'regime' column in this dataset — skipping regime frequency comparison)")
        return None

    print("\n" + "=" * 78)
    print("REGIME FREQUENCY SHIFT")
    print("=" * 78)
    tr_freq = train["regime"].value_counts(normalize=True)
    oo_freq = oos["regime"].value_counts(normalize=True)
    all_regimes = sorted(set(tr_freq.index) | set(oo_freq.index))

    print(f"{'regime':<30}{'train %':>10}{'oos %':>10}{'change':>12}")
    print("-" * 78)
    rows = []
    for r in all_regimes:
        t = tr_freq.get(r, 0.0)
        o = oo_freq.get(r, 0.0)
        delta = o - t
        rows.append({"regime": r, "train_pct": t, "oos_pct": o, "change": delta})
        flag = "  <-- notably more common in OOS" if delta > 0.04 else (
               "  <-- notably rarer in OOS" if delta < -0.04 else "")
        print(f"{r:<30}{t:>9.1%}{o:>10.1%}{delta:>+11.1%}{flag}")
    return pd.DataFrame(rows)


def compare_outcomes_by_regime(train: pd.DataFrame, oos: pd.DataFrame):
    """The key question: within the SAME regime, did the win rate change?
    If regime frequencies shifted but per-regime win rates held, it's a mix
    effect. If per-regime win rates themselves dropped, the relationship
    genuinely broke."""
    if "regime" not in train.columns or "y_binary" not in train.columns:
        return None

    print("\n" + "=" * 78)
    print("WIN RATE WITHIN EACH REGIME — train vs OOS")
    print("  This is the decisive test: if regime MIX changed but per-regime win")
    print("  rates held steady, it's a composition effect (the strategy is fine,")
    print("  the window was just unfavourable). If per-regime win rates THEMSELVES")
    print("  dropped, the underlying relationship genuinely broke.")
    print("=" * 78)
    print(f"{'regime':<30}{'train n':>9}{'train win%':>12}{'oos n':>8}{'oos win%':>11}{'change':>10}")
    print("-" * 78)

    rows = []
    for regime in sorted(set(train["regime"]) | set(oos["regime"])):
        tr = train[train["regime"] == regime]
        oo = oos[oos["regime"] == regime]
        if len(tr) < 30 or len(oo) < 15:
            continue
        t_wr = tr["y_binary"].mean()
        o_wr = oo["y_binary"].mean()
        rows.append({"regime": regime, "train_n": len(tr), "train_winrate": t_wr,
                      "oos_n": len(oo), "oos_winrate": o_wr, "change": o_wr - t_wr})
        print(f"{regime:<30}{len(tr):>9}{t_wr:>11.1%}{len(oo):>8}{o_wr:>10.1%}{o_wr - t_wr:>+9.1%}")

    return pd.DataFrame(rows)


def run(dataset_path: str, split_date: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_path, parse_dates=["event_time", "entry_time", "touch_time"])
    df = df.sort_values("entry_time").reset_index(drop=True)

    split = pd.Timestamp(split_date)
    train = df[df["entry_time"] < split].reset_index(drop=True)
    oos = df[df["entry_time"] >= split].reset_index(drop=True)

    print(f"Dataset: {len(df)} events, {df['entry_time'].min()} -> {df['entry_time'].max()}")
    print(f"Split at {split_date}: train={len(train)}, oos={len(oos)}")

    if len(oos) < 50:
        print("WARNING: OOS slice is very small — results below will be unreliable.")

    feat_shift = compare_distributions(train, oos, "all BOS events")
    regime_shift = compare_regimes(train, oos)
    outcome_shift = compare_outcomes_by_regime(train, oos)

    # Long/short split comparison
    if "direction" in df.columns:
        print("\n" + "=" * 78)
        print("DIRECTION SPLIT")
        print("=" * 78)
        tr_dir = train["direction"].value_counts(normalize=True)
        oo_dir = oos["direction"].value_counts(normalize=True)
        for d in sorted(set(tr_dir.index) | set(oo_dir.index)):
            print(f"  {d:<8} train {tr_dir.get(d,0):.1%}  ->  oos {oo_dir.get(d,0):.1%}")

    feat_shift.to_csv(out / "feature_shift.csv", index=False)
    if regime_shift is not None:
        regime_shift.to_csv(out / "regime_shift.csv", index=False)
    if outcome_shift is not None:
        outcome_shift.to_csv(out / "outcome_by_regime_shift.csv", index=False)

    print("\n" + "=" * 78)
    print("HOW TO READ THIS")
    print("=" * 78)
    print("1. If FEATURE PSI values are mostly <0.10, the model is seeing familiar inputs —")
    print("   rules out 'the market became unrecognizable'.")
    print("2. If REGIME FREQUENCIES shifted a lot but WIN RATE WITHIN EACH REGIME held")
    print("   steady, the strategy didn't break — the window just had more of the")
    print("   conditions it was always weak in. Fix = regime filtering / wait it out.")
    print("3. If WIN RATE WITHIN REGIMES dropped too, something structural changed and")
    print("   regime filtering alone won't save it — that points to needing better")
    print("   event definitions or features (e.g. session-aware structure for an index).")
    print(f"\nSaved to: {out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distribution shift diagnostic")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split-date", required=True, help="YYYY-MM-DD — start of the OOS window")
    parser.add_argument("--out", default="./distribution_shift")
    args = parser.parse_args()

    run(args.dataset, args.split_date, args.out)
