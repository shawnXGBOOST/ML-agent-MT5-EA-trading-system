"""
XAUUSD ML+Agent+EA — Phase 4b: Regime Detection
==================================================

Builds an interpretable, rule-based regime classification (per the blueprint's
guidance: "the first implementation should remain interpretable enough to
diagnose failures" — so ADX/ATR buckets rather than an HMM/clustering black
box, at least for v1):

  - Trend strength: ADX(14) -> ranging (<20) / transitional (20-25) / trending (>25)
  - Volatility: ATR percentile (already computed in Phase 1) -> low (<33%) /
    normal (33-66%) / high (>66%)

Combined into a regime label like "trending_high_vol". Then breaks down the
calibrated EV results from Phase 4a by regime, to see whether the model's edge
concentrates in specific conditions — this feeds directly into the agent's
strategy-switching logic in Phase 6.

Usage:
    python regime.py --m15-events phase1_output/events/M15_events.csv \
        --ev-dataset phase1_output/ev/M15_calibrated_ev_dataset.csv \
        --out phase1_output/regime
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_high, prev_low, prev_close = high.shift(1), low.shift(1), close.shift(1)

    plus_dm = (high - prev_high).clip(lower=0)
    minus_dm = (prev_low - low).clip(lower=0)
    # Only the larger of the two counts on any given bar
    plus_dm = np.where(plus_dm > minus_dm, plus_dm, 0.0)
    minus_dm = np.where(minus_dm > plus_dm, minus_dm, 0.0)

    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx


def bucket_trend(adx: pd.Series) -> pd.Series:
    return pd.cut(
        adx, bins=[-np.inf, 20, 25, np.inf], labels=["ranging", "transitional", "trending"]
    )


def bucket_volatility(atr_percentile: pd.Series) -> pd.Series:
    return pd.cut(
        atr_percentile, bins=[-np.inf, 0.33, 0.66, np.inf], labels=["low_vol", "normal_vol", "high_vol"]
    )


def run(m15_events_path: str, ev_dataset_path: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading M15 events and computing ADX...")
    m15 = pd.read_csv(m15_events_path, parse_dates=["datetime"])
    m15["adx_14"] = compute_adx(m15)
    m15["trend_regime"] = bucket_trend(m15["adx_14"])
    m15["vol_regime"] = bucket_volatility(m15["atr_percentile"])
    m15["regime"] = m15["trend_regime"].astype(str) + "_" + m15["vol_regime"].astype(str)

    m15 = m15.reset_index().rename(columns={"index": "event_idx"})
    regime_lookup = m15[["event_idx", "adx_14", "trend_regime", "vol_regime", "regime"]]

    print("Loading EV dataset and merging regime labels...")
    ev = pd.read_csv(ev_dataset_path, parse_dates=["event_time", "entry_time", "touch_time"])
    merged = ev.merge(regime_lookup, on="event_idx", how="left")

    merged = merged.dropna(subset=["regime"])
    merged = merged[~merged["regime"].str.contains("nan")]

    reward_R, risk_R = 2.0, 1.0

    print("\n" + "=" * 70)
    print("EXPECTANCY BY REGIME (TAKE decisions only, cost-inclusive)")
    print("=" * 70)
    taken = merged[merged["decision"] == "TAKE"]
    rows = []
    for regime, g in taken.groupby("regime"):
        if len(g) < 30:
            continue
        tp_rate = g["y_binary"].mean()
        expectancy_excl = tp_rate * reward_R - (1 - tp_rate) * risk_R
        expectancy_incl = expectancy_excl - g["cost_r"].mean()
        rows.append({
            "regime": regime, "n_trades": len(g), "tp_rate": tp_rate,
            "expectancy_excl_cost_R": expectancy_excl, "expectancy_incl_cost_R": expectancy_incl,
        })
    regime_summary = pd.DataFrame(rows).sort_values("expectancy_incl_cost_R", ascending=False)
    for _, r in regime_summary.iterrows():
        print(f"  {r['regime']:30s} n={r['n_trades']:5.0f}  TP%={r['tp_rate']:.1%}  "
              f"EV(excl cost)={r['expectancy_excl_cost_R']:+.3f}R  "
              f"EV(incl cost)={r['expectancy_incl_cost_R']:+.3f}R")

    print("\n" + "=" * 70)
    print("MODEL AUC/TP-RATE BY VOLATILITY REGIME (all events, not just TAKE)")
    print("=" * 70)
    for vol, g in merged.groupby("vol_regime", observed=True):
        if len(g) < 30:
            continue
        print(f"  {vol:12s} n={len(g):5d}  base TP rate={g['y_binary'].mean():.1%}  "
              f"avg calibrated_prob={g['calibrated_prob'].mean():.3f}")

    out_path = out / "M15_regime_dataset.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path.resolve()}")
    print("\nUse this breakdown to decide: should the agent apply a UNIFORM ev threshold "
          "across all regimes (simpler), or a regime-specific threshold/strategy "
          "(e.g. only trade in trending+normal_vol, skip ranging+high_vol)? "
          "This is a Phase 6 agent design decision informed by what you see above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4b: regime detection")
    parser.add_argument("--m15-events", required=True)
    parser.add_argument("--ev-dataset", required=True)
    parser.add_argument("--out", default="./phase1_output/regime")
    args = parser.parse_args()

    run(args.m15_events, args.ev_dataset, args.out)
