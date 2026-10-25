"""
XAUUSD ML+Agent+EA — Phase 3a: Feature Engineering
=====================================================

Builds the feature set for every M15 event (from Phase 1/2), covering the
categories in the blueprint:

  - Price/structure: multi-horizon returns, distance to recent swing high/low
  - Volatility: ATR percentile (carried from Phase 1), realized vol, range expansion
  - Momentum: RSI, ROC, moving-average slope
  - Session/time: session flags (carried from Phase 1), time since session open,
    distance to previous day's high/low
  - HTF context: H1 and H4 bias/alignment at the time of the M15 event
    (does H1/H4 structure agree with the M15 BOS direction?)

Cross-market features (DXY, yields, silver) are deliberately deferred — per our
plan, core pipeline first, cross-market context later if it earns its place.

Usage:
    python features.py --m15-events phase1_output/events/M15_events.csv \
        --h1-events phase1_output/events/H1_events.csv \
        --h4-events phase1_output/events/H4_events.csv \
        --labels phase1_output/labels/M15_triple_barrier_labels.csv \
        --out phase1_output/features
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Momentum / structure indicators (no external TA library needed)
# ---------------------------------------------------------------------------

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def roc(close: pd.Series, period: int = 10) -> pd.Series:
    return (close / close.shift(period) - 1) * 100


def ma_slope(close: pd.Series, period: int = 20) -> pd.Series:
    ma = close.rolling(period).mean()
    return (ma - ma.shift(5)) / ma.shift(5)  # normalized slope over last 5 bars


def distance_to_last_swing(df: pd.DataFrame) -> pd.DataFrame:
    """Distance (in ATR units) from current close to the most recent confirmed
    swing high and swing low."""
    df = df.copy()
    last_swing_high = np.nan
    last_swing_low = np.nan
    dist_high = np.full(len(df), np.nan)
    dist_low = np.full(len(df), np.nan)

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    atrs = df["atr"].values

    for i in range(len(df)):
        if df["swing_high"].iloc[i]:
            last_swing_high = highs[i]
        if df["swing_low"].iloc[i]:
            last_swing_low = lows[i]
        if not np.isnan(last_swing_high) and atrs[i] > 0:
            dist_high[i] = (closes[i] - last_swing_high) / atrs[i]
        if not np.isnan(last_swing_low) and atrs[i] > 0:
            dist_low[i] = (closes[i] - last_swing_low) / atrs[i]

    df["dist_to_last_swing_high_atr"] = dist_high
    df["dist_to_last_swing_low_atr"] = dist_low
    return df


def build_core_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds structure/momentum/volatility features to a single-timeframe events df.
    Assumes df already has: datetime, open, high, low, close, atr, atr_percentile,
    swing_high, swing_low (from Phase 1's build_events)."""
    df = df.copy()

    # Multi-horizon returns
    for h in (1, 4, 16, 48):
        df[f"return_{h}"] = df["close"].pct_change(h)

    # Candle shape
    df["candle_range"] = df["high"] - df["low"]
    df["candle_body_ratio"] = (df["close"] - df["open"]).abs() / df["candle_range"].replace(0, np.nan)

    # Momentum
    df["rsi_14"] = rsi(df["close"], 14)
    df["roc_10"] = roc(df["close"], 10)
    df["ma_slope_20"] = ma_slope(df["close"], 20)

    # Realized volatility (rolling std of returns)
    df["realized_vol_20"] = df["close"].pct_change().rolling(20).std()

    # Structure distance
    df = distance_to_last_swing(df)

    # Time since session open (bars since London/NY session flag turned True)
    for sess in ("session_london", "session_ny"):
        flag = df[sess].astype(int)
        # bars since last False->True transition, reset at each new session start
        session_start = (flag == 1) & (flag.shift(1, fill_value=0) == 0)
        group_id = session_start.cumsum()
        bars_since_open = df.groupby(group_id).cumcount()
        df[f"bars_since_{sess}_open"] = np.where(flag == 1, bars_since_open, np.nan)

    # Previous day high/low distance (approx: 96 M15 bars = 1 day; for H1/H4 this
    # constant is overridden by the caller via `bars_per_day`)
    return df


def add_previous_day_levels(df: pd.DataFrame, bars_per_day: int) -> pd.DataFrame:
    df = df.copy()
    daily_high = df["high"].rolling(bars_per_day).max().shift(bars_per_day)
    daily_low = df["low"].rolling(bars_per_day).min().shift(bars_per_day)
    df["dist_to_prev_day_high_atr"] = (df["close"] - daily_high) / df["atr"].replace(0, np.nan)
    df["dist_to_prev_day_low_atr"] = (df["close"] - daily_low) / df["atr"].replace(0, np.nan)
    return df


# ---------------------------------------------------------------------------
# HTF (H1/H4) context alignment
# ---------------------------------------------------------------------------

def htf_bias_series(htf_df: pd.DataFrame) -> pd.DataFrame:
    """Reduce an HTF events df to a simple per-bar bias: +1 if most recent BOS
    was bullish, -1 if bearish, 0 if none yet. Also carries ATR percentile as
    an HTF volatility-regime signal."""
    df = htf_df.copy()
    bias = np.zeros(len(df))
    last_bias = 0
    for i in range(len(df)):
        if df["bos_bullish"].iloc[i]:
            last_bias = 1
        elif df["bos_bearish"].iloc[i]:
            last_bias = -1
        bias[i] = last_bias
    df["htf_bias"] = bias
    return df[["datetime", "htf_bias", "atr_percentile"]].rename(
        columns={"atr_percentile": "htf_atr_percentile"}
    )


def merge_htf_context(m15: pd.DataFrame, htf: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """As-of merge: for each M15 bar, attach the most recent HTF bar's bias/vol
    (no lookahead — only HTF bars that have already closed)."""
    m15 = m15.sort_values("datetime")
    htf_bias = htf_bias_series(htf).sort_values("datetime")
    htf_bias = htf_bias.rename(
        columns={"htf_bias": f"{prefix}_bias", "htf_atr_percentile": f"{prefix}_atr_percentile"}
    )
    merged = pd.merge_asof(m15, htf_bias, on="datetime", direction="backward")
    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(m15_path: str, h1_path: str, h4_path: str, labels_path: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading M15/H1/H4 event files and labels...")
    m15 = pd.read_csv(m15_path, parse_dates=["datetime"])
    h1 = pd.read_csv(h1_path, parse_dates=["datetime"])
    h4 = pd.read_csv(h4_path, parse_dates=["datetime"])
    labels = pd.read_csv(labels_path, parse_dates=["event_time", "entry_time", "touch_time"])

    print("Building core M15 features (structure/momentum/volatility/session)...")
    m15_feat = build_core_features(m15)
    m15_feat = add_previous_day_levels(m15_feat, bars_per_day=96)  # 96 x M15 = 24h

    print("Merging H1 and H4 context (as-of, no lookahead)...")
    m15_feat = merge_htf_context(m15_feat, h1, prefix="h1")
    m15_feat = merge_htf_context(m15_feat, h4, prefix="h4")

    # Add M15's own BOS direction as +1/-1 for alignment features
    m15_feat["m15_bias"] = np.where(
        m15_feat["bos_bullish"], 1, np.where(m15_feat["bos_bearish"], -1, 0)
    )
    m15_feat["h1_aligned_with_m15"] = (
        np.sign(m15_feat["m15_bias"]) == np.sign(m15_feat["h1_bias"])
    ).astype(int)
    m15_feat["h4_aligned_with_m15"] = (
        np.sign(m15_feat["m15_bias"]) == np.sign(m15_feat["h4_bias"])
    ).astype(int)

    print("Joining features onto triple-barrier labeled events...")
    # labels['event_idx'] refers to the row index in the ORIGINAL events file passed
    # to triple_barrier.py — since we haven't reordered m15_feat, this index still lines up.
    m15_feat = m15_feat.reset_index().rename(columns={"index": "event_idx"})
    dataset = labels.merge(m15_feat, on="event_idx", suffixes=("", "_feat"))

    feature_cols = [
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
    label_cols = [
        "event_idx", "event_time", "direction", "entry_time", "touch_time",
        "label", "y_binary", "bars_to_touch", "mfe_in_atr", "mae_in_atr", "realized_return",
        "atr_at_event", "entry_price", "tp_price", "sl_price",
    ]
    # Optional columns only present in some label schemas (e.g. the staged-exit
    # labeling from triple_barrier_staged.py adds realized_r, activation_price,
    # lock_price). Carry them through when present instead of hardcoding a
    # single fixed schema, so this script works with both label formats.
    optional_label_cols = ["realized_r", "activation_price", "lock_price"]
    label_cols += [c for c in optional_label_cols if c in dataset.columns]

    final = dataset[label_cols + feature_cols].copy()
    n_before = len(final)
    sentinel_cols = ["bars_since_session_london_open", "bars_since_session_ny_open"]
    final[sentinel_cols] = final[sentinel_cols].fillna(-1)  # -1 = "not in this session"
    strict_cols = [c for c in feature_cols if c not in sentinel_cols]
    final = final.dropna(subset=strict_cols)
    n_after = len(final)

    print(f"\nDataset rows: {n_before} -> {n_after} after dropping rows with unwarmed-up features")
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    out_path = out / "M15_feature_dataset.csv"
    final.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path.resolve()}")
    print("\nNext: train_baselines.py — LogReg -> Random Forest -> XGBoost/LightGBM "
          "with purged + embargoed walk-forward validation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3a: feature engineering")
    parser.add_argument("--m15-events", required=True)
    parser.add_argument("--h1-events", required=True)
    parser.add_argument("--h4-events", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", default="./phase1_output/features")
    args = parser.parse_args()

    run(args.m15_events, args.h1_events, args.h4_events, args.labels, args.out)
