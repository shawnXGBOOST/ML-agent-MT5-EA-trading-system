"""
Index-Specific Session Features (NAS100)
============================================

Implements ChatGPT's Experiment 6/12 — the strongest structural idea in that
review. The core argument: for an index CFD, "BOS at 09:35" and "BOS at
14:20" are NOT the same event, and a generic ADX+ATR feature set literally
cannot tell them apart. Gold trades nearly 24h with no single dominant
session; NAS100 is anchored to the US cash session, so session-relative
structure carries information our current features throw away.

Adds these features on top of the existing set:

  Session-anchored structure:
    - overnight_range_atr        range of the overnight (pre-US-open) period
    - opening_gap_atr            gap between US open and previous US close
    - dist_from_prev_us_close    where price sits vs yesterday's cash close
    - dist_from_overnight_high   distance to overnight high
    - dist_from_overnight_low    distance to overnight low
    - opening_range_30m_atr      size of the first 30 minutes' range
    - dist_from_opening_range_high / _low

  Session phase (what part of the day this BOS actually happened in):
    - mins_since_us_open         minutes since US cash open (negative = pre-market)
    - is_pre_market              before US open
    - is_opening_30m             first 30 min (impulse phase)
    - is_us_morning              open+30m to midday
    - is_us_lunch                midday lull
    - is_us_afternoon            afternoon through close
    - is_post_close              after US cash close

US cash session is 09:30-16:00 US/Eastern, converted to the broker's
DST-aware server time using the same logic as data_engine.py, so this stays
correct across DST transitions.

Usage (produces a feature dataset with BOTH the original and new features):
    python features_session.py \\
        --m15-events phase1_output_nas100/events/M15_events.csv \\
        --h1-events phase1_output_nas100/events/H1_events.csv \\
        --h4-events phase1_output_nas100/events/H4_events.csv \\
        --labels phase1_output_nas100/labels/M15_triple_barrier_labels.csv \\
        --out phase1_output_nas100/features_session
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import features as features_mod
from mltrader.data import data_engine

# US cash session in UTC terms. NY is UTC-5 (EST) / UTC-4 (EDT), so the
# 09:30-16:00 ET cash session is 14:30-21:00 UTC in winter, 13:30-20:00 UTC
# in summer. We compute it per-bar from the same DST logic data_engine uses.
US_OPEN_ET_HOUR, US_OPEN_ET_MIN = 9, 30
US_CLOSE_ET_HOUR, US_CLOSE_ET_MIN = 16, 0


def add_us_session_clock(df: pd.DataFrame) -> pd.DataFrame:
    """Adds mins_since_us_open, using UTC hour derived from the broker's
    DST-aware offset (already computed by data_engine.tag_sessions)."""
    df = df.copy()

    if "broker_utc_offset" not in df.columns:
        df = data_engine.add_dynamic_broker_offset(df)

    # Broker offset is GMT+2 (US winter) or GMT+3 (US summer). Pepperstone
    # anchors its server clock to the NY close, so the US cash open lands at
    # 16:30 broker time in BOTH seasons — but the UTC hour differs:
    #   winter: 09:30 ET = 14:30 UTC, offset 2  -> utc_open_hour 14
    #   summer: 09:30 ET = 13:30 UTC, offset 3  -> utc_open_hour 13
    # so utc_open_hour = 16 - offset in both cases.
    utc_open_hour = 16 - df["broker_utc_offset"]

    utc_minutes = (df["datetime"].dt.hour - df["broker_utc_offset"]) % 24 * 60 + df["datetime"].dt.minute
    open_minutes = utc_open_hour * 60 + US_OPEN_ET_MIN

    mins_since = utc_minutes - open_minutes
    # Wrap so the value stays in a sensible [-12h, +12h] band rather than
    # jumping by a full day across midnight.
    mins_since = ((mins_since + 720) % 1440) - 720
    df["mins_since_us_open"] = mins_since

    session_len = (US_CLOSE_ET_HOUR - US_OPEN_ET_HOUR) * 60 + (US_CLOSE_ET_MIN - US_OPEN_ET_MIN)  # 390
    df["is_pre_market"] = (mins_since < 0).astype(int)
    df["is_opening_30m"] = ((mins_since >= 0) & (mins_since < 30)).astype(int)
    df["is_us_morning"] = ((mins_since >= 30) & (mins_since < 150)).astype(int)
    df["is_us_lunch"] = ((mins_since >= 150) & (mins_since < 240)).astype(int)
    df["is_us_afternoon"] = ((mins_since >= 240) & (mins_since < session_len)).astype(int)
    df["is_post_close"] = (mins_since >= session_len).astype(int)
    return df


def add_session_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Adds overnight range, opening gap, previous-US-close distance, and
    opening-range features. All normalized by ATR so they're comparable
    across the huge price-level changes an index sees over years."""
    df = df.copy()
    df = add_us_session_clock(df)

    # Identify each "US trading day" — group bars by the calendar date of the
    # most recent US open. Bars before today's open belong to the previous
    # session's overnight block.
    is_open_bar = (df["mins_since_us_open"] >= 0) & (df["mins_since_us_open"].shift(1) < 0)
    df["us_day_id"] = is_open_bar.cumsum()

    atr = df["atr"].replace(0, np.nan)

    # --- Previous US cash close ---
    cash = df[(df["mins_since_us_open"] >= 0) &
              (df["mins_since_us_open"] < 390)]
    prev_close_by_day = cash.groupby("us_day_id")["close"].last().shift(1)
    df["prev_us_close"] = df["us_day_id"].map(prev_close_by_day)
    df["dist_from_prev_us_close_atr"] = (df["close"] - df["prev_us_close"]) / atr

    # --- Overnight (pre-market) range, per US day ---
    overnight = df[df["mins_since_us_open"] < 0]
    on_high = overnight.groupby("us_day_id")["high"].max()
    on_low = overnight.groupby("us_day_id")["low"].min()
    df["overnight_high"] = df["us_day_id"].map(on_high)
    df["overnight_low"] = df["us_day_id"].map(on_low)
    df["overnight_range_atr"] = (df["overnight_high"] - df["overnight_low"]) / atr
    df["dist_from_overnight_high_atr"] = (df["close"] - df["overnight_high"]) / atr
    df["dist_from_overnight_low_atr"] = (df["close"] - df["overnight_low"]) / atr

    # --- Opening gap: first cash-session price vs previous cash close ---
    open_price_by_day = cash.groupby("us_day_id")["open"].first()
    df["us_open_price"] = df["us_day_id"].map(open_price_by_day)
    df["opening_gap_atr"] = (df["us_open_price"] - df["prev_us_close"]) / atr

    # --- Opening range (first 30 min of cash session) ---
    opening_30 = df[(df["mins_since_us_open"] >= 0) & (df["mins_since_us_open"] < 30)]
    or_high = opening_30.groupby("us_day_id")["high"].max()
    or_low = opening_30.groupby("us_day_id")["low"].min()
    df["opening_range_high"] = df["us_day_id"].map(or_high)
    df["opening_range_low"] = df["us_day_id"].map(or_low)
    df["opening_range_30m_atr"] = (df["opening_range_high"] - df["opening_range_low"]) / atr
    df["dist_from_opening_range_high_atr"] = (df["close"] - df["opening_range_high"]) / atr
    df["dist_from_opening_range_low_atr"] = (df["close"] - df["opening_range_low"]) / atr

    return df


SESSION_FEATURE_COLS = [
    "mins_since_us_open",
    "is_pre_market", "is_opening_30m", "is_us_morning",
    "is_us_lunch", "is_us_afternoon", "is_post_close",
    "dist_from_prev_us_close_atr",
    "overnight_range_atr", "dist_from_overnight_high_atr", "dist_from_overnight_low_atr",
    "opening_gap_atr",
    "opening_range_30m_atr", "dist_from_opening_range_high_atr", "dist_from_opening_range_low_atr",
]


def run(m15_path: str, h1_path: str, h4_path: str, labels_path: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading M15/H1/H4 event files and labels...")
    m15 = pd.read_csv(m15_path, parse_dates=["datetime"])
    h1 = pd.read_csv(h1_path, parse_dates=["datetime"])
    h4 = pd.read_csv(h4_path, parse_dates=["datetime"])
    labels = pd.read_csv(labels_path, parse_dates=["event_time", "entry_time", "touch_time"])

    print("Building original core features...")
    m15_feat = features_mod.build_core_features(m15)
    m15_feat = features_mod.add_previous_day_levels(m15_feat, bars_per_day=96)
    m15_feat = features_mod.merge_htf_context(m15_feat, h1, prefix="h1")
    m15_feat = features_mod.merge_htf_context(m15_feat, h4, prefix="h4")
    m15_feat["m15_bias"] = np.where(m15_feat["bos_bullish"], 1, np.where(m15_feat["bos_bearish"], -1, 0))
    m15_feat["h1_aligned_with_m15"] = (np.sign(m15_feat["m15_bias"]) == np.sign(m15_feat["h1_bias"])).astype(int)
    m15_feat["h4_aligned_with_m15"] = (np.sign(m15_feat["m15_bias"]) == np.sign(m15_feat["h4_bias"])).astype(int)

    print("Adding index-specific US session structure features...")
    m15_feat = add_session_structure(m15_feat)

    print("Joining onto labels...")
    m15_feat = m15_feat.reset_index().rename(columns={"index": "event_idx"})
    dataset = labels.merge(m15_feat, on="event_idx", suffixes=("", "_feat"))

    base_feature_cols = [
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
    all_feature_cols = base_feature_cols + SESSION_FEATURE_COLS

    label_cols = [
        "event_idx", "event_time", "direction", "entry_time", "touch_time",
        "label", "y_binary", "bars_to_touch", "mfe_in_atr", "mae_in_atr", "realized_return",
        "atr_at_event", "entry_price", "tp_price", "sl_price",
    ]

    final = dataset[label_cols + all_feature_cols].copy()

    sentinel_cols = ["bars_since_session_london_open", "bars_since_session_ny_open"]
    final[sentinel_cols] = final[sentinel_cols].fillna(-1)

    # Session features can legitimately be NaN on the very first partial day
    # (no previous US close / no overnight block yet) — drop only those rows.
    strict_cols = [c for c in all_feature_cols if c not in sentinel_cols]
    n_before = len(final)
    final = final.dropna(subset=strict_cols)
    n_after = len(final)

    print(f"\nDataset rows: {n_before} -> {n_after} after dropping unwarmed-up rows "
          f"({n_after/max(n_before,1):.1%} retained)")
    print(f"Total features: {len(all_feature_cols)} "
          f"({len(base_feature_cols)} original + {len(SESSION_FEATURE_COLS)} new session features)")

    print("\nNew session feature sanity check (should be non-degenerate):")
    for c in SESSION_FEATURE_COLS:
        print(f"  {c:<40} mean={final[c].mean():>10.4f}  std={final[c].std():>10.4f}  "
              f"nulls={final[c].isna().sum()}")

    out_path = out / "M15_feature_dataset_session.csv"
    final.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path.resolve()}")
    print("\nNext: train_baselines_session.py on this dataset to see whether the new")
    print("session features actually improve AUC over the original feature set.")
    print("If they DON'T improve it, that's a real answer too — it means NAS100's")
    print("problem isn't missing session structure, and we look elsewhere.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index-specific session features")
    parser.add_argument("--m15-events", required=True)
    parser.add_argument("--h1-events", required=True)
    parser.add_argument("--h4-events", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", default="./features_session")
    args = parser.parse_args()

    run(args.m15_events, args.h1_events, args.h4_events, args.labels, args.out)
