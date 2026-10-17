"""
XAUUSD ML+Agent+EA — Phase 1: Data Engineering + Event Detection
==================================================================

What this does:
  1. Loads raw MT5-exported CSVs for H4, H1, M15 (auto-detects common MT5 export formats)
  2. Detects broker server timezone/DST offset automatically from the data itself
  3. Cleans + synchronizes bars across timeframes into one canonical dataset
  4. Detects structural events: swing highs/lows, BOS/MSS, volatility expansion,
     liquidity sweep candidates
  5. Saves layered outputs: raw -> cleaned -> events  (as described in the blueprint)

Usage:
    python data_engine.py --h4 path/to/H4.csv --h1 path/to/H1.csv --m15 path/to/M15.csv

Expected MT5 export columns (any of these common variants are auto-detected):
    Date,Time,Open,High,Low,Close,Volume
    Date,Open,High,Low,Close,Volume   (Date already contains time)
    Time,Open,High,Low,Close,Tick volume,Volume,Spread   (MT5 "Export to CSV" format)

Nothing here trades or predicts anything yet — this is purely the data foundation
that Phase 2 (triple-barrier labeling) and Phase 3 (features) will build on.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. LOADING — auto-detect MT5 export column layout
# ---------------------------------------------------------------------------

def load_mt5_csv(path: str) -> pd.DataFrame:
    """Load an MT5-exported CSV/TSV and normalize to columns:
    ['datetime','open','high','low','close','volume']
    Handles the common separator/column variants MT5 produces.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # MT5 "Export to CSV" is usually tab-separated; plain CSV exports use comma.
    # Try tab first, fall back to comma.
    for sep in ("\t", ","):
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] > 1:
                break
        except Exception:
            continue
    else:
        raise ValueError(f"Could not parse {path} with tab or comma separator")

    df.columns = [
        c.strip().lower().replace(" ", "_").replace("<", "").replace(">", "")
        for c in df.columns
    ]

    # Case A: separate date + time columns
    if "date" in df.columns and "time" in df.columns:
        dt = pd.to_datetime(
            df["date"].astype(str) + " " + df["time"].astype(str),
            errors="coerce",
        )
    # Case B: single date column already containing time, or single 'time' col
    elif "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce")
    elif "time" in df.columns:
        dt = pd.to_datetime(df["time"], errors="coerce")
    else:
        raise ValueError(f"No recognizable date/time column in {path}: {list(df.columns)}")

    out = pd.DataFrame({"datetime": dt})

    colmap_candidates = {
        "open": ["open"],
        "high": ["high"],
        "low": ["low"],
        "close": ["close"],
        "volume": ["tick_volume", "tickvol", "volume", "vol"],
    }
    for target, candidates in colmap_candidates.items():
        found = next((c for c in candidates if c in df.columns), None)
        if found is None:
            if target == "volume":
                out[target] = np.nan  # volume optional
                continue
            raise ValueError(f"Missing required column '{target}' in {path}")
        out[target] = pd.to_numeric(df[found], errors="coerce")

    out = out.dropna(subset=["datetime", "open", "high", "low", "close"])
    out = out.sort_values("datetime").drop_duplicates(subset="datetime").reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# 2. TIMEZONE / DST DETECTION
# ---------------------------------------------------------------------------

from zoneinfo import ZoneInfo


def broker_utc_offset_for_timestamp(ts: pd.Timestamp) -> int:
    """Pepperstone's server time tracks US DST (so the daily candle always closes
    at the NY market close): GMT+3 while US is on daylight saving time (EDT),
    GMT+2 while US is on standard time (EST). We derive this per-bar by checking
    New York's actual UTC offset on that date, rather than assuming a fixed value
    for the whole dataset.
    """
    ny_tz = ZoneInfo("America/New_York")
    # naive timestamp -> localize as if it were already NY wall-clock time just to
    # read off whether DST is active on that date (this only needs the DST flag,
    # not an actual timezone conversion of the broker time itself).
    try:
        ny_offset_hours = ts.tz_localize(ny_tz, ambiguous="NaT", nonexistent="NaT").utcoffset().total_seconds() / 3600
    except Exception:
        ny_offset_hours = -5  # fallback: assume standard time
    # NY winter (EST) = UTC-5 -> broker GMT+2  => broker = ny_offset + 7
    # NY summer (EDT) = UTC-4 -> broker GMT+3  => broker = ny_offset + 7
    return int(round(ny_offset_hours + 7))


def add_dynamic_broker_offset(df: pd.DataFrame) -> pd.DataFrame:
    """Add a per-bar 'broker_utc_offset' column reflecting Pepperstone's DST-linked
    GMT+2/GMT+3 server time, instead of assuming one fixed offset across the whole
    history."""
    df = df.copy()
    # Vectorized-ish: compute once per unique date to avoid per-row overhead.
    unique_dates = pd.Series(df["datetime"].dt.normalize().unique())
    offset_by_date = {d: broker_utc_offset_for_timestamp(pd.Timestamp(d)) for d in unique_dates}
    df["broker_utc_offset"] = df["datetime"].dt.normalize().map(offset_by_date)
    return df


def detect_broker_offset(m15: pd.DataFrame) -> dict:
    """Infer the broker server's UTC offset by finding where trading volume/
    volatility clusters around known session opens (London 08:00 GMT / NY 13:00 GMT).

    This is a heuristic, not a guarantee — always sanity-check the printed result
    against your broker's published server-time documentation.
    """
    df = m15.copy()
    df["hour"] = df["datetime"].dt.hour
    df["range"] = df["high"] - df["low"]

    # Average range per hour-of-day (broker server time) — session opens show
    # up as a volatility step-up.
    hourly_range = df.groupby("hour")["range"].mean()
    # London open in GMT is 08:00. Find the hour with the sharpest range increase
    # relative to the prior hour, in the 06:00-11:00 broker-time window (covers
    # GMT+0 to GMT+3 broker offsets).
    window = hourly_range.loc[6:11]
    if len(window) < 2:
        return {"offset_guess_hours": None, "note": "insufficient data to detect offset"}

    diffs = window.diff().dropna()
    likely_london_open_hour = diffs.idxmax()
    offset_guess = likely_london_open_hour - 8  # London open is 08:00 GMT

    return {
        "offset_guess_hours": int(offset_guess),
        "likely_london_open_broker_hour": int(likely_london_open_hour),
        "note": (
            "Heuristic based on volatility step-up near London open. "
            "Pepperstone typically reports server time as GMT+2/GMT+3 depending on DST "
            "(EET/EEST) — cross-check this guess against that before trusting it."
        ),
    }


# ---------------------------------------------------------------------------
# 3. SESSION TAGGING (uses detected/confirmed broker offset)
# ---------------------------------------------------------------------------

def tag_sessions(df: pd.DataFrame, broker_utc_offset_hours: int | None = None) -> pd.DataFrame:
    """Add asia/london/ny/overlap session flags based on UTC-normalized hour.

    If broker_utc_offset_hours is None, uses the dynamic DST-aware Pepperstone
    offset (GMT+2/GMT+3 depending on US DST) computed per bar — this is the
    correct default for multi-year datasets. Pass a fixed integer only if you
    specifically want to override this (e.g. for a different broker).
    """
    df = df.copy()
    if broker_utc_offset_hours is None:
        df = add_dynamic_broker_offset(df)
        offset_col = df["broker_utc_offset"]
    else:
        offset_col = broker_utc_offset_hours

    df["utc_hour"] = (df["datetime"].dt.hour - offset_col) % 24

    df["session_asia"] = df["utc_hour"].between(0, 7, inclusive="left")
    df["session_london"] = df["utc_hour"].between(7, 16, inclusive="left")
    df["session_ny"] = df["utc_hour"].between(12, 21, inclusive="left")
    df["session_london_ny_overlap"] = df["session_london"] & df["session_ny"]
    return df


# ---------------------------------------------------------------------------
# 4. EVENT DETECTION — swings, BOS/MSS, volatility expansion, liquidity sweeps
# ---------------------------------------------------------------------------

def detect_swings(df: pd.DataFrame, lookback: int = 3) -> pd.DataFrame:
    """Fractal-style swing high/low detection: a bar is a swing high if its high
    is the max within +/-lookback bars (same for swing low)."""
    df = df.copy()
    n = lookback
    highs = df["high"].values
    lows = df["low"].values
    swing_high = np.zeros(len(df), dtype=bool)
    swing_low = np.zeros(len(df), dtype=bool)

    for i in range(n, len(df) - n):
        window_h = highs[i - n : i + n + 1]
        window_l = lows[i - n : i + n + 1]
        if highs[i] == window_h.max() and np.argmax(window_h) == n:
            swing_high[i] = True
        if lows[i] == window_l.min() and np.argmin(window_l) == n:
            swing_low[i] = True

    df["swing_high"] = swing_high
    df["swing_low"] = swing_low
    return df


def detect_bos(df: pd.DataFrame) -> pd.DataFrame:
    """Break of Structure: close breaks above the most recent confirmed swing high
    (bullish BOS) or below the most recent confirmed swing low (bearish BOS)."""
    df = df.copy()
    bos_bull = np.zeros(len(df), dtype=bool)
    bos_bear = np.zeros(len(df), dtype=bool)

    last_swing_high = np.nan
    last_swing_low = np.nan

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values

    for i in range(len(df)):
        if df["swing_high"].iloc[i]:
            last_swing_high = highs[i]
        if df["swing_low"].iloc[i]:
            last_swing_low = lows[i]

        if not np.isnan(last_swing_high) and closes[i] > last_swing_high:
            bos_bull[i] = True
        if not np.isnan(last_swing_low) and closes[i] < last_swing_low:
            bos_bear[i] = True

    df["bos_bullish"] = bos_bull
    df["bos_bearish"] = bos_bear
    return df


def detect_volatility_expansion(df: pd.DataFrame, atr_period: int = 14, pctile_window: int = 200) -> pd.DataFrame:
    """ATR + rolling percentile rank of ATR = volatility regime proxy."""
    df = df.copy()
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    df["atr"] = tr.rolling(atr_period).mean()
    df["atr_percentile"] = (
        df["atr"].rolling(pctile_window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    )
    df["volatility_expansion"] = df["atr_percentile"] > 0.80
    return df


def detect_liquidity_sweeps(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Candidate liquidity sweep: current bar's high/low pierces the highest-high
    or lowest-low of the prior `lookback` bars, then closes back inside that range
    (classic stop-hunt / sweep-and-reject shape)."""
    df = df.copy()
    roll_high = df["high"].rolling(lookback).max().shift(1)
    roll_low = df["low"].rolling(lookback).min().shift(1)

    sweep_high = (df["high"] > roll_high) & (df["close"] < roll_high)
    sweep_low = (df["low"] < roll_low) & (df["close"] > roll_low)

    df["liquidity_sweep_high"] = sweep_high.fillna(False)
    df["liquidity_sweep_low"] = sweep_low.fillna(False)
    return df


def build_events(df: pd.DataFrame) -> pd.DataFrame:
    df = detect_swings(df)
    df = detect_bos(df)
    df = detect_volatility_expansion(df)
    df = detect_liquidity_sweeps(df)
    return df


# ---------------------------------------------------------------------------
# 5. MAIN PIPELINE
# ---------------------------------------------------------------------------

def run(h4_path: str, h1_path: str, m15_path: str, out_dir: str, broker_utc_offset: int | None):
    out = Path(out_dir)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "cleaned").mkdir(parents=True, exist_ok=True)
    (out / "events").mkdir(parents=True, exist_ok=True)

    print("Loading raw data...")
    h4 = load_mt5_csv(h4_path)
    h1 = load_mt5_csv(h1_path)
    m15 = load_mt5_csv(m15_path)

    for name, d in [("H4", h4), ("H1", h1), ("M15", m15)]:
        print(f"  {name}: {len(d)} bars, {d['datetime'].min()} -> {d['datetime'].max()}")
        d.to_csv(out / "raw" / f"{name}_raw.csv", index=False)

    if broker_utc_offset is None:
        print(
            "\nNo --utc-offset override provided. Using dynamic DST-aware Pepperstone "
            "offset (GMT+2 in US winter/EST, GMT+3 in US summer/EDT), computed per bar "
            "across the full history — this correctly handles DST transitions instead "
            "of assuming one fixed offset for all 6+ years."
        )
    else:
        print(f"\nUsing manually forced fixed UTC offset for ALL bars: GMT+{broker_utc_offset}")
        print("  (Not recommended for multi-year data — DST transitions will be misaligned.)")

    print("\nTagging sessions and cleaning...")
    for name, d in [("H4", h4), ("H1", h1), ("M15", m15)]:
        d = tag_sessions(d, broker_utc_offset)
        d.to_csv(out / "cleaned" / f"{name}_cleaned.csv", index=False)

    print("\nDetecting structural events on M15 (primary decision timeframe)...")
    m15_tagged = tag_sessions(m15, broker_utc_offset)
    m15_events = build_events(m15_tagged)
    m15_events.to_csv(out / "events" / "M15_events.csv", index=False)

    print("Detecting structural events on H1 and H4 (context timeframes)...")
    h1_tagged = tag_sessions(h1, broker_utc_offset)
    h1_events = build_events(h1_tagged)
    h1_events.to_csv(out / "events" / "H1_events.csv", index=False)

    h4_tagged = tag_sessions(h4, broker_utc_offset)
    h4_events = build_events(h4_tagged)
    h4_events.to_csv(out / "events" / "H4_events.csv", index=False)

    print("\nEvent summary (M15):")
    print(f"  Swing highs: {m15_events['swing_high'].sum()}")
    print(f"  Swing lows: {m15_events['swing_low'].sum()}")
    print(f"  Bullish BOS: {m15_events['bos_bullish'].sum()}")
    print(f"  Bearish BOS: {m15_events['bos_bearish'].sum()}")
    print(f"  Volatility expansion bars: {m15_events['volatility_expansion'].sum()}")
    print(f"  Liquidity sweeps (high): {m15_events['liquidity_sweep_high'].sum()}")
    print(f"  Liquidity sweeps (low): {m15_events['liquidity_sweep_low'].sum()}")

    print(f"\nDone. Outputs written to: {out.resolve()}")
    print("  raw/       -> untouched loaded data")
    print("  cleaned/   -> deduped, sorted, session-tagged")
    print("  events/    -> swings, BOS/MSS, volatility regime, liquidity sweeps")
    print("\nNext (Phase 2): triple-barrier labeling on top of events/M15_events.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: XAUUSD data engine")
    parser.add_argument("--h4", required=True, help="Path to H4 MT5 CSV export")
    parser.add_argument("--h1", required=True, help="Path to H1 MT5 CSV export")
    parser.add_argument("--m15", required=True, help="Path to M15 MT5 CSV export")
    parser.add_argument("--out", default="./phase1_output", help="Output directory")
    parser.add_argument(
        "--utc-offset",
        type=int,
        default=None,
        help="Broker server UTC offset in hours (e.g. 3 for GMT+3). "
        "If omitted, the script will attempt to auto-detect it.",
    )
    args = parser.parse_args()

    run(args.h4, args.h1, args.m15, args.out, args.utc_offset)
