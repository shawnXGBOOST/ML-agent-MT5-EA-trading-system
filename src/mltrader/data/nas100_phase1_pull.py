"""
NAS100 - Phase 1: Data Pull + Event Detection (via MT5 API directly)
========================================================================

Same Phase 1 methodology as XAUUSD (event detection, DST-aware sessions),
but pulls data directly through the MT5 Python API instead of a manual CSV
export, and saves output in the same file layout the rest of the pipeline
(triple_barrier.py, features.py, train_baselines.py, calibrate_and_ev.py,
regime.py, final_holdout_eval.py) already expects - so every one of those
scripts works UNCHANGED, just pointed at these new files.

IMPORTANT: NAS100 is a different instrument with a different price scale,
volatility profile, and (per your research blueprint) likely different
behavior than gold. This is genuinely a fresh Phase 1-5 validation, not a
"reuse gold's numbers" exercise - the TP/SL ATR multipliers, thresholds,
and model all need to be re-validated on NAS100's own data before trusting
any of it, same rigor as XAUUSD got.

Usage:
    python nas100_phase1_pull.py --symbol NAS100 --out phase1_output_nas100
"""

import argparse
from pathlib import Path

import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from mltrader.data import data_engine


def fetch_max_history(symbol: str, timeframe, max_bars: int = 200000) -> pd.DataFrame:
    """Pulls as much history as the broker/terminal will give us. NAS100 index
    CFDs often have shorter available intraday history than XAUUSD - this
    just asks for a lot and reports what actually came back."""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, max_bars)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"Failed to fetch {symbol} bars from MT5: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["datetime"] = pd.to_datetime(df["time"], unit="s")
    df = df.rename(columns={"tick_volume": "volume"})
    return df[["datetime", "open", "high", "low", "close", "volume"]]


def run(symbol: str, out_dir: str):
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package not installed. Run: pip install MetaTrader5")
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        mt5.shutdown()
        raise RuntimeError(f"Symbol '{symbol}' not found on this broker/terminal. "
                            f"Check the exact name in Market Watch.")
    print(f"Symbol confirmed: {symbol} (digits={symbol_info.digits}, "
          f"point={symbol_info.point}, min_lot={symbol_info.volume_min})")

    out = Path(out_dir)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "cleaned").mkdir(parents=True, exist_ok=True)
    (out / "events").mkdir(parents=True, exist_ok=True)

    print(f"\nPulling max available history for {symbol}...")
    h4 = fetch_max_history(symbol, mt5.TIMEFRAME_H4)
    h1 = fetch_max_history(symbol, mt5.TIMEFRAME_H1)
    m15 = fetch_max_history(symbol, mt5.TIMEFRAME_M15)

    for name, d in [("H4", h4), ("H1", h1), ("M15", m15)]:
        print(f"  {name}: {len(d)} bars, {d['datetime'].min()} -> {d['datetime'].max()}")
        d.to_csv(out / "raw" / f"{name}_raw.csv", index=False)

    print("\nTagging sessions (dynamic DST-aware offset, same as XAUUSD) and detecting events...")
    for name, d in [("H4", h4), ("H1", h1), ("M15", m15)]:
        tagged = data_engine.tag_sessions(d)
        events = data_engine.build_events(tagged)
        events.to_csv(out / "events" / f"{name}_events.csv", index=False)
        tagged.to_csv(out / "cleaned" / f"{name}_cleaned.csv", index=False)

        print(f"  {name} events: swing_high={events['swing_high'].sum()}, "
              f"swing_low={events['swing_low'].sum()}, "
              f"bos_bullish={events['bos_bullish'].sum()}, "
              f"bos_bearish={events['bos_bearish'].sum()}, "
              f"vol_expansion={events['volatility_expansion'].sum()}")

    print(f"\nSaved to: {out.resolve()}")
    print("\nNext: triple_barrier.py on events/M15_events.csv - same as XAUUSD Phase 2. "
          "IMPORTANT: reconsider the TP/SL ATR multipliers (2.0/1.0 for gold) before assuming "
          "they fit NAS100 - check the event summary above and average bars-to-touch once "
          "labeled, since index CFDs often move differently than gold intraday.")

    mt5.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NAS100 Phase 1: data pull + event detection")
    parser.add_argument("--symbol", default="NAS100")
    parser.add_argument("--out", default="./phase1_output_nas100")
    args = parser.parse_args()

    run(args.symbol, args.out)
