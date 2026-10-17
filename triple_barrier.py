"""
XAUUSD ML+Agent+EA — Phase 2: Triple-Barrier Labeling
=======================================================

Takes the M15 events file from Phase 1 (data_engine.py) and generates
triple-barrier labels for every BOS event:

  - Entry candidates: bullish BOS -> long, bearish BOS -> short
  - Entry price: NEXT bar's open (never the event bar's own close — that
    would be lookahead, since in live trading you can only react after
    the event bar has fully closed)
  - Barriers (ATR-scaled at the entry bar):
        TP = entry +/- (tp_atr_mult * ATR)
        SL = entry -/+ (sl_atr_mult * ATR)
        Time = max_bars candles ahead
  - Label: which barrier is touched first, walking forward bar by bar
    using each bar's high/low (not just close) so we don't miss
    intrabar touches.
  - If TP and SL are both touched within the SAME bar, we can't know
    which happened first from OHLC alone — default resolution is
    "conservative" (assume SL first). This avoids overstating edge.

Also records auxiliary outcomes per the blueprint: MFE, MAE, time-to-barrier,
realized return — useful for calibration and expected-value work later.

Usage:
    python triple_barrier.py --events phase1_output/events/M15_events.csv \
        --out phase1_output/labels --tp-atr 2.0 --sl-atr 1.0 --max-bars 48
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compute_triple_barrier_labels(
    df: pd.DataFrame,
    tp_atr_mult: float = 2.0,
    sl_atr_mult: float = 1.0,
    max_bars: int = 48,
    ambiguous_resolution: str = "conservative",
) -> pd.DataFrame:
    """
    df must contain: datetime, open, high, low, close, atr, bos_bullish, bos_bearish
    Returns a DataFrame of one row per event candidate with triple-barrier outcomes.
    """
    required = {"datetime", "open", "high", "low", "close", "atr", "bos_bullish", "bos_bearish"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input events file missing required columns: {missing}")

    df = df.reset_index(drop=True)
    n = len(df)
    records = []

    event_idxs = df.index[(df["bos_bullish"]) | (df["bos_bearish"])].tolist()

    for i in event_idxs:
        direction = "long" if df.at[i, "bos_bullish"] else "short"

        entry_idx = i + 1
        if entry_idx >= n:
            continue  # no next bar to enter on (end of data)

        atr_at_event = df.at[i, "atr"]
        if pd.isna(atr_at_event) or atr_at_event <= 0:
            continue  # ATR not yet warmed up

        entry_price = df.at[entry_idx, "open"]
        entry_time = df.at[entry_idx, "datetime"]

        if direction == "long":
            tp_price = entry_price + tp_atr_mult * atr_at_event
            sl_price = entry_price - sl_atr_mult * atr_at_event
        else:
            tp_price = entry_price - tp_atr_mult * atr_at_event
            sl_price = entry_price + sl_atr_mult * atr_at_event

        time_barrier_idx = min(entry_idx + max_bars, n - 1)
        if time_barrier_idx <= entry_idx:
            continue  # not enough future bars remain

        window = df.iloc[entry_idx : time_barrier_idx + 1]

        label = None
        touch_idx = None
        for j, row in window.iterrows():
            hit_tp = row["high"] >= tp_price if direction == "long" else row["low"] <= tp_price
            hit_sl = row["low"] <= sl_price if direction == "long" else row["high"] >= sl_price

            if hit_tp and hit_sl:
                # Ambiguous same-bar touch — can't tell order from OHLC alone.
                label = "SL" if ambiguous_resolution == "conservative" else "TP"
                touch_idx = j
                break
            elif hit_tp:
                label = "TP"
                touch_idx = j
                break
            elif hit_sl:
                label = "SL"
                touch_idx = j
                break

        if label is None:
            label = "TIMEOUT"
            touch_idx = time_barrier_idx

        # Auxiliary outcomes over the realized path up to the touch (or full window if timeout)
        path = df.iloc[entry_idx : touch_idx + 1]
        if direction == "long":
            mfe = (path["high"].max() - entry_price) / atr_at_event
            mae = (entry_price - path["low"].min()) / atr_at_event
            realized_return = (df.at[touch_idx, "close"] - entry_price) / entry_price
        else:
            mfe = (entry_price - path["low"].min()) / atr_at_event
            mae = (path["high"].max() - entry_price) / atr_at_event
            realized_return = (entry_price - df.at[touch_idx, "close"]) / entry_price

        records.append(
            {
                "event_idx": i,
                "event_time": df.at[i, "datetime"],
                "direction": direction,
                "entry_time": entry_time,
                "entry_price": entry_price,
                "atr_at_event": atr_at_event,
                "tp_price": tp_price,
                "sl_price": sl_price,
                "label": label,  # TP / SL / TIMEOUT
                "y_binary": 1 if label == "TP" else 0,  # for simple binary classification use
                "touch_time": df.at[touch_idx, "datetime"],
                "bars_to_touch": touch_idx - entry_idx,
                "mfe_in_atr": mfe,
                "mae_in_atr": mae,
                "realized_return": realized_return,
            }
        )

    return pd.DataFrame(records)


def run(events_path: str, out_dir: str, tp_atr: float, sl_atr: float, max_bars: int):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading events from {events_path}...")
    df = pd.read_csv(events_path, parse_dates=["datetime"])

    print(
        f"Running triple-barrier labeling: TP={tp_atr}x ATR, SL={sl_atr}x ATR, "
        f"time barrier={max_bars} bars..."
    )
    labels = compute_triple_barrier_labels(
        df, tp_atr_mult=tp_atr, sl_atr_mult=sl_atr, max_bars=max_bars
    )

    if labels.empty:
        print("WARNING: zero labeled events produced. Check that ATR has warmed up "
              "and that BOS events exist in the input file (this mirrors the blueprint's "
              "'zero trades' failure mode — usually a data/column issue, not a bad strategy).")
    else:
        counts = labels["label"].value_counts()
        total = len(labels)
        print(f"\nLabeled {total} events:")
        for lbl, cnt in counts.items():
            print(f"  {lbl}: {cnt} ({cnt/total:.1%})")

        long_count = (labels["direction"] == "long").sum()
        short_count = (labels["direction"] == "short").sum()
        print(f"\n  Long candidates: {long_count}  |  Short candidates: {short_count}")
        print(f"  Avg bars to touch: {labels['bars_to_touch'].mean():.1f}")
        print(f"  Avg MFE (in ATR): {labels['mfe_in_atr'].mean():.2f}")
        print(f"  Avg MAE (in ATR): {labels['mae_in_atr'].mean():.2f}")

    out_path = out / "M15_triple_barrier_labels.csv"
    labels.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path.resolve()}")
    print("\nNext (Phase 3): build feature library + train tabular ML baselines "
          "(LogReg -> Random Forest -> XGBoost/LightGBM) predicting y_binary (TP vs not-TP), "
          "using purged/embargoed splits so overlapping label windows don't leak across folds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Triple-barrier labeling")
    parser.add_argument("--events", required=True, help="Path to M15_events.csv from Phase 1")
    parser.add_argument("--out", default="./phase1_output/labels", help="Output directory")
    parser.add_argument("--tp-atr", type=float, default=2.0, help="TP barrier as multiple of ATR")
    parser.add_argument("--sl-atr", type=float, default=1.0, help="SL barrier as multiple of ATR")
    parser.add_argument("--max-bars", type=int, default=48, help="Time barrier, in bars (48 = 12h on M15)")
    args = parser.parse_args()

    run(args.events, args.out, args.tp_atr, args.sl_atr, args.max_bars)
