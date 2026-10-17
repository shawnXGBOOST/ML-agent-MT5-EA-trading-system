"""
XAUUSD ML+Agent+EA — Phase 8: Outcome Tracking / Paper Trading Monitor
=========================================================================

Pulls closed trades from MT5 history, matches each one back to the signal
that generated it (via the "seqN" comment the EA already stamps on every
order — see PlacePendingOrder in the EA), joins with the signal log written
by signal_bridge.py, and reports realized vs predicted performance.

This is the actual answer to the blueprint's core question for this stage:
"does live behavior match what the backtest promised, or is there a
live/backtest gap?" Per the blueprint's own warning, expect SOME gap —
the question is whether it's a reasonable gap or a "something is wrong" gap.

Usage:
    python track_outcomes.py --signal-log "C:\\...\\MQL5\\Files\\signal.log.jsonl" \
        --magic 778001 --out phase1_output/live_tracking --days-back 30
"""

import argparse
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None


def load_signal_log(path: str) -> pd.DataFrame:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    if not df.empty:
        df["sequence_id"] = df["sequence_id"].astype(int)
    return df


def fetch_closed_trades(magic: int, days_back: int) -> pd.DataFrame:
    date_from = datetime.now() - timedelta(days=days_back)
    date_to = datetime.now() + timedelta(days=1)

    deals = mt5.history_deals_get(date_from, date_to)
    if deals is None:
        return pd.DataFrame()

    rows = []
    for d in deals:
        if d.magic != magic:
            continue
        rows.append({
            "ticket": d.ticket, "order": d.order, "position_id": d.position_id,
            "time": pd.to_datetime(d.time, unit="s"), "type": d.type,
            "entry": d.entry,  # 0 = IN, 1 = OUT
            "volume": d.volume, "price": d.price, "profit": d.profit,
            "comment": d.comment,
        })
    return pd.DataFrame(rows)


def extract_sequence_id(comment: str):
    if comment and comment.startswith("seq"):
        try:
            return int(comment[3:])
        except ValueError:
            return None
    return None


def run(signal_log_path: str, magic: int, out_dir: str, days_back: int):
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package not installed.")
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading signal log...")
    signals = load_signal_log(signal_log_path)
    print(f"  {len(signals)} signals logged")

    print(f"Fetching closed trades from MT5 (magic={magic}, last {days_back} days)...")
    deals = fetch_closed_trades(magic, days_back)
    print(f"  {len(deals)} deals found")

    if deals.empty:
        print("No closed trades yet. Nothing to compare — check back after some trades close.")
        mt5.shutdown()
        return

    # Positions have an IN deal and an OUT deal; we want closed positions only
    closed_positions = []
    for pos_id, group in deals.groupby("position_id"):
        in_deal = group[group["entry"] == 0]
        out_deal = group[group["entry"] == 1]
        if in_deal.empty or out_deal.empty:
            continue  # still open, or malformed — skip
        seq_id = extract_sequence_id(in_deal.iloc[0]["comment"])
        closed_positions.append({
            "position_id": pos_id,
            "sequence_id": seq_id,
            "open_time": in_deal.iloc[0]["time"],
            "close_time": out_deal.iloc[0]["time"],
            "volume": in_deal.iloc[0]["volume"],
            "entry_fill_price": in_deal.iloc[0]["price"],
            "exit_fill_price": out_deal.iloc[0]["price"],
            "realized_profit": out_deal["profit"].sum(),
        })

    trades = pd.DataFrame(closed_positions)
    print(f"  {len(trades)} fully closed positions matched")

    if trades.empty:
        print("No fully closed positions yet.")
        mt5.shutdown()
        return

    merged = trades.merge(signals, on="sequence_id", how="left", suffixes=("", "_signal"))
    unmatched = merged["calibrated_prob"].isna().sum()
    if unmatched > 0:
        print(f"  WARNING: {unmatched} closed trades couldn't be matched to a logged signal "
              f"(comment missing/corrupted, or signal log doesn't cover that period)")

    matched = merged.dropna(subset=["calibrated_prob"]).copy()
    if matched.empty:
        print("No matched trades to analyze.")
        mt5.shutdown()
        return

    matched["won"] = matched["realized_profit"] > 0
    matched["risk_dollar"] = matched["equity_at_signal"] * (matched["risk_pct"] / 100)
    matched["realized_r"] = matched["realized_profit"] / matched["risk_dollar"].replace(0, pd.NA)

    print("\n" + "=" * 60)
    print("LIVE OUTCOME SUMMARY")
    print("=" * 60)
    print(f"Matched trades: {len(matched)}")
    print(f"Win rate: {matched['won'].mean():.1%}")
    print(f"Avg realized R: {matched['realized_r'].mean():+.3f}")
    print(f"Total realized P&L: ${matched['realized_profit'].sum():+.2f}")
    print(f"Avg predicted calibrated_prob: {matched['calibrated_prob'].mean():.3f}")
    print(f"Avg predicted ev_r: {matched['ev_r'].mean():+.3f}")

    print("\nCOMPARISON TO VALIDATED HOLDOUT (Phase 5):")
    print(f"  Holdout: TP rate 44.5%, net expectancy +0.293R/trade")
    print(f"  Live:    win rate {matched['won'].mean():.1%}, avg realized {matched['realized_r'].mean():+.3f}R")
    gap = matched['realized_r'].mean() - 0.293
    print(f"  Gap: {gap:+.3f}R  "
          f"({'within normal range — small sample noise expected' if abs(gap) < 0.3 else 'notable — watch closely, do not scale up yet' if abs(gap) < 0.6 else 'LARGE gap — investigate before continuing live'})")

    print("\nBy regime:")
    for regime, g in matched.groupby("regime"):
        if len(g) < 3:
            print(f"  {regime}: only {len(g)} trades — too few to judge yet")
            continue
        print(f"  {regime:25s} n={len(g):3d}  win%={g['won'].mean():.1%}  avg_R={g['realized_r'].mean():+.3f}")

    out_path = out / "live_outcomes.csv"
    matched.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path.resolve()}")
    print("\nRun this again periodically (e.g. daily) as more trades close — "
          "with under ~30-50 trades, don't over-interpret the numbers yet; "
          "noise dominates at small sample sizes.")

    mt5.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 8: outcome tracking")
    parser.add_argument("--signal-log", required=True,
                         help="Path to the .log.jsonl file (same folder as signal.json)")
    parser.add_argument("--magic", type=int, default=778001)
    parser.add_argument("--out", default="./phase1_output/live_tracking")
    parser.add_argument("--days-back", type=int, default=30)
    args = parser.parse_args()

    run(args.signal_log, args.magic, args.out, args.days_back)
