"""
TP/SL Geometry Sweep (Diagnostic)
=====================================

Motivated by a real observation from live trading: trades frequently run
most of the way toward TP, then reverse and hit SL. Our own data already
showed why - average MFE is ~1.70 ATR while TP sits at 2.0 ATR, so a large
share of trades genuinely get close without touching target.

This sweeps several TP levels (and optionally SL levels) across the SAME
historical BOS events, showing how expectancy, win rate and profit factor
change with exit geometry.

IMPORTANT - read this as a DIAGNOSTIC, not an optimizer:
  - We're looking at the SHAPE of the relationship, not cherry-picking the
    single highest number. If expectancy is flat-ish across a wide range,
    that tells you geometry isn't the lever. If there's a clear, smooth
    peak away from 2.0, that's a real finding worth acting on.
  - Picking the single best-performing value here and adopting it would be
    exactly the kind of in-sample optimization that manufactures good
    backtests. Any change chosen here MUST then be re-validated end-to-end
    (labels -> training -> calibration -> untouched OOS) before going live.
  - Lower TP raises win rate but lowers reward:risk, which RAISES the
    breakeven win rate. At 2:1 breakeven is 33.3%; at 1.5:1 it's 40%.
    Higher win rate alone means nothing without checking expectancy.

Usage:
    python tp_sl_sweep.py --events phase1_output/events/M15_events.csv \\
        --out phase1_output/tp_sweep --max-bars 48
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def simulate_geometry(df: pd.DataFrame, tp_mult: float, sl_mult: float, max_bars: int,
                       ambiguous_resolution: str = "conservative") -> dict:
    """Runs triple-barrier simulation across all BOS events at a given TP/SL
    geometry. Returns aggregate statistics (not per-trade rows, to keep this
    fast across many geometry combinations)."""
    n = len(df)
    event_idxs = df.index[(df["bos_bullish"]) | (df["bos_bearish"])].tolist()

    wins, losses, timeouts = 0, 0, 0
    r_results = []

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    opens = df["open"].values
    atrs = df["atr"].values
    bull = df["bos_bullish"].values
    n_total = len(df)

    for i in event_idxs:
        entry_idx = i + 1
        if entry_idx >= n_total:
            continue
        atr = atrs[i]
        if np.isnan(atr) or atr <= 0:
            continue

        direction_long = bull[i]
        entry_price = opens[entry_idx]

        if direction_long:
            tp_price = entry_price + tp_mult * atr
            sl_price = entry_price - sl_mult * atr
        else:
            tp_price = entry_price - tp_mult * atr
            sl_price = entry_price + sl_mult * atr

        end_idx = min(entry_idx + max_bars, n_total - 1)
        outcome = None
        for j in range(entry_idx, end_idx + 1):
            if direction_long:
                hit_tp = highs[j] >= tp_price
                hit_sl = lows[j] <= sl_price
            else:
                hit_tp = lows[j] <= tp_price
                hit_sl = highs[j] >= sl_price

            if hit_tp and hit_sl:
                outcome = "SL" if ambiguous_resolution == "conservative" else "TP"
                break
            elif hit_tp:
                outcome = "TP"
                break
            elif hit_sl:
                outcome = "SL"
                break

        if outcome is None:
            outcome = "TIMEOUT"

        # R-multiple: risk unit is the SL distance, so a win = tp_mult/sl_mult
        if outcome == "TP":
            wins += 1
            r_results.append(tp_mult / sl_mult)
        elif outcome == "SL":
            losses += 1
            r_results.append(-1.0)
        else:
            timeouts += 1
            # Timeout marked to market at the time-barrier close
            exit_price = closes[end_idx]
            raw = (exit_price - entry_price) if direction_long else (entry_price - exit_price)
            r_results.append(raw / (sl_mult * atr))

    r_arr = np.array(r_results)
    total = len(r_arr)
    if total == 0:
        return None

    gross_profit = r_arr[r_arr > 0].sum()
    gross_loss = abs(r_arr[r_arr < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    reward_risk = tp_mult / sl_mult
    breakeven_wr = 1.0 / (1.0 + reward_risk)

    return {
        "tp_mult": tp_mult, "sl_mult": sl_mult, "reward_risk": reward_risk,
        "n_trades": total, "wins": wins, "losses": losses, "timeouts": timeouts,
        "win_rate": wins / total,
        "breakeven_win_rate": breakeven_wr,
        "margin_over_breakeven": (wins / total) - breakeven_wr,
        "avg_r": r_arr.mean(),
        "total_r": r_arr.sum(),
        "profit_factor": profit_factor,
        "std_r": r_arr.std(),
        "sharpe_like": r_arr.mean() / r_arr.std() if r_arr.std() > 0 else 0,
    }


def run(events_path: str, out_dir: str, max_bars: int, sl_mults: list, tp_mults: list):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading events from {events_path}...")
    df = pd.read_csv(events_path, parse_dates=["datetime"]).reset_index(drop=True)
    n_events = ((df["bos_bullish"]) | (df["bos_bearish"])).sum()
    print(f"  {len(df)} bars, {n_events} BOS events\n")

    results = []
    for sl in sl_mults:
        for tp in tp_mults:
            res = simulate_geometry(df, tp, sl, max_bars)
            if res:
                results.append(res)
                print(f"  TP={tp:.2f}x SL={sl:.2f}x (R:R {res['reward_risk']:.2f})  "
                      f"win={res['win_rate']:.1%} (breakeven {res['breakeven_win_rate']:.1%})  "
                      f"avgR={res['avg_r']:+.4f}  PF={res['profit_factor']:.3f}")

    rdf = pd.DataFrame(results)

    print("\n" + "=" * 88)
    print("FULL RESULTS — sorted by average R per trade")
    print("=" * 88)
    display_cols = ["tp_mult", "sl_mult", "reward_risk", "n_trades", "win_rate",
                    "breakeven_win_rate", "margin_over_breakeven", "avg_r", "profit_factor", "sharpe_like"]
    print(rdf[display_cols].sort_values("avg_r", ascending=False).to_string(index=False))

    baseline = rdf[(rdf["tp_mult"] == 2.0) & (rdf["sl_mult"] == 1.0)]
    best = rdf.loc[rdf["avg_r"].idxmax()]

    print("\n" + "=" * 88)
    print("INTERPRETATION")
    print("=" * 88)
    if not baseline.empty:
        b = baseline.iloc[0]
        print(f"Current live geometry (TP 2.0x / SL 1.0x): avgR={b['avg_r']:+.4f}, "
              f"win={b['win_rate']:.1%}, PF={b['profit_factor']:.3f}")
    print(f"Best avgR in this sweep:  TP {best['tp_mult']:.2f}x / SL {best['sl_mult']:.2f}x  "
          f"avgR={best['avg_r']:+.4f}, win={best['win_rate']:.1%}, PF={best['profit_factor']:.3f}")

    if not baseline.empty:
        improvement = best["avg_r"] - baseline.iloc[0]["avg_r"]
        print(f"\nImprovement over current geometry: {improvement:+.4f}R per trade")
        if improvement < 0.02:
            print("  -> NEGLIGIBLE. Exit geometry is not the lever here. Your live observation")
            print("     (trades reversing before TP) is real, but changing TP doesn't fix it —")
            print("     the trades that DO run to 2.0x carry enough of the edge to pay for the")
            print("     near-misses. Keep the current geometry.")
        elif improvement < 0.05:
            print("  -> MARGINAL. Real but small. Given that any change requires re-running the")
            print("     ENTIRE validation pipeline (labels -> training -> calibration -> untouched")
            print("     OOS), this probably isn't worth the disruption and re-validation risk.")
        else:
            print("  -> MEANINGFUL. Worth pursuing — BUT only via a full re-validation: regenerate")
            print("     labels at the new geometry, retrain, recalibrate, and confirm on a")
            print("     genuinely untouched OOS window before changing anything live.")
            print("     Do NOT just change the live TP and assume the validated numbers carry over.")

    print("\nAlso check the SHAPE of the curve above: if avgR is relatively flat across a wide")
    print("range of TP values, geometry genuinely doesn't matter much and any apparent 'best'")
    print("value is mostly noise. A smooth, clear peak is far more trustworthy than a single")
    print("spiky high value surrounded by worse neighbours.")

    rdf.to_csv(out / "tp_sl_sweep_results.csv", index=False)
    print(f"\nSaved to: {(out / 'tp_sl_sweep_results.csv').resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TP/SL geometry diagnostic sweep")
    parser.add_argument("--events", required=True)
    parser.add_argument("--out", default="./tp_sweep")
    parser.add_argument("--max-bars", type=int, default=48)
    parser.add_argument("--sl-mults", type=str, default="1.0",
                         help="Comma-separated SL multipliers to test, e.g. 0.75,1.0,1.25")
    parser.add_argument("--tp-mults", type=str, default="1.0,1.25,1.5,1.75,2.0,2.25,2.5,3.0",
                         help="Comma-separated TP multipliers to test")
    args = parser.parse_args()

    sl_list = [float(x) for x in args.sl_mults.split(",")]
    tp_list = [float(x) for x in args.tp_mults.split(",")]

    run(args.events, args.out, args.max_bars, sl_list, tp_list)
