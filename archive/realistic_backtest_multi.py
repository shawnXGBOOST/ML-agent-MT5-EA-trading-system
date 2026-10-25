"""
XAUUSD + NAS100 - Multi-Instrument Realistic Backtest
=========================================================

Same realistic fill simulation as realistic_backtest.py (pending-order fills,
real lot rounding, full Agent risk gating) but replays candidates from BOTH
instruments in one MERGED chronological timeline through a SINGLE shared
Agent + AgentState - matching exactly how live_runner_multi.py behaves live:
one shared concurrency cap, one shared daily loss limit, one shared
drawdown circuit breaker.

This answers a genuinely different question than running each instrument's
realistic_backtest.py separately: when both compete for the same single
concurrency slot and the same daily loss budget, how many of each
instrument's signals actually get through, and what does the COMBINED
equity curve look like?

Usage:
    python realistic_backtest_multi.py \\
        --xauusd-model-dir phase1_output/production_model \\
        --nas100-model-dir phase1_output_nas100/production_model \\
        --months 1 --starting-equity 200 \\
        --out phase1_output/realistic_backtest_multi
"""

import argparse
import json
from pathlib import Path
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from agent import Agent, AgentState, RiskConfig, TradeCandidate
from realistic_backtest import (
    fetch_mt5_range, build_full_feature_series, simulate_fill_and_outcome,
    calculate_realistic_lot_size,
)


def load_instrument_model(model_dir: str):
    model_dir = Path(model_dir)
    model = joblib.load(model_dir / "xgb_model.joblib")
    calibrator = joblib.load(model_dir / "calibrator.joblib")
    with open(model_dir / "model_meta.json") as f:
        config_meta = json.load(f)
    return model, calibrator, config_meta


def build_candidates_for_symbol(symbol: str, model, calibrator, config_meta,
                                 months: int, spread_price: float, commission_price: float):
    """Pulls data, builds features, and returns (full_df, candidates_list) where
    each candidate dict has everything needed to later evaluate + simulate a fill,
    tagged with which symbol/dataframe it belongs to."""
    print(f"\n[{symbol}] Fetching ~{months} months of data...")
    m15 = fetch_mt5_range(symbol, mt5.TIMEFRAME_M15, months, warmup_bars=500)
    h1 = fetch_mt5_range(symbol, mt5.TIMEFRAME_H1, months, warmup_bars=500)
    h4 = fetch_mt5_range(symbol, mt5.TIMEFRAME_H4, months, warmup_bars=500)
    print(f"  {len(m15)} M15 bars")

    print(f"[{symbol}] Building feature series...")
    full_df = build_full_feature_series(m15, h1, h4)

    backtest_start = datetime.now() - timedelta(days=months * 31)
    feature_cols = config_meta["feature_cols"]
    tp_atr_mult, sl_atr_mult = config_meta["tp_atr_mult"], config_meta["sl_atr_mult"]
    reward_R, risk_R = tp_atr_mult / sl_atr_mult, 1.0

    start_pos = full_df.index[full_df["datetime"] >= backtest_start]
    if len(start_pos) == 0:
        raise RuntimeError(f"[{symbol}] No data found after backtest_start={backtest_start} — "
                            f"check that enough history was fetched.")
    start_pos = start_pos[0]

    candidates = []
    for pos in range(start_pos, len(full_df) - 5):
        row = full_df.iloc[pos]
        if not (row["bos_bullish"] or row["bos_bearish"]):
            continue
        if row[feature_cols].isna().any():
            continue
        atr_at_event = row["atr"]
        if pd.isna(atr_at_event) or atr_at_event <= 0:
            continue

        direction = "long" if row["bos_bullish"] else "short"
        X = row[feature_cols].values.reshape(1, -1).astype(float)
        raw_prob = model.predict_proba(X)[0, 1]
        calibrated_prob = float(calibrator.predict([raw_prob])[0])

        entry_price = row["close"]
        if direction == "long":
            sl_price = entry_price - sl_atr_mult * atr_at_event
            tp_price = entry_price + tp_atr_mult * atr_at_event
        else:
            sl_price = entry_price + sl_atr_mult * atr_at_event
            tp_price = entry_price - tp_atr_mult * atr_at_event

        cost_r = (spread_price + commission_price) / (sl_atr_mult * atr_at_event)
        ev_r = calibrated_prob * reward_R - (1 - calibrated_prob) * risk_R - cost_r

        candidates.append({
            "symbol": symbol, "pos": pos, "event_time": row["datetime"],
            "direction": direction, "calibrated_prob": calibrated_prob, "ev_r": ev_r,
            "regime": row["regime"], "session_london": bool(row["session_london"]),
            "session_ny": bool(row["session_ny"]), "session_asia": bool(row["session_asia"]),
            "entry_price": float(entry_price), "sl_price": float(sl_price), "tp_price": float(tp_price),
            "cost_r": cost_r, "reward_R": reward_R, "risk_R": risk_R,
        })

    print(f"[{symbol}] {len(candidates)} BOS candidates in backtest window")
    return full_df, candidates


def run(xauusd_model_dir: str, nas100_model_dir: str, months: int, starting_equity: float,
        out_dir: str, xauusd_spread_price: float, xauusd_commission_price: float,
        nas100_spread_price: float, nas100_commission_price: float, min_ev: float,
        expiry_minutes: int, max_hold_bars: int):
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package not installed.")
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    xau_model, xau_cal, xau_meta = load_instrument_model(xauusd_model_dir)
    nas_model, nas_cal, nas_meta = load_instrument_model(nas100_model_dir)

    xau_df, xau_candidates = build_candidates_for_symbol(
        "XAUUSD", xau_model, xau_cal, xau_meta, months, xauusd_spread_price, xauusd_commission_price)
    nas_df, nas_candidates = build_candidates_for_symbol(
        "NAS100", nas_model, nas_cal, nas_meta, months, nas100_spread_price, nas100_commission_price)

    dfs = {"XAUUSD": xau_df, "NAS100": nas_df}
    all_candidates = sorted(xau_candidates + nas_candidates, key=lambda c: c["event_time"])
    print(f"\nMerged timeline: {len(all_candidates)} total candidates across both instruments")

    config = RiskConfig(min_ev_r=min_ev)
    agent = Agent(config)
    state = AgentState(equity=starting_equity, peak_equity=starting_equity, day_start_equity=starting_equity)

    trades, unfilled, decisions_log = [], [], []
    restarts = []
    true_peak_equity = starting_equity  # tracks the REAL all-time peak, independent of
                                          # the operational peak_equity that gets reset on restart
    true_max_dd_pct = 0.0  # tracks the WORST peak-to-trough drawdown at any point, not just at the end
    expiry_bars = max(1, round(expiry_minutes / 15))

    print("\nWalking forward through the MERGED timeline (shared risk state)...")
    for c in all_candidates:
        # BACKTEST-ONLY: simulate a manual review + restart after a drawdown halt.
        # This does NOT change live behavior (live_runner.py still requires a real
        # human to review and restart) — it's purely so a single rough stretch
        # doesn't end the entire backtest early and understate the full period's
        # results, the same way live_runner.py's actual halt behaves correctly.
        if state.trading_halted_drawdown:
            restarts.append({
                "time": str(c["event_time"]), "equity_at_restart": state.equity,
                "drawdown_from_peak_pct": state.drawdown_pct(),
            })
            state.trading_halted_drawdown = False
            state.peak_equity = state.equity  # restart tracking from current balance

        candidate = TradeCandidate(
            symbol=c["symbol"], event_time=c["event_time"], entry_time=c["event_time"],
            touch_time=c["event_time"], direction=c["direction"],
            calibrated_prob=c["calibrated_prob"], ev_r=c["ev_r"], regime=c["regime"],
            session_london=c["session_london"], session_ny=c["session_ny"], session_asia=c["session_asia"],
            entry_price=c["entry_price"], sl_price=c["sl_price"], tp_price=c["tp_price"],
            y_binary=None, cost_r=c["cost_r"],
        )
        decision = agent.evaluate(candidate, state)
        decisions_log.append({
            "time": str(c["event_time"]), "symbol": c["symbol"], "direction": c["direction"],
            "regime": c["regime"], "ev_r": c["ev_r"], "action": decision.action, "reasons": decision.reasons,
        })
        if decision.action != "APPROVE":
            continue

        fill_result = simulate_fill_and_outcome(
            dfs[c["symbol"]], c["pos"], c["direction"], c["entry_price"], c["sl_price"], c["tp_price"],
            expiry_bars=expiry_bars, max_bars=max_hold_bars,
        )
        if not fill_result["filled"]:
            unfilled.append({"time": str(c["event_time"]), "symbol": c["symbol"], "reason": fill_result["reason"]})
            continue

        lot, risk_exceeded, actual_risk_pct = calculate_realistic_lot_size(
            state.equity, decision.risk_pct, c["entry_price"], c["sl_price"], symbol=c["symbol"]
        )
        if risk_exceeded or lot <= 0:
            unfilled.append({"time": str(c["event_time"]), "symbol": c["symbol"], "reason": "lot_size_risk_exceeded"})
            continue

        candidate.y_binary = fill_result["y_binary"]
        candidate.touch_time = fill_result["touch_time"]
        agent.apply_outcome(state, decision, candidate, reward_R=c["reward_R"], risk_R=c["risk_R"])
        true_peak_equity = max(true_peak_equity, state.equity)
        true_max_dd_pct = max(true_max_dd_pct, (true_peak_equity - state.equity) / true_peak_equity * 100)

        trades.append({
            "symbol": c["symbol"], "event_time": c["event_time"], "fill_time": fill_result["fill_time"],
            "touch_time": fill_result["touch_time"], "direction": c["direction"], "regime": c["regime"],
            "calibrated_prob": c["calibrated_prob"], "ev_r": c["ev_r"], "lot": lot,
            "label": fill_result["label"], "won": fill_result["y_binary"] == 1,
            "risk_pct": decision.risk_pct, "actual_risk_pct": actual_risk_pct, "equity_after": state.equity,
        })

    trades_df = pd.DataFrame(trades)
    unfilled_df = pd.DataFrame(unfilled)

    print("\n" + "=" * 60)
    print("MULTI-INSTRUMENT REALISTIC BACKTEST RESULTS (shared risk)")
    print("=" * 60)
    print(f"Starting equity: ${starting_equity:.2f}")
    print(f"Total candidates (both instruments): {len(all_candidates)}")
    print(f"  Filled (real trades): {len(trades)}")
    print(f"  Unfilled/skipped: {len(unfilled)}")
    if not unfilled_df.empty:
        print(f"  Unfilled breakdown: {unfilled_df['reason'].value_counts().to_dict()}")

    if trades_df.empty:
        print("\nNo trades filled.")
    else:
        print(f"\nBy instrument:")
        for sym, g in trades_df.groupby("symbol"):
            print(f"  {sym}: {len(g)} trades, win rate {g['won'].mean():.1%}")

        print(f"\nWin rate (combined): {trades_df['won'].mean():.1%}")
        print(f"Ending equity: ${state.equity:.2f}")
        print(f"Return: {(state.equity / starting_equity - 1) * 100:+.1f}%")
        print(f"True all-time peak equity: ${true_peak_equity:.2f}")
        print(f"True max drawdown reached at any point (peak-to-trough): {true_max_dd_pct:.1f}%")
        print(f"Operational peak equity (since last restart): ${state.peak_equity:.2f}")
        print(f"Drawdown breaches (simulated manual restarts): {len(restarts)}")
        if restarts:
            for r in restarts:
                print(f"  Restarted at {r['time']}: equity ${r['equity_at_restart']:.2f} "
                      f"(was {r['drawdown_from_peak_pct']:.1f}% below peak)")
        print(f"Drawdown circuit breaker currently active at end of run: {state.trading_halted_drawdown}")

    trades_df.to_csv(out / "multi_backtest_trades.csv", index=False)
    unfilled_df.to_csv(out / "multi_backtest_unfilled.csv", index=False)
    pd.DataFrame(decisions_log).to_csv(out / "multi_backtest_decisions.csv", index=False)
    pd.DataFrame(restarts).to_csv(out / "multi_backtest_restarts.csv", index=False)
    print(f"\nSaved to: {out.resolve()}")

    mt5.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-instrument realistic backtest (shared risk)")
    parser.add_argument("--xauusd-model-dir", required=True)
    parser.add_argument("--nas100-model-dir", required=True)
    parser.add_argument("--months", type=int, default=1)
    parser.add_argument("--starting-equity", type=float, default=200.0)
    parser.add_argument("--out", default="./phase1_output/realistic_backtest_multi")
    parser.add_argument("--xauusd-spread-price", type=float, default=0.30)
    parser.add_argument("--xauusd-commission-price", type=float, default=0.05)
    parser.add_argument("--nas100-spread-price", type=float, default=1.0,
                         help="NAS100 trades in index points, not gold's price scale — "
                              "check your real Pepperstone NAS100 spread before trusting this default")
    parser.add_argument("--nas100-commission-price", type=float, default=0.5)
    parser.add_argument("--min-ev", type=float, default=0.05)
    parser.add_argument("--expiry-minutes", type=int, default=10)
    parser.add_argument("--max-hold-bars", type=int, default=48)
    args = parser.parse_args()

    run(
        args.xauusd_model_dir, args.nas100_model_dir, args.months, args.starting_equity,
        args.out, args.xauusd_spread_price, args.xauusd_commission_price,
        args.nas100_spread_price, args.nas100_commission_price, args.min_ev,
        args.expiry_minutes, args.max_hold_bars,
    )
