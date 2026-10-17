"""
XAUUSD + NAS100 - Multi-Symbol Live Runner
==============================================

Runs BOTH instruments through the SAME Agent + AgentState each 15-minute
cycle, per your shared-risk design decision: one shared concurrency cap
(max 1 position total, either instrument), one shared daily loss limit,
one shared drawdown circuit breaker.

This reuses run_one_cycle() from live_runner.py UNCHANGED for each
instrument - the only difference is that both calls pass the same `agent`
and `state` objects, so a position opened on one instrument correctly
blocks the other (matches the EA's now-symbol-agnostic concurrency check).

Each instrument still needs:
  - its own trained model/calibrator (from train_final_model.py, run
    separately per instrument)
  - its own signal file (so each EA instance, attached to its own chart,
    reads its own signals)

Setup required before running this:
  1. Two EA instances attached: one to an XAUUSD chart, one to a NAS100
     chart, BOTH using the same InpMagicNumber (this is what makes the
     concurrency check shared - see the updated PositionsTotalForSymbolMagic
     in XAUUSD_Agent_EA.mq5).
  2. Two separate --signal-path values, one per instrument, matching what
     each EA instance's InpSignalFilePath is set to.

Usage:
    python live_runner_multi.py \\
        --xauusd-model-dir phase1_output/production_model \\
        --xauusd-signal-path "C:\\...\\MQL5\\Files\\signal_xauusd.json" \\
        --nas100-model-dir phase1_output_nas100/production_model \\
        --nas100-signal-path "C:\\...\\MQL5\\Files\\signal_nas100.json" \\
        --starting-equity 1200
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime, timedelta

import joblib

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from agent import Agent, AgentState, RiskConfig
from signal_bridge import SignalBridge
from live_runner import run_one_cycle, load_state, save_state, wait_until_next_m15_close
import news_filter

STATE_FILE_DEFAULT = "live_agent_state_multi.json"
DECISIONS_LOG_DEFAULT = "live_decisions_multi.jsonl"


def load_instrument(model_dir: str, signal_path: str, symbol: str):
    model_dir = Path(model_dir)
    model = joblib.load(model_dir / "xgb_model.joblib")
    calibrator = joblib.load(model_dir / "calibrator.joblib")
    with open(model_dir / "model_meta.json") as f:
        config_meta = json.load(f)
    bridge = SignalBridge(signal_path)
    print(f"  {symbol}: loaded model trained on {config_meta['trained_on_events']} events "
          f"({config_meta['data_range_start']} -> {config_meta['data_range_end']})")
    return {"symbol": symbol, "model": model, "calibrator": calibrator,
            "config_meta": config_meta, "bridge": bridge}


def run(instruments: list, starting_equity: float, min_ev: float, dry_run: bool, once: bool,
        allowed_sessions: tuple, magic_number: int = 778001):
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package not installed. Run: pip install MetaTrader5")
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")
    print(f"Connected to MT5.")

    print("Loading models for each instrument...")
    loaded = [load_instrument(inst["model_dir"], inst["signal_path"], inst["symbol"])
              for inst in instruments]

    # ONE shared agent + state across all instruments - this is the actual
    # mechanism that implements shared concurrency/daily-loss/drawdown.
    symbol_min_ev = {i["symbol"]: i["min_ev"] for i in instruments if i["min_ev"] is not None}
    symbol_high_tier_ev = {i["symbol"]: i["high_tier_ev"] for i in instruments
                           if i["high_tier_ev"] is not None}
    config = RiskConfig(min_ev_r=min_ev, allowed_sessions=allowed_sessions,
                        symbol_min_ev=symbol_min_ev, symbol_high_tier_ev=symbol_high_tier_ev)
    print(f"Allowed sessions: {allowed_sessions}")
    for i in instruments:
        base = i["min_ev"] if i["min_ev"] is not None else min_ev
        tier = i["high_tier_ev"] if i["high_tier_ev"] is not None else "default (0.35/0.15)"
        print(f"  {i['symbol']}: min EV {base}  |  full-size tier at EV >= {tier}  |  "
              f"costs spread={i['spread_price']} commission={i['commission_price']}")
    agent = Agent(config)
    state_path = Path(STATE_FILE_DEFAULT)
    state = load_state(state_path, starting_equity)
    decisions_log_path = Path(DECISIONS_LOG_DEFAULT)

    last_news_refresh = None
    NEWS_REFRESH_INTERVAL = timedelta(hours=1)

    print(f"\n{'DRY RUN' if dry_run else 'LIVE'} - {len(loaded)} instruments, shared risk state")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            if last_news_refresh is None or datetime.now() - last_news_refresh > NEWS_REFRESH_INTERVAL:
                try:
                    calendar = news_filter.fetch_calendar()
                    windows = news_filter.build_blackout_windows(calendar)
                    agent.set_news_blackout_windows(windows)
                    last_news_refresh = datetime.now()
                    print(f"  News calendar refreshed: {len(windows)} blackout windows this week.")
                except Exception as e:
                    print(f"  WARNING: news refresh failed ({e}) - continuing with previous windows.")

            for inst, spec in zip(loaded, instruments):
                run_one_cycle(
                    inst["model"], inst["calibrator"], inst["config_meta"], inst["symbol"],
                    agent, state, spec["spread_price"], spec["commission_price"], min_ev, inst["bridge"],
                    dry_run, decisions_log_path=decisions_log_path, magic_number=magic_number,
                )

            save_state(state_path, state)
            if once:
                break
            wait_until_next_m15_close()
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-symbol live runner (shared risk state)")
    parser.add_argument("--xauusd-model-dir", required=True)
    parser.add_argument("--xauusd-signal-path", required=True)
    parser.add_argument("--nas100-model-dir", required=True)
    parser.add_argument("--nas100-signal-path", required=True)
    parser.add_argument("--starting-equity", type=float, default=1200.0)
    # Per-instrument costs (previously one shared value, which mis-stated NAS100's EV)
    parser.add_argument("--spread-price-xauusd", type=float, default=0.30)
    parser.add_argument("--commission-price-xauusd", type=float, default=0.05)
    parser.add_argument("--spread-price-nas100", type=float, default=1.0)
    parser.add_argument("--commission-price-nas100", type=float, default=0.5)
    parser.add_argument("--min-ev", type=float, default=0.05,
                         help="Global fallback, used only for a symbol with no per-symbol value.")
    parser.add_argument("--min-ev-xauusd", type=float, default=None,
                         help="Per-symbol EV threshold (e.g. top-40%% cutoff from CV data).")
    parser.add_argument("--min-ev-nas100", type=float, default=None)
    parser.add_argument("--high-tier-ev-xauusd", type=float, default=None,
                         help="EV at/above which XAUUSD trades get full size (e.g. top-20%% cutoff).")
    parser.add_argument("--high-tier-ev-nas100", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--sessions", type=str, default="session_london,session_ny",
                         help="Comma-separated allowed sessions. Options: session_london, "
                              "session_ny, session_asia. E.g. --sessions "
                              "session_london,session_ny,session_asia to allow all three.")
    parser.add_argument("--magic-number", type=int, default=778001,
                         help="Must match the EA's InpMagicNumber on BOTH charts — used to check "
                              "real MT5 position/order state for shared concurrency.")
    args = parser.parse_args()

    instruments = [
        {"symbol": "XAUUSD", "model_dir": args.xauusd_model_dir, "signal_path": args.xauusd_signal_path,
         "spread_price": args.spread_price_xauusd, "commission_price": args.commission_price_xauusd,
         "min_ev": args.min_ev_xauusd, "high_tier_ev": args.high_tier_ev_xauusd},
        {"symbol": "NAS100", "model_dir": args.nas100_model_dir, "signal_path": args.nas100_signal_path,
         "spread_price": args.spread_price_nas100, "commission_price": args.commission_price_nas100,
         "min_ev": args.min_ev_nas100, "high_tier_ev": args.high_tier_ev_nas100},
    ]

    run(instruments, args.starting_equity, args.min_ev, args.dry_run, args.once,
        tuple(args.sessions.split(",")), args.magic_number)
