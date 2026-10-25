"""
XAUUSD ML+Agent+EA — Phase 7d: Live Runner
=============================================

Runs every 15 minutes, aligned to M15 candle close:
  1. Pull latest H4/H1/M15 bars from MT5 (via the MetaTrader5 Python package)
  2. Run Phase 1 event detection + Phase 3 features on the fresh window
  3. Check if the just-closed M15 bar is a BOS event (bullish or bearish)
  4. If so: load the saved model/calibrator, compute calibrated P(TP), EV, regime
  5. Pull live account equity/positions from MT5, feed through Agent.evaluate()
  6. If APPROVED: write the signal via SignalBridge for the EA to pick up

REQUIRES: MetaTrader5 Python package (`pip install MetaTrader5`), and MUST be
run on the same Windows machine as your MT5 terminal (the package talks to a
running terminal, it doesn't connect remotely).

I could not test this against a live MT5 terminal (none available in the
environment this was written in) — only the individual pipeline pieces this
calls have been tested on historical data. Run with --dry-run first: it does
everything except write the actual signal file, so you can watch it compute
real numbers from your live terminal without risking an order being placed.

Usage:
    python live_runner.py --model-dir phase1_output/production_model \
        --signal-path "C:\\Users\\User\\AppData\\Roaming\\MetaQuotes\\Terminal\\<ID>\\MQL5\\Files\\signal.json" \
        --symbol XAUUSD --starting-equity 150 --dry-run
"""

import argparse
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from mltrader.data import data_engine
from mltrader.features import features as features_mod
from mltrader.features import regime as regime_mod
from mltrader.agent.agent import Agent, AgentState, RiskConfig, TradeCandidate
from mltrader.agent.signal_bridge import SignalBridge
from mltrader.agent import news_filter


STATE_FILE_DEFAULT = "live_agent_state.json"


def load_state(state_path: Path, starting_equity: float) -> AgentState:
    if state_path.exists():
        with open(state_path) as f:
            d = json.load(f)
        state = AgentState(
            equity=d["equity"], peak_equity=d["peak_equity"], day_start_equity=d["day_start_equity"],
        )
        state.current_date = datetime.fromisoformat(d["current_date"]).date() if d.get("current_date") else None
        state.trading_halted_drawdown = d.get("trading_halted_drawdown", False)
        state.trading_halted_today = d.get("trading_halted_today", False)
        return state
    return AgentState(equity=starting_equity, peak_equity=starting_equity, day_start_equity=starting_equity)


def save_state(state_path: Path, state: AgentState):
    with open(state_path, "w") as f:
        json.dump({
            "equity": state.equity, "peak_equity": state.peak_equity,
            "day_start_equity": state.day_start_equity,
            "current_date": state.current_date.isoformat() if state.current_date else None,
            "trading_halted_drawdown": state.trading_halted_drawdown,
            "trading_halted_today": state.trading_halted_today,
        }, f)


def fetch_mt5_bars(symbol: str, timeframe, n_bars: int) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"Failed to fetch {symbol} bars from MT5: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["datetime"] = pd.to_datetime(df["time"], unit="s")
    df = df.rename(columns={"tick_volume": "volume"})
    return df[["datetime", "open", "high", "low", "close", "volume"]]


DECISIONS_LOG_DEFAULT = "live_decisions.jsonl"


def log_decision(decisions_log_path: Path, candidate: TradeCandidate, decision, state: AgentState):
    """Logs EVERY decision (approve or veto) — this is what the dashboard reads
    to show 'what the agent is doing', not just the trades that got placed."""
    record = {
        "logged_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": candidate.symbol,
        "event_time": str(candidate.event_time),
        "session_london": candidate.session_london,
        "session_ny": candidate.session_ny,
        "session_asia": candidate.session_asia,
        "direction": candidate.direction,
        "regime": candidate.regime,
        "calibrated_prob": round(float(candidate.calibrated_prob), 4),
        "ev_r": round(float(candidate.ev_r), 4),
        "entry_price": float(candidate.entry_price) if candidate.entry_price else None,
        "sl_price": float(candidate.sl_price) if candidate.sl_price else None,
        "tp_price": float(candidate.tp_price) if candidate.tp_price else None,
        "action": decision.action,
        "reasons": decision.reasons,
        "risk_pct": round(float(decision.risk_pct), 4) if decision.action == "APPROVE" else None,
        "confidence_tier": decision.confidence_tier,
        "equity": state.equity,
        "peak_equity": state.peak_equity,
        "drawdown_pct": state.drawdown_pct(),
        "daily_loss_pct": state.daily_loss_pct(),
    }
    with open(decisions_log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def has_real_open_position_or_order(magic_number: int, symbol: str) -> bool:
    """Checks MT5 directly for any open position or pending order under the
    given magic number FOR THIS SPECIFIC SYMBOL - matches the EA's per-symbol
    concurrency check. Each instrument gets its own independent slot; this
    is the fix for a real bug where Python's own internal AgentState never
    learns whether a written signal actually became a real position
    (apply_outcome() is only called during backtesting simulation, never in
    live_runner.py), so without this check Python could keep approving new
    trades for an instrument that already has a real position open."""
    positions = mt5.positions_get(symbol=symbol)
    orders = mt5.orders_get(symbol=symbol)
    magic_positions = [p for p in (positions or []) if p.magic == magic_number]
    magic_orders = [o for o in (orders or []) if o.magic == magic_number]
    return len(magic_positions) > 0 or len(magic_orders) > 0


def sync_concurrency_state_from_mt5(state: AgentState, magic_number: int, symbol: str):
    """Overwrites this symbol's concurrency flag with REALITY from MT5,
    instead of trusting Python's own (live-mode-unreliable) simulated state.
    Only touches this symbol's entry in the per-symbol dict - does not affect
    other instruments' independent slots."""
    if has_real_open_position_or_order(magic_number, symbol):
        state.open_positions_free_at[symbol] = datetime.now() + timedelta(hours=24)
    else:
        state.open_positions_free_at[symbol] = None


def run_one_cycle(model, calibrator, config_meta, symbol: str, agent: Agent, state: AgentState,
                   spread_price: float, commission_price: float, min_ev: float,
                   bridge: SignalBridge, dry_run: bool, decisions_log_path: Path = None,
                   magic_number: int = 778001):
    print(f"\n[{datetime.now()}] Running decision cycle...")

    m15 = fetch_mt5_bars(symbol, mt5.TIMEFRAME_M15, 500)
    h1 = fetch_mt5_bars(symbol, mt5.TIMEFRAME_H1, 500)
    h4 = fetch_mt5_bars(symbol, mt5.TIMEFRAME_H4, 500)

    m15 = data_engine.tag_sessions(m15)  # dynamic DST-aware offset, same as Phase 1
    m15 = data_engine.build_events(m15)
    h1 = data_engine.tag_sessions(h1)
    h1 = data_engine.build_events(h1)
    h4 = data_engine.tag_sessions(h4)
    h4 = data_engine.build_events(h4)

    latest = m15.iloc[-2]  # -1 is the still-forming current bar; -2 is the last CLOSED bar
    if not (latest["bos_bullish"] or latest["bos_bearish"]):
        print("  No BOS event on the last closed M15 bar. No candidate this cycle.")
        return

    direction = "long" if latest["bos_bullish"] else "short"
    print(f"  BOS event detected: {direction} at {latest['datetime']}")

    m15_feat = features_mod.build_core_features(m15)
    m15_feat = features_mod.add_previous_day_levels(m15_feat, bars_per_day=96)
    m15_feat = features_mod.merge_htf_context(m15_feat, h1, prefix="h1")
    m15_feat = features_mod.merge_htf_context(m15_feat, h4, prefix="h4")
    m15_feat["m15_bias"] = np.where(m15_feat["bos_bullish"], 1, np.where(m15_feat["bos_bearish"], -1, 0))
    m15_feat["h1_aligned_with_m15"] = (np.sign(m15_feat["m15_bias"]) == np.sign(m15_feat["h1_bias"])).astype(int)
    m15_feat["h4_aligned_with_m15"] = (np.sign(m15_feat["m15_bias"]) == np.sign(m15_feat["h4_bias"])).astype(int)
    m15_feat[["bars_since_session_london_open", "bars_since_session_ny_open"]] = (
        m15_feat[["bars_since_session_london_open", "bars_since_session_ny_open"]].fillna(-1)
    )

    row = m15_feat.iloc[-2]  # same "last closed bar" as the BOS check above
    feature_cols = config_meta["feature_cols"]
    if row[feature_cols].isna().any():
        print("  WARNING: some features are NaN (likely still warming up — need more history). Skipping.")
        return

    X = row[feature_cols].values.reshape(1, -1)
    raw_prob = model.predict_proba(X)[0, 1]
    calibrated_prob = calibrator.predict([raw_prob])[0]

    atr_at_event = row["atr"]
    tp_atr_mult, sl_atr_mult = config_meta["tp_atr_mult"], config_meta["sl_atr_mult"]
    entry_price = row["close"]  # live: limit order placed at current close, not a future "next open"
    if direction == "long":
        sl_price = entry_price - sl_atr_mult * atr_at_event
        tp_price = entry_price + tp_atr_mult * atr_at_event
    else:
        sl_price = entry_price + sl_atr_mult * atr_at_event
        tp_price = entry_price - tp_atr_mult * atr_at_event

    # Staged-exit activation/lock prices — only meaningful when the loaded
    # model was trained on the staged scheme (config_meta["is_staged_exit"]).
    # For an old-style model, these stay None and the EA places a plain
    # fixed TP/SL order (see PlacePendingOrder's backward-compat handling).
    activation_price, lock_price = None, None
    if config_meta.get("is_staged_exit"):
        act_frac = config_meta.get("activation_fraction", 0.4)
        lock_frac = config_meta.get("lock_fraction", 1.0)
        tp_distance = tp_atr_mult * atr_at_event
        if direction == "long":
            activation_price = entry_price + act_frac * tp_distance
            lock_price = entry_price + act_frac * tp_distance * lock_frac
        else:
            activation_price = entry_price - act_frac * tp_distance
            lock_price = entry_price - act_frac * tp_distance * lock_frac

    cost_r = (spread_price + commission_price) / (sl_atr_mult * atr_at_event)
    # Use the model's EMPIRICAL win/loss reward (correct for staged exit),
    # falling back to the fixed +2R/-1R assumption for old-style models via
    # the defaults train_final_model.py now always writes to meta.json.
    win_r = config_meta.get("empirical_win_r", tp_atr_mult / sl_atr_mult)
    loss_r = config_meta.get("empirical_loss_r", -1.0)
    ev_r = calibrated_prob * win_r + (1 - calibrated_prob) * loss_r - cost_r

    adx = regime_mod.compute_adx(m15).iloc[-2]
    trend_regime = regime_mod.bucket_trend(pd.Series([adx])).iloc[0]
    vol_regime = regime_mod.bucket_volatility(pd.Series([row["atr_percentile"]])).iloc[0]
    regime_label = f"{trend_regime}_{vol_regime}"

    print(f"  calibrated_prob={calibrated_prob:.3f}  ev_r={ev_r:.3f}  regime={regime_label}")

    # Pull live account state
    account_info = mt5.account_info()
    if account_info is None:
        print("  ERROR: could not read MT5 account info.")
        return
    state.equity = account_info.equity
    state.peak_equity = max(state.peak_equity, state.equity)

    # Sync concurrency with REALITY from MT5 — see has_real_open_position_or_order
    # docstring for why this is necessary in live mode.
    sync_concurrency_state_from_mt5(state, magic_number, symbol)

    candidate = TradeCandidate(
        symbol=symbol,
        event_time=latest["datetime"], entry_time=datetime.now(), touch_time=datetime.now() + timedelta(hours=12),
        direction=direction, calibrated_prob=float(calibrated_prob), ev_r=float(ev_r), regime=regime_label,
        session_london=bool(row["session_london"]), session_ny=bool(row["session_ny"]),
        session_asia=bool(row["session_asia"]),
        entry_price=float(entry_price), sl_price=float(sl_price), tp_price=float(tp_price),
        activation_price=float(activation_price) if activation_price is not None else None,
        lock_price=float(lock_price) if lock_price is not None else None,
        y_binary=None, cost_r=float(cost_r),
    )

    decision = agent.evaluate(candidate, state)
    print(f"  DECISION: {decision.action}  reasons={decision.reasons}")

    if decisions_log_path is not None:
        log_decision(decisions_log_path, candidate, decision, state)

    if decision.action == "APPROVE" and not dry_run:
        payload = bridge.write_signal(candidate, decision, symbol=symbol, expiry_minutes=10,
                                       equity_at_signal=state.equity)
        print(f"  Signal written: seq={payload['sequence_id']}")
    elif decision.action == "APPROVE" and dry_run:
        print("  [DRY RUN] Would have written signal — not writing due to --dry-run.")


def wait_until_next_m15_close():
    now = datetime.now()
    minutes_to_next = 15 - (now.minute % 15)
    next_close = (now + timedelta(minutes=minutes_to_next)).replace(second=5, microsecond=0)
    sleep_seconds = (next_close - now).total_seconds()
    print(f"Sleeping {sleep_seconds:.0f}s until next M15 close check ({next_close})...")
    time.sleep(max(sleep_seconds, 1))


def run(model_dir: str, signal_path: str, symbol: str, starting_equity: float,
        spread_price: float, commission_price: float, min_ev: float, dry_run: bool, once: bool,
        magic_number: int = 778001):
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package not installed. Run: pip install MetaTrader5")

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")
    print(f"Connected to MT5. Terminal info: {mt5.terminal_info()}")

    model_dir = Path(model_dir)
    model = joblib.load(model_dir / "xgb_model.joblib")
    calibrator = joblib.load(model_dir / "calibrator.joblib")
    with open(model_dir / "model_meta.json") as f:
        config_meta = json.load(f)
    print(f"Loaded model trained on {config_meta['trained_on_events']} events "
          f"({config_meta['data_range_start']} -> {config_meta['data_range_end']})")

    config = RiskConfig(min_ev_r=min_ev)
    agent = Agent(config)

    state_path = Path(STATE_FILE_DEFAULT)
    state = load_state(state_path, starting_equity)
    decisions_log_path = Path(DECISIONS_LOG_DEFAULT)

    bridge = SignalBridge(signal_path)

    last_news_refresh = None
    NEWS_REFRESH_INTERVAL = timedelta(hours=1)

    print(f"\n{'DRY RUN — no signals will be written' if dry_run else 'LIVE — signals WILL be written'}")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            if last_news_refresh is None or datetime.now() - last_news_refresh > NEWS_REFRESH_INTERVAL:
                try:
                    calendar = news_filter.fetch_calendar()
                    windows = news_filter.build_blackout_windows(calendar)
                    agent.set_news_blackout_windows(windows)
                    last_news_refresh = datetime.now()
                    print(f"  News calendar refreshed: {len(windows)} blackout windows active this week.")
                except Exception as e:
                    print(f"  WARNING: news calendar refresh failed ({e}) — continuing with "
                          f"previous windows (or none if this is the first attempt).")

            run_one_cycle(model, calibrator, config_meta, symbol, agent, state,
                          spread_price, commission_price, min_ev, bridge, dry_run,
                          decisions_log_path=decisions_log_path, magic_number=magic_number)
            save_state(state_path, state)
            if once:
                break
            wait_until_next_m15_close()
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 7d: live runner")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--signal-path", required=True,
                         help="Full path to signal.json inside your MT5 terminal's MQL5/Files folder")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--starting-equity", type=float, default=150.0)
    parser.add_argument("--spread-price", type=float, default=0.30)
    parser.add_argument("--commission-price", type=float, default=0.05)
    parser.add_argument("--min-ev", type=float, default=0.05)
    parser.add_argument("--dry-run", action="store_true", help="Compute everything but don't write signals")
    parser.add_argument("--once", action="store_true", help="Run a single cycle and exit (for testing)")
    parser.add_argument("--magic-number", type=int, default=778001,
                         help="Must match the EA's InpMagicNumber — used to check real MT5 "
                              "position/order state for concurrency, matching the EA exactly.")
    args = parser.parse_args()

    run(
        args.model_dir, args.signal_path, args.symbol, args.starting_equity,
        args.spread_price, args.commission_price, args.min_ev, args.dry_run, args.once,
        args.magic_number,
    )
