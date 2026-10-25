"""
XAUUSD ML+Agent+EA — Realistic Backtester
=============================================

Pulls fresh data directly from MT5 (not pre-exported CSVs) and replays the
FULL live pipeline — event detection, features, calibrated ML probability,
EV, regime, and the actual Agent risk-gating logic — walking forward through
real time, exactly as live_runner.py would have seen it.

What makes this "realistic" rather than the earlier research-phase numbers:

1. PENDING ORDER FILL SIMULATION — earlier phases assumed every signal fills
   instantly at the next bar's open. This backtester instead simulates what
   your EA actually does: place a pending limit order, wait up to
   `expiry_minutes`, and only count it as a trade if price actually reaches
   the entry level in that window. Signals that would have expired unfilled
   are tracked separately and do NOT count as trades or block concurrency.

2. REAL LOT-SIZE ROUNDING — position size is rounded to the broker's actual
   lot step (0.01) and capped by InpMaxLotSize, exactly like the EA. On a
   small account, this means some approved signals get skipped because even
   the minimum lot exceeds the risk cap (same constraint we found live).

3. FULL AGENT RISK GATING — every hard rule (concurrency, daily loss,
   drawdown breaker, session filter, EV/regime threshold) runs exactly as
   Agent.evaluate() does live, via sequential state, not just averaged
   per-trade statistics.

4. REAL SPREAD + COMMISSION COSTS — pulled from the model's cost formula,
   same as the live EV engine.

Usage:
    python realistic_backtest.py --model-dir phase1_output/production_model \\
        --months 3 --starting-equity 200 --out phase1_output/realistic_backtest
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

import data_engine
import features as features_mod
import regime as regime_mod
from agent import Agent, AgentState, RiskConfig, TradeCandidate


def fetch_mt5_range(symbol: str, timeframe, months_back: int, warmup_bars: int) -> pd.DataFrame:
    """Fetch enough history to cover `months_back` of backtest PLUS a warmup
    buffer so features (ATR percentile, prev-day levels, etc.) aren't NaN at
    the very start of the backtest window."""
    date_to = datetime.now()
    date_from = date_to - timedelta(days=months_back * 31 + 30)  # extra buffer month for warmup
    rates = mt5.copy_rates_range(symbol, timeframe, date_from, date_to)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"Failed to fetch {symbol} range from MT5: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["datetime"] = pd.to_datetime(df["time"], unit="s")
    df = df.rename(columns={"tick_volume": "volume"})
    return df[["datetime", "open", "high", "low", "close", "volume"]]


def build_full_feature_series(m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame) -> pd.DataFrame:
    """Vectorized version of what live_runner.py does per-cycle — computed
    ONCE across the whole backtest window instead of recomputing every 15
    minutes, since we already have all the historical data up front."""
    m15 = data_engine.tag_sessions(m15)
    m15 = data_engine.build_events(m15)
    h1 = data_engine.tag_sessions(h1)
    h1 = data_engine.build_events(h1)
    h4 = data_engine.tag_sessions(h4)
    h4 = data_engine.build_events(h4)

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

    adx = regime_mod.compute_adx(m15_feat)
    m15_feat["trend_regime"] = regime_mod.bucket_trend(adx)
    m15_feat["vol_regime"] = regime_mod.bucket_volatility(m15_feat["atr_percentile"])
    m15_feat["regime"] = m15_feat["trend_regime"].astype(str) + "_" + m15_feat["vol_regime"].astype(str)

    return m15_feat.reset_index(drop=True)


def simulate_fill_and_outcome(df: pd.DataFrame, event_idx: int, direction: str,
                               entry_price: float, sl_price: float, tp_price: float,
                               expiry_bars: int, max_bars: int, ambiguous_resolution: str = "conservative"):
    """Realistic pending-limit-order simulation:
    1. Check if price actually reaches entry_price within expiry_bars (order
       placed at the bar AFTER the event bar, matching live_runner's timing)
    2. If filled, walk forward using triple-barrier logic from the fill point
    Returns a dict with fill status and (if filled) the outcome.
    """
    n = len(df)
    order_placed_idx = event_idx + 1
    if order_placed_idx >= n:
        return {"filled": False, "reason": "end_of_data"}

    fill_window_end = min(order_placed_idx + expiry_bars, n - 1)
    fill_idx = None
    for j in range(order_placed_idx, fill_window_end + 1):
        row = df.iloc[j]
        if direction == "long":  # BUY LIMIT fills when price dips to/through entry
            if row["low"] <= entry_price:
                fill_idx = j
                break
        else:  # SELL LIMIT fills when price rallies to/through entry
            if row["high"] >= entry_price:
                fill_idx = j
                break

    if fill_idx is None:
        return {"filled": False, "reason": "expired_unfilled"}

    time_barrier_idx = min(fill_idx + max_bars, n - 1)
    if time_barrier_idx <= fill_idx:
        return {"filled": False, "reason": "insufficient_future_data"}

    window = df.iloc[fill_idx:time_barrier_idx + 1]
    label, touch_idx = None, None
    for j, row in window.iterrows():
        hit_tp = row["high"] >= tp_price if direction == "long" else row["low"] <= tp_price
        hit_sl = row["low"] <= sl_price if direction == "long" else row["high"] >= sl_price
        if hit_tp and hit_sl:
            label = "SL" if ambiguous_resolution == "conservative" else "TP"
            touch_idx = j
            break
        elif hit_tp:
            label, touch_idx = "TP", j
            break
        elif hit_sl:
            label, touch_idx = "SL", j
            break
    if label is None:
        label, touch_idx = "TIMEOUT", time_barrier_idx

    return {
        "filled": True, "fill_idx": fill_idx, "fill_time": df.at[fill_idx, "datetime"],
        "label": label, "touch_idx": touch_idx, "touch_time": df.at[touch_idx, "datetime"],
        "y_binary": 1 if label == "TP" else 0,
    }


def calculate_realistic_lot_size(equity: float, risk_pct: float, entry_price: float, sl_price: float,
                                   symbol: str = None, lot_step: float = 0.01, min_lot: float = 0.01,
                                   max_lot: float = 0.05, ignore_risk_cap: bool = False):
    """Mirrors the EA's CalculateLotSize — rounds DOWN to broker lot step,
    refuses the trade if even minimum lot exceeds 1.5x intended risk (same
    small-account reality check we hit live), UNLESS ignore_risk_cap is set.

    IMPORTANT: uses REAL broker contract specs (tick value/tick size) fetched
    via MT5, not a hardcoded contract-size assumption. Different instruments
    have wildly different contract conventions (e.g. gold's 100oz/lot vs an
    index CFD's $1-or-so per point per lot) — hardcoding one instrument's
    convention for another silently produces wrong risk numbers.
    """
    risk_dollar = equity * (risk_pct / 100)
    sl_distance = abs(entry_price - sl_price)
    if sl_distance <= 0:
        return 0.0, True, risk_pct

    if symbol is not None and mt5 is not None:
        info = mt5.symbol_info(symbol)
        if info is None or info.trade_tick_size <= 0:
            raise RuntimeError(f"Could not fetch valid symbol_info for '{symbol}' — "
                                f"cannot compute real contract-based risk without it.")
        tick_value = info.trade_tick_value
        tick_size = info.trade_tick_size
        loss_per_lot = (sl_distance / tick_size) * tick_value
    else:
        # Fallback ONLY for contexts with no MT5 connection (shouldn't normally
        # happen in a live/backtest run) — assumes gold's 100oz convention.
        loss_per_lot = sl_distance * 100

    raw_lot = risk_dollar / loss_per_lot
    lot = np.floor(raw_lot / lot_step) * lot_step
    lot = min(lot, max_lot)

    if lot < min_lot:
        min_lot_risk_dollar = loss_per_lot * min_lot
        min_lot_risk_pct = (min_lot_risk_dollar / equity * 100) if equity > 0 else 999
        if min_lot_risk_pct > risk_pct * 1.5:
            if ignore_risk_cap:
                return min_lot, False, min_lot_risk_pct
            return 0.0, True, risk_pct
        return min_lot, False, min_lot_risk_pct

    return lot, False, risk_pct


def run(model_dir: str, months: int, starting_equity: float, symbol: str, out_dir: str,
        spread_price: float, commission_price: float, min_ev: float,
        expiry_minutes: int, max_hold_bars: int, ignore_lot_risk_cap: bool = False):
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package not installed. Run: pip install MetaTrader5")
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Fetching ~{months} months of {symbol} data from MT5 (plus warmup buffer)...")
    m15 = fetch_mt5_range(symbol, mt5.TIMEFRAME_M15, months, warmup_bars=500)
    h1 = fetch_mt5_range(symbol, mt5.TIMEFRAME_H1, months, warmup_bars=500)
    h4 = fetch_mt5_range(symbol, mt5.TIMEFRAME_H4, months, warmup_bars=500)
    print(f"  M15: {len(m15)} bars, {h1.shape[0]} H1 bars, {h4.shape[0]} H4 bars")

    print("Building full feature series (events, features, regime)...")
    m15_feat = build_full_feature_series(m15, h1, h4)

    backtest_start = datetime.now() - timedelta(days=months * 31)
    backtest_df = m15_feat[m15_feat["datetime"] >= backtest_start].reset_index(drop=True)
    # Re-index against the FULL m15_feat (not the filtered slice) so
    # simulate_fill_and_outcome can look at bars beyond the backtest window
    # end for barrier resolution — use original integer positions.
    full_df = m15_feat
    start_pos = full_df.index[full_df["datetime"] >= backtest_start][0]

    print(f"Backtest window: {backtest_df['datetime'].min()} -> {backtest_df['datetime'].max()} "
          f"({len(backtest_df)} bars)")
    if ignore_lot_risk_cap:
        print("WARNING: --ignore-lot-risk-cap is ON — trades below will use minimum lot even when "
              "that exceeds your intended risk %. Check 'actual_risk_pct' per trade in the output CSV. "
              "This is exploratory only, not a recommendation for real capital.")

    model_dir = Path(model_dir)
    model = joblib.load(model_dir / "xgb_model.joblib")
    calibrator = joblib.load(model_dir / "calibrator.joblib")
    with open(model_dir / "model_meta.json") as f:
        config_meta = json.load(f)
    feature_cols = config_meta["feature_cols"]
    tp_atr_mult, sl_atr_mult = config_meta["tp_atr_mult"], config_meta["sl_atr_mult"]
    reward_R, risk_R = tp_atr_mult / sl_atr_mult, 1.0

    config = RiskConfig(min_ev_r=min_ev)
    agent = Agent(config)
    state = AgentState(equity=starting_equity, peak_equity=starting_equity, day_start_equity=starting_equity)

    trades = []
    unfilled = []
    max_drawdown_seen = 0.0
    decisions_log = []

    print("\nWalking forward through time (this is the real backtest loop)...")
    for pos in range(start_pos, len(full_df) - max(max_hold_bars, 5)):
        row = full_df.iloc[pos]
        if not (row["bos_bullish"] or row["bos_bearish"]):
            continue
        if row[feature_cols].isna().any():
            continue

        direction = "long" if row["bos_bullish"] else "short"
        X = row[feature_cols].values.reshape(1, -1).astype(float)
        raw_prob = model.predict_proba(X)[0, 1]
        calibrated_prob = float(calibrator.predict([raw_prob])[0])

        atr_at_event = row["atr"]
        if pd.isna(atr_at_event) or atr_at_event <= 0:
            continue
        entry_price = row["close"]
        if direction == "long":
            sl_price = entry_price - sl_atr_mult * atr_at_event
            tp_price = entry_price + tp_atr_mult * atr_at_event
        else:
            sl_price = entry_price + sl_atr_mult * atr_at_event
            tp_price = entry_price - tp_atr_mult * atr_at_event

        cost_r = (spread_price + commission_price) / (sl_atr_mult * atr_at_event)
        ev_r = calibrated_prob * reward_R - (1 - calibrated_prob) * risk_R - cost_r

        candidate = TradeCandidate(
            symbol=symbol,
            event_time=row["datetime"], entry_time=row["datetime"], touch_time=row["datetime"],
            direction=direction, calibrated_prob=calibrated_prob, ev_r=ev_r, regime=row["regime"],
            session_london=bool(row["session_london"]), session_ny=bool(row["session_ny"]),
            session_asia=bool(row["session_asia"]),
            entry_price=float(entry_price), sl_price=float(sl_price), tp_price=float(tp_price),
            y_binary=None, cost_r=cost_r,
        )

        decision = agent.evaluate(candidate, state)
        decisions_log.append({
            "time": str(row["datetime"]), "direction": direction, "regime": row["regime"],
            "calibrated_prob": calibrated_prob, "ev_r": ev_r, "action": decision.action,
            "reasons": decision.reasons,
        })
        if decision.action != "APPROVE":
            continue

        # Realistic fill simulation — this is what makes this backtest different
        # from the earlier research-phase numbers
        expiry_bars = max(1, round(expiry_minutes / 15))
        fill_result = simulate_fill_and_outcome(
            full_df, pos, direction, entry_price, sl_price, tp_price,
            expiry_bars=expiry_bars, max_bars=max_hold_bars,
        )

        if not fill_result["filled"]:
            unfilled.append({"time": str(row["datetime"]), "direction": direction, "reason": fill_result["reason"]})
            continue

        lot, risk_exceeded, actual_risk_pct = calculate_realistic_lot_size(
            state.equity, decision.risk_pct, entry_price, sl_price, symbol=symbol,
            ignore_risk_cap=ignore_lot_risk_cap
        )
        if risk_exceeded or lot <= 0:
            unfilled.append({"time": str(row["datetime"]), "direction": direction, "reason": "lot_size_risk_exceeded"})
            continue

        candidate.y_binary = fill_result["y_binary"]
        candidate.touch_time = fill_result["touch_time"]
        agent.apply_outcome(state, decision, candidate, reward_R=reward_R, risk_R=risk_R)
        max_drawdown_seen = max(max_drawdown_seen, state.drawdown_pct())

        trades.append({
            "event_time": row["datetime"], "fill_time": fill_result["fill_time"],
            "touch_time": fill_result["touch_time"], "direction": direction,
            "regime": row["regime"], "calibrated_prob": calibrated_prob, "ev_r": ev_r,
            "entry_price": entry_price, "sl_price": sl_price, "tp_price": tp_price,
            "lot": lot, "label": fill_result["label"], "won": fill_result["y_binary"] == 1,
            "risk_pct": decision.risk_pct, "actual_risk_pct": actual_risk_pct, "equity_after": state.equity,
        })

    trades_df = pd.DataFrame(trades)
    unfilled_df = pd.DataFrame(unfilled)

    print("\n" + "=" * 60)
    print("REALISTIC BACKTEST RESULTS")
    print("=" * 60)
    print(f"Period: {backtest_df['datetime'].min().date()} -> {backtest_df['datetime'].max().date()}")
    print(f"Starting equity: ${starting_equity:.2f}")
    print(f"Approved signals: {len(trades) + len(unfilled)}")
    print(f"  Filled (real trades): {len(trades)}")
    print(f"  Unfilled/skipped: {len(unfilled)}")
    if len(unfilled) > 0:
        print(f"  Unfilled breakdown: {unfilled_df['reason'].value_counts().to_dict()}")

    if trades_df.empty:
        print("\nNo trades were actually filled — check EV thresholds and expiry window, "
              "this shouldn't happen if the pipeline is working correctly.")
    else:
        days = (backtest_df["datetime"].max() - backtest_df["datetime"].min()).days
        print(f"\nTrades/day: {len(trades_df) / max(days, 1):.2f}")
        print(f"Win rate: {trades_df['won'].mean():.1%}")
        print(f"Ending equity: ${state.equity:.2f}")
        print(f"Return: {(state.equity / starting_equity - 1) * 100:+.1f}%")
        print(f"Peak equity: ${state.peak_equity:.2f}")
        print(f"Max drawdown reached at any point (true peak-to-trough): {max_drawdown_seen:.1f}%")
        print(f"Drawdown at END of run (final equity vs final peak): {state.drawdown_pct():.1f}%")
        print(f"Drawdown circuit breaker triggered: {state.trading_halted_drawdown}")

        print(f"\nBy regime:")
        for regime, g in trades_df.groupby("regime"):
            print(f"  {regime:25s} n={len(g):3d}  win%={g['won'].mean():.1%}")

    trades_df.to_csv(out / "backtest_trades.csv", index=False)
    unfilled_df.to_csv(out / "backtest_unfilled_signals.csv", index=False)
    pd.DataFrame(decisions_log).to_csv(out / "backtest_all_decisions.csv", index=False)
    print(f"\nSaved to: {out.resolve()}")

    mt5.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Realistic backtester with fill simulation")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--months", type=int, default=3)
    parser.add_argument("--starting-equity", type=float, default=200.0)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--out", default="./phase1_output/realistic_backtest")
    parser.add_argument("--spread-price", type=float, default=0.30)
    parser.add_argument("--commission-price", type=float, default=0.05)
    parser.add_argument("--min-ev", type=float, default=0.05)
    parser.add_argument("--expiry-minutes", type=int, default=10)
    parser.add_argument("--max-hold-bars", type=int, default=48)
    parser.add_argument("--ignore-lot-risk-cap", action="store_true",
                         help="Use minimum lot even when it exceeds 1.5x intended risk — "
                              "EXPLORATORY ONLY, does not respect your real risk discipline. "
                              "Shows actual_risk_pct per trade so you can see the real overshoot.")
    args = parser.parse_args()

    run(
        args.model_dir, args.months, args.starting_equity, args.symbol, args.out,
        args.spread_price, args.commission_price, args.min_ev,
        args.expiry_minutes, args.max_hold_bars, args.ignore_lot_risk_cap,
    )
