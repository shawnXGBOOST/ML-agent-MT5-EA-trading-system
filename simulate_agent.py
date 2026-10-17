"""
XAUUSD ML+Agent+EA — Phase 6b: Sequential Agent Simulation
=============================================================

Phase 4/5 measured expectancy treating every TAKE-flagged event as an
independent trade. In reality, max_concurrent_positions=1 means you can't
take every signal — many get skipped because you're already in a trade.
Daily loss halts and the drawdown circuit breaker also remove trades.

This replays your regime/holdout dataset chronologically through the actual
Agent class, respecting all of that, so you see:
  - realistic trades/day (does it hit your 1-3/day target?)
  - realistic equity curve and max drawdown
  - how often each risk gate actually fires (useful for tuning)

Usage:
    python simulate_agent.py --dataset phase1_output/holdout/final_holdout_results.csv \
        --starting-equity 150 --out phase1_output/simulation
"""

import argparse
from pathlib import Path
from collections import Counter

import pandas as pd

from agent import Agent, AgentState, RiskConfig, TradeCandidate


def run(dataset_path: str, out_dir: str, starting_equity: float):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        dataset_path,
        parse_dates=["event_time", "entry_time", "touch_time"],
    )
    df = df.sort_values("entry_time").reset_index(drop=True)

    # regime column name may be 'regime' (from regime.py output) — build it if missing
    if "regime" not in df.columns and {"trend_regime", "vol_regime"}.issubset(df.columns):
        df["regime"] = df["trend_regime"].astype(str) + "_" + df["vol_regime"].astype(str)

    config = RiskConfig()
    agent = Agent(config)
    state = AgentState(equity=starting_equity, peak_equity=starting_equity, day_start_equity=starting_equity)

    veto_reasons = Counter()
    approved = []
    equity_curve = []

    for _, row in df.iterrows():
        candidate = TradeCandidate(
            symbol="XAUUSD",
            event_time=row["event_time"],
            entry_time=row["entry_time"],
            touch_time=row["touch_time"],
            direction=row["direction"],
            calibrated_prob=row["calibrated_prob"],
            ev_r=row["ev_r"],
            regime=row.get("regime", "unknown"),
            session_london=bool(row.get("session_london", False)),
            session_ny=bool(row.get("session_ny", False)),
            session_asia=bool(row.get("session_asia", False)),
            y_binary=row["y_binary"],
            cost_r=row.get("cost_r", 0.0),
        )

        decision = agent.evaluate(candidate, state)

        if decision.action == "VETO":
            # bucket the primary reason for reporting (rounded so similar EV
            # shortfalls group together instead of each unique float being its own bucket)
            if decision.reasons:
                r = decision.reasons[0]
                if r.startswith("EV "):
                    primary = "EV below threshold"
                else:
                    primary = r.split(" — ")[0].split(" (")[0][:40]
            else:
                primary = "unknown"
            veto_reasons[primary] += 1
            continue

        agent.apply_outcome(state, decision, candidate)
        approved.append({
            "entry_time": candidate.entry_time, "direction": candidate.direction,
            "regime": candidate.regime, "risk_pct": decision.risk_pct,
            "tier": decision.confidence_tier, "won": candidate.y_binary == 1,
            "equity_after": state.equity,
        })
        equity_curve.append({"time": candidate.entry_time, "equity": state.equity})

    approved_df = pd.DataFrame(approved)
    equity_df = pd.DataFrame(equity_curve)

    print(f"Total candidate events: {len(df)}")
    print(f"Trades actually taken: {len(approved_df)}")
    print(f"\nVeto reasons breakdown:")
    for reason, count in veto_reasons.most_common():
        print(f"  {reason}: {count}")

    if approved_df.empty:
        print("\nNo trades were taken — check EV thresholds/session config, this shouldn't happen "
              "if Phase 4/5 showed a healthy TAKE rate.")
        return

    date_range_days = (df["entry_time"].max() - df["entry_time"].min()).days
    trades_per_day = len(approved_df) / max(date_range_days, 1)

    print(f"\nDate range: {date_range_days} days")
    print(f"Trades/day (avg): {trades_per_day:.2f}  (target was 1-3/day)")

    win_rate = approved_df["won"].mean()
    print(f"\nActual win rate (post risk-gate filtering): {win_rate:.1%}")
    print(f"Starting equity: ${starting_equity:.2f}")
    print(f"Ending equity: ${state.equity:.2f}")
    print(f"Return: {(state.equity / starting_equity - 1) * 100:+.1f}%")
    print(f"Peak equity: ${state.peak_equity:.2f}")
    print(f"Max drawdown reached: {state.drawdown_pct():.1f}%")
    print(f"Drawdown circuit breaker triggered: {state.trading_halted_drawdown}")

    print(f"\nTrades by confidence tier:")
    print(approved_df["tier"].value_counts())

    approved_df.to_csv(out / "simulated_trades.csv", index=False)
    equity_df.to_csv(out / "equity_curve.csv", index=False)
    print(f"\nSaved to: {out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 6b: sequential agent simulation")
    parser.add_argument("--dataset", required=True,
                         help="Path to a dataset with calibrated_prob/ev_r/regime columns "
                              "(e.g. phase1_output/holdout/final_holdout_results.csv)")
    parser.add_argument("--starting-equity", type=float, default=150.0)
    parser.add_argument("--out", default="./phase1_output/simulation")
    args = parser.parse_args()

    run(args.dataset, args.out, args.starting_equity)
