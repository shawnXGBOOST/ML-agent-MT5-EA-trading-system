"""
Diagnostic: Does shared concurrency block better opportunities?
===================================================================

Reads the decisions log + trades CSV from realistic_backtest_multi.py's last
run and checks: for every "already in a position" veto, was the candidate
that got blocked actually HIGHER EV than the trade currently occupying the
slot? If yes often, that confirms the first-come-first-served side effect
of shared concurrency is costing real trade quality, not just correctly
limiting exposure.

Usage:
    python diagnose_concurrency_effect.py --decisions phase1_output/realistic_backtest_multi/multi_backtest_decisions.csv \\
        --trades phase1_output/realistic_backtest_multi/multi_backtest_trades.csv
"""

import argparse
import pandas as pd


def run(decisions_path: str, trades_path: str):
    decisions = pd.read_csv(decisions_path, parse_dates=["time"])
    trades = pd.read_csv(trades_path, parse_dates=["event_time", "touch_time"])
    trades = trades.sort_values("event_time").reset_index(drop=True)

    concurrency_vetoes = decisions[decisions["reasons"].str.contains("already in a position", na=False)].copy()
    print(f"Total 'already in a position' vetoes: {len(concurrency_vetoes)}")

    if concurrency_vetoes.empty:
        print("No concurrency vetoes found — shared concurrency isn't the bottleneck here.")
        return

    results = []
    for _, veto in concurrency_vetoes.iterrows():
        veto_time = veto["time"]
        # Find the trade whose occupancy window [event_time, touch_time] contains this veto
        blocking = trades[(trades["event_time"] <= veto_time) & (trades["touch_time"] >= veto_time)]
        if blocking.empty:
            continue
        blocker = blocking.iloc[0]
        results.append({
            "veto_time": veto_time, "veto_symbol": veto["symbol"], "veto_ev_r": veto["ev_r"],
            "blocker_symbol": blocker["symbol"], "blocker_ev_r": blocker["ev_r"],
            "veto_was_better": veto["ev_r"] > blocker["ev_r"],
        })

    results_df = pd.DataFrame(results)
    if results_df.empty:
        print("Could not match any vetoes to a blocking trade — check touch_time coverage.")
        return

    print(f"Matched {len(results_df)} vetoes to their blocking trade.")
    print(f"\nOf these, blocked candidate had HIGHER EV than the trade holding the slot: "
          f"{results_df['veto_was_better'].sum()} / {len(results_df)} "
          f"({results_df['veto_was_better'].mean():.1%})")

    print(f"\nAvg EV of blocked candidates: {results_df['veto_ev_r'].mean():+.3f}R")
    print(f"Avg EV of trades that held the slot: {results_df['blocker_ev_r'].mean():+.3f}R")

    print("\nCross-instrument blocking (one symbol blocking the other):")
    cross = results_df[results_df["veto_symbol"] != results_df["blocker_symbol"]]
    print(f"  {len(cross)} / {len(results_df)} vetoes were cross-instrument "
          f"({len(cross)/len(results_df):.1%})")
    if not cross.empty:
        print(f"  Of cross-instrument vetoes, blocked candidate was better: "
              f"{cross['veto_was_better'].mean():.1%}")

    print("\nInterpretation:")
    if results_df["veto_was_better"].mean() > 0.35:
        print("  A meaningful fraction of blocked signals were higher-EV than what got taken —")
        print("  this supports the first-come-first-served hypothesis. A short look-ahead")
        print("  tie-break (take the higher-EV candidate within a small time window) is")
        print("  likely worth implementing.")
    else:
        print("  Blocked signals were mostly lower-EV than what got taken — shared concurrency")
        print("  seems to be working as intended here, not the main driver of the win-rate gap.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", required=True)
    parser.add_argument("--trades", required=True)
    args = parser.parse_args()
    run(args.decisions, args.trades)
