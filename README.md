# ML + Rule-Based Agent + MT5 EA Swing-Trading System (XAUUSD & NAS100)

An end-to-end research and execution pipeline for swing trading gold and the Nasdaq 100 on MetaTrader 5.

## Pipeline

1. **Event detection**: break-of-structure (BOS) events on M15, with H1/H4 context.
2. **Labeling**: triple-barrier labels, plus a staged "stop-lock" exit variant.
3. **Model**: XGBoost with purged + embargoed walk-forward CV and an untouched holdout.
4. **Calibration**: isotonic (gold) / Platt (NAS100), then an expected-value (EV) filter using empirical win/loss R.
5. **Agent**: rule-based risk layer. Per-symbol EV thresholds, per-symbol position slots, shared daily-loss and drawdown limits.
6. **Execution**: Python writes `signal*.json`; an MQL5 Expert Advisor places pending limit orders and manages the stop-lock.
7. **Monitoring**: outcome tracking, dashboard, weekly scheduled retraining.

## Key files

| File | Purpose |
|---|---|
| `triple_barrier_staged.py`, `features.py` | Labels and features |
| `train_baselines.py`, `train_final_model.py` | CV and production training |
| `calibrate_and_ev_staged.py`, `calibrators.py` | Calibration and EV |
| `agent.py` | Risk and decision agent |
| `live_runner_multi.py`, `signal_bridge.py` | Live loop and EA bridge |
| `XAUUSD_Agent_EA.mq5` | MT5 Expert Advisor |
| `retrain_pipeline.py`, `weekly_retrain.bat` | Automated retraining |
| `track_outcomes.py`, `dashboard_server.py` | Monitoring |

## Honest status

- Demo account only. Live sample is still small; results under ~30-50 trades are noise.
- Holdout (6 months, net of costs): roughly +0.39R/trade for gold and +0.38R/trade for NAS100.
- The original NAS100 benchmark was likely optimistic; the real edge is thinner.
- Many hypotheses were tested; repeated testing risks overfitting, so live forward results are the real judge.

## Not included

Market data, trained models, logs, and account/signal files are git-ignored. Rebuild with the data engine and training scripts.

## Disclaimer

Research project, not financial advice. Trading carries risk of loss.
