# ML Agent MT5 EA — Trading System

An end-to-end ML swing-trading pipeline for MetaTrader 5 (MT5).
Detects Break-of-Structure (BOS) events on M15 bars, labels them with triple-barrier exits (staged stop-lock), trains an XGBoost classifier, calibrates probabilities, filters by expected value, and routes live decisions through a rule-based risk agent that writes signal files consumed by an MQL5 EA.

---

## Pipeline overview

```
MT5 export (M15 OHLCV)
        │
        ▼
  BOS event detection          data_engine.py
        │
        ▼
  Triple-barrier labeling       labeling/triple_barrier.py
  (staged stop-lock exit)
        │
        ▼
  Feature engineering           features/features.py + regime.py
        │
        ▼
  XGBoost training              models/train_final_model.py
  (purged + embargoed CV)
        │
        ▼
  Isotonic / Platt calibration  calibration/calibration.py
        │
        ▼
  EV filter                     calibration/calibration.py
  (per-symbol thresholds)
        │
        ▼
  Rule-based risk agent         agent/agent.py
  (lot sizing, risk gates)
        │
        ▼
  Signal files                  signal.json / signal_nas100.json
        │
        ▼
  MQL5 EA reads signals         mql5/XAUUSD_Agent_EA.mq5
```

---

## Directory layout

```
.
├── src/
│   └── mltrader/
│       ├── agent/          rule-based risk agent + signal bridge + news filter
│       ├── calibration/    probability calibration + EV filter
│       ├── data/           MT5 data pull + data engine
│       ├── execution/      live runner (single & multi-symbol)
│       ├── features/       feature engineering + regime detection
│       ├── labeling/       triple-barrier labeling (staged exit)
│       ├── models/         XGBoost training (baselines + final)
│       └── monitoring/     (placeholder for future alerting)
├── mql5/
│   └── XAUUSD_Agent_EA.mq5   MQL5 EA — reads signal files, sends orders
├── scripts/
│   └── retrain_pipeline.py   orchestrates the full retrain sequence
├── tests/                    pytest suite (no market data required)
├── archive/                  legacy / experiment scripts (not in pipeline)
├── pyproject.toml
└── weekly_retrain.bat        Windows scheduled task entry point
```

---

## Installation

> Requires Python ≥ 3.10 and a running MetaTrader 5 terminal (for live mode).

```bash
# 1. Clone
git clone https://github.com/shawnXGBOOST/ML-agent-MT5-EA-trading-system.git
cd ML-agent-MT5-EA-trading-system

# 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# 3. Install
pip install -e ".[dev]"
```

---

## Running the pipeline

### One-off retrain
```bash
python scripts/retrain_pipeline.py
```

### Automated weekly retrain (Windows)
The repo includes `weekly_retrain.bat`. Register it with Task Scheduler:
```bat
schtasks /create /tn "MLTrader weekly retrain" /tr "C:\...\weekly_retrain.bat" /sc weekly /d SUN /st 03:00
```

### Live runner (multi-symbol)
```bash
python -m mltrader.execution.live_runner_multi
```
MT5 must be running and logged in. No credentials are read from files —
`mt5.initialize()` attaches to the running terminal.

---

## What is NOT included

| Excluded | Why | How to rebuild |
|---|---|---|
| Raw OHLCV data (`*.csv`) | Too large; changes daily | Export M15 bars from Pepperstone MT5 terminal |
| Trained models (`*.pkl`, `*.joblib`) | Contain fitted weights; retrain instead | `python scripts/retrain_pipeline.py` |
| Signal files (`signal*.json`) | Live-trading state | Generated automatically by `live_runner_multi.py` |
| Agent state (`live_agent_state*.json`) | Per-session risk tracking | Generated automatically |
| Logs | Session-specific | Run live runner; logs appear in `logs/` |

**Data source:** Pepperstone MT5 terminal, M15 OHLCV for XAUUSD and NAS100.

---

## Running tests

```bash
pytest tests/ -v
```

All tests use synthetic data — no MT5 connection, no CSV files required.

---

## Honest status

This bot runs on a live Pepperstone demo account (magic number 778001).
Below is a snapshot from the most recent holdout evaluation:

| Symbol | Holdout trades | Win rate | Max drawdown |
|--------|---------------|----------|--------------|
| XAUUSD | *(update after holdout run)* | | |
| NAS100 | *(update after holdout run)* | | |

**Caveats:**
- Walk-forward CV uses purging + embargo (48 bars) to prevent leakage.
- The holdout set was untouched during model development.
- Staged stop-lock means "wins" include LOCKED_STOP exits (~0.8R), not only FULL_TP (2.0R).
- Live performance may differ from backtest due to spread, slippage, and regime shift.
- This is a portfolio/learning project, not financial advice.

---

## EV thresholds (current production)

| Symbol | Top-40% threshold | Top-20% threshold |
|--------|------------------|------------------|
| XAUUSD | 0.35 | 0.512 |
| NAS100 | 0.463 | 0.584 |

---

## Risk gates (hard-coded in agent)

- Max risk per trade: **1.5%** of account balance
- Daily loss limit: **4%**
- Drawdown breaker: **15%**
- Max positions per symbol: **1**
