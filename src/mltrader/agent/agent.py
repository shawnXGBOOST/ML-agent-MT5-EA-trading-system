"""
XAUUSD ML+Agent+EA — Phase 6: Rule-Based Agent
=================================================

The agent's job (per our architecture): consume ML/EV/regime output + current
account state, apply deterministic risk gates and session/regime filters,
size the position, and output an approve/veto decision with a logged reason.
No LLM, no black box — every decision traces to an explicit rule.

Components:
  - RiskConfig: your Phase 0 hard ceilings (not overridable by anything below)
  - AgentState: running account/position state, updated after every decision
  - TradeCandidate: one BOS event's ML/EV/regime output (from Phases 2-4)
  - Agent.evaluate(): the actual decision function

Also includes simulate_agent.py-style sequential replay logic so you can see
how the agent behaves as a STREAM of decisions over time (respecting
max-concurrent-positions=1, daily loss halts, drawdown circuit breaker) —
this is different from Phase 4/5's per-trade expectancy, which treated every
candidate as independent. Concurrency and risk halts mean you don't actually
take every TAKE-flagged trade — this shows what you'd realistically get.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

# ---------------------------------------------------------------------------
# Config — your confirmed Phase 0 hard ceilings
# ---------------------------------------------------------------------------

@dataclass
class RiskConfig:
    max_risk_per_trade_pct: float = 1.5       # % of equity, hard cap
    max_daily_loss_pct: float = 4.0           # % of equity, halts new trades for the day
    max_concurrent_positions: int = 1
    max_drawdown_pct: float = 15.0            # from peak equity, halts everything until manual review
    min_ev_r: float = 0.05                    # base EV threshold (R-multiples)
    # Regimes the agent will NOT trade at all, keyed by (symbol, regime).
    # NAS100's three regimes previously listed here were removed: the
    # population-level regime win-rate check (distribution_shift.py) showed
    # only mild, gold-comparable movement (e.g. trending_low_vol -8.8%,
    # ranging_normal_vol actually IMPROVED +2.6%) — nowhere near the 20-38%
    # figures that originally justified exclusion. Those figures came from
    # realistic_backtest.py's "by regime" tables, which show win rate only
    # among trades the model actually SELECTED within that regime, not the
    # full population — a different and much noisier quantity to draw a
    # structural conclusion from. Left empty pending real demo-trade evidence.
    excluded_regimes: frozenset = frozenset()
    # Per-regime EV threshold overrides for regimes confirmed weak by the
    # official Phase 5 holdout (not just windowed-backtest slicing). Keys can
    # be a plain regime string (any symbol) or a (symbol, regime) tuple.
    regime_ev_thresholds: dict = field(default_factory=lambda: {
        ("XAUUSD", "trending_low_vol"): 0.15,
    })
    # Per-symbol base EV threshold — overrides min_ev_r for that symbol. Needed
    # because EV scales differ by instrument (different costs), so one global
    # value can't select the same quality percentile on both. Values should
    # come from ev_r quantiles of the walk-forward CV dataset, not the holdout.
    symbol_min_ev: dict = field(default_factory=dict)
    # Per-symbol cutoff for full-size ("high") tier. When set, trades at or
    # above it get full risk, trades between symbol_min_ev and it get medium
    # risk. When unset, falls back to the original fixed 0.35/0.15 tiers.
    symbol_high_tier_ev: dict = field(default_factory=dict)
    allowed_sessions: tuple = ("session_london", "session_ny")  # switchable per Phase 0
    cooldown_bars_after_skip_near_level: int = 4  # avoid repeatedly re-evaluating near-identical setups


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class AgentState:
    equity: float
    peak_equity: float
    day_start_equity: float
    current_date: Optional[datetime] = None
    # Per-symbol concurrency tracking — each instrument gets its own slot,
    # so gold having a position open no longer blocks NAS100 (and vice versa).
    # Daily loss and drawdown stay SHARED (combined account risk budget) —
    # only concurrency is now independent per instrument.
    open_positions_free_at: dict = field(default_factory=dict)  # {symbol: datetime or None}
    trades_today: int = 0
    consecutive_losses: int = 0
    trading_halted_today: bool = False
    trading_halted_drawdown: bool = False  # requires manual reset
    last_decision_time: Optional[datetime] = None

    def roll_day_if_needed(self, event_time: datetime):
        event_date = event_time.date()
        if self.current_date is None:
            # First-ever call: adopt this date without resetting day_start_equity,
            # since that was already set intentionally at construction.
            self.current_date = event_date
            return
        if event_date != self.current_date:
            self.current_date = event_date
            self.day_start_equity = self.equity
            self.trades_today = 0
            self.trading_halted_today = False

    def drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.equity) / self.peak_equity * 100

    def daily_loss_pct(self) -> float:
        if self.day_start_equity <= 0:
            return 0.0
        return max(0.0, (self.day_start_equity - self.equity) / self.day_start_equity * 100)


# ---------------------------------------------------------------------------
# Candidate + decision
# ---------------------------------------------------------------------------

@dataclass
class TradeCandidate:
    symbol: str
    event_time: datetime
    entry_time: datetime
    touch_time: datetime  # used only for simulation bookkeeping (position occupancy)
    direction: str  # "long" / "short"
    calibrated_prob: float
    ev_r: float
    regime: str
    session_london: bool
    session_ny: bool
    session_asia: bool
    entry_price: float = None
    sl_price: float = None
    tp_price: float = None
    activation_price: float = None  # staged-exit: price that triggers the stop move
    lock_price: float = None        # staged-exit: where the stop moves to once triggered
    y_binary: Optional[int] = None  # ground truth, only present in historical replay
    cost_r: float = 0.0


@dataclass
class AgentDecision:
    action: str  # "APPROVE" / "VETO"
    reasons: list = field(default_factory=list)
    risk_pct: float = 0.0
    confidence_tier: str = ""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    def __init__(self, config: RiskConfig):
        self.config = config
        self.news_blackout_windows = []  # list of (start_utc, end_utc, title) — see set_news_blackout_windows

    def set_news_blackout_windows(self, windows: list):
        """Call this periodically (e.g. once per hour) with fresh windows from
        news_filter.build_blackout_windows(). Kept separate from RiskConfig
        since these need refreshing over time, unlike the static risk caps."""
        self.news_blackout_windows = windows

    def _confidence_tier(self, ev_r: float, symbol: str = None) -> tuple:
        """Tiered position sizing — explainable, not a continuous formula fit to noise."""
        if symbol is not None and symbol in self.config.symbol_high_tier_ev:
            if ev_r >= self.config.symbol_high_tier_ev[symbol]:
                return "high", self.config.max_risk_per_trade_pct
            return "medium", self.config.max_risk_per_trade_pct * 0.73  # ~1.1%
        if ev_r >= 0.35:
            return "high", self.config.max_risk_per_trade_pct
        elif ev_r >= 0.15:
            return "medium", self.config.max_risk_per_trade_pct * 0.73  # ~1.1%
        else:
            return "low", self.config.max_risk_per_trade_pct * 0.5     # ~0.75%

    def evaluate(self, candidate: TradeCandidate, state: AgentState) -> AgentDecision:
        state.roll_day_if_needed(candidate.event_time)
        reasons = []

        # --- Hard risk gates (checked first, cannot be overridden below) ---
        if state.trading_halted_drawdown:
            return AgentDecision("VETO", ["drawdown circuit breaker active — requires manual review"])

        if state.drawdown_pct() >= self.config.max_drawdown_pct:
            state.trading_halted_drawdown = True
            return AgentDecision("VETO", [f"drawdown {state.drawdown_pct():.1f}% hit max "
                                           f"{self.config.max_drawdown_pct}% — halting until manual review"])

        if state.daily_loss_pct() >= self.config.max_daily_loss_pct:
            state.trading_halted_today = True

        if state.trading_halted_today:
            return AgentDecision("VETO", [f"daily loss limit {self.config.max_daily_loss_pct}% reached — "
                                           f"no more trades today"])

        symbol_free_at = state.open_positions_free_at.get(candidate.symbol)
        if symbol_free_at is not None and candidate.entry_time < symbol_free_at:
            return AgentDecision("VETO", [f"{candidate.symbol}: already in a position (max concurrent = "
                                          f"{self.config.max_concurrent_positions})"])

        # --- Session filter ---
        session_ok = any([
            candidate.session_london and "session_london" in self.config.allowed_sessions,
            candidate.session_ny and "session_ny" in self.config.allowed_sessions,
            candidate.session_asia and "session_asia" in self.config.allowed_sessions,
        ])
        if not session_ok:
            return AgentDecision("VETO", ["outside allowed trading sessions"])

        # --- News blackout filter ---
        if self.news_blackout_windows:
            event_time = candidate.event_time
            if event_time.tzinfo is None:
                event_time = event_time.replace(tzinfo=timezone.utc)  # assume UTC if naive
            for start, end, title in self.news_blackout_windows:
                if start <= event_time <= end:
                    return AgentDecision("VETO", [f"news blackout active: {title}"])

        # --- EV / regime filter ---
        if (candidate.symbol, candidate.regime) in self.config.excluded_regimes:
            return AgentDecision("VETO", [
                f"regime '{candidate.regime}' is excluded for {candidate.symbol} "
                f"(validated as net-losing; excluded pending further research)"
            ])

        # Base threshold: per-symbol if configured, else global.
        threshold = self.config.symbol_min_ev.get(candidate.symbol, self.config.min_ev_r)
        # Regime overrides can only TIGHTEN the base, never loosen it — so a
        # stale low regime override (e.g. an old 0.15) can't sneak under a
        # stricter per-symbol threshold.
        symbol_regime_key = (candidate.symbol, candidate.regime)
        override = None
        if symbol_regime_key in self.config.regime_ev_thresholds:
            override = self.config.regime_ev_thresholds[symbol_regime_key]
        elif candidate.regime in self.config.regime_ev_thresholds:
            override = self.config.regime_ev_thresholds[candidate.regime]
        if override is not None and override > threshold:
            threshold = override
            reasons.append(f"{candidate.symbol} regime '{candidate.regime}' — "
                            f"using stricter EV threshold {threshold}")

        if candidate.ev_r < threshold:
            return AgentDecision("VETO", [f"EV {candidate.ev_r:.3f}R below threshold {threshold:.3f}R"] + reasons)

        # --- Approved: size the position ---
        if candidate.entry_price is None or candidate.sl_price is None or candidate.tp_price is None:
            return AgentDecision("VETO", ["missing entry/SL/TP price — cannot size or execute a trade "
                                          "without real price levels (data pipeline issue, not a market condition)"])

        tier, risk_pct = self._confidence_tier(candidate.ev_r, candidate.symbol)
        reasons.append(f"EV {candidate.ev_r:.3f}R >= threshold {threshold:.3f}R, "
                        f"calibrated_prob={candidate.calibrated_prob:.3f}, regime={candidate.regime}")
        reasons.append(f"confidence tier: {tier} -> risk {risk_pct:.2f}% of equity")

        return AgentDecision("APPROVE", reasons, risk_pct=risk_pct, confidence_tier=tier)

    def apply_outcome(self, state: AgentState, decision: AgentDecision, candidate: TradeCandidate,
                       reward_R: float = 2.0, risk_R: float = 1.0):
        """Updates state after a trade resolves (for sequential simulation only —
        in live trading, the EA reports the real outcome back)."""
        if decision.action != "APPROVE":
            return
        risk_dollar = state.equity * (decision.risk_pct / 100)
        realized_R = (reward_R if candidate.y_binary == 1 else -risk_R) - candidate.cost_r
        pnl_dollar = risk_dollar * realized_R

        state.equity += pnl_dollar
        state.peak_equity = max(state.peak_equity, state.equity)
        state.trades_today += 1
        state.open_positions_free_at[candidate.symbol] = candidate.touch_time
        state.last_decision_time = candidate.entry_time
        state.consecutive_losses = state.consecutive_losses + 1 if realized_R < 0 else 0
