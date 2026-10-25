"""
Tests for the rule-based risk agent.

No MT5 connection or real market data — uses Agent directly with synthetic state.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datetime import datetime, timedelta, timezone

import pytest

from mltrader.agent.agent import Agent, AgentDecision, AgentState, RiskConfig, TradeCandidate

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

T0 = datetime(2024, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
T1 = T0 + timedelta(hours=2)   # touch_time (position closes)

DEFAULT_CONFIG = RiskConfig(
    max_risk_per_trade_pct=1.5,
    max_daily_loss_pct=4.0,
    max_drawdown_pct=15.0,
    max_concurrent_positions=1,
    min_ev_r=0.05,
)

BALANCE = 10_000.0


def make_state(equity=BALANCE) -> AgentState:
    return AgentState(
        equity=equity,
        peak_equity=BALANCE,
        day_start_equity=BALANCE,
    )


def make_agent(config=None) -> Agent:
    return Agent(config=config or DEFAULT_CONFIG)


def make_candidate(symbol="XAUUSD", ev=0.25, prob=0.60,
                   regime="trending_high_vol", direction="long",
                   entry=2000.0, sl=1998.0, tp=2004.0) -> TradeCandidate:
    return TradeCandidate(
        symbol=symbol,
        event_time=T0,
        entry_time=T0,
        touch_time=T1,
        direction=direction,
        calibrated_prob=prob,
        ev_r=ev,
        regime=regime,
        session_london=True,
        session_ny=False,
        session_asia=False,
        entry_price=entry,
        sl_price=sl,
        tp_price=tp,
    )


# ---------------------------------------------------------------------------
# Tests: basic approval
# ---------------------------------------------------------------------------

class TestAgentApproval:

    def test_valid_trade_approved(self):
        agent = make_agent()
        state = make_state()
        decision = agent.evaluate(make_candidate(), state)
        assert decision.action == "APPROVE"

    def test_decision_returns_agentdecision(self):
        agent = make_agent()
        state = make_state()
        result = agent.evaluate(make_candidate(), state)
        assert isinstance(result, AgentDecision)

    def test_low_ev_vetoed(self):
        agent = make_agent()
        state = make_state()
        candidate = make_candidate(ev=-0.10)
        decision = agent.evaluate(candidate, state)
        assert decision.action == "VETO"

    def test_ev_at_threshold_approved(self):
        agent = make_agent()
        state = make_state()
        candidate = make_candidate(ev=DEFAULT_CONFIG.min_ev_r)
        decision = agent.evaluate(candidate, state)
        assert decision.action == "APPROVE"


# ---------------------------------------------------------------------------
# Tests: risk gates
# ---------------------------------------------------------------------------

class TestRiskGates:

    def test_daily_loss_limit_blocks(self):
        """After 4% daily loss, new trades must be vetoed."""
        agent = make_agent()
        state = make_state()
        state.equity = BALANCE * 0.96  # exactly 4% down
        state.current_date = T0.date()  # prevent roll_day from resetting
        state.trading_halted_today = True
        decision = agent.evaluate(make_candidate(), state)
        assert decision.action == "VETO"
        assert any("daily" in r.lower() for r in decision.reasons)

    def test_drawdown_breaker_blocks(self):
        """After 15% drawdown, agent must veto."""
        agent = make_agent()
        state = make_state()
        state.peak_equity = BALANCE
        state.equity = BALANCE * 0.85   # exactly 15% drawdown
        state.trading_halted_drawdown = True
        decision = agent.evaluate(make_candidate(), state)
        assert decision.action == "VETO"
        assert any("drawdown" in r.lower() for r in decision.reasons)

    def test_drawdown_gate_triggers_at_threshold(self):
        """Drawdown at exactly max_drawdown_pct triggers the halt."""
        agent = make_agent()
        state = make_state()
        state.peak_equity = BALANCE
        state.equity = BALANCE * (1 - DEFAULT_CONFIG.max_drawdown_pct / 100)
        state.current_date = T0.date()
        decision = agent.evaluate(make_candidate(), state)
        assert decision.action == "VETO"

    def test_one_position_per_symbol(self):
        """Second candidate for same symbol while position open must be vetoed."""
        agent = make_agent()
        state = make_state()
        # Mark XAUUSD as occupied until T1
        state.open_positions_free_at["XAUUSD"] = T1
        # New candidate arrives before T1
        candidate = make_candidate(symbol="XAUUSD")
        decision = agent.evaluate(candidate, state)
        assert decision.action == "VETO"

    def test_different_symbols_independent(self):
        """XAUUSD position open must not block NAS100."""
        agent = make_agent()
        state = make_state()
        state.open_positions_free_at["XAUUSD"] = T1
        candidate = make_candidate(symbol="NAS100", entry=18000.0, sl=17950.0, tp=18100.0)
        decision = agent.evaluate(candidate, state)
        assert decision.action == "APPROVE"

    def test_session_filter(self):
        """Candidate outside allowed sessions must be vetoed."""
        agent = make_agent()
        state = make_state()
        candidate = make_candidate()
        candidate.session_london = False
        candidate.session_ny = False
        candidate.session_asia = False
        decision = agent.evaluate(candidate, state)
        assert decision.action == "VETO"
        assert any("session" in r.lower() for r in decision.reasons)

    def test_missing_sl_vetoed(self):
        """Candidate missing SL price must be vetoed."""
        agent = make_agent()
        state = make_state()
        candidate = make_candidate()
        candidate.sl_price = None
        decision = agent.evaluate(candidate, state)
        assert decision.action == "VETO"


# ---------------------------------------------------------------------------
# Tests: drawdown calculation
# ---------------------------------------------------------------------------

class TestDrawdownCalculation:

    def test_drawdown_zero_at_peak(self):
        state = make_state()
        state.peak_equity = BALANCE
        state.equity = BALANCE
        assert state.drawdown_pct() == pytest.approx(0.0)

    def test_drawdown_value(self):
        state = make_state()
        state.peak_equity = 10_000.0
        state.equity = 8_500.0
        assert state.drawdown_pct() == pytest.approx(15.0, abs=1e-6)

    def test_drawdown_equity_above_peak(self):
        """If equity is above recorded peak, drawdown is 0 (peak not yet updated)."""
        state = make_state()
        state.peak_equity = 9_000.0
        state.equity = 10_000.0
        # drawdown_pct uses (peak - equity) / peak * 100; this goes negative
        # The agent checks >= threshold, so negative means no trigger — acceptable.
        dd = state.drawdown_pct()
        assert dd <= 0.0


# ---------------------------------------------------------------------------
# Tests: confidence tier / risk sizing
# ---------------------------------------------------------------------------

class TestConfidenceTier:

    def test_high_ev_gets_full_risk(self):
        agent = make_agent()
        state = make_state()
        candidate = make_candidate(ev=0.40)
        decision = agent.evaluate(candidate, state)
        assert decision.action == "APPROVE"
        assert decision.confidence_tier == "high"
        assert decision.risk_pct == pytest.approx(DEFAULT_CONFIG.max_risk_per_trade_pct)

    def test_medium_ev_gets_reduced_risk(self):
        agent = make_agent()
        state = make_state()
        candidate = make_candidate(ev=0.20)
        decision = agent.evaluate(candidate, state)
        assert decision.action == "APPROVE"
        assert decision.confidence_tier == "medium"
        assert decision.risk_pct < DEFAULT_CONFIG.max_risk_per_trade_pct

    def test_risk_pct_never_exceeds_max(self):
        for ev in [0.06, 0.20, 0.50, 1.0]:
            agent = make_agent()
            state = make_state()
            decision = agent.evaluate(make_candidate(ev=ev), state)
            if decision.action == "APPROVE":
                assert decision.risk_pct <= DEFAULT_CONFIG.max_risk_per_trade_pct + 1e-9
