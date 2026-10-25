"""
Tests for the rule-based risk agent.

No MT5 connection or real market data — uses in-process Agent directly.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from mltrader.agent.agent import Agent, AgentState, RiskConfig, TradeCandidate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = RiskConfig(
    max_risk_pct=0.015,        # 1.5% per trade
    daily_loss_limit_pct=0.04, # 4% daily
    max_drawdown_pct=0.15,     # 15% drawdown breaker
    max_positions_per_symbol=1,
)

ACCOUNT_BALANCE = 10_000.0


def make_agent(balance=ACCOUNT_BALANCE, config=None) -> Agent:
    state = AgentState(balance=balance, equity=balance)
    return Agent(state=state, config=config or DEFAULT_CONFIG)


def make_candidate(symbol="XAUUSD", direction=1, prob=0.60,
                   ev=0.20, atr=2.0, entry=2000.0) -> TradeCandidate:
    return TradeCandidate(
        symbol=symbol,
        direction=direction,
        calibrated_prob=prob,
        ev_r=ev,
        atr_at_event=atr,
        entry_price=entry,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAgentRiskGates:

    def test_valid_trade_approved(self):
        agent = make_agent()
        candidate = make_candidate()
        decision = agent.evaluate(candidate)
        assert decision.action == "TAKE"

    def test_max_risk_per_trade(self):
        """Lot size must never risk more than 1.5% of balance."""
        agent = make_agent()
        candidate = make_candidate()
        decision = agent.evaluate(candidate)
        if decision.action == "TAKE":
            # 1 lot XAUUSD = 100 oz; sl_distance * lot_size * 100 <= 1.5% * balance
            sl_distance = candidate.atr_at_event  # 1× ATR
            max_loss = ACCOUNT_BALANCE * 0.015
            actual_loss = decision.lot_size * sl_distance * 100
            assert actual_loss <= max_loss * 1.001  # 0.1% rounding tolerance

    def test_daily_loss_limit_blocks(self):
        """Agent must refuse new trades when daily loss >= 4%."""
        agent = make_agent()
        # Simulate a 4% daily loss
        agent.state.daily_loss = ACCOUNT_BALANCE * 0.04
        candidate = make_candidate()
        decision = agent.evaluate(candidate)
        assert decision.action == "SKIP"
        assert "daily" in decision.reason.lower()

    def test_drawdown_breaker(self):
        """Agent must refuse when drawdown >= 15%."""
        agent = make_agent()
        agent.state.peak_equity = ACCOUNT_BALANCE
        agent.state.equity = ACCOUNT_BALANCE * 0.85  # exactly 15% DD
        candidate = make_candidate()
        decision = agent.evaluate(candidate)
        assert decision.action == "SKIP"
        assert "drawdown" in decision.reason.lower()

    def test_one_position_per_symbol(self):
        """Second trade on same symbol must be refused."""
        agent = make_agent()
        agent.state.open_positions["XAUUSD"] = 1
        candidate = make_candidate(symbol="XAUUSD")
        decision = agent.evaluate(candidate)
        assert decision.action == "SKIP"

    def test_different_symbols_allowed(self):
        """Positions on different symbols are independent."""
        agent = make_agent()
        agent.state.open_positions["XAUUSD"] = 1
        candidate = make_candidate(symbol="NAS100")
        decision = agent.evaluate(candidate)
        assert decision.action == "TAKE"

    def test_low_ev_skipped(self):
        """Candidate with ev_r below threshold must be skipped."""
        agent = make_agent()
        candidate = make_candidate(ev=-0.10)
        decision = agent.evaluate(candidate)
        assert decision.action == "SKIP"


class TestLotRounding:

    def test_lot_is_multiple_of_01(self):
        """Lot sizes must be multiples of 0.01."""
        agent = make_agent()
        candidate = make_candidate()
        decision = agent.evaluate(candidate)
        if decision.action == "TAKE":
            lot = decision.lot_size
            assert abs(lot - round(lot, 2)) < 1e-9

    def test_lot_minimum(self):
        """Lot size must be at least 0.01."""
        agent = make_agent(balance=500.0)  # small account
        candidate = make_candidate()
        decision = agent.evaluate(candidate)
        if decision.action == "TAKE":
            assert decision.lot_size >= 0.01

    def test_lot_does_not_exceed_risk(self):
        """Regardless of balance, lot must not risk > 1.5%."""
        for balance in [1_000, 5_000, 50_000]:
            agent = make_agent(balance=balance)
            candidate = make_candidate(atr=1.5)
            decision = agent.evaluate(candidate)
            if decision.action == "TAKE":
                max_loss = balance * 0.015
                actual_loss = decision.lot_size * candidate.atr_at_event * 100
                assert actual_loss <= max_loss * 1.001


class TestDrawdownCalculation:

    def test_drawdown_zero_at_peak(self):
        agent = make_agent()
        agent.state.peak_equity = ACCOUNT_BALANCE
        agent.state.equity = ACCOUNT_BALANCE
        dd = agent.current_drawdown()
        assert dd == pytest.approx(0.0)

    def test_drawdown_calculation(self):
        agent = make_agent()
        agent.state.peak_equity = 10_000.0
        agent.state.equity = 8_500.0
        dd = agent.current_drawdown()
        assert dd == pytest.approx(0.15, abs=1e-6)

    def test_drawdown_never_negative(self):
        """Equity above peak (e.g. after peak not updated) must give 0."""
        agent = make_agent()
        agent.state.peak_equity = 9_000.0
        agent.state.equity = 10_000.0
        dd = agent.current_drawdown()
        assert dd == pytest.approx(0.0)
