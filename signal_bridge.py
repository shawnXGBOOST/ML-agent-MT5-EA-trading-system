"""
XAUUSD ML+Agent+EA — Phase 7a: Signal Bridge (Python side)
==============================================================

Writes agent APPROVE decisions to a JSON file the MT5 EA polls. Per the
blueprint: "signal bridge files must be timestamped and validated to avoid
stale signals." This writer includes:
  - a monotonically increasing sequence_id (EA ignores anything it's already
    processed, and ignores anything with a LOWER sequence_id than last seen —
    protects against a stale file being re-read after a restart)
  - an ISO timestamp (EA rejects anything older than a configurable max age —
    protects against the EA acting on a signal that's gone stale because the
    Python side hung, crashed, or fell behind)
  - atomic write (write to a temp file, then rename) so the EA never reads a
    half-written file mid-write

This does NOT replace the Agent's own decision logic — it's purely the
transport layer between Agent.evaluate() and the EA.
"""

import json
import os
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from agent import TradeCandidate, AgentDecision


class SignalBridge:
    def __init__(self, signal_path: str, state_path: str = None, signal_log_path: str = None):
        self.signal_path = Path(signal_path)
        self.state_path = Path(state_path) if state_path else self.signal_path.with_suffix(".state.json")
        self.signal_log_path = Path(signal_log_path) if signal_log_path else self.signal_path.with_suffix(".log.jsonl")
        self._sequence_id = self._load_last_sequence_id()

    def _load_last_sequence_id(self) -> int:
        if self.state_path.exists():
            try:
                with open(self.state_path) as f:
                    return json.load(f).get("last_sequence_id", 0)
            except Exception:
                return 0
        return 0

    def _save_sequence_id(self, seq_id: int):
        with open(self.state_path, "w") as f:
            json.dump({"last_sequence_id": seq_id}, f)

    def write_signal(
        self,
        candidate: TradeCandidate,
        decision: AgentDecision,
        symbol: str = "XAUUSD",
        expiry_minutes: int = 10,
        equity_at_signal: float = None,
    ):
        """Write an APPROVE decision to the bridge file. Does nothing for VETO
        decisions — the EA only ever sees actionable signals, never has to
        interpret a veto reason."""
        if decision.action != "APPROVE":
            return

        self._sequence_id += 1
        # MQL5's StringToTime() doesn't parse the ISO 'T' separator correctly —
        # it silently defaults to midnight instead of erroring, which is why the
        # EA computed a wildly wrong signal age. Use MQL5's native datetime format
        # instead (space separator, no timezone suffix — always UTC by convention
        # here, and the EA compares against TimeGMT()).
        written_at = datetime.now(timezone.utc).strftime("%Y.%m.%d %H:%M:%S")
        payload = {
            "sequence_id": self._sequence_id,
            "written_at_utc": written_at,
            "symbol": symbol,
            "direction": candidate.direction,
            "entry_price": float(candidate.entry_price),
            "sl_price": float(candidate.sl_price),
            "tp_price": float(candidate.tp_price),
            "activation_price": float(candidate.activation_price) if candidate.activation_price else 0.0,
            "lock_price": float(candidate.lock_price) if candidate.lock_price else 0.0,
            "regime": candidate.regime,
            "session_london": candidate.session_london,
            "session_ny": candidate.session_ny,
            "session_asia": candidate.session_asia,
            "calibrated_prob": round(float(candidate.calibrated_prob), 4),
            "ev_r": round(float(candidate.ev_r), 4),
            "risk_pct": round(float(decision.risk_pct), 4),
            "confidence_tier": decision.confidence_tier,
            "expiry_minutes": expiry_minutes,
            "reasons": decision.reasons,
        }

        # Atomic write: temp file in same directory, then rename (rename is
        # atomic on the same filesystem — the EA never sees a partial file)
        fd, tmp_path = tempfile.mkstemp(dir=self.signal_path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_path, self.signal_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

        self._save_sequence_id(self._sequence_id)

        # Append a permanent record to the signal log — this is what
        # track_outcomes.py joins against real MT5 trade history later, so it
        # needs equity_at_signal (to normalize realized $ P&L into R-multiples)
        # and everything needed to compare predicted vs actual outcome.
        log_record = dict(payload)
        log_record["equity_at_signal"] = equity_at_signal
        with open(self.signal_log_path, "a") as f:
            f.write(json.dumps(log_record) + "\n")

        return payload
