"""
XAUUSD ML+Agent+EA — Live Agent Dashboard v2
================================================

A proper "command center" view of the agent: live reasoning feed, equity
curve, a decision funnel showing where candidates get filtered, and
regime-level performance. Reads the same log files as v1 — no changes
needed to live_runner.py.

Usage:
    python dashboard_server.py --port 5000
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from flask import Flask, jsonify, render_template_string

app = Flask(__name__)

DECISIONS_LOG_CANDIDATES = [Path("live_decisions_multi.jsonl"), Path("live_decisions.jsonl")]
STATE_FILE_CANDIDATES = [Path("live_agent_state_multi.json"), Path("live_agent_state.json")]


def _first_existing(candidates):
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]  # default to first for a clear "not found" path in read functions


OUTCOMES_CSV = Path("phase1_output/live_tracking/live_outcomes.csv")


def read_all_decisions():
    path = _first_existing(DECISIONS_LOG_CANDIDATES)
    if not path.exists():
        return []
    lines = path.read_text().strip().splitlines()
    return [json.loads(line) for line in lines if line]


def read_state():
    path = _first_existing(STATE_FILE_CANDIDATES)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def read_outcomes_summary():
    if not OUTCOMES_CSV.exists():
        return None
    import pandas as pd
    df = pd.read_csv(OUTCOMES_CSV)
    if df.empty:
        return None
    overall = {
        "n_trades": len(df),
        "win_rate": float(df["won"].mean()),
        "avg_realized_r": float(df["realized_r"].mean()),
        "total_pnl": float(df["realized_profit"].sum()),
    }
    by_regime = []
    if "regime" in df.columns:
        for regime, g in df.groupby("regime"):
            by_regime.append({
                "regime": regime, "n": len(g),
                "win_rate": float(g["won"].mean()),
                "avg_r": float(g["realized_r"].mean()),
            })
    return {"overall": overall, "by_regime": by_regime}


def classify_veto_reason(reasons):
    """Bucket a veto's reasons list into a short category for the funnel chart."""
    if not reasons:
        return "unknown"
    text = " ".join(reasons).lower()
    if "drawdown circuit breaker" in text or ("drawdown" in text and "hit max" in text):
        return "drawdown_halt"
    if "daily loss" in text:
        return "daily_loss_halt"
    if "already in a position" in text:
        return "concurrency"
    if "session" in text:
        return "session_filter"
    if "ev" in text and "threshold" in text:
        return "ev_below_threshold"
    if "missing entry" in text:
        return "data_issue"
    return "other"


@app.route("/api/feed")
def api_feed():
    decisions = read_all_decisions()
    return jsonify(list(reversed(decisions[-60:])))  # newest first, last 60


@app.route("/api/state")
def api_state():
    state = read_state() or {}
    decisions = read_all_decisions()
    if decisions:
        last = decisions[-1]
        state["current_regime"] = last.get("regime")
        state["last_action"] = last.get("action")
        state["last_updated"] = last.get("logged_at_utc")
    today_approvals = 0
    if decisions:
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).date()
        for d in decisions:
            try:
                ts = datetime.fromisoformat(d["logged_at_utc"]).date()
                if ts == today and d["action"] == "APPROVE":
                    today_approvals += 1
            except Exception:
                pass
    state["approvals_today"] = today_approvals
    return jsonify(state)


@app.route("/api/equity_curve")
def api_equity_curve():
    decisions = read_all_decisions()
    points = [{"t": d["logged_at_utc"], "equity": d["equity"]} for d in decisions if "equity" in d]
    return jsonify(points)


@app.route("/api/funnel")
def api_funnel():
    decisions = read_all_decisions()
    total = len(decisions)
    approved = sum(1 for d in decisions if d["action"] == "APPROVE")
    veto_buckets = Counter()
    for d in decisions:
        if d["action"] == "VETO":
            veto_buckets[classify_veto_reason(d.get("reasons", []))] += 1
    return jsonify({
        "total_evaluated": total,
        "approved": approved,
        "veto_breakdown": dict(veto_buckets.most_common()),
    })


@app.route("/api/outcomes")
def api_outcomes():
    return jsonify(read_outcomes_summary() or {})


PAGE = """
<!DOCTYPE html>
<html>
<head>
  <title>XAUUSD Agent — Command Center</title>
  <meta charset="utf-8">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
  <style>
    * { box-sizing: border-box; }
    body { background:#0a0e14; color:#c9d1d9; font-family: 'SF Mono', Consolas, monospace; margin:0; padding:16px; }
    h1 { font-size:18px; color:#e6edf3; margin:0 0 2px; font-weight:600; letter-spacing:0.3px; }
    .subtitle { color:#5a6472; font-size:11px; margin-bottom:16px; }
    .statusbar { display:flex; gap:10px; margin-bottom:16px; flex-wrap:wrap; }
    .pill { background:#12161f; border:1px solid #232936; border-radius:20px; padding:6px 14px; font-size:12px; display:flex; align-items:center; gap:6px; }
    .dot { width:7px; height:7px; border-radius:50%; display:inline-block; }
    .dot-green { background:#3fb950; box-shadow:0 0 6px #3fb950; }
    .dot-red { background:#f85149; box-shadow:0 0 6px #f85149; }
    .dot-gray { background:#5a6472; }
    .grid { display:grid; grid-template-columns: 1.3fr 1fr; gap:14px; }
    .col { display:flex; flex-direction:column; gap:14px; }
    .panel { background:#0e1219; border:1px solid #1c2230; border-radius:10px; padding:16px; }
    .panel-title { font-size:11px; color:#5a6472; text-transform:uppercase; letter-spacing:1px; margin-bottom:12px; font-weight:600; }
    .green { color:#3fb950; } .red { color:#f85149; } .yellow { color:#d29922; } .blue { color:#58a6ff; }
    canvas { max-height:220px; }
    .feed { max-height:520px; overflow-y:auto; display:flex; flex-direction:column; gap:8px; }
    .feed-item { border-left:3px solid #232936; padding:8px 12px; background:#12161f; border-radius:0 6px 6px 0; font-size:12px; }
    .feed-item.approve { border-left-color:#3fb950; }
    .feed-item.veto { border-left-color:#3a4150; opacity:0.85; }
    .feed-time { color:#5a6472; font-size:10px; }
    .feed-headline { font-weight:600; margin:2px 0; }
    .feed-headline .action-approve { color:#3fb950; }
    .feed-headline .action-veto { color:#8b949e; }
    .feed-reason { color:#7d8590; font-size:11px; line-height:1.5; }
    .badge { display:inline-block; background:#1c2230; border-radius:4px; padding:1px 6px; font-size:10px; color:#8b949e; margin-right:4px; }
    .funnel-row { display:flex; align-items:center; gap:8px; margin-bottom:6px; font-size:11px; }
    .funnel-bar-wrap { flex:1; background:#12161f; border-radius:4px; height:18px; overflow:hidden; }
    .funnel-bar { height:100%; background:linear-gradient(90deg,#f85149,#d29922); }
    .funnel-label { width:130px; color:#8b949e; }
    .funnel-count { width:36px; text-align:right; color:#c9d1d9; }
    .regime-table { width:100%; font-size:12px; border-collapse:collapse; }
    .regime-table th { text-align:left; color:#5a6472; font-weight:500; font-size:10px; text-transform:uppercase; padding-bottom:6px; }
    .regime-table td { padding:5px 0; border-top:1px solid #1c2230; }
    .empty-note { color:#5a6472; font-style:italic; font-size:12px; padding:16px 0; text-align:center; }
  </style>
</head>
<body>
  <h1>XAUUSD Agent — Command Center</h1>
  <div class="subtitle">Live view · auto-refresh 10s · read-only, does not affect trading</div>

  <div class="statusbar" id="statusBar"></div>

  <div class="grid">
    <div class="col">
      <div class="panel">
        <div class="panel-title">Live Agent Feed</div>
        <div class="feed" id="feed"><div class="empty-note">Waiting for agent activity...</div></div>
      </div>
    </div>

    <div class="col">
      <div class="panel">
        <div class="panel-title">Equity Curve</div>
        <canvas id="equityChart"></canvas>
      </div>

      <div class="panel">
        <div class="panel-title">Decision Funnel (all-time)</div>
        <div id="funnel"></div>
      </div>

      <div class="panel">
        <div class="panel-title">Regime Performance</div>
        <table class="regime-table" id="regimeTable">
          <thead><tr><th>Regime</th><th>N</th><th>Win%</th><th>Avg R</th></tr></thead>
          <tbody></tbody>
        </table>
        <div id="regimeEmpty" class="empty-note" style="display:none;">No closed trades yet — run track_outcomes.py once trades close.</div>
      </div>
    </div>
  </div>

<script>
let equityChart = null;

function fmtTime(iso) {
  return new Date(iso).toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'});
}

async function refresh() {
  try {
    const [state, feed, equity, funnel, outcomes] = await Promise.all([
      fetch('/api/state').then(r => r.json()),
      fetch('/api/feed').then(r => r.json()),
      fetch('/api/equity_curve').then(r => r.json()),
      fetch('/api/funnel').then(r => r.json()),
      fetch('/api/outcomes').then(r => r.json()),
    ]);

    const dd = state.peak_equity ? ((state.peak_equity - state.equity) / state.peak_equity * 100) : 0;
    const halted = state.trading_halted_drawdown || state.trading_halted_today;
    const statusPills = [
      `<span class="pill"><span class="dot ${halted ? 'dot-red' : 'dot-green'}"></span>${halted ? 'HALTED' : 'ACTIVE'}</span>`,
      `<span class="pill">Equity <b>&nbsp;$${(state.equity||0).toFixed(2)}</b></span>`,
      `<span class="pill">Drawdown <b>&nbsp;${dd.toFixed(1)}%</b></span>`,
      `<span class="pill">Regime <b>&nbsp;${state.current_regime || '—'}</b></span>`,
      `<span class="pill">Approvals today <b>&nbsp;${state.approvals_today ?? 0}</b></span>`,
      `<span class="pill"><span class="dot dot-gray"></span>Last update ${state.last_updated ? fmtTime(state.last_updated) : '—'}</span>`,
    ];
    document.getElementById('statusBar').innerHTML = statusPills.join('');

    const feedEl = document.getElementById('feed');
    if (!feed.length) {
      feedEl.innerHTML = '<div class="empty-note">Waiting for agent activity...</div>';
    } else {
      feedEl.innerHTML = feed.map(d => {
        const cls = d.action === 'APPROVE' ? 'approve' : 'veto';
        const actionSpan = d.action === 'APPROVE'
          ? `<span class="action-approve">&#10003; APPROVED</span> [${d.symbol||'?'}] ${d.direction} &middot; ${d.confidence_tier} tier &middot; risk ${d.risk_pct}%`
          : `<span class="action-veto">&mdash; veto</span> [${d.symbol||'?'}] ${d.direction}`;
        return `<div class="feed-item ${cls}">
          <div class="feed-time">${fmtTime(d.logged_at_utc)}</div>
          <div class="feed-headline">${actionSpan}</div>
          <div class="feed-reason">
            <span class="badge">prob ${d.calibrated_prob.toFixed(3)}</span>
            <span class="badge">EV ${d.ev_r >= 0 ? '+' : ''}${d.ev_r.toFixed(3)}R</span>
            <span class="badge">${d.regime}</span>
          </div>
          <div class="feed-reason" style="margin-top:4px;">${d.reasons.join(' &middot; ')}</div>
        </div>`;
      }).join('');
    }

    const labels = equity.map(p => fmtTime(p.t));
    const values = equity.map(p => p.equity);
    if (!equityChart) {
      const ctx = document.getElementById('equityChart').getContext('2d');
      equityChart = new Chart(ctx, {
        type: 'line',
        data: { labels, datasets: [{ data: values, borderColor:'#58a6ff', backgroundColor:'rgba(88,166,255,0.08)',
          fill:true, tension:0.25, pointRadius:0, borderWidth:2 }] },
        options: { responsive:true, plugins:{legend:{display:false}},
          scales: { x:{display:false}, y:{ ticks:{color:'#5a6472', font:{size:10}}, grid:{color:'#1c2230'} } } }
      });
    } else {
      equityChart.data.labels = labels;
      equityChart.data.datasets[0].data = values;
      equityChart.update('none');
    }

    const funnelEl = document.getElementById('funnel');
    const maxCount = Math.max(funnel.total_evaluated, 1);
    let funnelHtml = `<div class="funnel-row"><span class="funnel-label">Total evaluated</span>
      <div class="funnel-bar-wrap"><div class="funnel-bar" style="width:100%;background:#58a6ff;"></div></div>
      <span class="funnel-count">${funnel.total_evaluated}</span></div>`;
    funnelHtml += `<div class="funnel-row"><span class="funnel-label">Approved</span>
      <div class="funnel-bar-wrap"><div class="funnel-bar" style="width:${funnel.approved/maxCount*100}%;background:#3fb950;"></div></div>
      <span class="funnel-count">${funnel.approved}</span></div>`;
    for (const [reason, count] of Object.entries(funnel.veto_breakdown || {})) {
      funnelHtml += `<div class="funnel-row"><span class="funnel-label">${reason.replace(/_/g,' ')}</span>
        <div class="funnel-bar-wrap"><div class="funnel-bar" style="width:${count/maxCount*100}%;"></div></div>
        <span class="funnel-count">${count}</span></div>`;
    }
    funnelEl.innerHTML = funnelHtml;

    const tbody = document.querySelector('#regimeTable tbody');
    const regimeEmpty = document.getElementById('regimeEmpty');
    if (outcomes && outcomes.by_regime && outcomes.by_regime.length) {
      regimeEmpty.style.display = 'none';
      tbody.innerHTML = outcomes.by_regime.map(r => `
        <tr><td>${r.regime}</td><td>${r.n}</td>
        <td class="${r.win_rate >= 0.4 ? 'green' : 'red'}">${(r.win_rate*100).toFixed(0)}%</td>
        <td class="${r.avg_r >= 0 ? 'green' : 'red'}">${r.avg_r >= 0 ? '+' : ''}${r.avg_r.toFixed(3)}</td></tr>
      `).join('');
    } else {
      tbody.innerHTML = '';
      regimeEmpty.style.display = 'block';
    }

  } catch (e) {
    console.error('Dashboard refresh failed:', e);
  }
}
refresh();
setInterval(refresh, 10000);
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live agent command center")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    print(f"Dashboard running at http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)
