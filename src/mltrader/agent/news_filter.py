"""
XAUUSD ML+Agent+EA — News/Session Filter
============================================

Fetches the current week's economic calendar from a free, widely-used public
feed (Forex Factory's JSON mirror — no API key needed) and builds "blackout
windows" around high-impact USD events (the ones that actually move gold —
CPI, NFP, FOMC, etc.). The agent then refuses to open new trades inside
those windows.

IMPORTANT LIMITATIONS, stated upfront:
  - This feed only covers the CURRENT week (and sometimes next week) — it is
    NOT a historical archive. That means this filter works for live/demo
    trading going forward, but you cannot cleanly backtest "how many past
    trades would have been filtered" without a paid historical calendar
    data source. Don't try to retrofit this into realistic_backtest.py
    without sourcing real historical calendar data first — a fake/assumed
    historical filter would just be guessing.
  - I could not test the actual live HTTP fetch against the real feed from
    my environment (network access there is restricted). The parsing logic
    below is tested against a realistic mock of the feed's known format —
    run it once yourself and paste back the output (or any error) so we can
    fix the parsing if the real feed's field names differ from what's coded.

Usage (standalone check):
    python news_filter.py --impact High --currencies USD --before 30 --after 30

Usage (from live_runner.py):
    from news_filter import fetch_calendar, build_blackout_windows, is_in_blackout
    calendar = fetch_calendar()
    windows = build_blackout_windows(calendar)
    if is_in_blackout(datetime.now(timezone.utc), windows):
        # skip this cycle
"""

import argparse
import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

# Public, free, no-API-key JSON mirror of the Forex Factory economic calendar.
# Widely used by the retail EA community. Covers roughly the current week.
CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"


def fetch_calendar(timeout: int = 10) -> pd.DataFrame:
    """Fetch and parse the current week's economic calendar.
    Returns a DataFrame with columns: title, country, datetime (UTC), impact.
    """
    resp = requests.get(CALENDAR_URL, timeout=timeout)
    resp.raise_for_status()
    raw = resp.json()

    rows = []
    for item in raw:
        # Field names based on the known public format — flexible lookup in
        # case the feed uses slightly different casing/keys than expected.
        title = item.get("title") or item.get("event") or item.get("name")
        country = item.get("country") or item.get("currency")
        impact = item.get("impact")
        date_str = item.get("date") or item.get("datetime")

        if not date_str:
            continue
        try:
            dt = pd.to_datetime(date_str, utc=True)
        except Exception:
            continue

        rows.append({"title": title, "country": country, "datetime": dt, "impact": impact})

    df = pd.DataFrame(rows)
    if df.empty:
        print("WARNING: fetched calendar but parsed zero rows — the feed's field names "
              "may differ from what this parser expects. Print raw JSON structure to debug:")
        print(json.dumps(raw[:2], indent=2) if raw else "(empty response)")
    return df


def build_blackout_windows(calendar_df: pd.DataFrame, impact_levels=("High",),
                            currencies=("USD",), before_minutes: int = 30,
                            after_minutes: int = 30) -> list:
    """Returns a list of (start_utc, end_utc, title) tuples — one blackout
    window per matching event."""
    if calendar_df.empty:
        return []

    filtered = calendar_df[
        calendar_df["impact"].isin(impact_levels) & calendar_df["country"].isin(currencies)
    ]

    windows = []
    for _, row in filtered.iterrows():
        start = row["datetime"] - timedelta(minutes=before_minutes)
        end = row["datetime"] + timedelta(minutes=after_minutes)
        windows.append((start, end, row["title"]))
    return windows


def is_in_blackout(dt: datetime, windows: list):
    """Returns (True, event_title) if dt falls within any blackout window,
    else (False, None). dt must be timezone-aware UTC."""
    for start, end, title in windows:
        if start <= dt <= end:
            return True, title
    return False, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check current news blackout status")
    parser.add_argument("--impact", nargs="+", default=["High"])
    parser.add_argument("--currencies", nargs="+", default=["USD"])
    parser.add_argument("--before", type=int, default=30)
    parser.add_argument("--after", type=int, default=30)
    args = parser.parse_args()

    print(f"Fetching calendar from {CALENDAR_URL}...")
    cal = fetch_calendar()
    print(f"\nParsed {len(cal)} events this week.")
    if not cal.empty:
        print("\nSample events:")
        print(cal.head(10).to_string())

    windows = build_blackout_windows(cal, impact_levels=tuple(args.impact),
                                      currencies=tuple(args.currencies),
                                      before_minutes=args.before, after_minutes=args.after)
    print(f"\n{len(windows)} blackout windows built (impact={args.impact}, currencies={args.currencies}, "
          f"+/-{args.before}/{args.after} min).")
    for start, end, title in windows:
        print(f"  {start} -> {end}  [{title}]")

    now = datetime.now(timezone.utc)
    blackout, title = is_in_blackout(now, windows)
    print(f"\nRight now ({now}): {'IN BLACKOUT — ' + str(title) if blackout else 'clear to trade'}")
