"""
Stock Deep Dive
----------------
For a single ticker you're currently interested in: runs the Phase 5 report
generator (saves you a PDF, same as always) and the SEC 10-K analyzer (Item 1
Business + Item 7 MD&A), bundles the computed financial data + filing text
into one JSON payload, and fires it at a Claude Code Routine's API trigger.
The routine (prompt in stock_deep_dive_routine_prompt.md, pasted in separately
at claude.ai/code/routines) does the actual research -- answering the 27
investment-thesis questions using this payload as a starting point plus its
own web search -- and posts the write-up to Slack.

Modeled on /Users/simonrozman/Signals Engine/signals_engine.py: compute
locally, POST the data to the routine's fire URL, let the routine do the
thinking.

Setup:
    pip install -r requirements.txt

    export STOCK_DEEP_DIVE_ROUTINE_FIRE_URL=https://api.anthropic.com/v1/claude_code/routines/YOUR_ROUTINE_ID/fire
    export STOCK_DEEP_DIVE_ROUTINE_TOKEN=your_routine_bearer_token   (from claude.ai/code/routines
                                                        when you add the API trigger -- this needs
                                                        to be its own routine, separate from the
                                                        signals-engine one, since it runs a
                                                        different prompt. Deliberately prefixed
                                                        (not the bare ROUTINE_FIRE_URL / ROUTINE_TOKEN
                                                        signals_engine.py uses) so the two scripts
                                                        never collide if both env vars are set in
                                                        the same shell.)

    Slack posting is configured on the routine itself (SLACK_BOT_TOKEN /
    SLACK_CHANNEL env vars on the routine) -- reuse the same values you
    already use for the signals-engine routine; no new Slack app needed.

Run:
    python stock_deep_dive.py NVDA
    (or edit TICKER below and run with no argument)

Output:
    Saves the PDF report to Financial Modeling/Phase 5/reports/ (unchanged
    behavior). Writes reports/{TICKER}_deep_dive.json locally for your own
    records. POSTs the same payload to the routine if ROUTINE_FIRE_URL /
    ROUTINE_TOKEN are set.
"""

import os
import sys
import json
import datetime

import requests

# ─────────────────────────────────────────────
# TICKER — change this to look at a different stock, or pass one as argv[1]
# ─────────────────────────────────────────────
TICKER = "TTD"

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIN_MODELING = os.path.dirname(_HERE)   # this folder now lives inside Financial Modeling/
sys.path.insert(0, os.path.join(_FIN_MODELING, "Phase 5"))
sys.path.insert(0, os.path.join(_FIN_MODELING, "sec_10k_analyzer"))

from phase5_report_generator import generate_report, get_report_data
from sec_analyzer import analyze_company

REPORT_DIR = os.path.join(_HERE, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

# SEC filing text is verbose -- cap what goes into the routine payload so a
# long MD&A doesn't blow up the fired message. Applied per item (1 and 7
# each get their own budget).
MAX_SECTIONS_PER_ITEM = 8
MAX_CHARS_PER_ITEM = 15000


def _cap_item(item: dict) -> dict:
    """Trim a {"sections": [...], "raw": ...} dict to a payload-safe size."""
    sections = item.get("sections", [])[:MAX_SECTIONS_PER_ITEM]
    capped, total = [], 0
    for sec in sections:
        remaining = MAX_CHARS_PER_ITEM - total
        if remaining <= 0:
            break
        text = sec["text"][:remaining]
        capped.append({"title": sec["title"], "text": text})
        total += len(text)
    return {"sections": capped}


def build_payload(ticker: str) -> dict:
    ticker = ticker.upper()

    print(f"  Generating PDF report for {ticker}...")
    generate_report(ticker)   # unchanged local PDF, for your own reading

    print(f"  Computing financial/valuation/technical data for {ticker}...")
    financials = get_report_data(ticker)

    print(f"  Fetching SEC 10-K Item 1 / Item 7 for {ticker}...")
    sec = analyze_company(ticker)

    return {
        "ticker": ticker,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "financials": financials,
        "sec_filing": {
            "company_name": sec["company_name"],
            "filing_date": sec["filing_date"],
            "item1_business": _cap_item(sec["item1"]),
            "item7_mdna": _cap_item(sec["item7"]),
        },
    }


def fire_routine(payload: dict) -> None:
    """POST the payload directly to the routine's API trigger."""
    fire_url = os.environ.get("STOCK_DEEP_DIVE_ROUTINE_FIRE_URL")
    token = os.environ.get("STOCK_DEEP_DIVE_ROUTINE_TOKEN")
    if not fire_url or not token:
        print("STOCK_DEEP_DIVE_ROUTINE_FIRE_URL / STOCK_DEEP_DIVE_ROUTINE_TOKEN not set -- skipping routine trigger, wrote local JSON only.")
        return

    headers = {
        "Authorization": f"Bearer {token}",
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "experimental-cc-routine-2026-04-01",
        "Content-Type": "application/json",
    }
    body = {"text": f"Stock deep-dive data for {payload['ticker']}:\n" + json.dumps(payload)}
    resp = requests.post(fire_url, headers=headers, json=body, timeout=30)
    resp.raise_for_status()
    session = resp.json()
    print(f"Routine fired. Session: {session.get('url', session)}")


def main():
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else TICKER

    payload = build_payload(ticker)

    out_path = os.path.join(REPORT_DIR, f"{ticker}_deep_dive.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_path}")

    fire_routine(payload)


if __name__ == "__main__":
    main()
