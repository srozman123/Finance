# ============================================================================
# momentum_signals.py — Multi-Signal Momentum Scanner
# ============================================================================
# This file identifies current momentum candidates using three independent
# signals — price momentum, earnings momentum, and technical momentum.
# Each signal captures a different dimension of momentum:
#   - Price momentum (12-1) reflects what the market has already rewarded
#   - Earnings momentum reflects fundamental surprise versus expectations
#   - Technical momentum reflects current trend structure and RSI positioning
#
# The composite score (0–5) represents the confluence of all three signals.
# High confluence across independent signals produces stronger, more reliable
# candidates than any single signal alone.
# ============================================================================

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta

# ──────────────────────────────────────────────────────────────────────────────
# USER CONFIGURATION — edit these two variables to change the run
# ──────────────────────────────────────────────────────────────────────────────
UNIVERSE = [
    'AAPL', 'MSFT', 'GOOG', 'META', 'AVGO', 'AMZN',
    'ADBE', 'PAYC', 'TTD', 'SNPS', 'NVDA', 'AMD',
    'NFLX', 'HOOD', 'RDDT', 'PLTR', 'DDOG'
]
ANALYSIS_DATE = None   # None = use today  |  'YYYY-MM-DD' = historical backfill


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_date():
    """Return the analysis date as a datetime.date object."""
    if ANALYSIS_DATE is None:
        return date.today()
    return datetime.strptime(ANALYSIS_DATE, "%Y-%m-%d").date()


def _wilder_rsi(close, period=14):
    """14-day Wilder EMA RSI. Returns a pandas Series aligned to close.index."""
    delta  = close.diff()
    gains  = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)
    alpha  = 1 / period
    avg_g  = gains.ewm(alpha=alpha, adjust=False).mean()
    avg_l  = losses.ewm(alpha=alpha, adjust=False).mean()
    return 100 - 100 / (1 + avg_g / avg_l)


def _safe_float(val):
    """Convert a value to float, returning None if it is NaN or unconvertible."""
    try:
        f = float(val)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


# ──────────────────────────────────────────────────────────────────────────────
# SIGNAL 1 — Price momentum (12-1)
# Academic standard: total return from 12 months ago to 1 month ago,
# deliberately skipping the most recent month to avoid short-term reversal.
# ──────────────────────────────────────────────────────────────────────────────

def calc_price_momentum(ticker_str, as_of):
    """
    Returns the 12-1 momentum as a decimal (e.g. 0.32 = +32%), or None.
    Downloads 13 months of daily data, then:
      - price_12m_ago  = close at ~252 trading days before as_of
      - price_1m_ago   = close at ~21 trading days before as_of
    momentum_12_1 = (price_1m_ago / price_12m_ago) - 1
    """
    start = as_of - timedelta(days=400)   # buffer for non-trading days
    raw   = yf.download(ticker_str, start=start, end=as_of,
                        interval="1d", auto_adjust=True, progress=False)
    if raw.empty:
        return None

    close = raw["Close"].squeeze().dropna()
    if len(close) < 230:
        return None

    price_12m_ago = float(close.iloc[-252]) if len(close) >= 252 else float(close.iloc[0])
    price_1m_ago  = float(close.iloc[-21])

    if price_12m_ago <= 0:
        return None
    return (price_1m_ago / price_12m_ago) - 1


# ──────────────────────────────────────────────────────────────────────────────
# SIGNAL 2 — Earnings momentum
# Earnings surprise = (reported EPS - estimated EPS) / |estimated EPS|.
# Also flags whether the most recent beat occurred within the last 60 days.
# ──────────────────────────────────────────────────────────────────────────────

def calc_earnings_momentum(ticker_obj, as_of):
    """
    Returns (earnings_surprise, recent_beat) where:
      earnings_surprise  float | None   — fractional surprise on last quarter
      recent_beat        bool           — True if positive surprise within 60 days
    Tries earnings_history first (newer yfinance), falls back to quarterly_earnings.
    """
    surprise       = None
    recent_beat    = False
    cutoff         = as_of - timedelta(days=60)

    # ── Attempt 1: earnings_history (yfinance ≥ 0.2) ──
    try:
        eh = ticker_obj.earnings_history
        if eh is not None and not eh.empty:
            # Keep rows with both estimate and actual
            eh = eh.dropna(subset=["epsEstimate", "epsActual"])
            if not eh.empty:
                # Sort by date descending, take the most recent
                if hasattr(eh.index[0], 'date'):
                    eh = eh.sort_index(ascending=False)
                else:
                    eh = eh.sort_values(by=eh.columns[0], ascending=False)

                row      = eh.iloc[0]
                reported = _safe_float(row.get("epsActual"))
                estimate = _safe_float(row.get("epsEstimate"))

                if reported is not None and estimate is not None and estimate != 0:
                    surprise = (reported - estimate) / abs(estimate)

                # Date of the most recent quarter
                try:
                    q_date = eh.index[0]
                    if hasattr(q_date, 'date'):
                        q_date = q_date.date()
                    else:
                        q_date = pd.Timestamp(q_date).date()
                    if surprise is not None and surprise > 0 and q_date >= cutoff:
                        recent_beat = True
                except Exception:
                    pass

                return surprise, recent_beat
    except Exception:
        pass

    # ── Attempt 2: quarterly_earnings (older yfinance) ──
    try:
        qe = ticker_obj.quarterly_earnings
        if qe is not None and not qe.empty:
            qe = qe.dropna()
            if not qe.empty:
                row      = qe.iloc[-1]   # most recent quarter
                reported = _safe_float(row.get("Reported") or row.get("Earnings"))
                estimate = _safe_float(row.get("Estimate"))
                if reported is not None and estimate is not None and estimate != 0:
                    surprise = (reported - estimate) / abs(estimate)
                # quarterly_earnings doesn't carry exact dates — can't check 60-day window
    except Exception:
        pass

    return surprise, recent_beat


# ──────────────────────────────────────────────────────────────────────────────
# SIGNAL 3 — Technical momentum
# Uses 200 days of price data.  Returns a dict of sub-signals.
# ──────────────────────────────────────────────────────────────────────────────

def calc_technical_momentum(ticker_str, as_of):
    """
    Returns dict with keys:
      above_200ma   bool   — close > 200-day MA
      ma50_above    bool   — 50MA > 200MA
      rsi           float  — current RSI
      rsi_above50   bool   — RSI > 50
      rsi_rising    bool   — RSI has risen over each of the last 5 days
    Returns None if insufficient data.
    """
    start = as_of - timedelta(days=400)
    raw   = yf.download(ticker_str, start=start, end=as_of,
                        interval="1d", auto_adjust=True, progress=False)
    if raw.empty:
        return None

    close = raw["Close"].squeeze().dropna()
    if len(close) < 210:
        return None

    ma50  = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    rsi   = _wilder_rsi(close)

    last_close = float(close.iloc[-1])
    last_ma50  = float(ma50.iloc[-1])
    last_ma200 = float(ma200.iloc[-1])
    last_rsi   = float(rsi.iloc[-1])

    # RSI rising: each of the last 5 RSI values is higher than the one before
    rsi_window = rsi.iloc[-6:]   # 6 values → 5 consecutive differences
    rsi_rising = bool((rsi_window.diff().dropna() > 0).all())

    return {
        "above_200ma": last_close > last_ma200,
        "ma50_above":  last_ma50  > last_ma200,
        "rsi":         round(last_rsi, 1),
        "rsi_above50": last_rsi > 50,
        "rsi_rising":  rsi_rising,
    }


# ──────────────────────────────────────────────────────────────────────────────
# COMPOSITE SCORE (0–5)
# One point for each of:
#   1. Top 50% of 12-1 momentum in the universe
#   2. Positive earnings surprise
#   3. Recent earnings beat within 60 days
#   4. Price above 200MA AND 50MA above 200MA
#   5. RSI above 50 AND rising
# ──────────────────────────────────────────────────────────────────────────────

def composite_score(mom_rank_top_half, surprise, recent_beat, tech):
    """
    Returns (score, criteria) where criteria is an ordered list of 5 booleans,
    one per criterion in display order.  Each dot in the score bar maps directly
    to the criterion at that position — no positional ambiguity.

    Criterion order (matches the legend printed below the table):
      [1] Top-50% price momentum
      [2] Positive earnings surprise
      [3] Recent earnings beat within 60 days  ← same condition as Beat? column
      [4] Price above 200MA and 50MA above 200MA
      [5] RSI above 50 and rising
    """
    c1 = bool(mom_rank_top_half)
    c2 = surprise is not None and surprise > 0
    c3 = bool(recent_beat)           # identical to the Beat? column — no divergence possible
    c4 = bool(tech and tech["above_200ma"] and tech["ma50_above"])
    c5 = bool(tech and tech["rsi_above50"] and tech["rsi_rising"])
    return sum([c1, c2, c3, c4, c5]), [c1, c2, c3, c4, c5]


def signal_label(score):
    if score >= 4:
        return "Strong Momentum"
    if score == 3:
        return "Moderate Momentum"
    if score == 2:
        return "Weak"
    return "No Signal"


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    as_of = _resolve_date()

    print("=" * 78)
    print(f"  MOMENTUM SCANNER  |  {len(UNIVERSE)} tickers  |  As of: {as_of}")
    print("=" * 78)
    print("  Collecting data... (this may take 1–2 minutes)\n")

    # ── Pass 1: collect raw signals ──
    raw_results = []
    for ticker in UNIVERSE:
        try:
            t = yf.Ticker(ticker)

            mom   = calc_price_momentum(ticker, as_of)
            surp, rbeat = calc_earnings_momentum(t, as_of)
            tech  = calc_technical_momentum(ticker, as_of)

            raw_results.append({
                "ticker":          ticker,
                "momentum_12_1":   mom,
                "earnings_surprise": surp,
                "recent_beat":     rbeat,
                "tech":            tech,
            })
            status = f"mom {mom*100:+.1f}%" if mom is not None else "mom N/A"
            print(f"  {ticker:<7} {status}")

        except Exception as exc:
            print(f"  {ticker:<7} SKIPPED — {exc}")

    # ── Pass 2: rank momentum within universe (top 50% gets the point) ──
    mom_vals = [(r["ticker"], r["momentum_12_1"])
                for r in raw_results if r["momentum_12_1"] is not None]
    if mom_vals:
        mom_series  = pd.Series({t: v for t, v in mom_vals})
        median_mom  = mom_series.median()
    else:
        median_mom  = 0.0

    # ── Pass 3: compute composite scores ──
    records = []
    for r in raw_results:
        mom   = r["momentum_12_1"]
        surp  = r["earnings_surprise"]
        rbeat = r["recent_beat"]
        tech  = r["tech"]

        top_half          = (mom is not None) and (mom >= median_mom)
        score, criteria   = composite_score(top_half, surp, rbeat, tech)
        label             = signal_label(score)

        # Technical status string
        if tech:
            trend_ok = tech["above_200ma"] and tech["ma50_above"]
            rsi_ok   = tech["rsi_above50"] and tech["rsi_rising"]
            trend_str = "Above MA✓" if trend_ok else "Below MA"
            rsi_str   = f"RSI {tech['rsi']} ↑" if rsi_ok else f"RSI {tech['rsi']}"
            tech_str  = f"{trend_str}  {rsi_str}"
        else:
            tech_str  = "N/A"

        records.append({
            "Ticker":        r["ticker"],
            "12-1 Mom%":     f"{mom*100:+.1f}%" if mom is not None else "N/A",
            "EPS Surprise%": f"{surp*100:+.1f}%" if surp is not None else "N/A",
            "Recent Beat":   "Yes" if rbeat else "No",
            "Technical":     tech_str,
            "Score":         score,
            "Signal":        label,
            "_criteria":     criteria,   # [c1, c2, c3, c4, c5] — drives the dot bar
            "_score":        score,
            "_mom":          mom if mom is not None else -999,
        })

    # Sort by score desc, then momentum desc as tiebreaker
    records.sort(key=lambda x: (x["_score"], x["_mom"]), reverse=True)

    # ── Print ranked table ──
    print()
    print(f"  {'─'*94}")
    print(f"  {'TICKER':<8} {'12-1 Mom%':>10} {'EPS Surp%':>10} {'Beat?':>6} "
          f"  {'Technical':<28} {'Dots':>7}  Signal")
    print(f"  {'─'*94}")

    for r in records:
        # Each dot maps to exactly one criterion — position 3 is always Beat?
        score_bar = "".join("●" if c else "○" for c in r["_criteria"])
        print(f"  {r['Ticker']:<8} "
              f"{r['12-1 Mom%']:>10} "
              f"{r['EPS Surprise%']:>10} "
              f"{r['Recent Beat']:>6}   "
              f"{r['Technical']:<28} "
              f"[{score_bar}]  "
              f"{r['Signal']}")

    print(f"  {'─'*94}")
    print(f"  Dots: [1]=Top-50% momentum  [2]=Positive EPS surprise  "
          f"[3]=Beat within 60 days  [4]=Above 200MA & 50MA>200MA  [5]=RSI>50 & rising")

    # ── Aggregate summary ──
    scores    = [r["_score"] for r in records]
    strong    = sum(1 for s in scores if s >= 4)
    moderate  = sum(1 for s in scores if s == 3)
    weak      = sum(1 for s in scores if s == 2)
    no_signal = sum(1 for s in scores if s <= 1)

    print()
    print(f"  UNIVERSE SUMMARY  |  median 12-1 momentum: {median_mom*100:+.1f}%")
    print(f"  Strong Momentum (4–5):  {strong} tickers")
    print(f"  Moderate Momentum (3):  {moderate} tickers")
    print(f"  Weak (2):               {weak} tickers")
    print(f"  No Signal (0–1):        {no_signal} tickers")
    print()


if __name__ == "__main__":
    main()
