# ============================================================================
# GMAIL APP PASSWORD SETUP — four steps to enable email alerts
# ============================================================================
# 1. Enable 2-Factor Authentication at myaccount.google.com/security
# 2. Search "App Passwords" in the Security settings search bar
# 3. Generate a new App Password for "Mail" — Google gives you a 16-char code
# 4. Copy config_template.py to config.py, fill in your credentials, and
#    set EMAIL_ENABLED = True. config.py is listed in .gitignore so it will
#    never be uploaded to GitHub.
#
# WARNING: Never commit config.py or any file containing EMAIL_PASSWORD to GitHub.
# ============================================================================

import requests
import yfinance as yf
import pandas as pd
import numpy as np
import json
import math
import os
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timezone
import anthropic

# Email credentials are stored in config.py (excluded from git via .gitignore).
# Copy config_template.py to config.py and fill in your details to get started.
from config import EMAIL_ENABLED, EMAIL_FROM, EMAIL_TO, EMAIL_PASSWORD, ANTHROPIC_API_KEY

anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def _nan_to_none(value):
    """Return None if value is NaN or infinite; otherwise return value unchanged.
    Prevents JSON serialization errors when yfinance returns NaN floats."""
    if value is None:
        return None
    try:
        if math.isnan(value) or math.isinf(value):
            return None
    except TypeError:
        pass
    return value


# ============================================================================
# WATCHLIST — same 38 tickers as phase5_screener.py
# ============================================================================
WATCHLIST = [
    "CVX", "MU", "MSFT", "HAL", "WMT", "BAC", "DKNG", "CRGO", "RZLV",
    "RGTI", "ALAB", "CTRE", "SMCI", "SOUN", "BE", "TATT", "KLAR", "SNPS",
    "NVO", "TEM", "PLTR", "META", "AVGO", "TSLA", "AAPL", "RDDT", "HOOD",
    "DDOG", "INTC", "OKLO", "ISRG", "PAYC", "TTD", "AMD", "AMBA", "GOOG",
    "QBTS", "ORCL", "AMZN", "NFLX", "ADBE", "NVDA", "NOW", "PSUS"
]

SNAPSHOT_PATH = os.path.join(os.path.dirname(__file__), "monitor_snapshot.json")

# Mandatory email send times (local, hour, minute). Emails always go out within
# ±10 minutes of each window regardless of alert count.
_SCHEDULED_TIMES = [(8, 0), (12, 0), (16, 30)]
_SCHEDULED_LABELS = {
    (8, 0):   "Morning Briefing",
    (12, 0):  "Midday Briefing",
    (16, 30): "Afternoon Briefing",
}

WACC           = 0.095   # 9.5%
TERMINAL_G     = 0.030   # 3.0%
PROJECTION_YRS = 5


# ============================================================================
# STEP 1 — Data collection
# For each ticker, calculate current price, RSI, cross signals, screener score,
# DCF implied price, fundamental ratios, and next earnings date.
# All fields default to None so downstream code can handle missing data safely.
# ============================================================================

def _safe_loc(df, label):
    try:
        return df.loc[label]
    except KeyError:
        return None


def _calc_rsi_and_crosses(ticker_str):
    """
    Download 1 year of daily price data and return:
      - RSI (14-day Wilder EMA)
      - death_cross_active: bool — 50MA currently below 200MA
      - golden_cross_5d:    bool — golden cross (50MA crossed above 200MA) within last 5 days
      - death_cross_5d:     bool — death cross (50MA crossed below 200MA) within last 5 days
    Returns (None, None, None, None) if insufficient history.
    """
    raw   = yf.download(ticker_str, period="1y", interval="1d",
                        auto_adjust=True, progress=False)
    close = raw["Close"].squeeze().dropna()
    if len(close) < 210:
        return None, None, None, None

    # Wilder RSI
    delta  = close.diff()
    gains  = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)
    alpha  = 1 / 14
    avg_g  = gains.ewm(alpha=alpha, adjust=False).mean()
    avg_l  = losses.ewm(alpha=alpha, adjust=False).mean()
    rsi    = _nan_to_none(float((100 - 100 / (1 + avg_g / avg_l)).iloc[-1]))

    # Moving averages
    ma50  = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()

    death_cross_active = bool(ma50.iloc[-1] < ma200.iloc[-1])

    # Detect crosses in last 5 trading days
    # A golden cross on day d means: ma50[d] > ma200[d] AND ma50[d-1] <= ma200[d-1]
    window          = 5
    golden_cross_5d = False
    death_cross_5d  = False
    for i in range(-window, 0):
        above_today = ma50.iloc[i]     > ma200.iloc[i]
        above_prev  = ma50.iloc[i - 1] > ma200.iloc[i - 1]
        if above_today and not above_prev:
            golden_cross_5d = True
        if not above_today and above_prev:
            death_cross_5d = True

    return rsi, death_cross_active, golden_cross_5d, death_cross_5d


def _run_dcf(base_fcf, cagr):
    """
    Project FCF for PROJECTION_YRS years using cagr, compute terminal value,
    discount at WACC, return sum of PV. Returns None if inputs are invalid.
    """
    if base_fcf is None or cagr is None or base_fcf <= 0:
        return None
    # Cap CAGR to a reasonable range to avoid extreme projections
    growth = float(np.clip(cagr, -0.05, 0.25))
    pv_sum = 0.0
    fcf    = base_fcf
    for yr in range(1, PROJECTION_YRS + 1):
        fcf    *= (1 + growth)
        pv_sum += fcf / (1 + WACC) ** yr
    fcf_yr5 = base_fcf * (1 + growth) ** PROJECTION_YRS
    # Gordon Growth terminal value
    tv = fcf_yr5 * (1 + TERMINAL_G) / (WACC - TERMINAL_G)
    pv_sum += tv / (1 + WACC) ** PROJECTION_YRS
    return pv_sum


def _screener_score(metrics):
    """
    Replicate the 8-criterion screener from phase5_screener.py.
    Returns an integer 0–8.
    """
    gm   = metrics.get("gross_margin")
    nm   = metrics.get("net_margin")
    fcfc = metrics.get("fcf_conversion")
    cagr = metrics.get("rev_cagr")
    de   = metrics.get("debt_equity")
    eve  = metrics.get("ev_ebitda")
    rsi  = metrics.get("rsi")
    dc   = metrics.get("death_cross_active")

    checks = [
        gm   is not None and gm   > 0.40,
        nm   is not None and nm   > 0.10,
        fcfc is not None and fcfc > 0.75,
        cagr is not None and cagr > 0.05,
        de   is not None and de   < 2.00,
        eve  is not None and eve  < 25.0,
        rsi  is not None and rsi  < 50.0,
        dc   is not None and dc   == False,
    ]
    return sum(checks)


def collect_metrics(ticker_str):
    """
    Fetch all monitored metrics for a single ticker.
    Returns a dict; any unavailable value is stored as None.
    """
    result = {
        "price":              None,
        "rsi":                None,
        "death_cross_active": None,
        "golden_cross_5d":    None,
        "death_cross_5d":     None,
        "gross_margin":       None,
        "net_margin":         None,
        "fcf_conversion":     None,
        "rev_cagr":           None,
        "debt_equity":        None,
        "ev_ebitda":          None,
        "screener_score":     None,
        "dcf_implied":        None,
        "dcf_vs_market_pct":  None,
        "next_earnings_date": None,
        "days_to_earnings":   None,
    }

    t    = yf.Ticker(ticker_str)
    info = t.info

    # --- Current price ---
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    result["price"] = _nan_to_none(float(price)) if price else None

    # --- Financial statements ---
    is_  = t.financials.dropna(axis=1, how="all")
    bs   = t.balance_sheet.dropna(axis=1, how="all")
    cf   = t.cashflow.dropna(axis=1, how="all")

    if not is_.empty and not bs.empty and not cf.empty:
        fy0 = is_.columns[0]

        rev_row    = _safe_loc(is_, "Total Revenue")
        gross_row  = _safe_loc(is_, "Gross Profit")
        ni_row     = _safe_loc(is_, "Net Income")
        ebitda_row = _safe_loc(is_, "EBITDA")
        fcf_row    = _safe_loc(cf,  "Free Cash Flow")
        debt_row   = _safe_loc(bs,  "Total Debt")
        equity_row = _safe_loc(bs,  "Stockholders Equity")
        cash_row   = _safe_loc(bs,  "Cash And Cash Equivalents")

        revenue    = _nan_to_none(float(rev_row[fy0]))    if rev_row    is not None else None
        gross      = _nan_to_none(float(gross_row[fy0]))  if gross_row  is not None else None
        net_income = _nan_to_none(float(ni_row[fy0]))     if ni_row     is not None else None
        ebitda     = _nan_to_none(float(ebitda_row[fy0])) if ebitda_row is not None else None
        fcf        = _nan_to_none(float(fcf_row[fy0]))    if fcf_row    is not None else None
        debt       = _nan_to_none(float(debt_row[fy0]))   if debt_row   is not None else None
        equity     = _nan_to_none(float(equity_row[fy0])) if equity_row is not None else None
        cash_val   = _nan_to_none(float(cash_row[fy0]))   if cash_row   is not None else None

        # Margins
        if revenue and revenue != 0:
            result["gross_margin"] = _nan_to_none(gross      / revenue) if gross      is not None else None
            result["net_margin"]   = _nan_to_none(net_income / revenue) if net_income is not None else None

        # FCF conversion
        if fcf is not None and net_income and net_income != 0:
            result["fcf_conversion"] = _nan_to_none(fcf / net_income)

        # Debt/equity
        if debt is not None and equity and equity != 0:
            result["debt_equity"] = _nan_to_none(debt / equity)

        # Revenue CAGR (oldest to newest available year, up to 4 cols)
        if rev_row is not None and len(is_.columns) >= 4:
            rev_old = _nan_to_none(float(rev_row.iloc[-1]))
            rev_new = _nan_to_none(float(rev_row.iloc[0]))
            n_yrs   = len(is_.columns) - 1
            if rev_old is not None and rev_new is not None and rev_old > 0 and rev_new > 0:
                result["rev_cagr"] = _nan_to_none((rev_new / rev_old) ** (1 / n_yrs) - 1)

        # EV/EBITDA
        shares = info.get("sharesOutstanding")
        if (result["price"] and shares and debt is not None
                and cash_val is not None and ebitda and ebitda != 0):
            mcap = result["price"] * shares
            ev   = mcap + debt - cash_val
            result["ev_ebitda"] = _nan_to_none(ev / ebitda)

        # DCF — 4-year average FCF as base
        if fcf_row is not None and len(cf.columns) >= 2:
            fcf_vals = [float(fcf_row.iloc[i]) for i in range(min(4, len(cf.columns)))
                        if not pd.isna(fcf_row.iloc[i]) and fcf_row.iloc[i] > 0]
            base_fcf   = _nan_to_none(float(np.mean(fcf_vals))) if fcf_vals else None
            dcf_equity = _run_dcf(base_fcf, result["rev_cagr"])
            if dcf_equity is not None and shares and shares > 0:
                dcf_per_share = _nan_to_none(dcf_equity / shares)
                result["dcf_implied"] = round(dcf_per_share, 2) if dcf_per_share is not None else None
                if result["price"] and result["price"] > 0 and dcf_per_share is not None:
                    result["dcf_vs_market_pct"] = _nan_to_none(round(
                        (dcf_per_share - result["price"]) / result["price"] * 100, 2
                    ))

    # --- Technical indicators ---
    rsi, dc_active, gc_5d, dc_5d = _calc_rsi_and_crosses(ticker_str)
    result["rsi"]                = rsi
    result["death_cross_active"] = dc_active
    result["golden_cross_5d"]    = gc_5d
    result["death_cross_5d"]     = dc_5d

    # --- Screener score ---
    result["screener_score"] = _screener_score(result)

    # --- Next earnings date ---
    try:
        cal = t.calendar
        if isinstance(cal, pd.DataFrame):
            if "Earnings Date" in cal.index:
                val = cal.loc["Earnings Date"].iloc[0]
                if pd.notna(val):
                    ed = pd.Timestamp(val).date()
                    result["next_earnings_date"] = str(ed)
                    result["days_to_earnings"]   = (ed - datetime.now(timezone.utc).date()).days
        elif isinstance(cal, dict):
            ed_raw = cal.get("Earnings Date")
            if ed_raw:
                ed_list = ed_raw if isinstance(ed_raw, list) else [ed_raw]
                today   = datetime.now(timezone.utc).date()
                future  = [pd.Timestamp(d).date() for d in ed_list
                           if pd.Timestamp(d).date() >= today]
                if future:
                    ed = min(future)
                    result["next_earnings_date"] = str(ed)
                    result["days_to_earnings"]   = (ed - today).days
    except Exception:
        pass

    return result


# ============================================================================
# AI SUMMARY — get_ai_summary
# Pull fresh data for any ticker and return a 3-sentence Claude analysis.
# Raises ValueError('Ticker not found') if yfinance has no data for the symbol.
# Raises RuntimeError('Analysis unavailable — API error') if the API call fails.
# ============================================================================

def get_ai_summary(ticker):
    """
    Pull fresh data for ticker and return a 3-sentence Claude AI analysis.
    Raises ValueError if ticker not found, RuntimeError if API call fails.
    """
    try:
        t    = yf.Ticker(ticker)
        info = t.info

        company_name = info.get("longName") or info.get("shortName")
        price_raw    = info.get("currentPrice") or info.get("regularMarketPrice")

        if not company_name and not price_raw:
            raise ValueError("Ticker not found")

        price = _nan_to_none(float(price_raw)) if price_raw else None

        # 1-year cumulative return
        raw   = yf.download(ticker, period="1y", interval="1d",
                            auto_adjust=True, progress=False)
        close = raw["Close"].squeeze().dropna()
        cum_return_1y = None
        if len(close) >= 2:
            cum_return_1y = float((1 + close.pct_change().dropna()).cumprod().iloc[-1] - 1)

        # Financial statements
        is_  = t.financials.dropna(axis=1, how="all")
        bs   = t.balance_sheet.dropna(axis=1, how="all")
        cf   = t.cashflow.dropna(axis=1, how="all")

        gross_margin = op_margin = net_margin = fcf_conversion = debt_equity = None
        rev_cagr = base_fcf = dcf_implied = None

        if not is_.empty and not bs.empty and not cf.empty:
            fy0 = is_.columns[0]

            rev_row   = _safe_loc(is_, "Total Revenue")
            gross_row = _safe_loc(is_, "Gross Profit")
            op_row    = _safe_loc(is_, "Operating Income")
            ni_row    = _safe_loc(is_, "Net Income")
            fcf_row   = _safe_loc(cf,  "Free Cash Flow")
            debt_row  = _safe_loc(bs,  "Total Debt")
            eq_row    = _safe_loc(bs,  "Stockholders Equity")

            revenue    = _nan_to_none(float(rev_row[fy0]))    if rev_row    is not None else None
            gross      = _nan_to_none(float(gross_row[fy0]))  if gross_row  is not None else None
            op_inc     = _nan_to_none(float(op_row[fy0]))     if op_row     is not None else None
            net_income = _nan_to_none(float(ni_row[fy0]))     if ni_row     is not None else None
            fcf        = _nan_to_none(float(fcf_row[fy0]))    if fcf_row    is not None else None
            debt       = _nan_to_none(float(debt_row[fy0]))   if debt_row   is not None else None
            equity     = _nan_to_none(float(eq_row[fy0]))     if eq_row     is not None else None

            if revenue and revenue != 0:
                gross_margin = _nan_to_none(gross  / revenue) if gross   is not None else None
                op_margin    = _nan_to_none(op_inc  / revenue) if op_inc  is not None else None
                net_margin   = _nan_to_none(net_income / revenue) if net_income is not None else None

            if fcf is not None and net_income and net_income != 0:
                fcf_conversion = _nan_to_none(fcf / net_income)

            if debt is not None and equity and equity != 0:
                debt_equity = _nan_to_none(debt / equity)

            # Revenue CAGR
            if rev_row is not None and len(is_.columns) >= 4:
                rev_old = _nan_to_none(float(rev_row.iloc[-1]))
                rev_new = _nan_to_none(float(rev_row.iloc[0]))
                n_yrs   = len(is_.columns) - 1
                if rev_old and rev_new and rev_old > 0 and rev_new > 0:
                    rev_cagr = _nan_to_none((rev_new / rev_old) ** (1 / n_yrs) - 1)

            # 4-year avg FCF for DCF
            shares = info.get("sharesOutstanding")
            if fcf_row is not None and len(cf.columns) >= 2:
                fcf_vals = [float(fcf_row.iloc[i]) for i in range(min(4, len(cf.columns)))
                            if not pd.isna(fcf_row.iloc[i]) and fcf_row.iloc[i] > 0]
                base_fcf = _nan_to_none(float(np.mean(fcf_vals))) if fcf_vals else None

            dcf_equity = _run_dcf(base_fcf, rev_cagr)
            if dcf_equity is not None and shares and shares > 0:
                dcf_per_share = _nan_to_none(dcf_equity / shares)
                dcf_implied   = round(dcf_per_share, 2) if dcf_per_share is not None else None

        # RSI and death cross
        rsi, death_cross_active, _, _ = _calc_rsi_and_crosses(ticker)

        # Screener score (reuse same 8-criterion logic)
        score = _screener_score({
            "gross_margin":       gross_margin,
            "net_margin":         net_margin,
            "fcf_conversion":     fcf_conversion,
            "rev_cagr":           rev_cagr,
            "debt_equity":        debt_equity,
            "ev_ebitda":          None,
            "rsi":                rsi,
            "death_cross_active": death_cross_active,
        })

        # Days to next earnings
        days_to_earnings = None
        try:
            cal = t.calendar
            if isinstance(cal, pd.DataFrame):
                if "Earnings Date" in cal.index:
                    val = cal.loc["Earnings Date"].iloc[0]
                    if pd.notna(val):
                        ed = pd.Timestamp(val).date()
                        days_to_earnings = (ed - datetime.now(timezone.utc).date()).days
            elif isinstance(cal, dict):
                ed_raw = cal.get("Earnings Date")
                if ed_raw:
                    ed_list = ed_raw if isinstance(ed_raw, list) else [ed_raw]
                    today   = datetime.now(timezone.utc).date()
                    future  = [pd.Timestamp(d).date() for d in ed_list
                               if pd.Timestamp(d).date() >= today]
                    if future:
                        days_to_earnings = (min(future) - today).days
        except Exception:
            pass

        def _fmt_pct(v):   return f"{v*100:.1f}%"  if v is not None else "N/A"
        def _fmt_num(v):   return f"{v:.2f}"        if v is not None else "N/A"
        def _fmt_price(v): return f"${v:.2f}"       if v is not None else "N/A"

        dc_str = ("Yes" if death_cross_active else "No") if death_cross_active is not None else "N/A"

        prompt = (
            f"You are a concise financial analyst. Analyze {company_name} ({ticker}) "
            f"based on the following data:\n\n"
            f"Current price: {_fmt_price(price)}\n"
            f"1-year cumulative return: {_fmt_pct(cum_return_1y)}\n"
            f"Gross margin: {_fmt_pct(gross_margin)}\n"
            f"Operating margin: {_fmt_pct(op_margin)}\n"
            f"Net margin: {_fmt_pct(net_margin)}\n"
            f"FCF conversion: {_fmt_num(fcf_conversion)}x\n"
            f"Debt to equity: {_fmt_num(debt_equity)}x\n"
            f"14-day RSI (Wilder): {_fmt_num(rsi)}\n"
            f"Death cross active: {dc_str}\n"
            f"Screener score: {score}/8\n"
            f"DCF implied price: {_fmt_price(dcf_implied)}\n"
            f"Days until next earnings: {days_to_earnings if days_to_earnings is not None else 'N/A'}\n\n"
            f"Respond in exactly 3 sentences: "
            f"Sentence 1 states what the current technical and fundamental picture means for this specific business. "
            f"Sentence 2 states whether the current situation reflects a fundamental problem, a macro-driven event, or a technical signal. "
            f"Sentence 3 states the single most important thing to watch over the next 30 days."
        )

    except ValueError:
        raise ValueError("Ticker not found")
    except Exception:
        raise ValueError("Ticker not found")

    try:
        message = anthropic_client.messages.create(
            model="claude-opus-4-5",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception:
        raise RuntimeError("Analysis unavailable — API error")


# ============================================================================
# STEP 2 — Snapshot management
# Persist the current run's metrics to JSON so the next run can detect changes.
# The snapshot includes a top-level timestamp alongside per-ticker data.
# ============================================================================

def save_snapshot(data: dict):
    """Write metrics dict + current UTC timestamp to SNAPSHOT_PATH."""
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tickers":   data,
    }
    os.makedirs(os.path.dirname(SNAPSHOT_PATH), exist_ok=True)
    with open(SNAPSHOT_PATH, "w") as f:
        json.dump(payload, f, indent=2)


def load_snapshot():
    """
    Load the previous snapshot from SNAPSHOT_PATH.
    Returns (timestamp_str, tickers_dict) or (None, None) if no file exists.
    """
    if not os.path.exists(SNAPSHOT_PATH):
        return None, None
    with open(SNAPSHOT_PATH) as f:
        payload = json.load(f)
    return payload.get("timestamp"), payload.get("tickers", {})


# ============================================================================
# STEP 3 — Alert generation
# Compare current metrics to the previous snapshot and classify changes into
# three priority levels: HIGH (risk / deterioration), WATCH (softer concerns),
# and POSITIVE (improving signals worth attention).
# ============================================================================

def generate_alerts(current: dict, previous: dict):
    """
    Returns a dict keyed by ticker, each value a list of (priority, message) tuples.
    priority is one of: "HIGH", "WATCH", "POSITIVE"
    """
    alerts = {}

    for ticker, cur in current.items():
        prev = previous.get(ticker, {})
        ticker_alerts = []

        def _c(key):
            return cur.get(key)

        def _p(key):
            return prev.get(key)

        # ── HIGH PRIORITY ──────────────────────────────────────────────────
        # Death cross just formed
        if _c("death_cross_5d") and not _p("death_cross_5d"):
            ticker_alerts.append(("HIGH", "Death cross formed in last 5 days (50MA crossed below 200MA)"))

        # RSI dropped below 30
        if (_p("rsi") is not None and _c("rsi") is not None
                and _p("rsi") >= 30 and _c("rsi") < 30):
            ticker_alerts.append(("HIGH", f"RSI dropped below 30 — now {_c('rsi'):.1f} (was {_p('rsi'):.1f})"))

        # Price dropped more than 10%
        if (_p("price") is not None and _c("price") is not None
                and _p("price") > 0):
            price_chg = (_c("price") - _p("price")) / _p("price") * 100
            if price_chg <= -10:
                ticker_alerts.append(("HIGH",
                    f"Price dropped {price_chg:.1f}% since last run "
                    f"(${_p('price'):.2f} → ${_c('price'):.2f})"))

        # Screener score dropped by 2+
        if (_p("screener_score") is not None and _c("screener_score") is not None
                and _p("screener_score") - _c("screener_score") >= 2):
            ticker_alerts.append(("HIGH",
                f"Screener score dropped {_p('screener_score')} → {_c('screener_score')}/8"))

        # ── WATCH ──────────────────────────────────────────────────────────
        # RSI crossed above 70
        if (_p("rsi") is not None and _c("rsi") is not None
                and _p("rsi") < 70 and _c("rsi") >= 70):
            ticker_alerts.append(("WATCH", f"RSI crossed above 70 — now {_c('rsi'):.1f} (overbought)"))

        # FCF conversion dropped below 0.75x
        if (_p("fcf_conversion") is not None and _c("fcf_conversion") is not None
                and _p("fcf_conversion") >= 0.75 and _c("fcf_conversion") < 0.75):
            ticker_alerts.append(("WATCH",
                f"FCF conversion fell below 0.75x — now {_c('fcf_conversion'):.2f}x"))

        # Gross margin compressed more than 1.5pp
        if (_p("gross_margin") is not None and _c("gross_margin") is not None):
            gm_chg = (_c("gross_margin") - _p("gross_margin")) * 100
            if gm_chg <= -1.5:
                ticker_alerts.append(("WATCH",
                    f"Gross margin compressed {gm_chg:.1f}pp "
                    f"({_p('gross_margin')*100:.1f}% → {_c('gross_margin')*100:.1f}%)"))

        # DCF implied price more than 20% below market
        if _c("dcf_vs_market_pct") is not None and _c("dcf_vs_market_pct") <= -20:
            ticker_alerts.append(("WATCH",
                f"DCF implies ${_c('dcf_implied'):.2f} — "
                f"{abs(_c('dcf_vs_market_pct')):.0f}% below market price"))

        # ── POSITIVE ───────────────────────────────────────────────────────
        # Golden cross just formed
        if _c("golden_cross_5d") and not _p("golden_cross_5d"):
            ticker_alerts.append(("POSITIVE", "Golden cross formed in last 5 days (50MA crossed above 200MA)"))

        # RSI recovering from oversold — between 30–40 and rising
        if (_p("rsi") is not None and _c("rsi") is not None
                and _p("rsi") < 30 and 30 <= _c("rsi") <= 40):
            ticker_alerts.append(("POSITIVE",
                f"RSI recovering from oversold — now {_c('rsi'):.1f} (was {_p('rsi'):.1f})"))

        # Screener score improved by 2+
        if (_p("screener_score") is not None and _c("screener_score") is not None
                and _c("screener_score") - _p("screener_score") >= 2):
            ticker_alerts.append(("POSITIVE",
                f"Screener score improved {_p('screener_score')} → {_c('screener_score')}/8"))

        # Earnings within 7 days
        if (_c("days_to_earnings") is not None
                and 0 <= _c("days_to_earnings") <= 7):
            ticker_alerts.append(("POSITIVE",
                f"Earnings in {_c('days_to_earnings')} day(s) — {_c('next_earnings_date')} (upcoming catalyst)"))

        if ticker_alerts:
            alerts[ticker] = ticker_alerts

    return alerts


# ============================================================================
# STEP 4 — Output
# Print results grouped by priority level.  Only tickers with at least one
# alert appear in the output; unchanged tickers are counted in the summary.
# A separate text formatter builds the same content as a string for email.
# ============================================================================

def _format_alerts_text(alerts: dict, current: dict, prev_timestamp) -> str:
    """
    Build the full alert report as a plain-text string.
    Used by both print_alerts (stdout) and the email body.
    """
    lines = []
    buckets  = {"HIGH": [], "WATCH": [], "POSITIVE": []}
    dividers = {"HIGH": "!", "WATCH": "-", "POSITIVE": "+"}
    labels   = {"HIGH": "HIGH PRIORITY", "WATCH": "WATCH", "POSITIVE": "POSITIVE"}

    for ticker, ticker_alerts in alerts.items():
        for priority, msg in ticker_alerts:
            buckets[priority].append((ticker, msg))

    lines.append("")
    for level in ("HIGH", "WATCH", "POSITIVE"):
        items = buckets[level]
        div   = dividers[level]
        lines.append(f"  {'─'*66}")
        lines.append(f"  {labels[level]}  ({len(items)} alert(s))")
        lines.append(f"  {'─'*66}")
        if not items:
            lines.append("  No alerts.")
        else:
            for ticker, msg in items:
                price = current.get(ticker, {}).get("price")
                price_str = f"  ${price:.2f}" if price is not None else ""
                lines.append(f"  [{div}] {ticker:<6}{price_str:<10}  {msg}")
        lines.append("")

    # Summary
    tickers_with_alerts = len(alerts)
    tickers_unchanged   = len(current) - tickers_with_alerts
    total_alerts        = sum(len(v) for v in alerts.values())
    lines.append(f"  {'─'*66}")
    lines.append(
        f"  SUMMARY  |  {len(current)} tickers monitored  |  "
        f"{tickers_with_alerts} with alerts ({total_alerts} total)  |  "
        f"{tickers_unchanged} unchanged"
    )

    if prev_timestamp:
        try:
            prev_dt    = datetime.fromisoformat(prev_timestamp)
            prev_local = prev_dt.strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            prev_local = prev_timestamp
        lines.append(f"  Compared against snapshot from: {prev_local}")
    lines.append(f"  {'─'*66}")
    lines.append("")
    return "\n".join(lines)


def get_sp500_losers(top_n: int = 10):
    """
    Fetch S&P 500 components from Wikipedia (using a browser User-Agent so the
    request isn't blocked), batch-download ~1 month of daily prices, and return
    the top_n losers for three windows:
      - daily:   today vs yesterday          (lookback = 1)
      - weekly:  today vs 5 trading days ago (lookback = 5)
      - monthly: today vs ~21 trading days ago (lookback = 21)
    Returns a dict with keys "daily", "weekly", "monthly", each a list of
    {ticker, name, price, change_pct} sorted ascending by change_pct.
    Returns {"daily": [], "weekly": [], "monthly": []} on any failure.
    """
    empty = {"daily": [], "weekly": [], "monthly": []}

    try:
        headers  = {"User-Agent": "Mozilla/5.0 (compatible; sp500-monitor/1.0)"}
        response = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers=headers, timeout=15,
        )
        response.raise_for_status()
        sp_df   = pd.read_html(response.text)[0]
        tickers = sp_df["Symbol"].str.replace(".", "-", regex=False).tolist()
        names   = dict(zip(
            sp_df["Symbol"].str.replace(".", "-", regex=False),
            sp_df["Security"],
        ))
    except Exception:
        return empty

    try:
        raw = yf.download(
            tickers, period="1mo", interval="1d",
            auto_adjust=True, progress=False, threads=True,
        )
        if raw.empty:
            return empty
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    except Exception:
        return empty

    windows = {"daily": 1, "weekly": 5, "monthly": 21}
    result  = {}

    for window_name, lookback in windows.items():
        valid = close.dropna(axis=1, thresh=lookback + 1)
        if valid.shape[0] < lookback + 1:
            result[window_name] = []
            continue

        latest = valid.iloc[-1]
        base   = valid.iloc[-(lookback + 1)]
        chg    = ((latest - base) / base * 100).dropna()

        rows = []
        for ticker, pct in chg.items():
            rows.append({
                "ticker":     ticker,
                "name":       names.get(ticker, ticker),
                "price":      round(float(latest[ticker]), 2),
                "change_pct": round(float(pct), 2),
            })
        rows.sort(key=lambda x: x["change_pct"])
        result[window_name] = rows[:top_n]

    return result


def _format_sp500_losers(losers_by_window: dict) -> str:
    """Build a plain-text section listing S&P 500 top losers for each window."""
    window_labels = {
        "daily":   "DAILY  (today vs yesterday)",
        "weekly":  "WEEKLY  (today vs 5 trading days ago)",
        "monthly": "MONTHLY  (today vs ~21 trading days ago)",
    }
    lines = []
    for key in ("daily", "weekly", "monthly"):
        losers = losers_by_window.get(key, [])
        lines.append(f"  {'─'*66}")
        lines.append(f"  S&P 500 TOP LOSERS — {window_labels[key]}")
        lines.append(f"  {'─'*66}")
        if not losers:
            lines.append("  Data unavailable.")
        else:
            for item in losers:
                name_trunc = item["name"][:28]
                lines.append(
                    f"  {item['ticker']:<7}  {name_trunc:<28}  "
                    f"${item['price']:<9.2f}  {item['change_pct']:+.2f}%"
                )
        lines.append("")
    return "\n".join(lines)


def _format_watchlist_snapshot(current: dict) -> str:
    """
    Build a watchlist snapshot section sorted by screener score descending.
    Each ticker is one line:
      GOOG    $273.14   RSI 24.0   DeathX No    Score 7/8   Earnings 22 days
    """
    lines = []
    lines.append(f"  {'─'*66}")
    lines.append("  WATCHLIST SNAPSHOT  (sorted by score, highest first)")
    lines.append(f"  {'─'*66}")

    sorted_tickers = sorted(
        current.items(),
        key=lambda x: x[1].get("screener_score") or 0,
        reverse=True,
    )

    for ticker, m in sorted_tickers:
        price_str = f"${m['price']:.2f}"    if m.get("price")    is not None else "N/A"
        rsi_str   = f"RSI {m['rsi']:.1f}"  if m.get("rsi")      is not None else "RSI N/A"
        dc        = m.get("death_cross_active")
        dc_str    = f"DeathX {'Yes' if dc else 'No '}" if dc is not None else "DeathX N/A"
        score     = m.get("screener_score")
        score_str = f"Score {score}/8" if score is not None else "Score N/A"
        dte       = m.get("days_to_earnings")
        earn_str  = f"Earnings {dte} days" if dte is not None else "Earnings N/A"

        lines.append(
            f"  {ticker:<7} {price_str:<10}  {rsi_str:<11}  {dc_str:<12}  "
            f"{score_str:<10}  {earn_str}"
        )

    lines.append(f"  {'─'*66}")
    lines.append("")
    return "\n".join(lines)


def print_alerts(alerts: dict, current: dict, prev_timestamp):
    """Print the formatted alert report to stdout."""
    print(_format_alerts_text(alerts, current, prev_timestamp))


# ============================================================================
# EMAIL — send_email_alert
# Connects to Gmail via STARTTLS and delivers a plain-text message.
# Failures print a warning but never crash the monitor.
# ============================================================================

def _scheduled_briefing_label():
    """Return the briefing label if now is within 10 min of a scheduled send time, else None."""
    now = datetime.now()
    now_min = now.hour * 60 + now.minute
    for h, m in _SCHEDULED_TIMES:
        if abs(now_min - (h * 60 + m)) <= 10:
            return _SCHEDULED_LABELS[(h, m)]
    return None


def send_email_alert(subject: str, body: str):
    """Send a plain-text email via Gmail SMTP. No-op if EMAIL_ENABLED is False."""
    if not EMAIL_ENABLED:
        return

    try:
        msg = MIMEMultipart()
        msg["From"]    = EMAIL_FROM
        msg["To"]      = EMAIL_TO
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())

        print(f"  Email sent → {EMAIL_TO}")
    except Exception as exc:
        print(f"  WARNING: Email could not be sent ({exc}). Monitor output was not emailed.")


# ============================================================================
# STEP 5 — First-run handling
# When no prior snapshot exists, explain that a baseline has been established.
# The snapshot is always saved so subsequent runs can detect changes.
# ============================================================================

# ============================================================================
# INTERACTIVE LOOP — run_interactive_loop
# This function only runs when the script is executed manually by a human and
# is automatically skipped during cron scheduled runs.
# ============================================================================

def run_interactive_loop(interactive_analyses: list):
    while True:
        print()
        answer = input("Would you like an AI summary on a company? (yes/no): ").strip().lower()
        if answer not in ("yes", "y"):
            break

        ticker = input("Enter ticker symbol: ").strip().upper()

        try:
            summary = get_ai_summary(ticker)
            bar = "─" * 48
            print(f"\n  ── {ticker} — AI Analysis {bar[:max(0, 46 - len(ticker))]}")
            print(f"  {summary}")
            print(f"  {'─' * 66}")
            interactive_analyses.append({
                "ticker":   ticker,
                "analysis": summary,
            })
        except ValueError:
            print("  Ticker not found — please try again")
            continue
        except RuntimeError:
            print("  Analysis unavailable — API error")
            answer2 = input("Would you like to try another company? (yes/no): ").strip().lower()
            if answer2 not in ("yes", "y"):
                break


def main():
    print("=" * 70)
    print(f"  PORTFOLIO MONITOR  |  {len(WATCHLIST)} tickers")
    print("=" * 70)
    print("  Collecting data... (this may take 2–3 minutes)\n")

    t_start = time.time()

    # Load prior snapshot before overwriting it
    prev_timestamp, prev_data = load_snapshot()
    first_run = (prev_data is None)

    # Collect fresh metrics for every ticker
    current_data = {}
    for ticker in WATCHLIST:
        try:
            current_data[ticker] = collect_metrics(ticker)
        except Exception:
            current_data[ticker] = {k: None for k in [
                "price", "rsi", "death_cross_active", "golden_cross_5d",
                "death_cross_5d", "gross_margin", "net_margin", "fcf_conversion",
                "rev_cagr", "debt_equity", "ev_ebitda", "screener_score",
                "dcf_implied", "dcf_vs_market_pct", "next_earnings_date",
                "days_to_earnings",
            ]}

    elapsed = time.time() - t_start

    # Always save the current snapshot
    save_snapshot(current_data)

    if first_run:
        print("  ── FIRST RUN ──────────────────────────────────────────────────────")
        print("  Baseline snapshot created. Run the monitor again to start detecting")
        print("  changes, alerts, and signal shifts across your watchlist.")
        print(f"  Snapshot saved → {SNAPSHOT_PATH}")
        print(f"  {'─'*66}")
        print(f"  {len(WATCHLIST)} tickers collected in {elapsed:.1f}s\n")
        return

    # Generate alerts and build output
    alerts      = generate_alerts(current_data, prev_data)
    alert_text  = _format_alerts_text(alerts, current_data, prev_timestamp)
    snap_text   = _format_watchlist_snapshot(current_data)

    # Fetch S&P 500 losers (daily / weekly / monthly)
    print("  Fetching S&P 500 losers...\n")
    sp500_losers     = get_sp500_losers()
    sp500_loser_text = _format_sp500_losers(sp500_losers)

    # Print to terminal
    print(alert_text)
    print(sp500_loser_text)

    # Build AI context for top 1-2 high priority tickers (included in email body)
    ai_context = ""
    high_tickers = [
        ticker for ticker, ta in alerts.items()
        if any(p == "HIGH" for p, _ in ta)
    ][:2]
    if high_tickers:
        ai_lines = [f"\n  {'─'*66}", "  AI ANALYSIS — HIGH PRIORITY TICKERS", f"  {'─'*66}"]
        for t in high_tickers:
            try:
                summary = get_ai_summary(t)
                ai_lines.append(f"\n  {t} — AI Analysis")
                ai_lines.append(f"  {summary}")
            except (ValueError, RuntimeError):
                pass
        ai_lines.append(f"  {'─'*66}\n")
        ai_context = "\n".join(ai_lines)

    # Decide whether to send email
    if EMAIL_ENABLED:
        total_alerts   = sum(len(v) for v in alerts.values())
        high_count     = sum(
            1 for ta in alerts.values()
            for priority, _ in ta if priority == "HIGH"
        )

        # Always send at 8 AM, 12 PM, and 4:30 PM; also send any time there are alerts.
        briefing_label = _scheduled_briefing_label()
        should_send    = briefing_label is not None or total_alerts > 0

        if should_send:
            run_date     = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            briefing_tag = f"  [{briefing_label}]" if briefing_label else ""
            subject = (
                f"Portfolio Monitor — {run_date} — "
                f"{total_alerts} alert(s) — {high_count} high priority"
                f"{briefing_tag}"
            )
            body = alert_text + snap_text + sp500_loser_text + ai_context
            send_email_alert(subject, body)
        else:
            print("  No alerts detected — email suppressed.")

    print(f"  Data collected in {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
