import yfinance as yf
import pandas as pd
import numpy as np
import time

# ============================================================================
# WATCHLIST — edit only this line to add or remove stocks
# ============================================================================
WATCHLIST = [
    "CVX", "MU", "MSFT", "HAL", "WMT", "BAC", "DKNG", "CRGO", "RZLV",
    "RGTI", "ALAB", "CTRE", "SMCI", "SOUN", "BE", "TATT", "KLAR", "SNPS",
    "NVO", "TEM", "PLTR", "META", "AVGO", "TSLA", "AAPL", "RDDT", "HOOD",
    "DDOG", "INTC", "OKLO", "ISRG", "PAYC", "TTD", "AMD", "AMBA", "GOOG",
    "QBTS", "ORCL"
]

# ============================================================================
# SCREENING CRITERIA
# ============================================================================
# F1–F5: Fundamental quality filters — measure business durability and safety
# V1:    Valuation filter — avoid overpaying relative to cash generation
# T1–T2: Technical filters — time entry to avoid catching falling knives

CRITERIA = {
    "F1 Gross Margin >40%":    lambda r: r["gross_margin"]   > 0.40  if r["gross_margin"]   is not None else False,
    "F2 Net Margin >10%":      lambda r: r["net_margin"]     > 0.10  if r["net_margin"]     is not None else False,
    "F3 FCF Conv >0.75x":      lambda r: r["fcf_conversion"] > 0.75  if r["fcf_conversion"] is not None else False,
    "F4 Rev CAGR >5%":         lambda r: r["rev_cagr"]       > 0.05  if r["rev_cagr"]       is not None else False,
    "F5 D/E <2.0x":            lambda r: r["debt_equity"]    < 2.00  if r["debt_equity"]    is not None else False,
    "V1 EV/EBITDA <25x":       lambda r: r["ev_ebitda"]      < 25.0  if r["ev_ebitda"]      is not None else False,
    "T1 RSI <50":              lambda r: r["rsi"]            < 50.0  if r["rsi"]            is not None else False,
    "T2 No Death Cross":       lambda r: r["death_cross"]    == False if r["death_cross"]   is not None else False,
}

CRITERIA_KEYS = list(CRITERIA.keys())

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calc_rsi_and_ma(ticker_str):
    """Download 1 year of daily prices; return RSI(14) and death cross bool."""
    raw   = yf.download(ticker_str, period="1y", interval="1d",
                        auto_adjust=True, progress=False)
    close = raw["Close"].squeeze().dropna()
    if len(close) < 210:
        return None, None

    # Wilder RSI
    delta  = close.diff()
    gains  = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)
    alpha  = 1 / 14
    avg_g  = gains.ewm(alpha=alpha, adjust=False).mean()
    avg_l  = losses.ewm(alpha=alpha, adjust=False).mean()
    rsi    = (100 - 100 / (1 + avg_g / avg_l)).iloc[-1]

    # Death cross: 50MA below 200MA today
    ma50   = close.rolling(50).mean().iloc[-1]
    ma200  = close.rolling(200).mean().iloc[-1]
    death  = bool(ma50 < ma200)

    return float(rsi), death


def safe_loc(df, label):
    """Return df.loc[label] or None if label not present."""
    try:
        return df.loc[label]
    except KeyError:
        return None


def analyse_ticker(ticker_str):
    """
    Pull fundamental and technical data for one ticker.
    Returns a dict of raw metrics (None for any missing value).
    """
    t    = yf.Ticker(ticker_str)
    info = t.info

    is_  = t.financials.dropna(axis=1, how="all")
    bs   = t.balance_sheet.dropna(axis=1, how="all")
    cf   = t.cashflow.dropna(axis=1, how="all")

    if is_.empty or bs.empty or cf.empty:
        raise ValueError("Empty financial statement")

    fy0 = is_.columns[0]  # most recent fiscal year

    # --- Income statement ---
    rev_row    = safe_loc(is_, "Total Revenue")
    gross_row  = safe_loc(is_, "Gross Profit")
    ni_row     = safe_loc(is_, "Net Income")
    ebitda_row = safe_loc(is_, "EBITDA")

    revenue    = float(rev_row[fy0])    if rev_row    is not None else None
    gross      = float(gross_row[fy0])  if gross_row  is not None else None
    net_income = float(ni_row[fy0])     if ni_row     is not None else None
    ebitda     = float(ebitda_row[fy0]) if ebitda_row is not None else None

    # Revenue CAGR: use oldest available year vs most recent (up to 4 cols)
    rev_cagr = None
    if rev_row is not None and len(is_.columns) >= 4:
        rev_old  = float(rev_row.iloc[-1])
        rev_new  = float(rev_row.iloc[0])
        n_years  = len(is_.columns) - 1
        if rev_old > 0 and rev_new > 0:
            rev_cagr = (rev_new / rev_old) ** (1 / n_years) - 1

    # --- Balance sheet ---
    debt_row   = safe_loc(bs, "Total Debt")
    equity_row = safe_loc(bs, "Stockholders Equity")
    cash_row   = safe_loc(bs, "Cash And Cash Equivalents")

    debt   = float(debt_row[fy0])   if debt_row   is not None else None
    equity = float(equity_row[fy0]) if equity_row is not None else None
    cash   = float(cash_row[fy0])   if cash_row   is not None else None

    # --- Cash flow ---
    fcf_row = safe_loc(cf, "Free Cash Flow")
    fcf     = float(fcf_row[fy0]) if fcf_row is not None else None

    # --- Derived metrics ---
    gross_margin   = gross      / revenue   if (gross      is not None and revenue and revenue != 0) else None
    net_margin     = net_income / revenue   if (net_income is not None and revenue and revenue != 0) else None
    fcf_conversion = fcf        / net_income if (fcf       is not None and net_income and net_income != 0) else None
    debt_equity    = debt       / equity    if (debt       is not None and equity and equity != 0) else None

    # EV / EBITDA
    ev_ebitda = None
    price  = info.get("currentPrice")
    shares = info.get("sharesOutstanding")
    if price and shares and debt is not None and cash is not None and ebitda and ebitda != 0:
        mcap      = price * shares
        ev        = mcap + debt - cash
        ev_ebitda = ev / ebitda

    # --- Technical ---
    rsi, death_cross = calc_rsi_and_ma(ticker_str)

    return {
        "gross_margin":   gross_margin,
        "net_margin":     net_margin,
        "fcf_conversion": fcf_conversion,
        "rev_cagr":       rev_cagr,
        "debt_equity":    debt_equity,
        "ev_ebitda":      ev_ebitda,
        "rsi":            rsi,
        "death_cross":    death_cross,
        "status":         "OK",
    }


def fmt(val, style):
    if val is None:
        return "  N/A"
    if style == "pct":
        return f"{val*100:>5.1f}%"
    if style == "x":
        return f"{val:>6.2f}x"
    if style == "rsi":
        return f"{val:>5.1f}"
    if style == "dc":
        return "Yes" if val else "No"
    return str(val)


# ============================================================================
# MAIN SCREENING LOOP
# ============================================================================

print("=" * 70)
print(f"  STOCK SCREENER  |  {len(WATCHLIST)} tickers")
print("=" * 70)
print("  Running... (this may take 1–2 minutes)\n")

t_start  = time.time()
results  = {}

for ticker in WATCHLIST:
    try:
        metrics = analyse_ticker(ticker)
    except Exception:
        metrics = {
            "gross_margin": None, "net_margin": None, "fcf_conversion": None,
            "rev_cagr": None, "debt_equity": None, "ev_ebitda": None,
            "rsi": None, "death_cross": None, "status": "Insufficient Data",
        }

    # Score
    if metrics["status"] == "Insufficient Data":
        score      = 0
        passes     = {k: False for k in CRITERIA_KEYS}
    else:
        passes     = {k: fn(metrics) for k, fn in CRITERIA.items()}
        score      = sum(passes.values())

    # Tier
    if metrics["status"] == "Insufficient Data":
        tier_num, tier_label = 4, "Insufficient Data — Thesis Driven Only"
    elif score >= 6:
        tier_num, tier_label = 1, "Top Candidate"
    elif score >= 3:
        tier_num, tier_label = 2, "Watch List"
    else:
        tier_num, tier_label = 3, "Does Not Qualify"

    results[ticker] = {**metrics, "score": score, "passes": passes,
                       "tier_num": tier_num, "tier_label": tier_label}

elapsed = time.time() - t_start

# ============================================================================
# OUTPUT — TIER 1
# ============================================================================
# Top Candidates (6–8 criteria): strong fundamentals, reasonable valuation,
# and a technical setup suggesting the market hasn't already priced in the
# quality. These are the names worth deepest diligence.

def print_tier(tier_num, tier_label, blurb):
    tier_tickers = {k: v for k, v in results.items() if v["tier_num"] == tier_num}
    if tier_num < 4:
        tier_tickers = dict(sorted(tier_tickers.items(), key=lambda x: -x[1]["score"]))

    print(f"\n{'─'*70}")
    print(f"  TIER {tier_num} — {tier_label.upper()}")
    print(f"  {blurb}")
    print(f"  {len(tier_tickers)} ticker(s)")
    print(f"{'─'*70}")

    if tier_num == 4:
        for ticker, r in tier_tickers.items():
            print(f"  {ticker:<7}  {r['tier_label']}")
        return

    # Column header
    hdr = (f"  {'Ticker':<7} {'Score':>5}  "
           f"{'GrsMgn':>7} {'NetMgn':>7} {'FCFConv':>7} "
           f"{'RevCAGR':>7} {'D/E':>7} {'EV/EBTDA':>9} "
           f"{'RSI':>5}  {'DeathX':>6}")
    print(hdr)
    print("  " + "-" * 68)

    for ticker, r in tier_tickers.items():
        crit_bar = "".join("●" if r["passes"][k] else "○" for k in CRITERIA_KEYS)
        print(f"  {ticker:<7} {r['score']:>2}/8   "
              f"{fmt(r['gross_margin'],   'pct')} "
              f"{fmt(r['net_margin'],     'pct')} "
              f"{fmt(r['fcf_conversion'], 'x'  )} "
              f"{fmt(r['rev_cagr'],       'pct')} "
              f"{fmt(r['debt_equity'],    'x'  )} "
              f"{fmt(r['ev_ebitda'],      'x'  )}  "
              f"{fmt(r['rsi'],            'rsi')}  "
              f"{fmt(r['death_cross'],    'dc' ):>6}  "
              f"{crit_bar}")

print_tier(1, "Top Candidate",
           "Strong fundamentals + reasonable valuation + constructive technicals. Prioritise for deep diligence.")
print_tier(2, "Watch List",
           "Partial fit. Monitor for improving setup or catalyst before adding to portfolio.")
print_tier(3, "Does Not Qualify",
           "Fails most criteria. Re-evaluate if fundamentals improve or valuation compresses significantly.")
print_tier(4, "Insufficient Data — Thesis Driven Only",
           "Data unavailable for systematic scoring. Evaluate manually on qualitative thesis.")

# ============================================================================
# CRITERIA KEY
# ============================================================================
print(f"\n  {'─'*70}")
print(f"  CRITERIA KEY  (● = pass  ○ = fail)  order: {' | '.join(CRITERIA_KEYS)}")

# ============================================================================
# SUMMARY
# ============================================================================
t1 = sum(1 for r in results.values() if r["tier_num"] == 1)
t2 = sum(1 for r in results.values() if r["tier_num"] == 2)
t3 = sum(1 for r in results.values() if r["tier_num"] == 3)
t4 = sum(1 for r in results.values() if r["tier_num"] == 4)

print(f"\n  {'─'*70}")
print(f"  SUMMARY  |  {len(WATCHLIST)} screened  |  {elapsed:.1f}s")
print(f"  Tier 1 (Top Candidate):      {t1}")
print(f"  Tier 2 (Watch List):         {t2}")
print(f"  Tier 3 (Does Not Qualify):   {t3}")
print(f"  Tier 4 (Insufficient Data):  {t4}")
print(f"  {'─'*70}\n")
