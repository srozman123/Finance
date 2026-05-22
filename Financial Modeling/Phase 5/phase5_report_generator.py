import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from datetime import date
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# REPORT_TICKER — change this to generate a
# different report when running the file directly
# ─────────────────────────────────────────────
REPORT_TICKER = "INTU"

REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

PLT_STYLE = "seaborn-v0_8-whitegrid"

# ============================================================
# HELPERS
# ============================================================

def add_page_chrome(fig, ticker, report_date, page_title):
    """Adds consistent header and footer to every page."""
    # Header line
    fig.add_artist(plt.Line2D([0.04, 0.96], [0.945, 0.945],
                              transform=fig.transFigure,
                              color="#cccccc", linewidth=0.8))
    fig.text(0.04, 0.955, ticker, transform=fig.transFigure,
             fontsize=11, fontweight="bold", va="bottom")
    fig.text(0.96, 0.955, page_title, transform=fig.transFigure,
             fontsize=9, color="gray", va="bottom", ha="right")
    # Footer line
    fig.add_artist(plt.Line2D([0.04, 0.96], [0.05, 0.05],
                              transform=fig.transFigure,
                              color="#cccccc", linewidth=0.8))
    fig.text(0.04, 0.035, f"Generated: {report_date}",
             transform=fig.transFigure, fontsize=7, color="gray", va="top")
    fig.text(0.96, 0.035,
             "For educational purposes only. Not investment advice.",
             transform=fig.transFigure, fontsize=7, color="gray",
             va="top", ha="right")


def calc_rsi(close_series, period=14):
    """14-day Wilder EMA RSI."""
    delta  = close_series.diff()
    gains  = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)
    alpha  = 1 / period
    avg_g  = gains.ewm(alpha=alpha, adjust=False).mean()
    avg_l  = losses.ewm(alpha=alpha, adjust=False).mean()
    return 100 - 100 / (1 + avg_g / avg_l)


def safe_loc(df, label):
    try:
        return df.loc[label]
    except KeyError:
        return None


def run_dcf_inner(base_fcf, fcf_growth, wacc, terminal_growth,
                  debt, cash, shares, n_years=5):
    """Gordon Growth DCF → implied share price."""
    if wacc <= terminal_growth:
        return np.nan
    proj   = [base_fcf * (1 + fcf_growth) ** i for i in range(1, n_years + 1)]
    tv     = proj[-1] * (1 + terminal_growth) / (wacc - terminal_growth)
    pv_cf  = sum(cf / (1 + wacc) ** i for i, cf in enumerate(proj, 1))
    pv_tv  = tv / (1 + wacc) ** n_years
    eq     = (pv_cf + pv_tv) - debt + cash
    return eq / shares


def screener_score(m):
    """Returns (score, passes_dict). m = metrics dict."""
    criteria = {
        "F1 Gross Margin >40%": m["gross_margin"]   > 0.40  if m["gross_margin"]   is not None else False,
        "F2 Net Margin >10%":   m["net_margin"]     > 0.10  if m["net_margin"]     is not None else False,
        "F3 FCF Conv >0.75x":   m["fcf_conversion"] > 0.75  if m["fcf_conversion"] is not None else False,
        "F4 Rev CAGR >5%":      m["rev_cagr"]       > 0.05  if m["rev_cagr"]       is not None else False,
        "F5 D/E <2.0x":         m["debt_equity"]    < 2.00  if m["debt_equity"]    is not None else False,
        "V1 EV/EBITDA <25x":    m["ev_ebitda"]      < 25.0  if m["ev_ebitda"]      is not None else False,
        "T1 RSI <50":           m["rsi"]            < 50.0  if m["rsi"]            is not None else False,
        "T2 No Death Cross":    m["death_cross"]    == False if m["death_cross"]   is not None else False,
    }
    score = sum(criteria.values())
    return score, criteria


def fmt_b(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"${val/1e9:.2f}B"


def fmt_pct(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val*100:.1f}%"


def fmt_x(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.2f}x"


# ============================================================
# JSON DATA EXTRACTOR  (used by the web API)
# ============================================================

def _safe_float(val):
    """Return Python float or None; maps NaN/Inf → None."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def get_report_data(ticker: str) -> dict:
    """
    Identical computation to generate_report() but returns a JSON-serializable
    dict instead of building a PDF.  Called by the FastAPI web endpoint.
    """
    ticker = ticker.upper()
    report_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    t    = yf.Ticker(ticker)
    info = t.info

    raw_1y = yf.download(ticker, period="1y", interval="1d",
                          auto_adjust=True, progress=False)
    raw_4y = yf.download(ticker, period="5y", interval="1d",
                          auto_adjust=True, progress=False)

    close_1y = raw_1y["Close"].squeeze().dropna()
    vol_1y   = raw_1y["Volume"].squeeze().dropna()
    close_4y = raw_4y["Close"].squeeze().dropna()
    vol_4y   = raw_4y["Volume"].squeeze().dropna()

    is_  = t.financials.dropna(axis=1, how="all").iloc[:, :4]
    bs   = t.balance_sheet.dropna(axis=1, how="all").iloc[:, :4]
    cf   = t.cashflow.dropna(axis=1, how="all").iloc[:, :4]

    fy_cols  = is_.columns
    fy_years = [c.year for c in fy_cols]
    fy0      = fy_cols[0]

    rev_row    = safe_loc(is_, "Total Revenue")
    gp_row     = safe_loc(is_, "Gross Profit")
    oi_row     = safe_loc(is_, "Operating Income")
    ni_row     = safe_loc(is_, "Net Income")
    ebitda_row = safe_loc(is_, "EBITDA")
    opcf_row   = safe_loc(cf,  "Operating Cash Flow")
    fcf_row    = safe_loc(cf,  "Free Cash Flow")
    capex_row  = safe_loc(cf,  "Capital Expenditure")
    debt_row   = safe_loc(bs,  "Total Debt")
    eq_row     = safe_loc(bs,  "Stockholders Equity")
    ca_row     = safe_loc(bs,  "Current Assets")
    cl_row     = safe_loc(bs,  "Current Liabilities")
    cash_row   = safe_loc(bs,  "Cash And Cash Equivalents")

    def fv(row, col):
        try:
            v = row[col]
            return float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else None
        except Exception:
            return None

    revenue    = fv(rev_row,    fy0)
    gross_p    = fv(gp_row,     fy0)
    op_income  = fv(oi_row,     fy0)
    net_income = fv(ni_row,     fy0)
    ebitda     = fv(ebitda_row, fy0)
    free_cf    = fv(fcf_row,    fy0)
    total_debt = fv(debt_row,   fy0)
    equity_val = fv(eq_row,     fy0)
    cash       = fv(cash_row,   fy0)
    cur_assets = fv(ca_row,     fy0)
    cur_liab   = fv(cl_row,     fy0)

    current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    shares        = info.get("sharesOutstanding")
    company_name  = info.get("longName", ticker)
    market_cap    = current_price * shares if current_price and shares else None
    market_cap_b  = market_cap / 1e9 if market_cap else None

    gross_margin = gross_p    / revenue    if gross_p    and revenue    else None
    net_margin   = net_income / revenue    if net_income and revenue    else None
    op_margin    = op_income  / revenue    if op_income  and revenue    else None
    fcf_conv     = free_cf    / net_income if free_cf    and net_income else None
    debt_equity  = total_debt / equity_val if total_debt is not None and equity_val else None
    cur_ratio    = cur_assets / cur_liab   if cur_assets and cur_liab  else None

    ev_ebitda = None
    if market_cap and total_debt is not None and cash is not None and ebitda and ebitda != 0:
        ev        = market_cap + total_debt - cash
        ev_ebitda = ev / ebitda

    pe = market_cap / net_income if market_cap and net_income and net_income > 0 else None
    ps = market_cap / revenue    if market_cap and revenue    else None

    rev_cagr = None
    if rev_row is not None and len(fy_cols) >= 2:
        rv0 = fv(rev_row, fy_cols[0])
        rvn = fv(rev_row, fy_cols[-1])
        if rv0 and rvn and rvn > 0 and rv0 > 0:
            rev_cagr = (rv0 / rvn) ** (1 / (len(fy_cols) - 1)) - 1

    fcf_vals = [fv(fcf_row, c) for c in fy_cols] if fcf_row is not None else []
    fcf_vals = [v for v in fcf_vals if v is not None]
    base_fcf_4yr = float(np.mean(fcf_vals)) if fcf_vals else None
    fcf_cagr = None
    if len(fcf_vals) >= 2 and fcf_vals[-1] != 0 and fcf_vals[0] / fcf_vals[-1] > 0:
        try:
            fcf_cagr = (fcf_vals[0] / fcf_vals[-1]) ** (1 / (len(fcf_vals) - 1)) - 1
            fcf_cagr = max(min(fcf_cagr, 0.30), -0.20)
        except Exception:
            fcf_cagr = 0.0

    rsi_1y      = calc_rsi(close_1y)
    current_rsi = float(rsi_1y.iloc[-1]) if not rsi_1y.empty else None
    ma50_last   = float(close_4y.rolling(50).mean().iloc[-1])
    ma200_last  = float(close_4y.rolling(200).mean().iloc[-1])
    death_cross = bool(ma50_last < ma200_last)

    rsi_interp = ("OVERBOUGHT" if current_rsi and current_rsi > 70
                  else "OVERSOLD" if current_rsi and current_rsi < 30
                  else "NEUTRAL")

    daily_rets = close_1y.pct_change().dropna()
    total_ret  = float(close_1y.iloc[-1] / close_1y.iloc[0]) - 1
    ann_vol    = float(daily_rets.std() * np.sqrt(252))
    best_day   = float(daily_rets.max())
    worst_day  = float(daily_rets.min())
    best_date  = daily_rets.idxmax().strftime("%Y-%m-%d")
    worst_date = daily_rets.idxmin().strftime("%Y-%m-%d")

    screen_m = {
        "gross_margin": gross_margin, "net_margin": net_margin,
        "fcf_conversion": fcf_conv,   "rev_cagr": rev_cagr,
        "debt_equity": debt_equity,   "ev_ebitda": ev_ebitda,
        "rsi": current_rsi,           "death_cross": death_cross,
    }
    score, passes = screener_score(screen_m)
    tier_label = ("Top Candidate" if score >= 6
                  else "Watch List" if score >= 3
                  else "Does Not Qualify")

    WACC    = 0.095
    TERM_G  = 0.030
    N_YEARS = 5
    g = fcf_cagr if fcf_cagr is not None else 0.0

    tv1 = tv2 = tv_blend = pv_tv_blend = sum_pv_fcf = implied_price = None
    diff_pct = exit_multiple_note = None

    if base_fcf_4yr and total_debt is not None and cash is not None and shares:
        proj   = [base_fcf_4yr * (1 + g) ** i for i in range(1, N_YEARS + 1)]
        tv1    = proj[-1] * (1 + TERM_G) / (WACC - TERM_G)
        tv_use = tv1
        ev_ebitda_valid = (ev_ebitda is not None
                           and not np.isnan(ev_ebitda) and ev_ebitda > 0)
        if ebitda and ev_ebitda_valid:
            exit_mult = 25.0 if ev_ebitda > 50 else ev_ebitda
            if ev_ebitda > 50:
                exit_multiple_note = (f"Exit multiple capped at 25x — current EV/EBITDA "
                                      f"({ev_ebitda:.0f}x) not meaningful for valuation")
            ebitda_y5 = ebitda * (1 + g) ** N_YEARS
            tv2       = ebitda_y5 * exit_mult
            tv_use    = (tv1 + tv2) / 2
        elif ebitda and ev_ebitda is not None and not ev_ebitda_valid:
            exit_multiple_note = "Exit multiple capped at 25x — EV/EBITDA not meaningful"
            ebitda_y5 = ebitda * (1 + g) ** N_YEARS
            tv2       = ebitda_y5 * 25.0
            tv_use    = (tv1 + tv2) / 2
        tv_blend    = tv_use
        pv_fcfs     = [pv / (1 + WACC) ** i for i, pv in enumerate(proj, 1)]
        sum_pv_fcf  = sum(pv_fcfs)
        pv_tv_blend = tv_blend / (1 + WACC) ** N_YEARS
        total_pv    = sum_pv_fcf + pv_tv_blend
        eq_dcf      = total_pv - total_debt + cash
        implied_price = eq_dcf / shares
        if current_price:
            diff_pct = (implied_price - current_price) / current_price * 100

    MC_GROWTH_CAP  = 0.25
    g_raw          = g if g else 0.0
    g_mc           = min(g_raw, MC_GROWTH_CAP)
    mc_growth_note = (f"Historical FCF CAGR {g_raw*100:.1f}% — "
                      f"Monte Carlo mean capped at {MC_GROWTH_CAP*100:.0f}%"
                      if g_raw > MC_GROWTH_CAP else None)

    N_SIMS = 10_000
    np.random.seed(42)
    mc_mean = mc_median = mc_p10 = mc_p90 = prob_above = None
    mc_prices_sample = []

    if base_fcf_4yr and total_debt is not None and cash is not None and shares:
        g_samples  = np.clip(np.random.normal(g_mc, 0.03, N_SIMS), -0.10, 0.20)
        w_samples  = np.clip(np.random.normal(0.095, 0.01, N_SIMS),  0.05, 0.15)
        tg_samples = np.random.normal(0.03, 0.005, N_SIMS)
        tg_samples = np.minimum(tg_samples, w_samples - 0.01)
        mc_arr     = np.array([
            run_dcf_inner(base_fcf_4yr, gg, ww, tg, total_debt, cash, shares)
            for gg, ww, tg in zip(g_samples, w_samples, tg_samples)
        ])
        mc_arr = mc_arr[~np.isnan(mc_arr)]
        if len(mc_arr) > 0:
            mc_mean   = float(np.mean(mc_arr))
            mc_median = float(np.median(mc_arr))
            mc_p10    = float(np.percentile(mc_arr, 10))
            mc_p90    = float(np.percentile(mc_arr, 90))
            if current_price:
                prob_above = float(np.mean(mc_arr > current_price) * 100)
            idx = np.random.choice(len(mc_arr), min(300, len(mc_arr)), replace=False)
            mc_prices_sample = [float(x) for x in mc_arr[idx]]

    current_signal = "NEUTRAL — No active signal"
    if len(close_4y) >= 210:
        _df          = pd.DataFrame({"Close": close_4y})
        _df["MA20"]  = close_4y.rolling(20).mean()
        _df["MA50"]  = close_4y.rolling(50).mean()
        _df["MA200"] = close_4y.rolling(200).mean()
        _df["RSI"]   = calc_rsi(close_4y)
        _vol20       = vol_4y.rolling(20).mean() if len(vol_4y) >= 20 else None
        _df = _df.dropna()
        if len(_df) > 0:
            _rsi_ov  = _df["RSI"].rolling(10).min().shift(1) <= 30
            _rsi_2d  = ((_df["RSI"] > _df["RSI"].shift(1)) &
                        (_df["RSI"].shift(1) > _df["RSI"].shift(2)))
            _pr      = _df["Close"] > _df["Close"].shift(1)
            _dc_cont = (_df["MA50"] > _df["MA200"]).astype(int).rolling(30).min() == 1
            _pma200  = (_df["MA200"] - _df["Close"]) / _df["MA200"] <= 0.08
            _ma5020  = _df["MA50"] > _df["MA200"]
            if _vol20 is not None and len(_vol20.dropna()) > 0:
                _va = vol_4y.reindex(_df.index) > _vol20.reindex(_df.index)
            else:
                _va = pd.Series(True, index=_df.index)
            _entry = _rsi_ov & _rsi_2d & _pr & _va & _dc_cont & _pma200 & _ma5020
            if _entry.iloc[-1]:
                current_signal = "ENTRY SIGNAL — RSI recovery + volume + healthy trend"
            elif _df["MA50"].iloc[-1] < _df["MA200"].iloc[-1]:
                current_signal = "DEATH CROSS ACTIVE — No entries permitted (trend filter)"
            elif _df["RSI"].iloc[-1] > 70:
                current_signal = "RSI OVERBOUGHT — Monitor for exit"
            elif _df["RSI"].iloc[-1] < 30:
                current_signal = "RSI OVERSOLD — Watch for recovery signal"

    earnings_date_str = days_until_earnings = forward_eps = None
    try:
        cal = t.calendar
        earnings_date = None
        if cal is not None:
            if isinstance(cal, pd.DataFrame) and not cal.empty:
                earnings_date = pd.Timestamp(cal.columns[0])
            elif isinstance(cal, dict):
                ed_list = cal.get("Earnings Date", [])
                if ed_list:
                    earnings_date = pd.Timestamp(ed_list[0])
        if earnings_date:
            days_until_earnings = int((earnings_date - pd.Timestamp.today()).days)
            earnings_date_str   = earnings_date.strftime("%Y-%m-%d")
        forward_eps = info.get("forwardEps")
    except Exception:
        pass

    fs_rows = {
        "Total Revenue":       rev_row,
        "Gross Profit":        gp_row,
        "Operating Income":    oi_row,
        "Net Income":          ni_row,
        "Operating Cash Flow": opcf_row,
        "Free Cash Flow":      fcf_row,
        "Capital Expenditure": capex_row,
        "Total Debt":          debt_row,
        "Stockholders Equity": eq_row,
    }
    financials_table = {
        label: ([_safe_float(fv(row, c)) for c in fy_cols]
                if row is not None else [None] * len(fy_cols))
        for label, row in fs_rows.items()
    }

    fy_asc = fy_cols[::-1]
    ratio_trends = {}
    for rname, num_row, den_row, is_pct in [
        ("Gross Margin",     gp_row,   rev_row, True),
        ("Operating Margin", oi_row,   rev_row, True),
        ("Net Margin",       ni_row,   rev_row, True),
        ("Debt to Equity",   debt_row, eq_row,  False),
        ("Current Ratio",    ca_row,   cl_row,  False),
        ("FCF Conversion",   fcf_row,  ni_row,  False),
    ]:
        vals = []
        for c in fy_asc:
            n   = fv(num_row, c) if num_row is not None else None
            dv  = fv(den_row, c) if den_row is not None else None
            vals.append(round(n / dv * (100 if is_pct else 1), 4)
                        if n is not None and dv and dv != 0 else None)
        ratio_trends[rname] = {"years": [c.year for c in fy_asc],
                                "values": vals, "is_pct": is_pct}

    _cagr    = fcf_cagr if fcf_cagr is not None else 0.0
    _g_min   = min(-0.05, _cagr - 0.20)
    _g_max   = max(0.15,  _cagr + 0.05)
    _g_step  = 0.02
    _n_steps = round((_g_max - _g_min) / _g_step)
    sens_g   = [round(_g_min + i * _g_step, 4) for i in range(_n_steps + 1)]
    sens_w   = [0.08, 0.09, 0.10, 0.11]
    sens_p   = []
    for gr in sens_g:
        row_p = []
        for wr in sens_w:
            if base_fcf_4yr and total_debt is not None and cash is not None and shares:
                p = run_dcf_inner(base_fcf_4yr, gr, wr, TERM_G, total_debt, cash, shares)
                row_p.append(_safe_float(p))
            else:
                row_p.append(None)
        sens_p.append(row_p)

    sf = _safe_float
    return {
        "ticker":       ticker,
        "company_name": company_name,
        "report_date":  report_date,
        "current_price": sf(current_price),
        "market_cap_b":  sf(market_cap_b),
        "fy_years":      fy_years,
        "financials_table": financials_table,
        "metrics": {
            "gross_margin":   sf(gross_margin),
            "net_margin":     sf(net_margin),
            "op_margin":      sf(op_margin),
            "fcf_conversion": sf(fcf_conv),
            "debt_equity":    sf(debt_equity),
            "current_ratio":  sf(cur_ratio),
            "ev_ebitda":      sf(ev_ebitda),
            "pe":             sf(pe),
            "ps":             sf(ps),
            "rev_cagr":       sf(rev_cagr),
            "fcf_cagr":       sf(fcf_cagr),
        },
        "technicals": {
            "rsi":                sf(current_rsi),
            "rsi_interpretation": rsi_interp,
            "ma50":               sf(ma50_last),
            "ma200":              sf(ma200_last),
            "death_cross":        death_cross,
            "signal":             current_signal,
        },
        "price_stats": {
            "total_return_1y": sf(total_ret),
            "ann_volatility":  sf(ann_vol),
            "best_day":        sf(best_day),
            "best_date":       best_date,
            "worst_day":       sf(worst_day),
            "worst_date":      worst_date,
        },
        "screener": {
            "score":    score,
            "tier":     tier_label,
            "criteria": {k: bool(v) for k, v in passes.items()},
        },
        "dcf": {
            "base_fcf":           sf(base_fcf_4yr),
            "fcf_cagr_used":      sf(g),
            "wacc":               WACC,
            "terminal_growth":    TERM_G,
            "tv_gordon":          sf(tv1),
            "tv_exit":            sf(tv2),
            "tv_blend":           sf(tv_blend),
            "pv_fcf":             sf(sum_pv_fcf),
            "pv_tv":              sf(pv_tv_blend),
            "implied_price":      sf(implied_price),
            "diff_pct":           sf(diff_pct),
            "exit_multiple_note": exit_multiple_note,
        },
        "monte_carlo": {
            "mean":          sf(mc_mean),
            "median":        sf(mc_median),
            "p10":           sf(mc_p10),
            "p90":           sf(mc_p90),
            "prob_above":    sf(prob_above),
            "growth_note":   mc_growth_note,
            "prices_sample": mc_prices_sample,
        },
        "earnings": {
            "date":        earnings_date_str,
            "days_until":  days_until_earnings,
            "forward_eps": sf(forward_eps),
        },
        "ratio_trends": ratio_trends,
        "sensitivity": {
            "g_rates":       sens_g,
            "w_rates":       sens_w,
            "prices":        sens_p,
            "current_price": sf(current_price),
        },
    }


# ============================================================
# MAIN GENERATOR
# ============================================================

def generate_report(ticker):
    ticker = ticker.upper()
    report_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    pdf_path    = os.path.join(REPORT_DIR, f"{ticker}_report.pdf")

    print(f"  Generating report for {ticker}...")

    try:
        # ────────────────────────────────────
        # DATA FETCH
        # ────────────────────────────────────
        t    = yf.Ticker(ticker)
        info = t.info

        raw_1y  = yf.download(ticker, period="1y",  interval="1d",
                               auto_adjust=True, progress=False)
        raw_4y  = yf.download(ticker, period="5y",  interval="1d",
                               auto_adjust=True, progress=False)

        close_1y = raw_1y["Close"].squeeze().dropna()
        vol_1y   = raw_1y["Volume"].squeeze().dropna()
        close_4y = raw_4y["Close"].squeeze().dropna()
        vol_4y   = raw_4y["Volume"].squeeze().dropna()

        is_  = t.financials.dropna(axis=1, how="all").iloc[:, :4]
        bs   = t.balance_sheet.dropna(axis=1, how="all").iloc[:, :4]
        cf   = t.cashflow.dropna(axis=1, how="all").iloc[:, :4]

        fy_cols  = is_.columns
        fy_years = [c.year for c in fy_cols]
        fy0      = fy_cols[0]

        # ── Financial statement rows ──
        rev_row   = safe_loc(is_, "Total Revenue")
        gp_row    = safe_loc(is_, "Gross Profit")
        oi_row    = safe_loc(is_, "Operating Income")
        ni_row    = safe_loc(is_, "Net Income")
        ebitda_row = safe_loc(is_, "EBITDA")
        opcf_row  = safe_loc(cf,  "Operating Cash Flow")
        fcf_row   = safe_loc(cf,  "Free Cash Flow")
        capex_row = safe_loc(cf,  "Capital Expenditure")
        debt_row  = safe_loc(bs,  "Total Debt")
        eq_row    = safe_loc(bs,  "Stockholders Equity")
        ca_row    = safe_loc(bs,  "Current Assets")
        cl_row    = safe_loc(bs,  "Current Liabilities")
        cash_row  = safe_loc(bs,  "Cash And Cash Equivalents")

        def fv(row, col):
            """Safe float from row/col."""
            try:
                v = row[col]
                return float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else None
            except Exception:
                return None

        revenue    = fv(rev_row,   fy0)
        gross_p    = fv(gp_row,    fy0)
        op_income  = fv(oi_row,    fy0)
        net_income = fv(ni_row,    fy0)
        ebitda     = fv(ebitda_row, fy0)
        free_cf    = fv(fcf_row,   fy0)
        total_debt = fv(debt_row,  fy0)
        equity_val = fv(eq_row,    fy0)
        cash       = fv(cash_row,  fy0)
        cur_assets = fv(ca_row,    fy0)
        cur_liab   = fv(cl_row,    fy0)

        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        shares        = info.get("sharesOutstanding")
        company_name  = info.get("longName", ticker)
        market_cap    = current_price * shares if current_price and shares else None
        market_cap_b  = market_cap / 1e9 if market_cap else None

        # ── Derived metrics ──
        gross_margin   = gross_p    / revenue    if gross_p    and revenue    else None
        net_margin     = net_income / revenue    if net_income and revenue    else None
        op_margin      = op_income  / revenue    if op_income  and revenue    else None
        fcf_conv       = free_cf    / net_income if free_cf    and net_income else None
        debt_equity    = total_debt / equity_val if total_debt is not None and equity_val else None
        cur_ratio      = cur_assets / cur_liab   if cur_assets and cur_liab  else None

        ev_ebitda = None
        if market_cap and total_debt is not None and cash is not None and ebitda and ebitda != 0:
            ev        = market_cap + total_debt - cash
            ev_ebitda = ev / ebitda

        pe = market_cap / net_income if market_cap and net_income and net_income > 0 else None
        ps = market_cap / revenue    if market_cap and revenue    else None

        # Revenue CAGR
        rev_cagr = None
        if rev_row is not None and len(fy_cols) >= 2:
            rv0 = fv(rev_row, fy_cols[0])
            rvn = fv(rev_row, fy_cols[-1])
            if rv0 and rvn and rvn > 0 and rv0 > 0:
                rev_cagr = (rv0 / rvn) ** (1 / (len(fy_cols) - 1)) - 1

        # FCF CAGR
        fcf_vals = [fv(fcf_row, c) for c in fy_cols] if fcf_row is not None else []
        fcf_vals = [v for v in fcf_vals if v is not None]
        base_fcf_4yr = np.mean(fcf_vals) if fcf_vals else None
        fcf_cagr = None
        if len(fcf_vals) >= 2 and fcf_vals[-1] != 0 and fcf_vals[0] / fcf_vals[-1] > 0:
            try:
                fcf_cagr = (fcf_vals[0] / fcf_vals[-1]) ** (1 / (len(fcf_vals) - 1)) - 1
                fcf_cagr = max(min(fcf_cagr, 0.30), -0.20)
            except Exception:
                fcf_cagr = 0.0

        # ── Technical ──
        rsi_1y       = calc_rsi(close_1y)
        current_rsi  = float(rsi_1y.iloc[-1]) if not rsi_1y.empty else None
        ma50_last    = float(close_4y.rolling(50).mean().iloc[-1])
        ma200_last   = float(close_4y.rolling(200).mean().iloc[-1])
        death_cross  = bool(ma50_last < ma200_last)

        rsi_interp = ("OVERBOUGHT" if current_rsi and current_rsi > 70
                      else "OVERSOLD" if current_rsi and current_rsi < 30
                      else "NEUTRAL")

        # ── Price stats ──
        daily_rets = close_1y.pct_change().dropna()
        total_ret  = float(close_1y.iloc[-1] / close_1y.iloc[0]) - 1
        ann_vol    = float(daily_rets.std() * np.sqrt(252))
        best_day   = float(daily_rets.max())
        worst_day  = float(daily_rets.min())
        best_date  = daily_rets.idxmax().strftime("%Y-%m-%d")
        worst_date = daily_rets.idxmin().strftime("%Y-%m-%d")

        # ── Screener score ──
        screen_m = {
            "gross_margin": gross_margin, "net_margin": net_margin,
            "fcf_conversion": fcf_conv, "rev_cagr": rev_cagr,
            "debt_equity": debt_equity, "ev_ebitda": ev_ebitda,
            "rsi": current_rsi, "death_cross": death_cross,
        }
        score, passes = screener_score(screen_m)
        tier_label = ("Top Candidate" if score >= 6
                      else "Watch List" if score >= 3
                      else "Does Not Qualify")

        # ── DCF (blended) ──
        WACC        = 0.095
        TERM_G      = 0.030
        N_YEARS     = 5
        g           = fcf_cagr if fcf_cagr is not None else 0.0
        dcf_price   = None
        tv1 = tv2 = tv_blend = pv_tv_blend = sum_pv_fcf = implied_price = None
        diff_pct = None
        exit_multiple_note = None

        if base_fcf_4yr and total_debt is not None and cash is not None and shares:
            proj   = [base_fcf_4yr * (1 + g) ** i for i in range(1, N_YEARS + 1)]
            tv1    = proj[-1] * (1 + TERM_G) / (WACC - TERM_G)
            tv_use = tv1
            # Fix 2 — exit multiple sanity check: cap at 50x; use 25x default
            # when EV/EBITDA is above 50x, negative, or not meaningful.
            ev_ebitda_valid = (ev_ebitda is not None and not np.isnan(ev_ebitda)
                               and ev_ebitda > 0)
            if ebitda and ev_ebitda_valid:
                if ev_ebitda > 50:
                    exit_mult = 25.0
                    exit_multiple_note = (
                        f"Exit multiple capped at 25x — current EV/EBITDA "
                        f"({ev_ebitda:.0f}x) not meaningful for valuation"
                    )
                else:
                    exit_mult = ev_ebitda
                ebitda_y5 = ebitda * (1 + g) ** N_YEARS
                tv2       = ebitda_y5 * exit_mult
                tv_use    = (tv1 + tv2) / 2
            elif ebitda and ev_ebitda is not None and not ev_ebitda_valid:
                exit_mult = 25.0
                exit_multiple_note = (
                    "Exit multiple capped at 25x — current EV/EBITDA "
                    "not meaningful for valuation"
                )
                ebitda_y5 = ebitda * (1 + g) ** N_YEARS
                tv2       = ebitda_y5 * exit_mult
                tv_use    = (tv1 + tv2) / 2
            tv_blend   = tv_use
            pv_fcfs    = [cf / (1 + WACC) ** i for i, cf in enumerate(proj, 1)]
            sum_pv_fcf = sum(pv_fcfs)
            pv_tv_blend = tv_blend / (1 + WACC) ** N_YEARS
            total_pv   = sum_pv_fcf + pv_tv_blend
            eq_dcf     = total_pv - total_debt + cash
            implied_price = eq_dcf / shares
            if current_price:
                diff_pct = (implied_price - current_price) / current_price * 100

        # ── Monte Carlo ──
        # Fix 3 — cap Monte Carlo mean FCF growth at 25% to prevent unrealistic
        # simulations for high-growth companies with extreme historical CAGRs.
        MC_GROWTH_CAP = 0.25
        g_raw     = g if g else 0.0
        g_mc      = min(g_raw, MC_GROWTH_CAP)
        mc_growth_note = None
        if g_raw > MC_GROWTH_CAP:
            mc_growth_note = (
                f"Historical FCF CAGR {g_raw*100:.1f}% — "
                f"Monte Carlo mean capped at {MC_GROWTH_CAP*100:.0f}%"
            )

        N_SIMS = 10_000
        np.random.seed(42)
        mc_mean = mc_median = mc_p10 = mc_p90 = prob_above = None

        if base_fcf_4yr and total_debt is not None and cash is not None and shares:
            g_samples  = np.clip(np.random.normal(g_mc, 0.03, N_SIMS), -0.10, 0.20)
            w_samples  = np.clip(np.random.normal(0.095, 0.01,    N_SIMS),  0.05, 0.15)
            tg_samples = np.random.normal(0.03, 0.005, N_SIMS)
            tg_samples = np.minimum(tg_samples, w_samples - 0.01)
            mc_prices  = np.array([
                run_dcf_inner(base_fcf_4yr, gg, ww, tg,
                              total_debt, cash, shares)
                for gg, ww, tg in zip(g_samples, w_samples, tg_samples)
            ])
            mc_prices = mc_prices[~np.isnan(mc_prices)]
            if len(mc_prices) > 0:
                mc_mean   = float(np.mean(mc_prices))
                mc_median = float(np.median(mc_prices))
                mc_p10    = float(np.percentile(mc_prices, 10))
                mc_p90    = float(np.percentile(mc_prices, 90))
                if current_price:
                    prob_above = float(np.mean(mc_prices > current_price) * 100)

        # ── Signal status (trend filter logic) ──
        current_signal = "NEUTRAL — No active signal"
        if len(close_4y) >= 210:
            _df         = pd.DataFrame({"Close": close_4y})
            _df["MA20"] = close_4y.rolling(20).mean()
            _df["MA50"] = close_4y.rolling(50).mean()
            _df["MA200"]= close_4y.rolling(200).mean()
            _df["RSI"]  = calc_rsi(close_4y)
            _vol20      = vol_4y.rolling(20).mean() if len(vol_4y) >= 20 else None
            _df = _df.dropna()

            if len(_df) > 0:
                _rsi_ov  = _df["RSI"].rolling(10).min().shift(1) <= 30
                _rsi_2d  = (_df["RSI"] > _df["RSI"].shift(1)) & \
                           (_df["RSI"].shift(1) > _df["RSI"].shift(2))
                _pr      = _df["Close"] > _df["Close"].shift(1)
                _dc_cont = (_df["MA50"] > _df["MA200"]).astype(int).rolling(30).min() == 1
                _pma200  = (_df["MA200"] - _df["Close"]) / _df["MA200"] <= 0.08
                _ma5020  = _df["MA50"] > _df["MA200"]

                if _vol20 is not None and len(_vol20.dropna()) > 0:
                    _vol20_aligned = _vol20.reindex(_df.index)
                    _va = vol_4y.reindex(_df.index) > _vol20_aligned
                else:
                    _va = pd.Series(True, index=_df.index)

                _entry = _rsi_ov & _rsi_2d & _pr & _va & _dc_cont & _pma200 & _ma5020

                if _entry.iloc[-1]:
                    current_signal = "ENTRY SIGNAL — RSI recovery + volume + healthy trend"
                elif _df["MA50"].iloc[-1] < _df["MA200"].iloc[-1]:
                    current_signal = "DEATH CROSS ACTIVE — No entries permitted (trend filter)"
                elif _df["RSI"].iloc[-1] > 70:
                    current_signal = "RSI OVERBOUGHT — Monitor for exit"
                elif _df["RSI"].iloc[-1] < 30:
                    current_signal = "RSI OVERSOLD — Watch for recovery signal"

        # ────────────────────────────────────────────────────────────
        # BUILD PDF
        # ────────────────────────────────────────────────────────────
        with PdfPages(pdf_path) as pdf:

            # ── PAGE 1: COVER ────────────────────────────────────────
            fig = plt.figure(figsize=(8.5, 11))
            fig.patch.set_facecolor("white")

            cx = 0.5
            fig.text(cx, 0.80, ticker, ha="center", fontsize=40,
                     fontweight="bold", color="#1a1a2e")
            fig.text(cx, 0.73, company_name, ha="center",
                     fontsize=14, color="#555555")

            price_str = f"${current_price:.2f}" if current_price else "N/A"
            mcap_str  = f"${market_cap_b:.1f}B" if market_cap_b else "N/A"
            fig.text(cx, 0.66, f"Current Price: {price_str}   |   Market Cap: {mcap_str}",
                     ha="center", fontsize=12)
            fig.text(cx, 0.60, f"Report Date: {report_date}",
                     ha="center", fontsize=11, color="#555555")

            # Divider
            fig.add_artist(plt.Line2D([0.1, 0.9], [0.56, 0.56],
                                      transform=fig.transFigure,
                                      color="#cccccc", linewidth=1))

            fig.text(cx, 0.52, "SCREENER SUMMARY", ha="center",
                     fontsize=11, fontweight="bold")
            tier_color = ("#2ecc71" if score >= 6 else
                          "#e67e22" if score >= 3 else "#e74c3c")
            fig.text(cx, 0.47, f"Score: {score}/8 — {tier_label}",
                     ha="center", fontsize=13, color=tier_color, fontweight="bold")

            crit_items = list(passes.items())
            for i, (name, passed) in enumerate(crit_items):
                row, col = divmod(i, 2)
                x_pos = 0.20 if col == 0 else 0.55
                y_pos = 0.41 - row * 0.05
                symbol = "●" if passed else "○"
                color  = "#2ecc71" if passed else "#e74c3c"
                fig.text(x_pos, y_pos, f"{symbol} {name}",
                         fontsize=9, color=color)

            fig.text(cx, 0.14, "INVESTMENT ANALYSIS REPORT",
                     ha="center", fontsize=10, color="#aaaaaa",
                     fontweight="bold", style="italic")

            add_page_chrome(fig, ticker, report_date, "Page 1 — Cover")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # ── PAGE 2: MARKET DATA ──────────────────────────────────
            plt.style.use(PLT_STYLE)
            fig = plt.figure(figsize=(11, 8.5))
            gs  = GridSpec(3, 1, figure=fig, height_ratios=[3, 1.2, 0.8],
                           hspace=0.45, top=0.91, bottom=0.08,
                           left=0.07, right=0.96)

            # Price chart
            ax1 = fig.add_subplot(gs[0])
            ma20_1y  = close_1y.rolling(20).mean()
            ma50_1y  = close_1y.rolling(50).mean()
            ma200_1y = close_1y.rolling(200).mean()
            ax1.plot(close_1y.index,  close_1y.values,  color="lightgray",    lw=0.9,
                     label="Close")
            ax1.plot(ma20_1y.index,   ma20_1y.values,   color="seagreen",     lw=1.2,
                     label="20MA")
            ax1.plot(ma50_1y.index,   ma50_1y.values,   color="steelblue",    lw=1.5,
                     label="50MA")
            ax1.plot(ma200_1y.index,  ma200_1y.values,  color="mediumpurple", lw=1.5,
                     linestyle="--", label="200MA")
            ax1.set_title(f"{ticker} — Closing Price with Moving Averages (1 Year)",
                          fontsize=10, fontweight="bold")
            ax1.set_ylabel("Price (USD)", fontsize=8)
            ax1.legend(fontsize=7, loc="upper left")
            ax1.grid(True, linestyle="--", alpha=0.4)

            # RSI
            ax2 = fig.add_subplot(gs[1])
            ax2.plot(rsi_1y.index, rsi_1y.values, color="darkorange", lw=1.2,
                     label="RSI(14)")
            ax2.axhline(70, color="crimson",  lw=1, linestyle="--")
            ax2.axhline(30, color="seagreen", lw=1, linestyle="--")
            ax2.fill_between(rsi_1y.index, 70, rsi_1y.clip(lower=70),
                             color="crimson",  alpha=0.15)
            ax2.fill_between(rsi_1y.index, rsi_1y.clip(upper=30), 30,
                             color="seagreen", alpha=0.15)
            rsi_title = (f"RSI (14-day Wilder)  —  Current: "
                         f"{current_rsi:.1f}  [{rsi_interp}]"
                         if current_rsi else "RSI (14-day Wilder)")
            ax2.set_title(rsi_title, fontsize=9, fontweight="bold")
            ax2.set_ylim(0, 100)
            ax2.set_yticks([30, 50, 70])
            ax2.set_ylabel("RSI", fontsize=8)
            ax2.legend(fontsize=7)
            ax2.grid(True, linestyle="--", alpha=0.4)

            # Stats table
            ax3 = fig.add_subplot(gs[2])
            ax3.axis("off")
            rsi_v = f"{current_rsi:.1f} ({rsi_interp})" if current_rsi else "N/A"
            stats = [
                ["1-Year Cumulative Return", f"{total_ret*100:+.2f}%",
                 "Annualised Volatility",    f"{ann_vol*100:.1f}%"],
                ["Best Single Day",  f"{best_day*100:.2f}%  ({best_date})",
                 "Worst Single Day", f"{worst_day*100:.2f}%  ({worst_date})"],
                ["Current RSI", rsi_v, "Death Cross Active", "Yes" if death_cross else "No"],
            ]
            tbl = ax3.table(cellText=stats,
                            colLabels=["Metric A", "Value A", "Metric B", "Value B"],
                            cellLoc="center", loc="center")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)
            tbl.scale(1, 1.3)
            for (r, c), cell in tbl.get_celld().items():
                cell.set_edgecolor("#dddddd")
                if r == 0:
                    cell.set_facecolor("#dce6f1")
                elif r % 2 == 0:
                    cell.set_facecolor("#f7f9fc")
                else:
                    cell.set_facecolor("white")

            add_page_chrome(fig, ticker, report_date, "Page 2 — Market Data")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # ── PAGE 3: FINANCIAL STATEMENTS ────────────────────────
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis("off")
            fig.subplots_adjust(top=0.88, bottom=0.08, left=0.04, right=0.96)

            fs_items = [
                ("Total Revenue  (top line)",                     rev_row),
                ("Gross Profit  (= Revenue − COGS)",              gp_row),
                ("Operating Income  (= Gross Profit − OpEx)",     oi_row),
                ("Net Income  (= Op Income − Interest − Tax)",    ni_row),
                ("Operating Cash Flow",                           opcf_row),
                ("Free Cash Flow  (= OpCF − CapEx)",              fcf_row),
                ("Capital Expenditure  (negative = outflow)",     capex_row),
                ("Total Debt",                                    debt_row),
                ("Stockholders Equity",                           eq_row),
            ]

            col_headers = ["Line Item"] + [f"FY{y}" for y in fy_years]
            table_data  = []
            for label, row in fs_items:
                cells = [label]
                for c in fy_cols:
                    v = fv(row, c) if row is not None else None
                    cells.append(fmt_b(v))
                table_data.append(cells)

            ax.set_title(f"{ticker} — Key Financial Statement Line Items (in $B)",
                         fontsize=11, fontweight="bold", pad=12)

            tbl = ax.table(cellText=table_data,
                           colLabels=col_headers,
                           cellLoc="center", loc="center")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)
            tbl.scale(1, 1.6)
            for (r, c), cell in tbl.get_celld().items():
                cell.set_edgecolor("#dddddd")
                if r == 0:
                    cell.set_facecolor("#dce6f1")
                    cell.set_text_props(fontweight="bold")
                elif r % 2 == 0:
                    cell.set_facecolor("#f7f9fc")
                else:
                    cell.set_facecolor("white")
                if c == 0:
                    cell.set_text_props(ha="left")
                    cell.PAD = 0.02

            add_page_chrome(fig, ticker, report_date, "Page 3 — Financial Statements")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # ── PAGE 4: RATIO DASHBOARD ──────────────────────────────
            fig = plt.figure(figsize=(11, 8.5))
            fig.subplots_adjust(top=0.88, bottom=0.08, left=0.07,
                                right=0.96, hspace=0.55, wspace=0.35)
            axes = [fig.add_subplot(2, 3, i+1) for i in range(6)]

            fy_asc = fy_cols[::-1]   # oldest → newest for x-axis
            fy_asc_years = [c.year for c in fy_asc]

            ratio_specs = [
                ("Gross Margin",    "(= Gross Profit / Revenue)",        gp_row,   rev_row, True,  None),
                ("Operating Margin","(= Operating Income / Revenue)",    oi_row,   rev_row, True,  None),
                ("Net Margin",      "(= Net Income / Revenue)",          ni_row,   rev_row, True,  None),
                ("Debt to Equity",  "(= Total Debt / Equity)",           debt_row, eq_row,  False, 1.0),
                ("Current Ratio",   "(= Current Assets / Curr Liab)",   ca_row,   cl_row,  False, 1.0),
                ("FCF Conversion",  "(= Free Cash Flow / Net Income)",   fcf_row,  ni_row,  False, 1.0),
            ]

            for ax, (name, formula, num_row, den_row, is_pct, refline) in \
                    zip(axes, ratio_specs):
                xs, ys = [], []
                for c in fy_asc:
                    n = fv(num_row, c) if num_row is not None else None
                    d = fv(den_row, c) if den_row is not None else None
                    if n is not None and d and d != 0:
                        xs.append(c.year)
                        ys.append(n / d * (100 if is_pct else 1))

                if xs:
                    ax.plot(xs, ys, marker="o", color="steelblue", lw=2)
                    for x, y in zip(xs, ys):
                        lbl = f"{y:.1f}%" if is_pct else f"{y:.2f}x"
                        ax.annotate(lbl, (x, y),
                                    textcoords="offset points", xytext=(0, 6),
                                    ha="center", fontsize=7)
                if refline is not None:
                    ax.axhline(refline, color="tomato", lw=1,
                               linestyle="--", alpha=0.7)
                ax.set_title(name, fontsize=9, fontweight="bold")
                ax.set_xlabel(formula, fontsize=7, color="gray")
                ax.set_ylabel("%" if is_pct else "x", fontsize=7)
                ax.tick_params(axis="both", labelsize=7)
                ax.grid(True, linestyle="--", alpha=0.4)
                if xs:
                    ax.set_xticks(xs)

            fig.suptitle(f"{ticker} — Ratio Dashboard (4 Fiscal Years)",
                         fontsize=11, fontweight="bold", y=0.96)
            add_page_chrome(fig, ticker, report_date, "Page 4 — Ratio Dashboard")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # ── PAGE 5: VALUATION ───────────────────────────────────
            fig = plt.figure(figsize=(11, 8.5))
            fig.patch.set_facecolor("white")
            fig.subplots_adjust(top=0.88, bottom=0.10, left=0.06,
                                right=0.96)

            # Top section: text
            y = 0.87
            def fline(label, val, y_pos):
                fig.text(0.06, y_pos, label, fontsize=8,
                         color="#555555", fontweight="bold")
                fig.text(0.32, y_pos, val, fontsize=8)
                return y_pos - 0.028

            fig.text(0.06, y + 0.01,
                     f"{ticker} — Valuation  (FY end: {fy0.strftime('%Y-%m-%d')})",
                     fontsize=10, fontweight="bold")
            y -= 0.01

            y = fline("P/E Ratio  (Market Cap / Net Income)",
                      f"{pe:.1f}x" if pe else "N/A", y)
            y = fline("P/S Ratio  (Market Cap / Revenue)",
                      f"{ps:.1f}x" if ps else "N/A", y)
            y = fline("EV/EBITDA  (EV / EBITDA)",
                      f"{ev_ebitda:.1f}x" if ev_ebitda else "N/A", y)
            y -= 0.01
            y = fline("─── DCF Inputs ───", "", y)
            y = fline("Base FCF  (4-yr average)",
                      fmt_b(base_fcf_4yr), y)
            y = fline("Historical FCF CAGR",
                      f"{fcf_cagr*100:.1f}%" if fcf_cagr is not None else "N/A", y)
            y = fline("WACC", "9.5%", y)
            y = fline("Terminal Growth Rate", "3.0%", y)
            y -= 0.01
            y = fline("─── DCF Output ───", "", y)
            y = fline("Terminal Value — Gordon Growth (Method 1)",
                      fmt_b(tv1), y)
            tv2_str = fmt_b(tv2) if tv2 else "N/A (EV/EBITDA not available)"
            y = fline("Terminal Value — Exit Multiple (Method 2)", tv2_str, y)
            if exit_multiple_note:
                fig.text(0.32, y + 0.008, exit_multiple_note,
                         fontsize=6.5, color="darkorange", style="italic")
            y = fline("Blended Terminal Value  (avg of M1 & M2)",
                      fmt_b(tv_blend), y)
            y = fline("PV of FCFs",   fmt_b(sum_pv_fcf), y)
            y = fline("PV of Terminal Value", fmt_b(pv_tv_blend), y)
            y = fline("Implied Share Price",
                      f"${implied_price:.2f}" if implied_price else "N/A", y)
            mkt_str = f"${current_price:.2f}" if current_price else "N/A"
            diff_str = (f"{diff_pct:+.1f}%  "
                        f"({'UNDERVALUED' if diff_pct and diff_pct>0 else 'OVERVALUED'})"
                        if diff_pct is not None else "N/A")
            y = fline(f"Current Market Price", mkt_str, y)
            y = fline("DCF vs Market", diff_str, y)

            # Divider
            fig.add_artist(plt.Line2D([0.06, 0.96], [y - 0.01, y - 0.01],
                                      transform=fig.transFigure,
                                      color="#cccccc", linewidth=0.8))
            y -= 0.04

            # Build a dynamic FCF growth range anchored to the historical CAGR so
            # the base-case assumption always falls within the table's range.
            _cagr = fcf_cagr if fcf_cagr is not None else 0.0
            _g_min   = min(-0.05, _cagr - 0.20)
            _g_max   = max(0.15,  _cagr + 0.05)
            _g_step  = 0.02
            _n_steps = round((_g_max - _g_min) / _g_step)
            g_rates  = [round(_g_min + i * _g_step, 4) for i in range(_n_steps + 1)]
            w_rates  = [0.08, 0.09, 0.10, 0.11]

            s_data      = []
            cell_colors = []
            for gr in g_rates:
                row_d = []
                row_c = []
                for wr in w_rates:
                    if base_fcf_4yr and total_debt is not None and cash is not None and shares:
                        p = run_dcf_inner(base_fcf_4yr, gr, wr, TERM_G,
                                         total_debt, cash, shares)
                        row_d.append(f"${p:.0f}" if not np.isnan(p) else "N/A")
                        if current_price and not np.isnan(p):
                            row_c.append("#c8e6c9" if p > current_price * 1.1
                                         else "#ffcdd2" if p < current_price * 0.9
                                         else "#fff9c4")
                        else:
                            row_c.append("white")
                    else:
                        row_d.append("N/A")
                        row_c.append("white")
                s_data.append(row_d)
                cell_colors.append(row_c)

            col_lbls = [f"WACC {int(w*100)}%" for w in w_rates]
            row_lbls = [f"FCF {g*100:+.1f}%" for g in g_rates]
            cp_note  = (f"  Green = implied > ${current_price*1.1:.0f}   "
                        f"Red = implied < ${current_price*0.9:.0f}   "
                        f"Yellow = within 10%  |  Current: ${current_price:.2f}"
                        if current_price else "")

            def _render_sens_table(target_fig, left, bottom, width, height):
                """Draw the sensitivity table into target_fig at the given axes rect."""
                sens_ax = target_fig.add_axes([left, bottom, width, height])
                sens_ax.axis("off")
                tbl = sens_ax.table(
                    cellText=s_data,
                    rowLabels=row_lbls,
                    colLabels=col_lbls,
                    cellColours=cell_colors,
                    cellLoc="center", loc="center"
                )
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(8)
                tbl.scale(1, 1.4)
                for (r, _), cell in tbl.get_celld().items():
                    cell.set_edgecolor("#dddddd")
                    if r == 0:
                        cell.set_facecolor("#dce6f1")
                        cell.set_text_props(fontweight="bold")
                target_fig.text(0.06, 0.035, cp_note, fontsize=7, color="#555555")

            # Fix 1 — if the table has more than 10 rows start it on a new page
            # so it doesn't overlap the DCF text section above.
            if len(g_rates) > 10:
                # Close page 5 without the table, just the DCF text
                add_page_chrome(fig, ticker, report_date, "Page 5 — Valuation")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

                # New page: sensitivity table only
                fig_sens = plt.figure(figsize=(11, 8.5))
                fig_sens.patch.set_facecolor("white")
                fig_sens.text(0.06, 0.90,
                              "Sensitivity Table — Implied Share Price",
                              fontsize=9, fontweight="bold")
                _render_sens_table(fig_sens, 0.06, 0.08, 0.90, 0.80)
                add_page_chrome(fig_sens, ticker, report_date,
                                "Page 5b — Sensitivity Table")
                pdf.savefig(fig_sens, bbox_inches="tight")
                plt.close(fig_sens)
            else:
                # Table fits on the same page as the DCF text
                fig.text(0.06, y, "Sensitivity Table — Implied Share Price",
                         fontsize=9, fontweight="bold")
                y -= 0.03
                _render_sens_table(fig, 0.06, 0.06, 0.90, y - 0.06)
                add_page_chrome(fig, ticker, report_date, "Page 5 — Valuation")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

            # ── PAGE 6: MONTE CARLO ──────────────────────────────────
            fig, (ax_hist, ax_cdf) = plt.subplots(1, 2, figsize=(11, 8.5))
            fig.subplots_adjust(top=0.84, bottom=0.18, left=0.08, right=0.96)

            if mc_prices is not None and len(mc_prices) > 0:
                bins = np.linspace(mc_prices.min(), mc_prices.max(), 80)
                ax_hist.hist(mc_prices, bins=bins, density=True,
                             color="steelblue", edgecolor="white",
                             linewidth=0.3, alpha=0.8)
                if current_price:
                    ax_hist.axvline(current_price, color="crimson", lw=2,
                                    label=f"Market  ${current_price:.0f}")
                ax_hist.axvline(mc_p10, color="darkorange", lw=1.5, linestyle="--",
                                label=f"10th pct  ${mc_p10:.0f}")
                ax_hist.axvline(mc_p90, color="seagreen", lw=1.5, linestyle="--",
                                label=f"90th pct  ${mc_p90:.0f}")
                ax_hist.axvline(mc_mean, color="navy", lw=1.5, linestyle=":",
                                label=f"Mean  ${mc_mean:.0f}")

                sorted_p = np.sort(mc_prices)
                cdf_vals = np.arange(1, len(sorted_p) + 1) / len(sorted_p)
                ax_cdf.plot(sorted_p, cdf_vals * 100, color="steelblue", lw=2)
                if current_price:
                    ax_cdf.axvline(current_price, color="crimson", lw=2,
                                   label=f"Market  ${current_price:.0f}")
                    prob_below = 100 - (prob_above if prob_above else 0)
                    ax_cdf.annotate(f"{prob_below:.1f}% below\nmarket price",
                                    xy=(current_price, prob_below),
                                    xytext=(current_price * 0.6, prob_below - 15),
                                    fontsize=7, color="crimson",
                                    arrowprops=dict(arrowstyle="->",
                                                    color="crimson", lw=1))
                ax_cdf.axvline(mc_p10, color="darkorange", lw=1.5, linestyle="--",
                               label=f"10th ${mc_p10:.0f}")
                ax_cdf.axvline(mc_p90, color="seagreen",  lw=1.5, linestyle="--",
                               label=f"90th ${mc_p90:.0f}")
                ax_cdf.set_ylim(0, 100)
                ax_cdf.set_ylabel("Cumulative Probability (%)", fontsize=8)
            else:
                ax_hist.text(0.5, 0.5, "Insufficient data for Monte Carlo",
                             ha="center", va="center", transform=ax_hist.transAxes)
                ax_cdf.text(0.5, 0.5, "N/A", ha="center", va="center",
                            transform=ax_cdf.transAxes)

            ax_hist.set_title("Distribution of Implied Share Prices\n(10,000 simulations)",
                              fontsize=9, fontweight="bold")
            ax_hist.set_xlabel("Implied Price (USD)", fontsize=8)
            ax_hist.set_ylabel("Density", fontsize=8)
            ax_hist.legend(fontsize=7)
            ax_hist.grid(True, linestyle="--", alpha=0.4)

            ax_cdf.set_title("Cumulative Distribution",
                             fontsize=9, fontweight="bold")
            ax_cdf.set_xlabel("Implied Price (USD)", fontsize=8)
            ax_cdf.legend(fontsize=7)
            ax_cdf.grid(True, linestyle="--", alpha=0.4)

            g_mean_pct = f"{g_mc*100:.1f}%"
            fig.text(0.5, 0.10,
                     f"Mean: ${mc_mean:.2f}   Median: ${mc_median:.2f}   "
                     f"10th: ${mc_p10:.2f}   90th: ${mc_p90:.2f}   "
                     f"P(implied > market): {prob_above:.1f}%"
                     if all(v is not None for v in [mc_mean, mc_median, mc_p10, mc_p90, prob_above])
                     else "Insufficient data",
                     ha="center", fontsize=8)
            fig.text(0.5, 0.07,
                     f"FCF growth ~ N({g_mean_pct}, 3%)  |  "
                     "WACC ~ N(9.5%, 1%)  |  Terminal g ~ N(3.0%, 0.5%)",
                     ha="center", fontsize=7, color="#555555")
            if mc_growth_note:
                fig.text(0.5, 0.04, mc_growth_note,
                         ha="center", fontsize=7, color="darkorange", style="italic")

            add_page_chrome(fig, ticker, report_date, "Page 6 — Monte Carlo")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # ── PAGE 7: TECHNICAL SIGNALS ────────────────────────────
            fig = plt.figure(figsize=(11, 8.5))
            gs7 = GridSpec(2, 1, figure=fig, height_ratios=[3, 1],
                           hspace=0.3, top=0.88, bottom=0.12,
                           left=0.07, right=0.96)
            ax_p = fig.add_subplot(gs7[0])
            ax_r = fig.add_subplot(gs7[1], sharex=ax_p)

            _ma20  = close_4y.rolling(20).mean()
            _ma50  = close_4y.rolling(50).mean()
            _ma200 = close_4y.rolling(200).mean()
            _rsi2  = calc_rsi(close_4y)

            ax_p.plot(close_4y.index, close_4y.values,  color="lightgray",    lw=0.9,
                      label="Close")
            ax_p.plot(_ma20.index,    _ma20.values,     color="seagreen",     lw=1.2,
                      label="20MA")
            ax_p.plot(_ma50.index,    _ma50.values,     color="steelblue",    lw=1.5,
                      label="50MA")
            ax_p.plot(_ma200.index,   _ma200.values,    color="mediumpurple", lw=1.5,
                      linestyle="--", label="200MA")

            # Crossovers
            _prev_above = _ma50.shift(1) > _ma200.shift(1)
            _curr_above = _ma50 > _ma200
            _gold = close_4y[(~_prev_above) & _curr_above].dropna()
            _dead = close_4y[( _prev_above) & (~_curr_above)].dropna()
            if not _gold.empty:
                ax_p.scatter(_gold.index, _gold.values, marker="^",
                             color="limegreen", s=100, zorder=5,
                             label="Golden Cross")
            if not _dead.empty:
                ax_p.scatter(_dead.index, _dead.values, marker="v",
                             color="crimson", s=100, zorder=5,
                             label="Death Cross")

            ax_p.set_title(f"{ticker} — 2-Year Price with MAs & Crossover Signals",
                           fontsize=10, fontweight="bold")
            ax_p.set_ylabel("Price (USD)", fontsize=8)
            ax_p.legend(fontsize=7, loc="upper left")
            ax_p.grid(True, linestyle="--", alpha=0.4)

            ax_r.plot(_rsi2.index, _rsi2.values, color="darkorange", lw=1.2,
                      label="RSI(14)")
            ax_r.axhline(70, color="crimson",  lw=1, linestyle="--")
            ax_r.axhline(30, color="seagreen", lw=1, linestyle="--")
            ax_r.fill_between(_rsi2.index, 70, _rsi2.clip(lower=70),
                              color="crimson",  alpha=0.15)
            ax_r.fill_between(_rsi2.index, _rsi2.clip(upper=30), 30,
                              color="seagreen", alpha=0.15)
            ax_r.set_title(f"RSI (14-day)  —  Current: {current_rsi:.1f}" if current_rsi else "RSI",
                           fontsize=9, fontweight="bold")
            ax_r.set_ylim(0, 100)
            ax_r.set_yticks([30, 50, 70])
            ax_r.set_ylabel("RSI", fontsize=8)
            ax_r.grid(True, linestyle="--", alpha=0.4)

            sig_color = ("#2ecc71" if "ENTRY" in current_signal
                         else "#e74c3c" if "DEATH" in current_signal or "OVERBOUGHT" in current_signal
                         else "#e67e22" if "OVERSOLD" in current_signal
                         else "#555555")
            fig.text(0.5, 0.065, f"Signal Status: {current_signal}",
                     ha="center", fontsize=9, fontweight="bold", color=sig_color)

            add_page_chrome(fig, ticker, report_date, "Page 7 — Technical Signals")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # ── PAGE 8: EARNINGS & SUMMARY ───────────────────────────
            fig = plt.figure(figsize=(11, 8.5))
            fig.patch.set_facecolor("white")
            fig.subplots_adjust(top=0.88, bottom=0.08, left=0.06, right=0.96)

            y8 = 0.87
            fig.text(0.06, y8, "EARNINGS CALENDAR",
                     fontsize=10, fontweight="bold")
            y8 -= 0.04

            try:
                cal = t.calendar
                earnings_date = None
                if cal is not None:
                    if isinstance(cal, pd.DataFrame) and not cal.empty:
                        earnings_date = pd.Timestamp(cal.columns[0])
                    elif isinstance(cal, dict):
                        ed_list = cal.get("Earnings Date", [])
                        if ed_list:
                            earnings_date = pd.Timestamp(ed_list[0])

                if earnings_date:
                    days_until = (earnings_date - pd.Timestamp.today()).days
                    fig.text(0.06, y8,
                             f"Next Earnings Date: {earnings_date.strftime('%Y-%m-%d')}   "
                             f"|   Days Until Earnings: {days_until}",
                             fontsize=9)
                    y8 -= 0.03
                    fwd_eps = info.get("forwardEps")
                    if fwd_eps:
                        fig.text(0.06, y8,
                                 f"Forward EPS Estimate: ${fwd_eps:.2f}   "
                                 "(source: yfinance .info)",
                                 fontsize=8, color="#555555")
                        y8 -= 0.03
                else:
                    fig.text(0.06, y8, "Earnings date unavailable",
                             fontsize=9, color="#888888")
                    y8 -= 0.03
            except Exception:
                fig.text(0.06, y8, "Earnings date unavailable",
                         fontsize=9, color="#888888")
                y8 -= 0.03

            # Divider
            fig.add_artist(plt.Line2D([0.06, 0.96], [y8 - 0.01, y8 - 0.01],
                                      transform=fig.transFigure,
                                      color="#cccccc", linewidth=0.8))
            y8 -= 0.04

            fig.text(0.5, y8, f"{ticker} — Investment Summary",
                     ha="center", fontsize=11, fontweight="bold")
            y8 -= 0.03

            # Summary table
            def sf(v, style):
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return "N/A"
                if style == "price":  return f"${v:.2f}"
                if style == "bln":    return f"${v/1e9:.1f}B"
                if style == "pct":    return f"{v*100:.1f}%"
                if style == "pct+":   return f"{v*100:+.2f}%"
                if style == "x":      return f"{v:.1f}x"
                if style == "x2":     return f"{v:.2f}x"
                if style == "rsi":    return f"{v:.1f} ({rsi_interp})"
                return str(v)

            summary_rows = [
                ["Current Price",        sf(current_price, "price"),
                 "1-Year Return",         sf(total_ret, "pct+")],
                ["Annualised Volatility", sf(ann_vol, "pct"),
                 "Current RSI",           sf(current_rsi, "rsi")],
                ["Gross Margin",         sf(gross_margin, "pct"),
                 "Operating Margin",      sf(op_margin, "pct")],
                ["Net Margin",           sf(net_margin, "pct"),
                 "Debt to Equity",        sf(debt_equity, "x2")],
                ["FCF Conversion",       sf(fcf_conv, "x2"),
                 "Current Ratio",         sf(cur_ratio, "x2")],
                ["P/E Ratio",            sf(pe, "x"),
                 "EV/EBITDA",             sf(ev_ebitda, "x")],
                ["DCF Implied Price",    (f"${implied_price:.2f} ({diff_pct:+.1f}%)"
                                          if implied_price and diff_pct is not None
                                          else "N/A"),
                 "Monte Carlo Median",    (f"${mc_median:.2f}" if mc_median else "N/A")],
                ["P(Undervalued) MC",    (f"{prob_above:.1f}%" if prob_above is not None else "N/A"),
                 "Screener Score",        f"{score}/8 — {tier_label}"],
                ["Signal Status",        current_signal, "", ""],
            ]

            sum_ax = fig.add_axes([0.06, 0.08, 0.90, y8 - 0.12])
            sum_ax.axis("off")

            tbl = sum_ax.table(
                cellText=summary_rows,
                colLabels=["Metric A", "Value A", "Metric B", "Value B"],
                cellLoc="center", loc="center"
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)
            tbl.scale(1, 1.5)
            for (r, c), cell in tbl.get_celld().items():
                cell.set_edgecolor("#dddddd")
                if r == 0:
                    cell.set_facecolor("#dce6f1")
                    cell.set_text_props(fontweight="bold")
                elif r % 2 == 0:
                    cell.set_facecolor("#f7f9fc")
                else:
                    cell.set_facecolor("white")
                if c in (0, 2):
                    cell.set_text_props(fontweight="bold")

            add_page_chrome(fig, ticker, report_date, "Page 8 — Earnings & Summary")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        print(f"  Report saved → {pdf_path}")

    except Exception as e:
        print(f"  ERROR generating report for {ticker}: {e}")
        import traceback
        traceback.print_exc()


# ─────────────────────────────────────────────
if __name__ == "__main__":
    generate_report(REPORT_TICKER)
