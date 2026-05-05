import yfinance as yf

ticker = "GOOG"
t = yf.Ticker(ticker)

WACC          = 0.095  # Weighted Average Cost of Capital — the discount rate
TERMINAL_GROW = 0.030  # Long-run perpetual growth rate (roughly GDP growth)
# EXIT_MULTIPLE is derived dynamically from live EV/EBITDA — see Step 1 below.
# No manual input needed; the model calibrates automatically to any ticker.
PROJ_YEARS    = 5

B = 1e9

print("=" * 60)
print(f"DCF MODEL — {ticker}")
print(f"  WACC: {WACC*100:.1f}%   Terminal Growth Rate: {TERMINAL_GROW*100:.1f}%")
print("=" * 60)

# ============================================================================
# STEP 1 — Historical FCF growth rate and projected FCFs
# ============================================================================
# Free Cash Flow is what a company actually generates for its owners after
# reinvesting in the business. We look at the past 4 years to estimate a
# sustainable growth rate, then project that rate forward 5 years.
# We use the geometric mean (CAGR) rather than arithmetic mean because returns
# compound — a 50% gain followed by a 50% loss is NOT a 0% average return.

cf  = t.cashflow.dropna(axis=1, how="all")
fcf_series = cf.loc["Free Cash Flow"].sort_index()  # oldest → newest

# Take the 4 most recent years
fcf_hist = fcf_series.iloc[-4:]
fcf_values = fcf_hist.values.tolist()
fcf_years  = [d.year for d in fcf_hist.index]

# CAGR = (ending / beginning) ^ (1 / n) - 1
cagr = (fcf_values[-1] / fcf_values[0]) ** (1 / (len(fcf_values) - 1)) - 1
base_fcf = fcf_values[-1]

print("\n[ STEP 1 — FCF Projections ]")
print(f"  Historical FCF ({fcf_years[0]}–{fcf_years[-1]}):")
for yr, val in zip(fcf_years, fcf_values):
    print(f"    FY{yr}: ${val / B:.2f}B")
print(f"\n  Historical FCF CAGR: {cagr * 100:.1f}%")
print(f"  Base FCF (FY{fcf_years[-1]}): ${base_fcf / B:.2f}B")

# Pull EBITDA for the most recent fiscal year — used in the exit multiple TV method
is_         = t.financials.dropna(axis=1, how="all")
fy0         = is_.columns[0]
base_ebitda = is_.loc["EBITDA", fy0]
print(f"  Base EBITDA (FY{fy0.year}): ${base_ebitda / B:.2f}B  (used in exit multiple TV)")

# Pull balance sheet, market data, and compute live EV/EBITDA now so the exit
# multiple scenarios in Step 2 can be derived automatically from the market.
# This means EXIT_MULTIPLE self-calibrates to whatever ticker is set above —
# no manual lookup required when switching between companies.
bs          = t.balance_sheet.dropna(axis=1, how="all")
fy          = bs.columns[0]
total_debt  = bs.loc["Total Debt", fy]
cash        = bs.loc["Cash And Cash Equivalents", fy]

shares        = t.info["sharesOutstanding"]
current_price = t.info["currentPrice"]
market_cap    = current_price * shares
ev_live       = market_cap + total_debt - cash
ev_ebitda_live = ev_live / base_ebitda      # live EV/EBITDA — the base exit multiple

# Three exit multiple scenarios anchored to the live EV/EBITDA
exit_multiple_bear = ev_ebitda_live * 0.8   # compression: re-rating down 20%
exit_multiple_base = ev_ebitda_live * 1.0   # steady state: current market multiple holds
exit_multiple_bull = ev_ebitda_live * 1.3   # expansion: re-rating up 30%

print(f"\n  Live EV/EBITDA (market-implied):  {ev_ebitda_live:.1f}x")
print(f"  Exit multiple scenarios:")
print(f"    Bear  (×0.8):  {exit_multiple_bear:.1f}x")
print(f"    Base  (×1.0):  {exit_multiple_base:.1f}x")
print(f"    Bull  (×1.3):  {exit_multiple_bull:.1f}x")

projected_fcf = []
print(f"\n  Projected FCFs:")
for i in range(1, PROJ_YEARS + 1):
    fcf_proj = base_fcf * (1 + cagr) ** i
    projected_fcf.append(fcf_proj)
    print(f"    Year {i}: ${fcf_proj / B:.2f}B")

# ============================================================================
# STEP 2 — Terminal Value  (blended: perpetuity growth + exit multiple)
# ============================================================================
# After the 5-year projection window we need a single number that captures all
# remaining cash flows. Two complementary methods are used and averaged:
#
#  Method A — Perpetuity Growth (Gordon Growth Model):  captures intrinsic cash-
#    flow value assuming stable, GDP-rate growth forever. Sensitive to WACC and
#    the terminal growth rate; best anchored to long-run fundamentals.
#
#  Method B — Exit Multiple:  anchors the terminal value to what a rational buyer
#    would pay in Year 5 as a multiple of EBITDA. Uses the live EV/EBITDA
#    applied to Year-5 projected EBITDA, scaled by scenario (bear/base/bull).
#    This method is grounded in market-observable pricing rather than perpetuity
#    math, reducing reliance on a single discount rate assumption.
#
#  Blending the two gives a more robust estimate — one model anchors to cash-flow
#  theory, the other to market pricing, and the average balances both biases.
#  Running three scenarios makes the range of outcomes explicit.

# Method A — Perpetuity growth (same for all scenarios — WACC and g are fixed)
tv_perpetuity  = projected_fcf[-1] * (1 + TERMINAL_GROW) / (WACC - TERMINAL_GROW)

# Method B — Exit multiple (EBITDA projected forward at the same CAGR as FCF)
ebitda_proj_y5 = base_ebitda * (1 + cagr) ** PROJ_YEARS

tv_exit_bear   = ebitda_proj_y5 * exit_multiple_bear
tv_exit_base   = ebitda_proj_y5 * exit_multiple_base
tv_exit_bull   = ebitda_proj_y5 * exit_multiple_bull

# Blended terminal values (average of Method A and Method B per scenario)
tv_bear        = (tv_perpetuity + tv_exit_bear) / 2
tv_base        = (tv_perpetuity + tv_exit_base) / 2
tv_bull        = (tv_perpetuity + tv_exit_bull) / 2

print(f"\n[ STEP 2 — Terminal Value ]")
print(f"  Method A — Perpetuity Growth (Gordon Growth Model):")
print(f"    Formula: FCF_Year5 × (1 + g) / (WACC − g)")
print(f"           = ${projected_fcf[-1]/B:.2f}B × (1 + {TERMINAL_GROW:.3f}) / ({WACC:.3f} − {TERMINAL_GROW:.3f})")
print(f"    TV (Perpetuity): ${tv_perpetuity / B:.2f}B  [same across all scenarios]")
print(f"\n  Method B — Exit Multiple:")
print(f"    Formula: Projected_EBITDA_Year5 × Exit_Multiple")
print(f"    EBITDA Year 5: ${ebitda_proj_y5 / B:.2f}B  (Base ${base_ebitda/B:.2f}B grown at {cagr*100:.1f}%/yr × 5yr)")
print(f"    Bear  ({exit_multiple_bear:.1f}x):  TV = ${tv_exit_bear / B:.2f}B")
print(f"    Base  ({exit_multiple_base:.1f}x):  TV = ${tv_exit_base / B:.2f}B")
print(f"    Bull  ({exit_multiple_bull:.1f}x):  TV = ${tv_exit_bull / B:.2f}B")
print(f"\n  Blended TV (avg of Method A + Method B):")
print(f"    Bear:  ${tv_bear / B:.2f}B")
print(f"    Base:  ${tv_base / B:.2f}B")
print(f"    Bull:  ${tv_bull / B:.2f}B")

# ============================================================================
# STEP 3 — Discount to Present Value
# ============================================================================
# A dollar received in the future is worth less than a dollar today because of
# the time value of money — you could invest today's dollar and earn a return.
# We discount each future cash flow by (1 + WACC)^year to express everything
# in today's dollars, making the values directly comparable and summable.

print(f"\n[ STEP 3 — Discounted Cash Flows ]")
pv_fcfs = []
for i, fcf_proj in enumerate(projected_fcf, start=1):
    pv = fcf_proj / (1 + WACC) ** i
    pv_fcfs.append(pv)
    print(f"    Year {i}: ${fcf_proj/B:.2f}B  →  PV = ${pv/B:.2f}B")

pv_tv_bear = tv_bear / (1 + WACC) ** PROJ_YEARS
pv_tv_base = tv_base / (1 + WACC) ** PROJ_YEARS
pv_tv_bull = tv_bull / (1 + WACC) ** PROJ_YEARS
print(f"    Terminal PV  — Bear: ${pv_tv_bear/B:.2f}B  |  Base: ${pv_tv_base/B:.2f}B  |  Bull: ${pv_tv_bull/B:.2f}B")

# ============================================================================
# STEP 4 — Enterprise Value and Equity Value
# ============================================================================
# Summing all discounted cash flows gives Enterprise Value — the value of the
# entire business regardless of how it's financed. To get to Equity Value
# (what belongs to shareholders), we subtract debt (which must be repaid first)
# and add back cash (which belongs to shareholders and isn't needed to operate).

sum_pv_fcf = sum(pv_fcfs)

total_pv_bear  = sum_pv_fcf + pv_tv_bear
total_pv_base  = sum_pv_fcf + pv_tv_base
total_pv_bull  = sum_pv_fcf + pv_tv_bull

equity_bear    = total_pv_bear - total_debt + cash
equity_base    = total_pv_base - total_debt + cash
equity_bull    = total_pv_bull - total_debt + cash

print(f"\n[ STEP 4 — Enterprise Value → Equity Value ]")
print(f"  Sum of PV(FCFs):      ${sum_pv_fcf / B:.2f}B  [same across all scenarios]")
print(f"  {'Scenario':<10}  {'PV(TV)':<14}  {'% of EV':<10}  {'EV':<14}  {'Equity Value'}")
print(f"  {'-'*68}")
for label, pv_tv, total_pv, eq in [
    ("Bear",  pv_tv_bear, total_pv_bear, equity_bear),
    ("Base",  pv_tv_base, total_pv_base, equity_base),
    ("Bull",  pv_tv_bull, total_pv_bull, equity_bull),
]:
    print(f"  {label:<10}  ${pv_tv/B:<13.2f}  {pv_tv/total_pv*100:<9.0f}%  ${total_pv/B:<13.2f}  ${eq/B:.2f}B")
print(f"\n  − Total Debt: ${total_debt/B:.2f}B   + Cash: ${cash/B:.2f}B  (applied in all scenarios)")

# ============================================================================
# STEP 5 — Implied Share Price vs Current Market Price
# ============================================================================
# Dividing equity value by shares outstanding gives the intrinsic value per
# share — what the model says the stock is "worth" based on its future cash
# flows. Comparing this to the current price tells you whether the stock
# appears overvalued, undervalued, or fairly priced under these assumptions.
# Note: DCF outputs are highly sensitive to WACC and growth assumptions —
# treat the result as a range, not a point estimate.

price_bear = equity_bear / shares
price_base = equity_base / shares
price_bull = equity_bull / shares

def pct_diff(p):
    return (p - current_price) / current_price * 100

print(f"\n[ STEP 5 — Implied Share Price ]")
print(f"  Shares Outstanding:    {shares / B:.2f}B")
print(f"  Current Market Price:  ${current_price:.2f}")
print()
print(f"  {'Scenario':<8}  {'Exit Multiple':<16}  {'Implied Price':<16}  {'vs Market':<12}  Verdict")
print(f"  {'-'*72}")
for label, mult, price in [
    ("Bear",  exit_multiple_bear, price_bear),
    ("Base",  exit_multiple_base, price_base),
    ("Bull",  exit_multiple_bull, price_bull),
]:
    diff = pct_diff(price)
    verdict = "UNDERVALUED" if diff > 0 else "OVERVALUED"
    print(f"  {label:<8}  {mult:<16.1f}  ${price:<15.2f}  {diff:+.1f}%{'':6}  {verdict}")

# ============================================================================
# STEP 6 — Sensitivity Analysis
# ============================================================================
# A single DCF output is misleading because it implies false precision. Two
# inputs dominate the result: the FCF growth rate and WACC. The sensitivity
# table shows the implied share price across a realistic range of each,
# making the model's uncertainty explicit. Analysts use this to define a
# valuation range (e.g. "$180–$320") rather than a single point estimate.
# The "closest to market" cell reveals what assumptions the market is pricing
# in — a useful sanity check on whether those assumptions are reasonable.
# The table uses the base exit multiple (live EV/EBITDA × 1.0) so each cell
# is internally consistent with the base scenario in Step 5.

def dcf_implied_price(fcf_growth, wacc):
    # Project FCFs
    proj = [base_fcf * (1 + fcf_growth) ** i for i in range(1, PROJ_YEARS + 1)]
    # Terminal value — same blended approach as the main model (base scenario)
    tv_perp   = proj[-1] * (1 + TERMINAL_GROW) / (wacc - TERMINAL_GROW)
    ebitda_y5 = base_ebitda * (1 + fcf_growth) ** PROJ_YEARS
    tv_exit_  = ebitda_y5 * exit_multiple_base
    tv = (tv_perp + tv_exit_) / 2
    # Discount everything
    pv_fcf = sum(cf / (1 + wacc) ** i for i, cf in enumerate(proj, start=1))
    pv_tv  = tv / (1 + wacc) ** PROJ_YEARS
    # Equity value → price
    eq = (pv_fcf + pv_tv) - total_debt + cash
    return eq / shares

growth_rates = [-0.04, -0.02, 0.00, 0.02, 0.04, 0.06, 0.08, 0.10]
wacc_values  = [0.08, 0.09, 0.10, 0.11]

print("\n\n" + "=" * 60)
print("SENSITIVITY ANALYSIS — Implied Share Price")
print(f"  Terminal Growth Rate held constant at {TERMINAL_GROW*100:.1f}%")
print(f"  Exit Multiple: base scenario ({exit_multiple_base:.1f}x live EV/EBITDA)")
print("=" * 60)

# Build table and track closest-to-market cell
col_w = 10
row_header = "FCF \\ WACC"
header = f"{row_header:<12}" + "".join(f"{'WACC '+str(int(w*100))+'%':>{col_w}}" for w in wacc_values)
print("\n" + header)
print("-" * len(header))

best_diff  = float("inf")
best_combo = None

table = {}
for g in growth_rates:
    row_label = f"FCF {g*100:+.0f}%"
    row = f"{row_label:<12}"
    for w in wacc_values:
        if w <= TERMINAL_GROW:
            price_cell = "  N/A"
        else:
            price_cell = dcf_implied_price(g, w)
            diff = abs(price_cell - current_price)
            if diff < best_diff:
                best_diff  = diff
                best_combo = (g, w, price_cell)
            price_cell = f"${price_cell:>7.2f}"
        row += f"{price_cell:>{col_w}}"
    print(row)

print(f"\n  Current Market Price: ${current_price:.2f}")
g_best, w_best, p_best = best_combo
print(f"  Closest implied price: ${p_best:.2f}  "
      f"→  FCF growth {g_best*100:+.0f}%  |  WACC {w_best*100:.0f}%")
