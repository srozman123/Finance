import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# SECTION 1: Pull raw inputs from yfinance
# ============================================================================
# Valuation multiples connect a company's market price to its fundamentals.
# They answer the question: how much is the market paying for each dollar of
# revenue, earnings, or cash flow? Multiples are the primary tool for
# comparing whether a stock is cheap or expensive relative to peers or history.

ticker = "CTRE"
t = yf.Ticker(ticker)

# --- Market inputs (live, from .info) ---
current_price       = t.info["currentPrice"]
shares_outstanding  = t.info["sharesOutstanding"]

# --- Financial statement inputs (most recent fiscal year) ---
is_ = t.financials.dropna(axis=1, how="all")
bs  = t.balance_sheet.dropna(axis=1, how="all")
cf  = t.cashflow.dropna(axis=1, how="all")

fy = is_.columns[0]  # most recent fiscal year end

total_revenue = is_.loc["Total Revenue", fy]
net_income    = is_.loc["Net Income",    fy]
ebitda        = is_.loc["EBITDA",        fy]
total_debt    = bs.loc["Total Debt",     fy]
cash          = bs.loc["Cash And Cash Equivalents", fy]

# ============================================================================
# SECTION 2: Calculate valuation metrics
# ============================================================================
# Market Cap is the market's total assessed value of the equity.
# Enterprise Value adds debt and subtracts cash to capture the full cost of
# buying the entire business — what an acquirer would actually pay.

market_cap        = current_price * shares_outstanding
enterprise_value  = market_cap + total_debt - cash

# Multiples
pe_ratio   = market_cap       / net_income    # Price-to-Earnings
ps_ratio   = market_cap       / total_revenue # Price-to-Sales
ev_ebitda  = enterprise_value / ebitda        # EV/EBITDA

# ============================================================================
# SECTION 3: Print results
# ============================================================================

B = 1e9  # billions

print("=" * 60)
print(f"VALUATION MULTIPLES — {ticker}  (FY end: {fy.strftime('%Y-%m-%d')})")
print("=" * 60)

print("\n[ INPUTS ]")
print(f"  {'Current Stock Price':<35} ${current_price:>10.2f}")
print(f"  {'Shares Outstanding':<35} {shares_outstanding / B:>9.2f}B")
print(f"  {'Total Revenue':<35} ${total_revenue / B:>9.2f}B")
print(f"  {'Net Income':<35} ${net_income    / B:>9.2f}B")
print(f"  {'EBITDA':<35} ${ebitda        / B:>9.2f}B")
print(f"  {'Total Debt':<35} ${total_debt    / B:>9.2f}B")
print(f"  {'Cash & Cash Equivalents':<35} ${cash          / B:>9.2f}B")

print("\n[ CALCULATED VALUES ]")
print(f"  {'Market Cap  (= Price × Shares)':<35} ${market_cap       / B:>9.2f}B")
print(f"  {'Enterprise Value  (= MCap + Debt − Cash)':<35} ${enterprise_value / B:>9.2f}B")

print("\n[ VALUATION MULTIPLES ]")
print(f"  {'P/E  (= Market Cap / Net Income)':<35} {pe_ratio:>9.1f}x")
print(f"  {'P/S  (= Market Cap / Revenue)':<35} {ps_ratio:>9.1f}x")
print(f"  {'EV/EBITDA  (= EV / EBITDA)':<35} {ev_ebitda:>9.1f}x")

# ============================================================================
# SECTION 4: Peer group comparison
# ============================================================================
# Multiples are only meaningful in context. Comparing AAPL's P/E to MSFT and
# GOOGL shows whether the market is pricing it at a premium or discount to
# its closest peers — and which company the market views as the best value.

peers   = ["CTRE", "OHI", "SBRA"]
COLORS  = {"AAPL": "steelblue", "MSFT": "darkorange", "GOOGL": "seagreen"}
records = {}

for tk in peers:
    info = yf.Ticker(tk).info
    is_p = yf.Ticker(tk).financials.dropna(axis=1, how="all")
    bs_p = yf.Ticker(tk).balance_sheet.dropna(axis=1, how="all")
    fy_p = is_p.columns[0]

    price  = info["currentPrice"]
    shares = info["sharesOutstanding"]
    rev    = is_p.loc["Total Revenue", fy_p]
    ni     = is_p.loc["Net Income",    fy_p]
    ebit   = is_p.loc["EBITDA",        fy_p]
    debt   = bs_p.loc["Total Debt",    fy_p]
    cash_p = bs_p.loc["Cash And Cash Equivalents", fy_p]

    mcap = price * shares
    ev   = mcap + debt - cash_p

    records[tk] = {
        "FY End":    fy_p.strftime("%Y-%m-%d"),
        "P/E":       mcap / ni,
        "P/S":       mcap / rev,
        "EV/EBITDA": ev   / ebit,
    }

df = pd.DataFrame(records).T

# Print comparison table
print("\n\n" + "=" * 55)
print("PEER MULTIPLES COMPARISON — Most Recent Fiscal Year")
print("=" * 55)
print(f"\n{'':20}{'AAPL':>10}{'MSFT':>10}{'GOOGL':>10}{'Formula':>15}")
print("-" * 55)
for col, formula in [("P/E", "MCap/NI"), ("P/S", "MCap/Rev"), ("EV/EBITDA", "EV/EBITDA")]:
    row = f"{col:<20}"
    for tk in peers:
        row += f"  {df.loc[tk, col]:>6.1f}x"
    row += f"   ({formula})"
    print(row)

# ============================================================================
# SECTION 5: Grouped bar chart
# ============================================================================

multiples = ["P/E", "P/S", "EV/EBITDA"]
formulas  = {"P/E": "Market Cap / Net Income",
             "P/S": "Market Cap / Revenue",
             "EV/EBITDA": "Enterprise Value / EBITDA"}

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

for ax, multiple in zip(axes, multiples):
    values  = [df.loc[tk, multiple] for tk in peers]
    colors  = [COLORS[tk] for tk in peers]
    x       = np.arange(len(peers))
    bars    = ax.bar(x, values, color=colors, width=0.5, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.1f}x", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.set_title(f"{multiple}\n({formulas[multiple]})", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(peers)
    ax.set_ylabel("Multiple (x)", fontsize=8)

handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[tk]) for tk in peers]
fig.legend(handles, peers, loc="upper right", fontsize=9, title="Company")
fig.suptitle("Peer Valuation Multiples — AAPL vs MSFT vs GOOGL", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("Phase 3/peer_multiples_comparison.png", dpi=150, bbox_inches="tight")
print("\nChart saved as Phase 3/peer_multiples_comparison.png")
