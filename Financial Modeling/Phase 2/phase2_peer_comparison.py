import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ticker1 = "TPL"
ticker2 = "APTV"
ticker3 = "NOW"
tickers = [ticker1, ticker2, ticker3]
COLORS  = {ticker1: "steelblue", ticker2: "darkorange", ticker3: "lightgreen"}

records = {}

for ticker in tickers:
    t  = yf.Ticker(ticker)
    is_ = t.financials.dropna(axis=1, how="all")
    bs  = t.balance_sheet.dropna(axis=1, how="all")
    cf  = t.cashflow.dropna(axis=1, how="all")

    fy = is_.columns[0]

    def get(df, label):
        try:
            return df.loc[label, fy]
        except KeyError:
            return float("nan")

    revenue    = get(is_, "Total Revenue")
    gross      = get(is_, "Gross Profit")
    op_income  = get(is_, "Operating Income")
    net_income = get(is_, "Net Income")
    debt       = get(bs,  "Total Debt")
    equity     = get(bs,  "Stockholders Equity")
    cur_assets = get(bs,  "Current Assets")
    cur_liab   = get(bs,  "Current Liabilities")
    op_cf      = get(cf,  "Operating Cash Flow")
    fcf        = get(cf,  "Free Cash Flow")

    records[ticker] = {
        "FY End":            fy.strftime("%Y-%m-%d"),
        "Gross Margin %":    gross      / revenue   * 100,
        "Operating Margin %":op_income  / revenue   * 100,
        "Net Margin %":      net_income / revenue   * 100,
        "Debt to Equity":    debt       / equity,
        "Current Ratio":     cur_assets / cur_liab,
        "FCF Conversion":    fcf        / net_income,
    }

df = pd.DataFrame(records).T

print("=" * 70)
print("PEER COMPARISON — Most Recent Fiscal Year")
print("=" * 70)
print(f"\n{'':20}{'AAPL':>15}{'MSFT':>15}{'GOOGL':>15}")
print("-" * 65)

fmt_pct = "{:>14.1f}%"
fmt_num = "{:>15.2f}"

for col, fmt in [
    ("FY End",             "{:>15}"),
    ("Gross Margin %",     fmt_pct),
    ("Operating Margin %", fmt_pct),
    ("Net Margin %",       fmt_pct),
    ("Debt to Equity",     fmt_num),
    ("Current Ratio",      fmt_num),
    ("FCF Conversion",     fmt_num),
]:
    row = f"{col:<20}"
    for ticker in tickers:
        val = df.loc[ticker, col]
        row += fmt.format(val)
    print(row)

ratio_meta = [
    ("Gross Margin %",     "Gross Profit / Revenue",              "%"),
    ("Operating Margin %", "Operating Income / Revenue",          "%"),
    ("Net Margin %",       "Net Income / Revenue",                "%"),
    ("Debt to Equity",     "Total Debt / Stockholders Equity",    "x"),
    ("Current Ratio",      "Current Assets / Current Liabilities","x"),
    ("FCF Conversion",     "Free Cash Flow / Net Income",         "x"),
]

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

x = range(len(tickers))

for ax, (col, formula, unit) in zip(axes, ratio_meta):
    values = [df.loc[t, col] for t in tickers]
    bars   = ax.bar(tickers, values,
                    color=[COLORS[t] for t in tickers],
                    width=0.5, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, values):
        label = f"{val:.1f}%" if unit == "%" else f"{val:.2f}x"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + abs(bar.get_height()) * 0.02,
                label, ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_title(col, fontsize=10, fontweight="bold")
    ax.set_xlabel(formula, fontsize=8, color="gray")
    ax.set_ylabel(unit, fontsize=8)
    ax.set_xticks(range(len(tickers)))
    ax.set_xticklabels(tickers)

handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[t]) for t in tickers]
fig.legend(handles, tickers, loc="upper right", fontsize=9, title="Company")

fig.suptitle("Peer Comparison — AAPL vs MSFT vs GOOGL (Most Recent FY)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("Phase 2/peer_comparison_ratios.png", dpi=150, bbox_inches="tight")
print("\nChart saved as Phase 2/peer_comparison_ratios.png")
