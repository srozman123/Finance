import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Create a Ticker object for the selected symbol.
# yfinance's Ticker class is the entry point for all fundamental data —
# financial statements, balance sheets, cash flows, and more.
ticker_symbol = "HOOD"
aapl = yf.Ticker(ticker_symbol)

# Pull the three core financial statements.
# Each is returned as a pandas DataFrame where:
#   - columns = fiscal year end dates
#   - rows    = individual line items (revenue, net income, total assets, etc.)
income_statement = aapl.financials.iloc[:, :4]
balance_sheet    = aapl.balance_sheet.iloc[:, :4]
cash_flow        = aapl.cashflow.iloc[:, :4]

# --- Income Statement ---
print("=" * 60)
print("INCOME STATEMENT")
print("=" * 60)
print(f"Shape: {income_statement.shape}")
print(f"\nFiscal years (columns):")
print(income_statement.columns.tolist())
print(f"\nLine items (rows):")
for item in income_statement.index.tolist():
    print(f"  {item}")

# --- Balance Sheet ---
print("\n" + "=" * 60)
print("BALANCE SHEET")
print("=" * 60)
print(f"Shape: {balance_sheet.shape}")
print(f"\nFiscal years (columns):")
print(balance_sheet.columns.tolist())
print(f"\nLine items (rows):")
for item in balance_sheet.index.tolist():
    print(f"  {item}")

# --- Cash Flow Statement ---
print("\n" + "=" * 60)
print("CASH FLOW STATEMENT")
print("=" * 60)
print(f"Shape: {cash_flow.shape}")
print(f"\nFiscal years (columns):")
print(cash_flow.columns.tolist())
print(f"\nLine items (rows):")
for item in cash_flow.index.tolist():
    print(f"  {item}")

# =============================================================================
# SECTION 2: Extract and display key line items in billions
# =============================================================================

# Helper to print a line item's values across all fiscal years in billions.
def print_line_item(name, series):
    print(f"\n  {name}:")
    for date, value in series.items():
        year = date.year
        print(f"    {year}: ${value / 1e9:.2f}B")

years = [col.year for col in income_statement.columns]
print("\n" + "=" * 60)
print(f"KEY LINE ITEMS — FY{years[-1]} to FY{years[0]}")
print("=" * 60)

# --- Income Statement items ---
# The income statement measures profitability over a period of time. It shows
# how much revenue a company generated, what it cost to deliver that revenue,
# and what was left over as profit after all expenses and taxes.
print("\n[ INCOME STATEMENT ]")
total_revenue    = income_statement.loc["Total Revenue"]
gross_profit     = income_statement.loc["Gross Profit"]
operating_income = income_statement.loc["Operating Income"]
net_income       = income_statement.loc["Net Income"]

print_line_item("Total Revenue    (source: income statement, top line)",        total_revenue)
print_line_item("Gross Profit     (= Revenue − Cost of Revenue)",              gross_profit)
print_line_item("Operating Income (= Gross Profit − Operating Expenses)",      operating_income)
print_line_item("Net Income       (= Operating Income − Interest − Taxes)",    net_income)

# --- Balance Sheet items ---
# The balance sheet is a snapshot of what a company owns (assets) and owes
# (liabilities) at a single point in time. The difference is shareholders'
# equity — the net worth attributable to owners. It answers: how is the
# company financed, and how much financial cushion does it have?
print("\n[ BALANCE SHEET ]")
total_debt         = balance_sheet.loc["Total Debt"]
stockholders_equity = balance_sheet.loc["Stockholders Equity"]
current_assets     = balance_sheet.loc["Current Assets"]
current_liabilities = balance_sheet.loc["Current Liabilities"]

print_line_item("Total Debt          (= Short Term Debt + Long Term Debt)",                    total_debt)
print_line_item("Stockholders Equity (= Total Assets − Total Liabilities)",                   stockholders_equity)
print_line_item("Current Assets      (source: balance sheet — due within 12 months)",         current_assets)
print_line_item("Current Liabilities (source: balance sheet — owed within 12 months)",        current_liabilities)

# --- Cash Flow Statement items ---
# The cash flow statement tracks the actual movement of cash in and out of the
# business. Unlike net income, it cannot be manipulated by accounting choices —
# cash is either in the bank or it isn't. Free cash flow in particular is one
# of the most important metrics in valuation: it's the cash a company generates
# after maintaining and growing its asset base.
print("\n[ CASH FLOW STATEMENT ]")
operating_cash_flow = cash_flow.loc["Operating Cash Flow"]
free_cash_flow      = cash_flow.loc["Free Cash Flow"]
capex               = cash_flow.loc["Capital Expenditure"]

print_line_item("Operating Cash Flow  (= Net Income + Non-Cash Items + Working Capital Changes)", operating_cash_flow)
print_line_item("Free Cash Flow       (= Operating Cash Flow − Capital Expenditure)",            free_cash_flow)
print_line_item("Capital Expenditure  (= cash spent on physical assets — negative = outflow)",   capex)

# =============================================================================
# SECTION 3: Ratio Analysis
# =============================================================================
# Ratios compress raw dollar figures into comparable metrics that reveal how
# efficiently and safely a company operates. They are the primary tool analysts
# use to compare companies of different sizes or to track one company over time.

years = [col.year for col in income_statement.columns]

print("\n" + "=" * 60)
print("RATIO ANALYSIS")
print("=" * 60)

# Column header row — one column per fiscal year
header = f"{'Ratio':<45}" + "".join(f"  FY{y}" for y in years)
print("\n" + header)
print("-" * len(header))

def print_ratio_pct(label, series):
    row = f"{label:<45}"
    for y in years:
        col = next(c for c in income_statement.columns if c.year == y)
        row += f"  {series[col] * 100:>5.1f}%"
    print(row)

def print_ratio_num(label, series):
    row = f"{label:<45}"
    for y in years:
        col = next(c for c in income_statement.columns if c.year == y)
        row += f"  {series[col]:>6.2f}"
    print(row)

# Margin ratios — measure how much of each revenue dollar becomes profit
# at different stages of the income statement
gross_margin     = gross_profit     / total_revenue
net_margin       = net_income       / total_revenue
operating_margin = operating_income / total_revenue

print_ratio_pct("Gross Margin       (= Gross Profit / Revenue)",       gross_margin)
print_ratio_pct("Operating Margin   (= Operating Income / Revenue)",   operating_margin)
print_ratio_pct("Net Margin         (= Net Income / Revenue)",         net_margin)

print()

# Leverage & liquidity ratios — measure financial risk and short-term health
debt_to_equity = total_debt         / stockholders_equity
current_ratio  = current_assets     / current_liabilities

print_ratio_num("Debt to Equity     (= Total Debt / Stockholders Equity)", debt_to_equity)
print_ratio_num("Current Ratio      (= Current Assets / Current Liabilities)", current_ratio)

print()

# Cash quality ratio — measures how much of net income converts to real cash.
# A value above 1.0 means the company generates more free cash than it reports
# in net income, which is a sign of high earnings quality.
fcf_conversion = free_cash_flow / net_income

print_ratio_num("FCF Conversion     (= Free Cash Flow / Net Income)",   fcf_conversion)

# =============================================================================
# SECTION 4: Ratio Dashboard Chart
# =============================================================================

# Build aligned index — fiscal years as integers, oldest to newest (left→right)
fy = [col.year for col in income_statement.columns][::-1]
cols_ordered = income_statement.columns[::-1]

def get_values(series):
    return [series[c] for c in cols_ordered]

ratios = [
    {
        "title": "Gross Margin",
        "formula": "Gross Profit / Revenue",
        "values": [v * 100 for v in get_values(gross_margin)],
        "ylabel": "%",
        "refline": None,
    },
    {
        "title": "Operating Margin",
        "formula": "Operating Income / Revenue",
        "values": [v * 100 for v in get_values(operating_margin)],
        "ylabel": "%",
        "refline": None,
    },
    {
        "title": "Net Margin",
        "formula": "Net Income / Revenue",
        "values": [v * 100 for v in get_values(net_margin)],
        "ylabel": "%",
        "refline": None,
    },
    {
        "title": "Debt to Equity",
        "formula": "Total Debt / Stockholders Equity",
        "values": get_values(debt_to_equity),
        "ylabel": "x",
        "refline": 1.0,
    },
    {
        "title": "Current Ratio",
        "formula": "Current Assets / Current Liabilities",
        "values": get_values(current_ratio),
        "ylabel": "x",
        "refline": 1.0,
    },
    {
        "title": "FCF Conversion",
        "formula": "Free Cash Flow / Net Income",
        "values": get_values(fcf_conversion),
        "ylabel": "x",
        "refline": 1.0,
    },
]

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for ax, r in zip(axes, ratios):
    ax.plot(fy, r["values"], marker="o", linewidth=2, color="steelblue")
    for x, y in zip(fy, r["values"]):
        label = f"{y:.1f}%" if r["ylabel"] == "%" else f"{y:.2f}x"
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    if r["refline"] is not None:
        ax.axhline(r["refline"], color="tomato", linewidth=1,
                   linestyle="--", label=f"Ref: {r['refline']:.1f}")
        ax.legend(fontsize=8)
    ax.set_title(f"{r['title']}\n({r['formula']})", fontsize=9, fontweight="bold")
    ax.set_xlabel("Fiscal Year", fontsize=8)
    ax.set_ylabel(r["ylabel"], fontsize=8)
    ax.set_xticks(fy)

fig.suptitle(f"{ticker_symbol} — Ratio Dashboard (FY{fy[0]}–FY{fy[-1]})", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
filename = f"Phase 2/{ticker_symbol.lower()}_ratio_dashboard.png"
plt.savefig(filename, dpi=150, bbox_inches="tight")
print(f"\nChart saved as {filename}")
