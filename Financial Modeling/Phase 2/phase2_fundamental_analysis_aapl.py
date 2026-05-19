import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ticker_symbol = "HOOD"
aapl = yf.Ticker(ticker_symbol)

income_statement = aapl.financials.iloc[:, :4]
balance_sheet    = aapl.balance_sheet.iloc[:, :4]
cash_flow        = aapl.cashflow.iloc[:, :4]

print("=" * 60)
print("INCOME STATEMENT")
print("=" * 60)
print(f"Shape: {income_statement.shape}")
print(f"\nFiscal years (columns):")
print(income_statement.columns.tolist())
print(f"\nLine items (rows):")
for item in income_statement.index.tolist():
    print(f"  {item}")

print("\n" + "=" * 60)
print("BALANCE SHEET")
print("=" * 60)
print(f"Shape: {balance_sheet.shape}")
print(f"\nFiscal years (columns):")
print(balance_sheet.columns.tolist())
print(f"\nLine items (rows):")
for item in balance_sheet.index.tolist():
    print(f"  {item}")

print("\n" + "=" * 60)
print("CASH FLOW STATEMENT")
print("=" * 60)
print(f"Shape: {cash_flow.shape}")
print(f"\nFiscal years (columns):")
print(cash_flow.columns.tolist())
print(f"\nLine items (rows):")
for item in cash_flow.index.tolist():
    print(f"  {item}")

def print_line_item(name, series):
    print(f"\n  {name}:")
    for date, value in series.items():
        year = date.year
        print(f"    {year}: ${value / 1e9:.2f}B")

years = [col.year for col in income_statement.columns]
print("\n" + "=" * 60)
print(f"KEY LINE ITEMS — FY{years[-1]} to FY{years[0]}")
print("=" * 60)

print("\n[ INCOME STATEMENT ]")
total_revenue    = income_statement.loc["Total Revenue"]
gross_profit     = income_statement.loc["Gross Profit"]
operating_income = income_statement.loc["Operating Income"]
net_income       = income_statement.loc["Net Income"]

print_line_item("Total Revenue    (source: income statement, top line)",        total_revenue)
print_line_item("Gross Profit     (= Revenue − Cost of Revenue)",              gross_profit)
print_line_item("Operating Income (= Gross Profit − Operating Expenses)",      operating_income)
print_line_item("Net Income       (= Operating Income − Interest − Taxes)",    net_income)

print("\n[ BALANCE SHEET ]")
total_debt         = balance_sheet.loc["Total Debt"]
stockholders_equity = balance_sheet.loc["Stockholders Equity"]
current_assets     = balance_sheet.loc["Current Assets"]
current_liabilities = balance_sheet.loc["Current Liabilities"]

print_line_item("Total Debt          (= Short Term Debt + Long Term Debt)",                    total_debt)
print_line_item("Stockholders Equity (= Total Assets − Total Liabilities)",                   stockholders_equity)
print_line_item("Current Assets      (source: balance sheet — due within 12 months)",         current_assets)
print_line_item("Current Liabilities (source: balance sheet — owed within 12 months)",        current_liabilities)

print("\n[ CASH FLOW STATEMENT ]")
operating_cash_flow = cash_flow.loc["Operating Cash Flow"]
free_cash_flow      = cash_flow.loc["Free Cash Flow"]
capex               = cash_flow.loc["Capital Expenditure"]

print_line_item("Operating Cash Flow  (= Net Income + Non-Cash Items + Working Capital Changes)", operating_cash_flow)
print_line_item("Free Cash Flow       (= Operating Cash Flow − Capital Expenditure)",            free_cash_flow)
print_line_item("Capital Expenditure  (= cash spent on physical assets — negative = outflow)",   capex)

years = [col.year for col in income_statement.columns]

print("\n" + "=" * 60)
print("RATIO ANALYSIS")
print("=" * 60)

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

gross_margin     = gross_profit     / total_revenue
net_margin       = net_income       / total_revenue
operating_margin = operating_income / total_revenue

print_ratio_pct("Gross Margin       (= Gross Profit / Revenue)",       gross_margin)
print_ratio_pct("Operating Margin   (= Operating Income / Revenue)",   operating_margin)
print_ratio_pct("Net Margin         (= Net Income / Revenue)",         net_margin)

print()

debt_to_equity = total_debt         / stockholders_equity
current_ratio  = current_assets     / current_liabilities

print_ratio_num("Debt to Equity     (= Total Debt / Stockholders Equity)", debt_to_equity)
print_ratio_num("Current Ratio      (= Current Assets / Current Liabilities)", current_ratio)

print()

fcf_conversion = free_cash_flow / net_income

print_ratio_num("FCF Conversion     (= Free Cash Flow / Net Income)",   fcf_conversion)

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
