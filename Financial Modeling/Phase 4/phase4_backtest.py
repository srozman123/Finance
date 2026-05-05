import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TICKER         = "MSFT"
STARTING_CAP   = 10_000.0
RISK_FREE_RATE = 0.042   # 4.2% annualised

# ============================================================================
# SECTION 1 — Download data and calculate indicators
# ============================================================================
# Replicates the indicator stack from phase4_signal_generator.py exactly so
# the signals produced here are identical to those used in signal analysis.

raw    = yf.download(TICKER, period="2y", interval="1d", auto_adjust=True, progress=False)
close  = raw["Close"].squeeze()
volume = raw["Volume"].squeeze()

df = pd.DataFrame({"Close": close, "Volume": volume})

df["MA20"]  = df["Close"].rolling(20).mean()
df["MA50"]  = df["Close"].rolling(50).mean()
df["MA200"] = df["Close"].rolling(200).mean()

delta     = df["Close"].diff()
gains     = delta.clip(lower=0)
losses    = (-delta).clip(lower=0)
alpha     = 1 / 14
df["RSI"] = 100 - (100 / (1 + gains.ewm(alpha=alpha, adjust=False).mean()
                               / losses.ewm(alpha=alpha, adjust=False).mean()))
df["Vol20"] = df["Volume"].rolling(20).mean()
df = df.dropna()

# ============================================================================
# SECTION 2 — Replicate signal logic (with strict trend filter)
# ============================================================================
# Entry conditions (all required):
#   - RSI was ≤ 30 at any point in last 10 days
#   - RSI has risen for 2 consecutive days
#   - Close is higher than prior close (price confirming recovery)
#   - Volume above 20-day average
# Trend filter (all required — blocks entries in downtrends):
#   - 50MA has been above 200MA for every one of the last 30 days
#   - Close is within 8% below 200MA
#   - 50MA is strictly above 200MA today

rsi_was_oversold       = df["RSI"].rolling(10).min().shift(1) <= 30
rsi_rising_2d          = (df["RSI"] > df["RSI"].shift(1)) & (df["RSI"].shift(1) > df["RSI"].shift(2))
price_rising           = df["Close"] > df["Close"].shift(1)
volume_above_avg       = df["Volume"] > df["Vol20"]

ma50_above_ma200_daily = (df["MA50"] > df["MA200"]).astype(int)
no_recent_death_cross  = ma50_above_ma200_daily.rolling(30).min() == 1
price_near_ma200       = (df["MA200"] - df["Close"]) / df["MA200"] <= 0.08
ma50_above_ma200       = df["MA50"] > df["MA200"]
trend_filter           = no_recent_death_cross & price_near_ma200 & ma50_above_ma200

pre_filter_entry       = rsi_was_oversold & rsi_rising_2d & price_rising & volume_above_avg
df["entry_signal"]     = pre_filter_entry & trend_filter

rsi_above70            = (df["RSI"] >= 70).astype(int)
exit_overbought        = rsi_above70.rolling(3).sum() >= 3
df["exit_overbought"]  = exit_overbought

# Stop loss is computed dynamically inside the backtest loop (entry-price dependent)

# ============================================================================
# SECTION 3 — Backtest simulation
# ============================================================================
# We simulate trading sequentially day by day, using closing prices for both
# entry and exit (next-bar execution would be more realistic but requires OHLC
# intraday data). Starting capital is $10,000. We trade whole shares only —
# no fractional shares — to reflect realistic brokerage constraints.
# The stop loss is recalculated on each day relative to the actual entry price,
# matching the logic in phase4_signal_generator.py exactly.

cash          = STARTING_CAP
shares_held   = 0
entry_price   = None
entry_date    = None
trades        = []
portfolio_val = []   # daily portfolio value for drawdown / Sharpe calculation

for date, row in df.iterrows():
    price = row["Close"]

    # Mark portfolio value before any trade today
    port_val = cash + shares_held * price
    portfolio_val.append({"Date": date, "Value": port_val})

    # --- Entry ---
    if shares_held == 0 and row["entry_signal"]:
        shares_held = int(cash // price)
        if shares_held > 0:
            cost        = shares_held * price
            cash       -= cost
            entry_price = price
            entry_date  = date

    # --- Exit: overbought or stop loss ---
    elif shares_held > 0:
        stop_triggered     = price < entry_price * 0.92
        ob_triggered       = row["exit_overbought"]

        if stop_triggered or ob_triggered:
            proceeds   = shares_held * price
            cash      += proceeds
            ret_pct    = (price - entry_price) / entry_price * 100
            exit_type  = "Stop loss" if stop_triggered else "Overbought"
            trades.append({
                "Entry date":  entry_date,
                "Entry price": entry_price,
                "Exit date":   date,
                "Exit price":  price,
                "Shares":      shares_held,
                "Return %":    ret_pct,
                "Exit type":   exit_type,
            })
            shares_held = 0
            entry_price = None
            entry_date  = None

pv_df = pd.DataFrame(portfolio_val).set_index("Date")

# Mark open position to last price
last_price = df["Close"].iloc[-1]
final_value = cash + shares_held * last_price
if shares_held > 0:
    ret_pct    = (last_price - entry_price) / entry_price * 100
    trades.append({
        "Entry date":  entry_date,
        "Entry price": entry_price,
        "Exit date":   df.index[-1],
        "Exit price":  last_price,
        "Shares":      shares_held,
        "Return %":    ret_pct,
        "Exit type":   "Open (marked to market)",
    })

# ============================================================================
# SECTION 4 — Performance metrics
# ============================================================================
# These metrics are the standard toolkit for evaluating a strategy objectively.
# Win rate and average return tell you the quality of individual trades.
# Max drawdown measures risk — how much you could have lost from peak to trough.
# Sharpe ratio normalises return by volatility, making it comparable across
# strategies regardless of how aggressively they trade.

trades_df = pd.DataFrame(trades)

n_trades   = len(trades_df)
returns    = trades_df["Return %"].values if n_trades > 0 else [0]
win_rate   = (trades_df["Return %"] > 0).mean() * 100 if n_trades > 0 else 0
avg_ret    = trades_df["Return %"].mean()        if n_trades > 0 else 0
best_trade = trades_df["Return %"].max()         if n_trades > 0 else 0
worst_trade= trades_df["Return %"].min()         if n_trades > 0 else 0
total_ret  = (final_value - STARTING_CAP) / STARTING_CAP * 100

# Maximum drawdown
rolling_peak = pv_df["Value"].cummax()
drawdown     = (pv_df["Value"] - rolling_peak) / rolling_peak * 100
max_drawdown = drawdown.min()

# Sharpe ratio (daily returns → annualised)
daily_rets     = pv_df["Value"].pct_change().dropna()
ann_return     = (final_value / STARTING_CAP) ** (252 / len(pv_df)) - 1
ann_vol        = daily_rets.std() * np.sqrt(252)
sharpe         = (ann_return - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0

print("=" * 62)
print(f"  BACKTEST — {TICKER}  |  Starting Capital: ${STARTING_CAP:,.0f}")
print("=" * 62)

print(f"\n  [ TRADE LOG ]")
print(f"  {'Entry Date':<13} {'Entry $':>8} {'Exit Date':<13} {'Exit $':>8} {'Return':>8}  Exit Type")
print("  " + "-" * 70)
for _, t in trades_df.iterrows():
    print(f"  {t['Entry date'].strftime('%Y-%m-%d'):<13} "
          f"${t['Entry price']:>7.2f} "
          f"{t['Exit date'].strftime('%Y-%m-%d'):<13} "
          f"${t['Exit price']:>7.2f} "
          f"{t['Return %']:>+7.1f}%  "
          f"{t['Exit type']}")

print(f"\n  [ PERFORMANCE METRICS ]")
print(f"  {'Starting Capital':<40} ${STARTING_CAP:>10,.2f}")
print(f"  {'Final Portfolio Value':<40} ${final_value:>10,.2f}")
print(f"  {'Total Strategy Return':<40} {total_ret:>+10.2f}%")
print(f"  {'Number of Trades':<40} {n_trades:>10}")
print(f"  {'Win Rate':<40} {win_rate:>10.1f}%")
print(f"  {'Average Return per Trade':<40} {avg_ret:>+10.2f}%")
print(f"  {'Best Single Trade':<40} {best_trade:>+10.2f}%")
print(f"  {'Worst Single Trade':<40} {worst_trade:>+10.2f}%")
print(f"  {'Maximum Drawdown':<40} {max_drawdown:>+10.2f}%")
print(f"  {'Annualised Volatility':<40} {ann_vol*100:>10.2f}%")
print(f"  {'Sharpe Ratio':<40} {sharpe:>10.2f}")

# ============================================================================
# SECTION 5 — Buy and hold benchmark
# ============================================================================
# The benchmark answers the simplest possible question: would you have been
# better off just buying on day one and doing nothing? Any active strategy
# must justify its complexity by beating this baseline — otherwise the effort
# adds no value and introduces unnecessary transaction costs and tax events.

bh_shares     = int(STARTING_CAP // df["Close"].iloc[0])
bh_cost       = bh_shares * df["Close"].iloc[0]
bh_cash_left  = STARTING_CAP - bh_cost
bh_final      = bh_shares * last_price + bh_cash_left
bh_return     = (bh_final - STARTING_CAP) / STARTING_CAP * 100

bh_port       = bh_shares * df["Close"] + bh_cash_left
bh_peak       = bh_port.cummax()
bh_drawdown   = (bh_port - bh_peak) / bh_peak * 100

print(f"\n  [ BUY AND HOLD BENCHMARK ]")
print(f"  {'Shares bought on day 1':<40} {bh_shares:>10}")
print(f"  {'Buy price':<40} ${df['Close'].iloc[0]:>9.2f}")
print(f"  {'Final Value':<40} ${bh_final:>10,.2f}")
print(f"  {'Buy & Hold Return':<40} {bh_return:>+10.2f}%")
print(f"  {'B&H Max Drawdown':<40} {bh_drawdown.min():>+10.2f}%")

# ============================================================================
# SECTION 6 — Strategy vs benchmark comparison
# ============================================================================

outperformed = total_ret > bh_return
alpha        = total_ret - bh_return

print(f"\n  [ STRATEGY vs BUY & HOLD ]")
print(f"  {'Strategy Return':<35} {total_ret:>+8.2f}%")
print(f"  {'Buy & Hold Return':<35} {bh_return:>+8.2f}%")
print(f"  {'Alpha (Strategy − B&H)':<35} {alpha:>+8.2f}%")
print(f"  {'Result':<35} {'OUTPERFORMED' if outperformed else 'UNDERPERFORMED'} buy & hold")

# ============================================================================
# SECTION 7 — Chart
# ============================================================================

entry_dates = [t["Entry date"] for _, t in trades_df.iterrows()
               if t["Exit type"] != "Open (marked to market)"]
exit_dates  = [t["Exit date"]  for _, t in trades_df.iterrows()
               if t["Exit type"] != "Open (marked to market)"]

plt.style.use("seaborn-v0_8-whitegrid")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                gridspec_kw={"height_ratios": [2, 1]},
                                sharex=True)

# --- Portfolio value ---
ax1.plot(pv_df.index, pv_df["Value"], color="steelblue",  linewidth=1.5,
         label=f"Strategy  (${final_value:,.0f})")
ax1.plot(bh_port.index, bh_port.values, color="lightgray", linewidth=1.2,
         linestyle="--", label=f"Buy & Hold  (${bh_final:,.0f})")
ax1.axhline(STARTING_CAP, color="black", linewidth=0.8, linestyle=":", alpha=0.5)

for d in entry_dates:
    ax1.axvline(d, color="limegreen", linewidth=1.2, alpha=0.7)
for d in exit_dates:
    ax1.axvline(d, color="crimson",   linewidth=1.2, alpha=0.5)

# Dummy patches for legend
from matplotlib.patches import Patch
entry_patch = Patch(color="limegreen", label="Entry")
exit_patch  = Patch(color="crimson",   label="Exit")

ax1.set_title(f"{TICKER} — Backtest: Strategy vs Buy & Hold  (${STARTING_CAP:,.0f} starting capital)",
              fontsize=12, fontweight="bold")
ax1.set_ylabel("Portfolio Value (USD)", fontsize=9)
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles + [entry_patch, exit_patch], labels + ["Entry", "Exit"], fontsize=8)
ax1.grid(True, linestyle="--", alpha=0.4)

# --- Drawdown ---
ax2.fill_between(drawdown.index,    drawdown.values,    0, color="steelblue",
                 alpha=0.4, label=f"Strategy  (max {max_drawdown:.1f}%)")
ax2.fill_between(bh_drawdown.index, bh_drawdown.values, 0, color="lightgray",
                 alpha=0.5, label=f"Buy & Hold  (max {bh_drawdown.min():.1f}%)")
ax2.axhline(0, color="black", linewidth=0.6)

ax2.set_title("Drawdown — Distance from Portfolio Peak", fontsize=10, fontweight="bold")
ax2.set_ylabel("Drawdown (%)", fontsize=9)
ax2.set_xlabel("Date", fontsize=9)
ax2.legend(fontsize=8)
ax2.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("Phase 4/msft_backtest.png", dpi=150, bbox_inches="tight")
print(f"\n  Chart saved → Phase 4/msft_backtest.png")
