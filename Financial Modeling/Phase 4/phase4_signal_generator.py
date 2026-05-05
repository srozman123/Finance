import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TICKER = "MSFT"

# ============================================================================
# SECTION 1 — Download data and calculate indicators
# ============================================================================

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
# SECTION 2 — Signal generation
# ============================================================================
# Entry logic: pure RSI recovery + price recovery + volume confirmation.
# Requiring 2 consecutive days of rising RSI (not just one) filters single-day
# noise — a genuine recovery tends to show momentum over multiple sessions.
# Requiring close > prior close confirms price is following RSI upward.
# Removing MA proximity conditions broadens the signal to catch recoveries at
# any price level, including deep drawdowns where MAs are far above price.
#
# Exit logic:
#   A — RSI overbought for 3+ consecutive days: sustained momentum exhaustion
#   B — 8% hard stop loss from most recent entry: limits drawdown if the
#       recovery thesis fails. This is dollar-based risk management, not a
#       trend signal — it fires regardless of where RSI is.

# --- Entry: all conditions required ---
rsi_was_oversold = df["RSI"].rolling(10).min().shift(1) <= 30
rsi_rising_2d    = (df["RSI"] > df["RSI"].shift(1)) & (df["RSI"].shift(1) > df["RSI"].shift(2))
price_rising     = df["Close"] > df["Close"].shift(1)
volume_above_avg = df["Volume"] > df["Vol20"]

# --- Trend filter (all three conditions required) ---
# Prevents mean-reversion entries during sustained downtrends where oversold
# RSI readings reflect genuine fundamental deterioration rather than temporary
# panic selling. A death cross (50MA below 200MA) signals a structural regime
# change — RSI bounces in this environment are frequently dead-cat recoveries.
# The 30-day recency check adds a cooldown: even if the 50MA just crossed back
# above the 200MA, that crossover is too fresh to trust as a confirmed uptrend.
# The 8% price-to-200MA cap stops entries in freefall — if the stock is already
# well below its long-term average, the downtrend is too established for a
# mean-reversion strategy to work reliably.

# Condition 1: 50MA has been above 200MA continuously for at least 30 days.
# We check that the minimum of (50MA - 200MA) over the past 30 days is positive.
ma50_above_ma200_daily = (df["MA50"] > df["MA200"]).astype(int)
no_recent_death_cross  = ma50_above_ma200_daily.rolling(30).min() == 1

# Condition 2: Closing price is above 200MA or within 8% below it.
pct_below_ma200    = (df["MA200"] - df["Close"]) / df["MA200"]
price_near_ma200   = pct_below_ma200 <= 0.08

# Condition 3: 50MA is strictly above 200MA today (no tolerance).
ma50_above_ma200   = df["MA50"] > df["MA200"]

trend_filter = no_recent_death_cross & price_near_ma200 & ma50_above_ma200

# Compute pre-filter entry (for blocked/passed breakdown)
pre_filter_entry = rsi_was_oversold & rsi_rising_2d & price_rising & volume_above_avg

df["entry_signal"] = pre_filter_entry & trend_filter

# --- Exit A: RSI ≥ 70 for 3+ consecutive days ---
rsi_above70      = (df["RSI"] >= 70).astype(int)
exit_overbought  = rsi_above70.rolling(3).sum() >= 3

# --- Exit B: 8% hard stop loss from most recent entry price ---
# Walk forward through the data: at each row, find the last entry signal
# that fired before or on that date. If close drops 8% below that price, exit.
reference_price = df["Close"].iloc[0]   # fallback if no entry has fired yet
stop_loss_flags = []

last_entry_price = reference_price
for date, row in df.iterrows():
    if row["entry_signal"]:
        last_entry_price = row["Close"]
    triggered = row["Close"] < last_entry_price * 0.92
    stop_loss_flags.append(triggered)

df["exit_stop"] = stop_loss_flags
df["exit_signal"] = exit_overbought | df["exit_stop"]

# ============================================================================
# SECTION 3 — Print signal tables
# ============================================================================

entries      = df[df["entry_signal"]]
exits_ob     = df[exit_overbought]
exits_stop   = df[df["exit_stop"]]
exits_all    = df[df["exit_signal"]]

print("=" * 72)
print(f"  SIGNAL GENERATOR — {TICKER}  (2-Year Window)")
print("=" * 72)

print(f"\n  [ ENTRY SIGNALS  ({len(entries)} fired) ]")
print(f"  {'Date':<13} {'Close':>8} {'RSI':>7} {'RSI -2d':>8} {'RSI Δ2d':>8} {'Vol Ratio':>10}")
print("  " + "-" * 58)
for date, row in entries.iterrows():
    rsi_2d_ago = df["RSI"].shift(2).loc[date]
    rsi_delta  = row["RSI"] - rsi_2d_ago
    vol_ratio  = row["Volume"] / row["Vol20"]
    print(f"  {date.strftime('%Y-%m-%d'):<13} "
          f"${row['Close']:>7.2f} "
          f"{row['RSI']:>7.1f} "
          f"{rsi_2d_ago:>8.1f} "
          f"{rsi_delta:>+7.1f} "
          f"{vol_ratio:>9.2f}x")

print(f"\n  [ EXIT SIGNALS — Condition A: RSI ≥ 70 for 3+ days  ({len(exits_ob)} fired) ]")
print(f"  {'Date':<13} {'Close':>8} {'RSI':>7}")
print("  " + "-" * 32)
for date, row in exits_ob.iterrows():
    print(f"  {date.strftime('%Y-%m-%d'):<13} ${row['Close']:>7.2f} {row['RSI']:>7.1f}")

print(f"\n  [ EXIT SIGNALS — Condition B: 8% stop loss  ({len(exits_stop)} fired) ]")
print(f"  {'Date':<13} {'Close':>8} {'RSI':>7}  {'Stop Ref':>10}  {'% Drop':>8}")
print("  " + "-" * 52)
last_entry_price = reference_price
for date, row in df.iterrows():
    if row["entry_signal"]:
        last_entry_price = row["Close"]
    if row["exit_stop"]:
        pct_drop = (row["Close"] - last_entry_price) / last_entry_price * 100
        print(f"  {date.strftime('%Y-%m-%d'):<13} "
              f"${row['Close']:>7.2f} "
              f"{row['RSI']:>7.1f}  "
              f"${last_entry_price:>9.2f}  "
              f"{pct_drop:>+7.1f}%")

# ============================================================================
# SECTION 4 — Chart
# ============================================================================

plt.style.use("seaborn-v0_8-whitegrid")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9),
                                gridspec_kw={"height_ratios": [3, 1]},
                                sharex=True)

# --- Price subplot ---
ax1.plot(df.index, df["Close"], color="lightgray",    linewidth=0.9, label="Close",      zorder=1)
ax1.plot(df.index, df["MA20"],  color="seagreen",     linewidth=1.2, label="20-day MA",  zorder=2)
ax1.plot(df.index, df["MA50"],  color="steelblue",    linewidth=1.5, label="50-day MA",  zorder=2)
ax1.plot(df.index, df["MA200"], color="mediumpurple", linewidth=1.5,
         linestyle="--", label="200-day MA", zorder=2)

if not entries.empty:
    ax1.scatter(entries.index, entries["Close"], marker="^", color="limegreen",
                s=160, zorder=5, label=f"Entry ({len(entries)})")
if not exits_ob.empty:
    ax1.scatter(exits_ob.index, exits_ob["Close"], marker="v", color="crimson",
                s=130, zorder=5, label=f"Exit: overbought ({len(exits_ob)})")
if not exits_stop.empty:
    ax1.scatter(exits_stop.index, exits_stop["Close"], marker="x", color="darkorange",
                s=100, linewidths=1.5, zorder=5, label=f"Exit: stop loss ({len(exits_stop)})")

ax1.set_title(f"{TICKER} — Price, Moving Averages & Trade Signals", fontsize=12, fontweight="bold")
ax1.set_ylabel("Price (USD)", fontsize=9)
ax1.legend(fontsize=8)
ax1.grid(True, linestyle="--", alpha=0.4)

# --- RSI subplot ---
ax2.plot(df.index, df["RSI"], color="darkorange", linewidth=1.2, label="RSI (14)", zorder=2)
ax2.axhline(70, color="crimson",  linewidth=1, linestyle="--", label="Overbought (70)")
ax2.axhline(30, color="seagreen", linewidth=1, linestyle="--", label="Oversold (30)")
ax2.fill_between(df.index, 70, df["RSI"].clip(lower=70), color="crimson",  alpha=0.15)
ax2.fill_between(df.index, df["RSI"].clip(upper=30), 30,  color="seagreen", alpha=0.15)

for date in entries.index:
    ax2.axvspan(date, date + pd.Timedelta(days=1), color="limegreen",  alpha=0.30, zorder=1)
for date in exits_ob.index:
    ax2.axvspan(date, date + pd.Timedelta(days=1), color="crimson",    alpha=0.15, zorder=1)
for date in exits_stop.index:
    ax2.axvspan(date, date + pd.Timedelta(days=1), color="darkorange", alpha=0.20, zorder=1)

ax2.set_ylabel("RSI", fontsize=9)
ax2.set_xlabel("Date", fontsize=9)
ax2.set_ylim(0, 100)
ax2.set_yticks([30, 50, 70])
ax2.legend(fontsize=8, loc="upper left")
ax2.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("Phase 4/msft_signals.png", dpi=150, bbox_inches="tight")

# ============================================================================
# SECTION 5 — Summary
# ============================================================================

current_entry     = df["entry_signal"].iloc[-1]
current_exit_ob   = exit_overbought.iloc[-1]
current_exit_stop = df["exit_stop"].iloc[-1]

if current_entry:
    status = "ENTRY SIGNAL ACTIVE — all entry conditions met today"
elif current_exit_ob:
    status = "EXIT ACTIVE — RSI overbought for 3+ consecutive days"
elif current_exit_stop:
    status = "EXIT ACTIVE — 8% stop loss triggered"
else:
    status = "NEUTRAL — no signal active today"

n_pre_filter  = int(pre_filter_entry.sum())
n_passed      = len(entries)
n_blocked     = n_pre_filter - n_passed

# Per-condition breakdown of what blocked filtered-out signals
blocked_mask          = pre_filter_entry & ~trend_filter
n_blocked_dc          = int((blocked_mask & ~no_recent_death_cross).sum())
n_blocked_price       = int((blocked_mask & no_recent_death_cross & ~price_near_ma200).sum())
n_blocked_ma50        = int((blocked_mask & no_recent_death_cross & price_near_ma200 & ~ma50_above_ma200).sum())

print(f"\n{'='*72}")
print(f"  SUMMARY")
print(f"{'='*72}")
print(f"  Pre-filter entries (base conditions only): {n_pre_filter}")
print(f"  Blocked by trend filter:                   {n_blocked}")
print(f"    → Blocked by Cond 1 (recent death cross): {n_blocked_dc}")
print(f"    → Blocked by Cond 2 (price >8% below 200MA): {n_blocked_price}")
print(f"    → Blocked by Cond 3 (50MA not above 200MA): {n_blocked_ma50}")
print(f"  Passed through (final entry signals):      {n_passed}")
print(f"  Total exit signals (overbought A):         {len(exits_ob)}")
print(f"  Total exit signals (stop loss B):          {len(exits_stop)}")
print(f"  Current signal status:              {status}")
print(f"\n  Chart saved → Phase 4/msft_signals.png")
