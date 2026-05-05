import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TICKER = "MSFT"

# ============================================================================
# SECTION 1 — Download price data and calculate moving averages
# ============================================================================
# Moving averages smooth out day-to-day noise and reveal the underlying trend.
# The 20MA tracks short-term momentum, the 50MA medium-term trend, and the
# 200MA long-term trend direction. When shorter MAs cross longer MAs, it
# signals a potential shift in market sentiment — the basis of crossover strategies.

raw   = yf.download(TICKER, period="5y", interval="1d", auto_adjust=True, progress=False)
close = raw["Close"].squeeze()

df = pd.DataFrame({"Close": close})
df["MA20"]  = df["Close"].rolling(window=20).mean()
df["MA50"]  = df["Close"].rolling(window=50).mean()
df["MA200"] = df["Close"].rolling(window=200).mean()
df = df.dropna()

# ============================================================================
# SECTION 2 — Identify golden cross and death cross events
# ============================================================================
# A golden cross (50MA crosses above 200MA) is a widely-watched bullish signal —
# it suggests short/medium-term momentum is turning upward relative to the
# long-term trend. The death cross (50MA crosses below 200MA) is the bearish
# counterpart. These are lagging signals by nature, but institutional traders
# watch them closely, which can make them self-fulfilling.

# For each day, check whether the relationship between 50MA and 200MA flipped
# relative to the prior day. prev_above: True if 50MA was above 200MA yesterday.
prev_above = df["MA50"].shift(1) > df["MA200"].shift(1)
curr_above = df["MA50"] > df["MA200"]

golden_cross = df[(~prev_above) & curr_above]
death_cross  = df[( prev_above) & (~curr_above)]

print(f"{'='*55}")
print(f"  MOVING AVERAGE CROSSOVERS — {TICKER} (2-Year Window)")
print(f"{'='*55}")

print(f"\n  [ Golden Cross Events  (50MA crosses ABOVE 200MA) ]")
if golden_cross.empty:
    print("    None found in this period.")
else:
    for date, row in golden_cross.iterrows():
        print(f"    {date.strftime('%Y-%m-%d')}  |  Close: ${row['Close']:.2f}"
              f"  |  50MA: ${row['MA50']:.2f}  |  200MA: ${row['MA200']:.2f}")

print(f"\n  [ Death Cross Events  (50MA crosses BELOW 200MA) ]")
if death_cross.empty:
    print("    None found in this period.")
else:
    for date, row in death_cross.iterrows():
        print(f"    {date.strftime('%Y-%m-%d')}  |  Close: ${row['Close']:.2f}"
              f"  |  50MA: ${row['MA50']:.2f}  |  200MA: ${row['MA200']:.2f}")

# ============================================================================
# SECTION 3 — RSI (14-day, Wilder method)
# ============================================================================
# RSI (Relative Strength Index) measures the speed and magnitude of recent
# price changes to evaluate whether a stock is overbought or oversold.
# Wilder's method uses an exponential moving average (EMA) with α = 1/14,
# which gives more weight to recent data and smooths out short-term noise.
# RSI above 70 suggests the stock may be overextended to the upside;
# below 30 suggests it may be oversold and due for a bounce.
# Like all momentum indicators, RSI is most useful as a confirmation signal
# rather than a standalone buy/sell trigger.

delta  = df["Close"].diff()
gains  = delta.clip(lower=0)
losses = (-delta).clip(lower=0)

# Wilder EMA: span = 2*period - 1 maps to Wilder's α = 1/period
alpha       = 1 / 14
avg_gain    = gains.ewm(alpha=alpha, adjust=False).mean()
avg_loss    = losses.ewm(alpha=alpha, adjust=False).mean()
rs          = avg_gain / avg_loss
df["RSI"]   = 100 - (100 / (1 + rs))

current_rsi = df["RSI"].iloc[-1]
if current_rsi >= 70:
    interpretation = "OVERBOUGHT — momentum is extended to the upside; watch for a potential pullback"
elif current_rsi <= 30:
    interpretation = "OVERSOLD — selling may be exhausted; watch for a potential bounce"
else:
    interpretation = "NEUTRAL — no extreme momentum signal in either direction"

print(f"\n  Current RSI (14-day): {current_rsi:.1f}  →  {interpretation}")

# ============================================================================
# SECTION 4 — Chart
# ============================================================================

plt.style.use("seaborn-v0_8-whitegrid")
fig, (ax, ax_rsi) = plt.subplots(2, 1, figsize=(14, 9),
                                  gridspec_kw={"height_ratios": [3, 1]},
                                  sharex=True)

ax.plot(df.index, df["Close"], color="lightgray",  linewidth=0.9, label="Close price",  zorder=1)
ax.plot(df.index, df["MA20"],  color="seagreen",   linewidth=1.2, label="20-day MA",    zorder=2)
ax.plot(df.index, df["MA50"],  color="steelblue",  linewidth=1.5, label="50-day MA",    zorder=2)
ax.plot(df.index, df["MA200"], color="mediumpurple", linewidth=1.5,
        linestyle="--", label="200-day MA", zorder=2)

# Golden cross markers
if not golden_cross.empty:
    ax.scatter(golden_cross.index, golden_cross["Close"],
               marker="^", color="limegreen", s=120, zorder=5,
               label="Golden cross (50MA > 200MA)")

# Death cross markers
if not death_cross.empty:
    ax.scatter(death_cross.index, death_cross["Close"],
               marker="v", color="crimson", s=120, zorder=5,
               label="Death cross (50MA < 200MA)")

ax.set_title(f"{TICKER} — Closing Price with Moving Averages & Crossover Signals (5 Years)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Date", fontsize=9)
ax.set_ylabel("Price (USD)", fontsize=9)
ax.legend(fontsize=8)
ax.grid(True, linestyle="--", alpha=0.4)

# --- RSI subplot ---
ax_rsi.plot(df.index, df["RSI"], color="darkorange", linewidth=1.2, label="RSI (14)")
ax_rsi.axhline(70, color="crimson",   linewidth=1,   linestyle="--", label="Overbought (70)")
ax_rsi.axhline(30, color="seagreen",  linewidth=1,   linestyle="--", label="Oversold (30)")
ax_rsi.fill_between(df.index, 70, df["RSI"].clip(lower=70),
                    color="crimson",  alpha=0.15)
ax_rsi.fill_between(df.index, df["RSI"].clip(upper=30), 30,
                    color="seagreen", alpha=0.15)

ax_rsi.set_ylabel("RSI", fontsize=9)
ax_rsi.set_xlabel("Date", fontsize=9)
ax_rsi.set_ylim(0, 100)
ax_rsi.set_yticks([30, 50, 70])
ax_rsi.legend(fontsize=8, loc="upper left")
ax_rsi.grid(True, linestyle="--", alpha=0.4)
ax_rsi.set_title(f"RSI (14-day Wilder)  —  Current: {current_rsi:.1f}",
                 fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig("Phase 4/moving_averages.png", dpi=150, bbox_inches="tight")
print(f"  Chart saved → Phase 4/msft_moving_averages.png")
