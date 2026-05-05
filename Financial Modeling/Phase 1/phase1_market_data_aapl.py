import yfinance as yf # a library that pulls stock data from Yahoo Finance.
import pandas as pd #table structure that holds data in rows and columns, like a spreadsheet. It has powerful tools for data manipulation and analysis.
import matplotlib #used for charting and visualizing data. We set the backend to "Agg" to allow saving charts without a display (useful for scripts).
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Download 1 year of daily price data for Apple
ticker = "AAPL"
df = yf.download(ticker, period="1y", interval="1d", auto_adjust=True) #downloads 1 year of daily price data for Apple, adjusting for splits/dividends.

# Print shape and first 10 rows
print(f"DataFrame shape: {df.shape}")
print(f"\nFirst 10 rows of {ticker} daily price data:")
print(df.head(10))

# --- Section 1: Isolate the Close price ---
# The closing price is the most commonly used price in finance because it reflects
# the final consensus value of a stock for the day. Isolating it makes it easy to
# perform calculations without dragging along unneeded columns.
close = df["Close"]
print(f"\nType of 'close': {type(close)}")
print("\nFirst 5 closing prices:")
print(close.head(5))

# --- Section 2: Daily Return ---
# Daily return measures how much a stock gained or lost each day as a percentage.
# It's a core building block in finance — used to measure performance, volatility,
# and risk. We use percentage change rather than raw price change so returns are
# comparable across different stocks and time periods.
df["daily_return"] = (df["Close"] - df["Close"].shift(1)) / df["Close"].shift(1)
print("\nFirst 10 rows (Close and daily_return):")
print(df[["Close", "daily_return"]].head(10))

# --- Section 3: Summary Statistics for Daily Return ---
# These stats give a quick risk profile of the stock. The mean tells you the average
# daily gain/loss, std (standard deviation) measures volatility, and min/max show
# the worst and best single-day moves over the period.
print("\nDaily return summary statistics:")
print(df["daily_return"].describe()[["mean", "std", "min", "max"]])

# --- Section 4: Cumulative Return ---
# Cumulative return shows how much a $1 investment would have grown since day one.
# We multiply (1 + r1) * (1 + r2) * ... rather than adding returns because returns
# compound — each day's gain or loss is applied to a changing base, not the original.
# Adding returns would ignore this compounding effect and give the wrong total.
df["cum_return"] = (1 + df["daily_return"]).cumprod() - 1
print("\nLast 5 rows (Close and cum_return):")
print(df[["Close", "cum_return"]].tail(5))

total_return = df["cum_return"].iloc[-1]
print(f"\nTotal return over the period: {total_return * 100:.2f}%")

# --- Section 5: Visualization ---
# Plotting price and daily returns side by side gives a quick visual risk/reward
# summary. The price chart shows trend direction; the return bars highlight
# volatility clusters — days of outsized moves that matter for risk assessment.
close_series = df["Close"].squeeze()
return_series = df["daily_return"].squeeze()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

ax1.plot(close_series.index, close_series.values, color="steelblue", linewidth=1.5)
ax1.set_title("AAPL — Close Price & Daily Returns (1 Year)", fontsize=14)
ax1.set_ylabel("Close Price (USD)")
ax1.grid(True, linestyle="--", alpha=0.5)

colors = ["green" if v >= 0 else "red" for v in return_series.values]
ax2.bar(return_series.index, return_series.values, color=colors, width=0.8)
ax2.axhline(0, color="black", linewidth=0.8)
ax2.set_ylabel("Daily Return")
ax2.set_xlabel("Date")
ax2.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("aapl_overview.png", dpi=150)
print("\nChart saved as aapl_overview.png")
