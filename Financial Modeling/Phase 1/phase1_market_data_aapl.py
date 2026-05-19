import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ticker = "AAPL"
df = yf.download(ticker, period="1y", interval="1d", auto_adjust=True)

print(f"DataFrame shape: {df.shape}")
print(f"\nFirst 10 rows of {ticker} daily price data:")
print(df.head(10))

close = df["Close"]
print(f"\nType of 'close': {type(close)}")
print("\nFirst 5 closing prices:")
print(close.head(5))

df["daily_return"] = (df["Close"] - df["Close"].shift(1)) / df["Close"].shift(1)
print("\nFirst 10 rows (Close and daily_return):")
print(df[["Close", "daily_return"]].head(10))

print("\nDaily return summary statistics:")
print(df["daily_return"].describe()[["mean", "std", "min", "max"]])

df["cum_return"] = (1 + df["daily_return"]).cumprod() - 1
print("\nLast 5 rows (Close and cum_return):")
print(df[["Close", "cum_return"]].tail(5))

total_return = df["cum_return"].iloc[-1]
print(f"\nTotal return over the period: {total_return * 100:.2f}%")

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
