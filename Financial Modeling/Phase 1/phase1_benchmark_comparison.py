import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Download 1 year of daily data for AAPL and SPY.
# SPY tracks the S&P 500 and serves as a market benchmark — comparing a stock
# against it tells you whether the stock is outperforming or underperforming
# the broader market.
tickers = ["AAPL", "SPY"]
raw = yf.download(tickers, period="1y", interval="1d", auto_adjust=True)

# Isolate the Close prices for both tickers and drop any rows where
# either ticker has missing data to keep the comparison aligned.
close = raw["Close"][tickers].dropna()

# Calculate daily return and cumulative return for each ticker.
# We use the same approach as session01: pct_change() for daily returns and
# (1 + r).cumprod() - 1 for cumulative returns, which correctly accounts
# for the compounding effect of gains and losses over time.
returns = close.pct_change()
cum_returns = (1 + returns).cumprod() - 1

# Print total return for each ticker side by side.
total_aapl = cum_returns["AAPL"].iloc[-1]
total_spy  = cum_returns["SPY"].iloc[-1]
print(f"Total return over 1 year:")
print(f"  AAPL: {total_aapl * 100:.2f}%")
print(f"  SPY:  {total_spy  * 100:.2f}%")

# Plot both cumulative return series on the same axes.
# Overlaying them on a single chart makes outperformance/underperformance
# immediately visible — the gap between the lines is the alpha (excess return).
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(cum_returns.index, cum_returns["AAPL"], color="steelblue",  linewidth=1.5, label="AAPL")
ax.plot(cum_returns.index, cum_returns["SPY"],  color="darkorange", linewidth=1.5, label="SPY")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

ax.set_title("AAPL vs SPY — Cumulative Return (1 Year)", fontsize=14)
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative Return")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("aapl_vs_spy.png", dpi=150)
print("Chart saved as aapl_vs_spy.png")
