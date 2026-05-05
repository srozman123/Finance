# ============================================================================
# phase4_backtest_multi.py — Multi-Stock Backtest
# ============================================================================
# This file runs the identical signal strategy from phase4_backtest.py across
# a list of tickers to determine whether the strategy has genuine edge across
# multiple companies rather than being a one-stock coincidence. A strategy that
# consistently beats buy-and-hold across a diverse set of high-quality names is
# far more credible than one that happens to work on a single ticker. This is
# the first step toward validating the signal's statistical robustness.
# ============================================================================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
# Highest-scoring fundamental stocks from the phase5 screener results.
TICKERS = ['AAPL', 'MSFT', 'GOOG', 'META', 'AVGO', 'AMZN', 'ADBE', 'PAYC', 'TTD', 'SNPS']

STARTING_CAP   = 10_000.0
RISK_FREE_RATE = 0.042   # 4.2% annualised — matches phase4_backtest.py
STOP_LOSS_PCT  = 0.92    # 8% stop loss — matches phase4_backtest.py
DATA_PERIOD    = "2y"


# ============================================================================
# CORE BACKTEST FUNCTION
# Encapsulates the full backtest logic from phase4_backtest.py.
# Returns a dict of per-ticker metrics, or None if data is insufficient.
# ============================================================================

def run_backtest(ticker):
    # ── SECTION 1: Download and indicators ──────────────────────────────────
    raw = yf.download(ticker, period=DATA_PERIOD, interval="1d",
                      auto_adjust=True, progress=False)
    if raw.empty or len(raw) < 210:
        return None

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
    df["RSI"]   = 100 - (100 / (1 + gains.ewm(alpha=alpha, adjust=False).mean()
                                    / losses.ewm(alpha=alpha, adjust=False).mean()))
    df["Vol20"] = df["Volume"].rolling(20).mean()
    df = df.dropna()

    if len(df) < 50:
        return None

    # ── SECTION 2: Signal logic — identical to phase4_backtest.py ───────────
    rsi_was_oversold       = df["RSI"].rolling(10).min().shift(1) <= 30
    rsi_rising_2d          = ((df["RSI"] > df["RSI"].shift(1)) &
                              (df["RSI"].shift(1) > df["RSI"].shift(2)))
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
    df["exit_overbought"]  = rsi_above70.rolling(3).sum() >= 3

    # ── SECTION 3: Trade simulation ─────────────────────────────────────────
    cash          = STARTING_CAP
    shares_held   = 0
    entry_price   = None
    entry_date    = None
    trades        = []
    portfolio_val = []

    for date, row in df.iterrows():
        price    = row["Close"]
        port_val = cash + shares_held * price
        portfolio_val.append({"Date": date, "Value": port_val})

        if shares_held == 0 and row["entry_signal"]:
            shares_held = int(cash // price)
            if shares_held > 0:
                cash       -= shares_held * price
                entry_price = price
                entry_date  = date

        elif shares_held > 0:
            stop_triggered = price < entry_price * STOP_LOSS_PCT
            ob_triggered   = row["exit_overbought"]
            if stop_triggered or ob_triggered:
                cash         += shares_held * price
                ret_pct       = (price - entry_price) / entry_price * 100
                trades.append({
                    "Entry date":  entry_date,
                    "Exit date":   date,
                    "Entry price": entry_price,
                    "Exit price":  price,
                    "Return %":    ret_pct,
                    "Exit type":   "Stop" if stop_triggered else "Overbought",
                })
                shares_held = 0
                entry_price = None
                entry_date  = None

    pv_df      = pd.DataFrame(portfolio_val).set_index("Date")
    last_price = df["Close"].iloc[-1]
    final_val  = cash + shares_held * last_price

    # Mark open position to market
    if shares_held > 0:
        trades.append({
            "Entry date":  entry_date,
            "Exit date":   df.index[-1],
            "Entry price": entry_price,
            "Exit price":  last_price,
            "Return %":    (last_price - entry_price) / entry_price * 100,
            "Exit type":   "Open",
        })

    # ── SECTION 4: Performance metrics ──────────────────────────────────────
    trades_df  = pd.DataFrame(trades)
    n_trades   = len(trades_df)
    win_rate   = float((trades_df["Return %"] > 0).mean() * 100) if n_trades > 0 else 0.0
    total_ret  = (final_val - STARTING_CAP) / STARTING_CAP * 100

    rolling_peak = pv_df["Value"].cummax()
    drawdown     = (pv_df["Value"] - rolling_peak) / rolling_peak * 100
    max_drawdown = float(drawdown.min())

    daily_rets = pv_df["Value"].pct_change().dropna()
    ann_return = (final_val / STARTING_CAP) ** (252 / len(pv_df)) - 1
    ann_vol    = float(daily_rets.std() * np.sqrt(252))
    sharpe     = (ann_return - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0.0

    # ── SECTION 5: Buy and hold benchmark ───────────────────────────────────
    bh_shares   = int(STARTING_CAP // df["Close"].iloc[0])
    bh_cash_rem = STARTING_CAP - bh_shares * df["Close"].iloc[0]
    bh_final    = bh_shares * last_price + bh_cash_rem
    bh_return   = (bh_final - STARTING_CAP) / STARTING_CAP * 100
    bh_port     = bh_shares * df["Close"] + bh_cash_rem
    bh_draw     = (bh_port - bh_port.cummax()) / bh_port.cummax() * 100

    return {
        "ticker":         ticker,
        "strategy_ret":   round(total_ret, 2),
        "bh_ret":         round(bh_return, 2),
        "outperformance": round(total_ret - bh_return, 2),
        "n_trades":       n_trades,
        "win_rate":       round(win_rate, 1),
        "max_drawdown":   round(max_drawdown, 2),
        "bh_max_dd":      round(float(bh_draw.min()), 2),
        "sharpe":         round(sharpe, 2),
    }


# ============================================================================
# MAIN — run all tickers, compile results, print and chart
# ============================================================================

print("=" * 68)
print(f"  MULTI-STOCK BACKTEST  |  {len(TICKERS)} tickers  |  "
      f"${STARTING_CAP:,.0f} starting capital")
print("=" * 68)
print()

results = []
for ticker in TICKERS:
    print(f"  Running {ticker}...", end="", flush=True)
    try:
        r = run_backtest(ticker)
        if r is None:
            print("  skipped (insufficient data)")
        else:
            results.append(r)
            beat = "BEAT" if r["outperformance"] > 0 else "MISS"
            print(f"  {beat}  strategy {r['strategy_ret']:+.1f}%  "
                  f"B&H {r['bh_ret']:+.1f}%  "
                  f"alpha {r['outperformance']:+.1f}%  "
                  f"trades {r['n_trades']}")
    except Exception as exc:
        print(f"  ERROR: {exc}")

if not results:
    print("  No results to display.")
    raise SystemExit

results_df = pd.DataFrame(results).set_index("ticker")
results_df.sort_values("outperformance", ascending=False, inplace=True)

# ── Full results table ───────────────────────────────────────────────────────
print()
print("─" * 100)
print(f"  {'TICKER':<8} {'Strategy':>9} {'B&H':>9} {'Alpha':>8} "
      f"{'Trades':>7} {'Win%':>7} {'MaxDD':>8} {'B&H DD':>8} {'Sharpe':>8}")
print("─" * 100)

for ticker, row in results_df.iterrows():
    beat_marker = "▲" if row["outperformance"] > 0 else "▼"
    print(f"  {ticker:<8} "
          f"{row['strategy_ret']:>+8.1f}% "
          f"{row['bh_ret']:>+8.1f}% "
          f"{row['outperformance']:>+7.1f}% "
          f"{int(row['n_trades']):>7} "
          f"{row['win_rate']:>6.0f}% "
          f"{row['max_drawdown']:>+7.1f}% "
          f"{row['bh_max_dd']:>+7.1f}% "
          f"{row['sharpe']:>8.2f}  {beat_marker}")

print("─" * 100)

# ── Aggregate statistics ─────────────────────────────────────────────────────
n           = len(results_df)
beat_count  = (results_df["outperformance"] > 0).sum()
beat_pct    = beat_count / n * 100

print()
print("  AGGREGATE STATISTICS")
print(f"  {'Tickers run':<42} {n}")
print(f"  {'Tickers beat buy & hold':<42} {beat_count} / {n}  ({beat_pct:.0f}%)")
print(f"  {'Average strategy return':<42} {results_df['strategy_ret'].mean():>+.2f}%")
print(f"  {'Average buy & hold return':<42} {results_df['bh_ret'].mean():>+.2f}%")
print(f"  {'Average outperformance (alpha)':<42} {results_df['outperformance'].mean():>+.2f}%")
print(f"  {'Average trades per ticker':<42} {results_df['n_trades'].mean():.1f}")
print(f"  {'Average win rate':<42} {results_df['win_rate'].mean():.1f}%")
print(f"  {'Average max drawdown (strategy)':<42} {results_df['max_drawdown'].mean():>+.2f}%")
print(f"  {'Average max drawdown (buy & hold)':<42} {results_df['bh_max_dd'].mean():>+.2f}%")
print()

# ============================================================================
# CHART — scatter plot: strategy return vs buy and hold return
# ============================================================================

plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(10, 8))

x = results_df["bh_ret"].values
y = results_df["strategy_ret"].values
tickers_plot = results_df.index.tolist()

# Diagonal line: strategy == buy and hold
all_vals = np.concatenate([x, y])
diag_min = all_vals.min() - 5
diag_max = all_vals.max() + 5
ax.plot([diag_min, diag_max], [diag_min, diag_max],
        color="gray", linewidth=1.2, linestyle="--",
        alpha=0.7, label="Strategy = Buy & Hold")

# Shade regions above/below the diagonal
ax.fill_between([diag_min, diag_max], [diag_min, diag_max], diag_max,
                color="#c8e6c9", alpha=0.15, label="Strategy beats B&H")
ax.fill_between([diag_min, diag_max], diag_min, [diag_min, diag_max],
                color="#ffcdd2", alpha=0.15, label="B&H beats strategy")

# Plot each ticker
colors = ["#2e7d32" if v > 0 else "#c62828" for v in results_df["outperformance"].values]
ax.scatter(x, y, c=colors, s=120, zorder=5, edgecolors="white", linewidths=1.2)

# Label each point
for xi, yi, label in zip(x, y, tickers_plot):
    ax.annotate(label, (xi, yi),
                textcoords="offset points", xytext=(8, 4),
                fontsize=9, fontweight="bold",
                color="#2e7d32" if yi > xi else "#c62828")

ax.set_xlabel("Buy & Hold Return (%)", fontsize=11)
ax.set_ylabel("Strategy Return (%)", fontsize=11)
ax.set_title(
    f"Strategy vs Buy & Hold — {n} Tickers\n"
    f"{beat_count}/{n} tickers outperformed  |  "
    f"Avg alpha: {results_df['outperformance'].mean():+.1f}%",
    fontsize=12, fontweight="bold"
)
ax.legend(fontsize=9)
ax.set_xlim(diag_min, diag_max)
ax.set_ylim(diag_min, diag_max)
ax.set_aspect("equal")
ax.grid(True, linestyle="--", alpha=0.4)

out_path = "Phase 4/multi_backtest_results.png"
plt.tight_layout()
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Chart saved → {out_path}")
