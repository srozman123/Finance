# ============================================================================
# pershing_square.py — Pershing Square Holdings Buy-and-Hold Backtest
# ============================================================================
# Strategy: pure buy-and-hold of Pershing Square's disclosed equity positions,
#   equal-weighted at the start of each window.  No rebalancing, no stops,
#   no signals — just hold and compare to SPY.
#
# Non-equity line items from Pershing Square attribution reports
#   (Share Buyback Accretion, Bond Interest Expense, All Other Positions
#    and Other Income/Expense) are excluded — they are not investable tickers.
#
# Holdings modelled:
#   GOOGL  — Alphabet Inc.
#   BN     — Brookfield Corporation
#   FNMA   — Federal National Mortgage Association (OTC)
#   FMCC   — Federal Home Loan Mortgage Corporation (OTC)
#   UBER   — Uber Technologies, Inc.
#   AMZN   — Amazon.com, Inc.
#   HLT    — Hilton Worldwide Holdings Inc.
#   UMGNF  — Universal Music Group N.V. (OTC pink sheets; EUR-listed UMG.AS)
#   META   — Meta Platforms, Inc.
#   QSR    — Restaurant Brands International Inc.
#   NKE    — Nike, Inc.
#   CMG    — Chipotle Mexican Grill, Inc.
#
# Two backtest windows (matching other strategies in this folder):
#   Window 1 (2022–2024)  :  2022-04-01 → 2024-04-01
#   Window 2 (2024–2026)  :  2024-04-01 → 2026-04-12
# ============================================================================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, date
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# UNIVERSE
# ──────────────────────────────────────────────────────────────────────────────
HOLDINGS = [
    'GOOGL',   # Alphabet Inc.
    'BN',      # Brookfield Corporation
    'FNMA',    # Federal National Mortgage Association
    'FMCC',    # Federal Home Loan Mortgage Corporation
    'UBER',    # Uber Technologies, Inc.
    'AMZN',    # Amazon.com, Inc.
    'HLT',     # Hilton Worldwide Holdings Inc.
    'UMGNF',   # Universal Music Group N.V. (OTC)
    'META',    # Meta Platforms, Inc.
    'QSR',     # Restaurant Brands International Inc.
    'NKE',     # Nike, Inc.
    'CMG',     # Chipotle Mexican Grill, Inc.
]

# ──────────────────────────────────────────────────────────────────────────────
# PARAMETERS
# ──────────────────────────────────────────────────────────────────────────────
STARTING_CAPITAL   = 10_000

WINDOW1_START      = '2022-04-01'
WINDOW1_END        = '2024-04-01'
WINDOW2_START      = '2024-04-01'
WINDOW2_END        = '2026-04-12'
DATA_DOWNLOAD_START = '2021-01-01'

CHART_OUTPUT_PATH  = 'momentum_backtesting/pershing_square_charts.png'


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def _to_date(d):
    if isinstance(d, (datetime, pd.Timestamp)):
        return d.date()
    if isinstance(d, str):
        return datetime.strptime(d, '%Y-%m-%d').date()
    return d


# ──────────────────────────────────────────────────────────────────────────────
# DATA DOWNLOAD
# ──────────────────────────────────────────────────────────────────────────────
def download_all_data():
    """Download adjusted close prices for all holdings + SPY benchmark."""
    all_tickers = HOLDINGS + ['SPY']
    print(f"Downloading price data for {len(all_tickers)} tickers ...")
    raw = yf.download(
        all_tickers,
        start=DATA_DOWNLOAD_START,
        end=WINDOW2_END,
        auto_adjust=True,
        progress=False
    )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw['Close'].copy()
    else:
        # Single ticker fallback
        prices = raw[['Close']].copy()
        prices.columns = [all_tickers[0]]

    prices.index = pd.to_datetime(prices.index)

    # Report any tickers with no data at all
    missing = [t for t in all_tickers if t not in prices.columns or prices[t].isna().all()]
    if missing:
        print(f"  WARNING — no data found for: {missing}")

    print(f"Data ready — {len(prices)} trading days "
          f"from {prices.index[0].date()} to {prices.index[-1].date()}")
    return prices


# ──────────────────────────────────────────────────────────────────────────────
# BUY-AND-HOLD BACKTEST
# ──────────────────────────────────────────────────────────────────────────────
def run_backtest(prices, start_date, end_date, label=''):
    """
    Equal-weight buy-and-hold of all HOLDINGS over [start_date, end_date].

    Tickers that have no price on the first trading day of the window are
    excluded from that window's portfolio with a printed warning.

    Returns:
        portfolio_series : pd.Series — daily total portfolio value
        position_log     : dict      — per-ticker entry/exit/return details
        active_tickers   : list      — tickers that were actually held
    """
    start = _to_date(start_date)
    end   = _to_date(end_date)

    # Slice to window
    window_prices = prices[
        (prices.index >= pd.Timestamp(start)) &
        (prices.index <= pd.Timestamp(end))
    ].copy()

    if window_prices.empty:
        print(f"  ERROR: no price data in window {start} → {end}")
        return pd.Series(dtype=float), {}, []

    first_day = window_prices.index[0]
    last_day  = window_prices.index[-1]

    # Determine which tickers have a valid price on the first day
    active = []
    skipped = []
    for tkr in HOLDINGS:
        if tkr in window_prices.columns and not pd.isna(window_prices.at[first_day, tkr]):
            active.append(tkr)
        else:
            skipped.append(tkr)

    if skipped:
        print(f"  [{label}] Skipped (no data on {first_day.date()}): {skipped}")

    n = len(active)
    if n == 0:
        print(f"  ERROR: no active tickers for window {label}")
        return pd.Series(dtype=float), {}, []

    allocation_per_ticker = STARTING_CAPITAL / n

    # Compute shares purchased on day-1
    shares = {}
    entry_prices = {}
    for tkr in active:
        px = float(window_prices.at[first_day, tkr])
        shares[tkr] = allocation_per_ticker / px
        entry_prices[tkr] = px

    # Daily portfolio value = sum of (shares × price) for all active tickers
    portfolio_values = {}
    for ts, row in window_prices.iterrows():
        total = 0.0
        for tkr in active:
            if tkr in row.index and not pd.isna(row[tkr]):
                total += shares[tkr] * float(row[tkr])
            else:
                # Use last known price if data is missing mid-window
                last_known = window_prices[tkr].dropna()
                last_known = last_known[last_known.index <= ts]
                if not last_known.empty:
                    total += shares[tkr] * float(last_known.iloc[-1])
        portfolio_values[ts] = total

    portfolio_series = pd.Series(portfolio_values)

    # Per-ticker return summary
    position_log = {}
    for tkr in active:
        col = window_prices[tkr].dropna()
        col = col[col.index <= last_day]
        if col.empty:
            continue
        exit_px  = float(col.iloc[-1])
        entry_px = entry_prices[tkr]
        ret      = exit_px / entry_px - 1
        position_log[tkr] = {
            'entry_date':  first_day.date(),
            'exit_date':   last_day.date(),
            'entry_price': entry_px,
            'exit_price':  exit_px,
            'return':      ret,
            'shares':      shares[tkr],
        }

    # Print summary
    print(f"\n{'='*60}")
    print(f"  BACKTEST: {label}  ({start} → {end})")
    print(f"{'='*60}")
    print(f"  Active holdings ({n}): {active}")
    print(f"  Allocation per ticker: ${allocation_per_ticker:,.2f}")
    print(f"\n  Per-ticker returns:")
    sorted_log = sorted(position_log.items(), key=lambda x: x[1]['return'], reverse=True)
    for tkr, rec in sorted_log:
        print(f"    {tkr:8s}  {rec['entry_price']:8.4f} → {rec['exit_price']:8.4f}  "
              f"ret={rec['return']:+.2%}")

    return portfolio_series, position_log, active


# ──────────────────────────────────────────────────────────────────────────────
# STATISTICS
# ──────────────────────────────────────────────────────────────────────────────
def compute_stats(portfolio_series, position_log, prices, start_date, end_date,
                  label='', active_tickers=None):
    start = _to_date(start_date)
    end   = _to_date(end_date)

    if len(portfolio_series) < 2:
        print(f"  [{label}] Insufficient data for stats.")
        return {}

    total_ret = portfolio_series.iloc[-1] / portfolio_series.iloc[0] - 1

    # Annualized return
    n_years = (end - start).days / 365.25
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else float('nan')

    # Daily returns for Sharpe / max drawdown
    daily_rets = portfolio_series.pct_change().dropna()
    sharpe = (daily_rets.mean() / daily_rets.std() * np.sqrt(252)
              if daily_rets.std() > 0 else float('nan'))

    # Max drawdown
    rolling_max = portfolio_series.cummax()
    drawdowns   = (portfolio_series - rolling_max) / rolling_max
    max_dd      = float(drawdowns.min())

    # SPY return over same window
    spy = prices['SPY'].dropna()
    spy = spy[(spy.index >= pd.Timestamp(start)) & (spy.index <= pd.Timestamp(end))]
    spy_ret = float(spy.iloc[-1]) / float(spy.iloc[0]) - 1 if len(spy) >= 2 else float('nan')
    spy_ann = (1 + spy_ret) ** (1 / n_years) - 1 if n_years > 0 else float('nan')

    print(f"\n{'─'*60}")
    print(f"  RESULTS: {label}  ({start} → {end})")
    print(f"{'─'*60}")
    print(f"  Portfolio total return     : {total_ret:+.2%}")
    print(f"  Portfolio annualised return: {ann_ret:+.2%}")
    print(f"  SPY total return           : {spy_ret:+.2%}")
    print(f"  SPY annualised return      : {spy_ann:+.2%}")
    print(f"  Outperformance vs SPY      : {total_ret - spy_ret:+.2%}")
    print(f"  Sharpe ratio (daily)       : {sharpe:.2f}" if not np.isnan(sharpe) else "  Sharpe ratio               : N/A")
    print(f"  Max drawdown               : {max_dd:.2%}")

    if position_log:
        returns = [v['return'] for v in position_log.values()]
        best    = max(position_log.items(), key=lambda x: x[1]['return'])
        worst   = min(position_log.items(), key=lambda x: x[1]['return'])
        print(f"  Best  holding              : {best[0]} ({best[1]['return']:+.2%})")
        print(f"  Worst holding              : {worst[0]} ({worst[1]['return']:+.2%})")
        print(f"  Avg per-stock return       : {np.mean(returns):+.2%}")
        winners = sum(1 for r in returns if r > 0)
        print(f"  Winners / total            : {winners} / {len(returns)}")

    return {
        'label': label, 'total_ret': total_ret, 'ann_ret': ann_ret,
        'spy_ret': spy_ret, 'spy_ann': spy_ann,
        'sharpe': sharpe, 'max_dd': max_dd,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CHARTING
# ──────────────────────────────────────────────────────────────────────────────
def _spy_normalised(prices, start_date, end_date, base_capital):
    """Return SPY price series normalised to base_capital."""
    start = pd.Timestamp(start_date)
    end   = pd.Timestamp(end_date)
    spy   = prices['SPY'].dropna()
    spy   = spy[(spy.index >= start) & (spy.index <= end)]
    if spy.empty:
        return pd.Series(dtype=float)
    return spy / spy.iloc[0] * base_capital


def _plot_window(ax_port, ax_stocks,
                 portfolio_series, position_log, active_tickers,
                 prices, start_date, end_date, base_capital, title_prefix):
    """Fill one row: (a) portfolio vs SPY, (b) per-stock total-return bar chart."""
    start = pd.Timestamp(start_date)
    end   = pd.Timestamp(end_date)

    # ── Left: Portfolio value vs SPY ─────────────────────────────────────────
    spy_norm = _spy_normalised(prices, start_date, end_date, base_capital)

    ax_port.plot(portfolio_series.index, portfolio_series.values,
                 color='royalblue', linewidth=1.8, label='Pershing Square Holdings')
    ax_port.plot(spy_norm.index, spy_norm.values,
                 color='darkorange', linewidth=1.2, linestyle='--', label='SPY (benchmark)')
    ax_port.set_title(f'{title_prefix} — Portfolio vs SPY', fontsize=10, fontweight='bold')
    ax_port.set_ylabel('Portfolio Value ($)')
    ax_port.legend(fontsize=8)
    ax_port.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    ax_port.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax_port.tick_params(axis='x', rotation=30, labelsize=7)
    ax_port.grid(alpha=0.25)

    strat_ret = (portfolio_series.iloc[-1] / portfolio_series.iloc[0] - 1
                 if len(portfolio_series) >= 2 else float('nan'))
    spy_ret   = (spy_norm.iloc[-1] / spy_norm.iloc[0] - 1
                 if not spy_norm.empty else float('nan'))
    ax_port.text(
        0.02, 0.05,
        f"Holdings: {strat_ret:+.1%}\nSPY: {spy_ret:+.1%}",
        transform=ax_port.transAxes, fontsize=8, verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
    )

    # ── Right: Per-stock return bar chart ────────────────────────────────────
    if position_log:
        sorted_items = sorted(position_log.items(), key=lambda x: x[1]['return'])
        tickers  = [item[0] for item in sorted_items]
        returns  = [item[1]['return'] * 100 for item in sorted_items]
        colors   = ['tomato' if r < 0 else 'mediumseagreen' for r in returns]

        y_pos = range(len(tickers))
        ax_stocks.barh(y_pos, returns, color=colors, edgecolor='white', linewidth=0.5)
        ax_stocks.set_yticks(list(y_pos))
        ax_stocks.set_yticklabels(tickers, fontsize=8)
        ax_stocks.axvline(0, color='black', linewidth=0.8)

        # SPY return as reference line
        if not np.isnan(spy_ret):
            ax_stocks.axvline(spy_ret * 100, color='darkorange', linewidth=1.2,
                              linestyle='--', label=f'SPY ({spy_ret:+.1%})')
            ax_stocks.legend(fontsize=7)

        # Annotate bars with return values
        for i, (r, tkr) in enumerate(zip(returns, tickers)):
            ax_stocks.text(
                r + (0.5 if r >= 0 else -0.5), i,
                f'{r:+.1f}%', va='center',
                ha='left' if r >= 0 else 'right', fontsize=7
            )

        ax_stocks.set_xlabel('Total Return (%)')
        ax_stocks.set_title(f'{title_prefix} — Per-Stock Returns', fontsize=10, fontweight='bold')
        ax_stocks.grid(axis='x', alpha=0.25)
    else:
        ax_stocks.text(0.5, 0.5, 'No data', transform=ax_stocks.transAxes,
                       ha='center', va='center')


def generate_charts(
    w1_portfolio, w1_log, w1_active,
    w2_portfolio, w2_log, w2_active,
    prices
):
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        'Pershing Square Holdings — Buy-and-Hold Backtest',
        fontsize=13, fontweight='bold', y=1.01
    )

    _plot_window(
        axes[0, 0], axes[0, 1],
        w1_portfolio, w1_log, w1_active,
        prices, WINDOW1_START, WINDOW1_END, STARTING_CAPITAL,
        'Window 1 — Apr 2022 to Apr 2024'
    )
    _plot_window(
        axes[1, 0], axes[1, 1],
        w2_portfolio, w2_log, w2_active,
        prices, WINDOW2_START, WINDOW2_END, STARTING_CAPITAL,
        'Window 2 — Apr 2024 to Apr 2026'
    )

    plt.tight_layout()
    plt.savefig(CHART_OUTPUT_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nChart saved to: {CHART_OUTPUT_PATH}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    prices = download_all_data()

    print("\nRunning Window 1 backtest (Apr 2022 – Apr 2024) ...")
    w1_portfolio, w1_log, w1_active = run_backtest(
        prices, WINDOW1_START, WINDOW1_END, label='Window 1 (2022–2024)'
    )

    print("\nRunning Window 2 backtest (Apr 2024 – Apr 2026) ...")
    w2_portfolio, w2_log, w2_active = run_backtest(
        prices, WINDOW2_START, WINDOW2_END, label='Window 2 (2024–2026)'
    )

    compute_stats(w1_portfolio, w1_log, prices,
                  WINDOW1_START, WINDOW1_END,
                  label='Window 1 (2022–2024)', active_tickers=w1_active)

    compute_stats(w2_portfolio, w2_log, prices,
                  WINDOW2_START, WINDOW2_END,
                  label='Window 2 (2024–2026)', active_tickers=w2_active)

    generate_charts(
        w1_portfolio, w1_log, w1_active,
        w2_portfolio, w2_log, w2_active,
        prices
    )


if __name__ == '__main__':
    main()
