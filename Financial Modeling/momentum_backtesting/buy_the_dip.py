# ============================================================================
# buy_the_dip.py — "Buy Quality on Unjustified Dips" Backtesting Framework
# ============================================================================
# Strategy rules:
#   - Universe : 29 quality compounders  (QUALITY_WATCHLIST)
#   - Entry requires ALL THREE conditions simultaneously:
#       A. Stock is 20–40% off its rolling 52-week high
#       B. SPY is above its 200-day MA (bull regime only — no entries in bear)
#       C. Stock underperforms its sector ETF by ≥10pp over trailing 60 days
#   - Optional panic confirmation (+1 conviction if VIX > 25 OR RSI < 35)
#   - Max 4 simultaneous positions, equal-weighted at 25% each
#   - Three exit triggers:
#       A. Thesis recovery: stock recovers to within 10% of 52-wk high at entry
#       B. Stop loss: stock falls 20% below entry price
#       C. Max hold: 365 days — if thesis hasn't played out, exit and review
#
# Two-window validation (identical windows to capital_preservation_strategy):
#   - In-sample      2024-04-01 → 2026-04-12
#   - Out-of-sample  2022-04-01 → 2024-04-01
# ============================================================================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, date, timedelta
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# VERBOSE FLAG — set True to print every signal and exit with full details
# ──────────────────────────────────────────────────────────────────────────────
VERBOSE = True

# ──────────────────────────────────────────────────────────────────────────────
# UNIVERSE
# ──────────────────────────────────────────────────────────────────────────────
QUALITY_WATCHLIST = [
    # Tier 1 — existing monitor stocks
    'AAPL', 'MSFT', 'GOOG', 'META', 'AMZN', 'ADBE', 'SNPS',
    'PAYC', 'NVO', 'AMD', 'NVDA', 'NFLX', 'DDOG', 'NOW', 'CRM',

    # New quality additions — wide moat, strong FCF, proven business models
    # They grind upward rather than spike, making them ideal dip-buy candidates
    'V', 'MA', 'COST', 'BRK-B', 'ASML', 'TMO', 'UNH',
    'SPGI', 'MCO', 'IDEXX', 'MSCI', 'ROP', 'FAST', 'WST'
]

SECTOR_ETFS = {
    'AAPL': 'XLK', 'MSFT': 'XLK', 'GOOG': 'XLK', 'META': 'XLK',
    'AMZN': 'XLY', 'ADBE': 'XLK', 'SNPS': 'XLK', 'NVDA': 'XLK',
    'AMD':  'XLK', 'NFLX': 'XLY', 'DDOG': 'XLK', 'NOW':  'XLK',
    'CRM':  'XLK', 'PAYC': 'XLK', 'NVO':  'XLV',
    'V':    'XLF', 'MA':   'XLF', 'COST': 'XLP', 'BRK-B':'XLF',
    'ASML': 'XLK', 'TMO':  'XLV', 'UNH':  'XLV',
    'SPGI': 'XLF', 'MCO':  'XLF', 'IDEXX':'XLV',
    'MSCI': 'XLF', 'ROP':  'XLK', 'FAST': 'XLI', 'WST':  'XLV'
}

# ──────────────────────────────────────────────────────────────────────────────
# PARAMETERS
# ──────────────────────────────────────────────────────────────────────────────
STARTING_CAPITAL      = 10_000
MAX_POSITIONS         = 4           # concentration is intentional
POSITION_SIZE         = 0.25        # equal weight — 25% per position

# Entry thresholds
DIP_MIN_PCT           = 0.20        # minimum drawdown from 52-wk high to flag
DIP_MAX_PCT           = 0.40        # above this raises value-trap risk
SECTOR_UNDERPERF_DAYS = 60          # rolling window for stock vs sector comparison
SECTOR_UNDERPERF_MIN  = 0.10        # stock must lag sector by at least 10pp

# Panic confirmation
VIX_PANIC_THRESHOLD   = 25          # VIX above this → +1 conviction
RSI_PANIC_THRESHOLD   = 35          # RSI below this → +1 conviction

# Exit thresholds
RECOVERY_BUFFER       = 0.10        # Exit A: within 10% of 52-wk high at entry
STOP_LOSS_PCT         = 0.20        # Exit B: 20% below entry price
MAX_HOLD_DAYS         = 365         # Exit C: maximum holding period in calendar days

# Backtest windows
INSAMPLE_START        = '2024-04-01'
INSAMPLE_END          = '2026-04-12'
OUTSAMPLE_START       = '2022-04-01'
OUTSAMPLE_END         = '2024-04-01'
DATA_DOWNLOAD_START   = '2021-01-01'

CHART_OUTPUT_PATH     = 'momentum_backtesting/buy_the_dip_charts.png'


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def _to_date(d):
    if isinstance(d, (datetime, pd.Timestamp)):
        return d.date()
    if isinstance(d, str):
        return datetime.strptime(d, '%Y-%m-%d').date()
    return d


def _wilder_rsi(close, period=14):
    delta  = close.diff()
    gains  = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)
    alpha  = 1 / period
    avg_g  = gains.ewm(alpha=alpha, adjust=False).mean()
    avg_l  = losses.ewm(alpha=alpha, adjust=False).mean()
    return 100 - 100 / (1 + avg_g / avg_l)


def _prev_trading_date(prices_index, target_date):
    """Return the last index date on or before target_date."""
    dates = [d for d in prices_index if _to_date(d) <= _to_date(target_date)]
    return dates[-1] if dates else None


# ──────────────────────────────────────────────────────────────────────────────
# DATA DOWNLOAD
# ──────────────────────────────────────────────────────────────────────────────
def download_all_data():
    """Download price data for all tickers needed across both windows."""
    all_tickers = (
        QUALITY_WATCHLIST
        + list(set(SECTOR_ETFS.values()))
        + ['SPY', 'VIX']  # VIX downloaded as ^VIX
    )
    unique = sorted(set(all_tickers))
    vix_tickers = ['SPY'] + QUALITY_WATCHLIST + sorted(set(SECTOR_ETFS.values()))

    print(f"Downloading price data for {len(vix_tickers)} tickers + ^VIX ...")
    raw = yf.download(
        vix_tickers,
        start=DATA_DOWNLOAD_START,
        end=INSAMPLE_END,
        auto_adjust=True,
        progress=False
    )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw['Close'].copy()
    else:
        prices = raw[['Close']].copy()
        prices.columns = [vix_tickers[0]]

    prices.index = pd.to_datetime(prices.index)

    print("Downloading ^VIX ...")
    vix_raw = yf.download('^VIX', start=DATA_DOWNLOAD_START, end=INSAMPLE_END,
                          auto_adjust=True, progress=False)
    if isinstance(vix_raw.columns, pd.MultiIndex):
        vix_close = vix_raw['Close'].iloc[:, 0]
    else:
        vix_close = vix_raw['Close']
    vix_close.index = pd.to_datetime(vix_close.index)
    prices['VIX'] = vix_close

    print(f"Data ready — {len(prices)} trading days from {prices.index[0].date()} "
          f"to {prices.index[-1].date()}")
    return prices


# ──────────────────────────────────────────────────────────────────────────────
# SIGNAL EVALUATION
# ──────────────────────────────────────────────────────────────────────────────
def _drawdown_from_52wk_high(ticker, prices, sim_date):
    """
    Return (drawdown_pct, high_52wk) where drawdown_pct is the fraction below
    the rolling 252-day high.  Positive value = below the high.
    """
    if ticker not in prices.columns:
        return None, None
    col = prices[ticker].dropna()
    col = col[col.index <= sim_date]
    if len(col) < 30:
        return None, None
    window = col.iloc[-252:] if len(col) >= 252 else col
    high_52 = float(window.max())
    current = float(col.iloc[-1])
    if high_52 <= 0:
        return None, None
    drawdown = (high_52 - current) / high_52
    return drawdown, high_52


def _spy_above_200ma(prices, sim_date):
    """Condition B: SPY must be above its 200-day MA."""
    if 'SPY' not in prices.columns:
        return False
    spy = prices['SPY'].dropna()
    spy = spy[spy.index <= sim_date]
    if len(spy) < 200:
        return False
    ma200 = float(spy.rolling(200).mean().iloc[-1])
    current = float(spy.iloc[-1])
    return current > ma200


def _sector_underperformance(ticker, prices, sim_date, window_days=60):
    """
    Condition C: Return stock_return - sector_return over trailing window.
    Negative means stock lagged sector.  We want this < -SECTOR_UNDERPERF_MIN.
    """
    sector_etf = SECTOR_ETFS.get(ticker)
    if sector_etf is None or sector_etf not in prices.columns:
        return None
    if ticker not in prices.columns:
        return None

    stock_col  = prices[ticker].dropna()
    sector_col = prices[sector_etf].dropna()

    stock_col  = stock_col[stock_col.index <= sim_date]
    sector_col = sector_col[sector_col.index <= sim_date]

    if len(stock_col) < window_days + 5 or len(sector_col) < window_days + 5:
        return None

    # Align on common dates
    common = stock_col.index.intersection(sector_col.index)
    stock_w  = stock_col.loc[common].iloc[-window_days:]
    sector_w = sector_col.loc[common].iloc[-window_days:]

    if len(stock_w) < 20:
        return None

    stock_ret  = float(stock_w.iloc[-1]) / float(stock_w.iloc[0]) - 1
    sector_ret = float(sector_w.iloc[-1]) / float(sector_w.iloc[0]) - 1
    return stock_ret - sector_ret  # negative = stock lagged sector


def _get_rsi(ticker, prices, sim_date, period=14):
    if ticker not in prices.columns:
        return None
    col = prices[ticker].dropna()
    col = col[col.index <= sim_date]
    if len(col) < period + 5:
        return None
    return float(_wilder_rsi(col, period).iloc[-1])


def _get_vix(prices, sim_date):
    if 'VIX' not in prices.columns:
        return None
    vix = prices['VIX'].dropna()
    vix = vix[vix.index <= sim_date]
    if vix.empty:
        return None
    return float(vix.iloc[-1])


def evaluate_entry(ticker, prices, sim_date):
    """
    Returns a dict with all condition details and whether the entry is valid.
    conviction_score: 0 or 1 (panic confirmation adds 1 if VIX>25 or RSI<35).
    """
    result = {
        'ticker': ticker,
        'date': sim_date,
        'drawdown': None,
        'high_52wk': None,
        'cond_A': False,   # drawdown 20–40%
        'spy_above_200': False,
        'cond_B': False,   # SPY regime
        'sector_underperf': None,
        'cond_C': False,   # idiosyncratic dip
        'vix': None,
        'rsi': None,
        'panic_confirmed': False,
        'conviction_score': 0,
        'entry_valid': False,
    }

    # ── Condition A ───────────────────────────────────────────────────────────
    drawdown, high_52 = _drawdown_from_52wk_high(ticker, prices, sim_date)
    result['drawdown']  = drawdown
    result['high_52wk'] = high_52
    if drawdown is not None:
        result['cond_A'] = DIP_MIN_PCT <= drawdown <= DIP_MAX_PCT

    # ── Condition B ───────────────────────────────────────────────────────────
    result['spy_above_200'] = _spy_above_200ma(prices, sim_date)
    result['cond_B'] = result['spy_above_200']

    # ── Condition C ───────────────────────────────────────────────────────────
    underperf = _sector_underperformance(ticker, prices, sim_date, SECTOR_UNDERPERF_DAYS)
    result['sector_underperf'] = underperf
    if underperf is not None:
        result['cond_C'] = underperf < -SECTOR_UNDERPERF_MIN

    # ── Panic confirmation (optional +1) ─────────────────────────────────────
    vix = _get_vix(prices, sim_date)
    rsi = _get_rsi(ticker, prices, sim_date)
    result['vix'] = vix
    result['rsi'] = rsi
    panic = (vix is not None and vix > VIX_PANIC_THRESHOLD) or \
            (rsi is not None and rsi < RSI_PANIC_THRESHOLD)
    result['panic_confirmed']  = panic
    result['conviction_score'] = 1 if panic else 0

    result['entry_valid'] = result['cond_A'] and result['cond_B'] and result['cond_C']
    return result


# ──────────────────────────────────────────────────────────────────────────────
# BACKTEST ENGINE
# ──────────────────────────────────────────────────────────────────────────────
def run_backtest(prices, start_date, end_date, label=''):
    """
    Main simulation loop.  Returns:
        portfolio_values : pd.Series (daily portfolio value)
        trades           : list of dicts (one per closed trade)
        entry_events     : list of (date, ticker, price)
        exit_events      : list of (date, ticker, price, reason)
    """
    start = _to_date(start_date)
    end   = _to_date(end_date)

    trading_days = [
        d.date() for d in prices.index
        if start <= d.date() <= end
    ]

    # ── State ─────────────────────────────────────────────────────────────────
    cash      = float(STARTING_CAPITAL)
    positions = {}   # ticker → {entry_price, shares, entry_date, high_52wk_at_entry}
    portfolio_values = {}
    trades           = []
    entry_events     = []
    exit_events      = []

    # Track idle periods and peak simultaneous positions
    last_open_date  = None
    peak_open_count = 0

    if VERBOSE:
        print(f"\n{'='*70}")
        print(f"  BACKTEST: {label}  ({start} → {end})")
        print(f"{'='*70}")

    for sim_date in trading_days:
        # ── Locate today's prices ─────────────────────────────────────────────
        ts = pd.Timestamp(sim_date)
        if ts not in prices.index:
            continue

        def price(tkr):
            if tkr in prices.columns and not pd.isna(prices.at[ts, tkr]):
                return float(prices.at[ts, tkr])
            return None

        # ── Mark open positions to market ─────────────────────────────────────
        total_equity = cash
        for tkr, pos in positions.items():
            p = price(tkr)
            if p is not None:
                total_equity += pos['shares'] * p

        # ── Check exits on existing positions ────────────────────────────────
        to_exit = {}
        for tkr, pos in list(positions.items()):
            p = price(tkr)
            if p is None:
                continue

            entry_px   = pos['entry_price']
            high_52    = pos['high_52wk_at_entry']
            entry_date = pos['entry_date']
            hold_days  = (sim_date - entry_date).days

            reason = None

            # Exit A — thesis recovery: within 10% of 52-wk-high at entry
            if high_52 > 0 and p >= high_52 * (1 - RECOVERY_BUFFER):
                reason = 'A_RECOVERY'

            # Exit B — stop loss: 20% below entry
            elif p <= entry_px * (1 - STOP_LOSS_PCT):
                reason = 'B_STOP_LOSS'

            # Exit C — max hold period
            elif hold_days >= MAX_HOLD_DAYS:
                reason = 'C_MAX_HOLD'

            if reason:
                to_exit[tkr] = (p, reason, entry_px, entry_date, hold_days)

        for tkr, (p, reason, entry_px, entry_date, hold_days) in to_exit.items():
            proceeds = positions[tkr]['shares'] * p
            cash += proceeds
            trade_ret = (p / entry_px) - 1
            trades.append({
                'ticker':      tkr,
                'entry_date':  entry_date,
                'exit_date':   sim_date,
                'entry_price': entry_px,
                'exit_price':  p,
                'return':      trade_ret,
                'hold_days':   hold_days,
                'exit_reason': reason,
            })
            exit_events.append((sim_date, tkr, p, reason))
            if VERBOSE:
                emoji_map = {
                    'A_RECOVERY':  '[EXIT A — RECOVERY]',
                    'B_STOP_LOSS': '[EXIT B — STOP LOSS]',
                    'C_MAX_HOLD':  '[EXIT C — MAX HOLD ]',
                }
                label_str = emoji_map.get(reason, f'[EXIT {reason}]')
                print(f"  {sim_date}  {label_str}  {tkr:6s}  "
                      f"entry={entry_px:.2f}  exit={p:.2f}  "
                      f"ret={trade_ret:+.1%}  held={hold_days}d")
                if reason == 'B_STOP_LOSS':
                    print(f"    >> Potential value trap — flagged for review")
                if reason == 'C_MAX_HOLD':
                    print(f"    >> Thesis did not play out in {MAX_HOLD_DAYS}d — flagged for pattern analysis")
            del positions[tkr]

        # ── Check entries ─────────────────────────────────────────────────────
        n_open = len(positions)
        if n_open < MAX_POSITIONS and cash > 0:
            candidates = []
            for tkr in QUALITY_WATCHLIST:
                if tkr in positions:
                    continue  # already holding
                sig = evaluate_entry(tkr, prices, ts)
                if sig['entry_valid']:
                    candidates.append(sig)

            # Sort by conviction (panic-confirmed first), then drawdown depth
            candidates.sort(key=lambda s: (-s['conviction_score'], -s['drawdown']))

            slots_available = MAX_POSITIONS - n_open
            for sig in candidates[:slots_available]:
                tkr   = sig['ticker']
                p     = price(tkr)
                if p is None:
                    continue

                # Position size: 25% of current total equity (not just cash)
                target_value = total_equity * POSITION_SIZE
                target_value = min(target_value, cash)  # can't exceed available cash
                if target_value < 1:
                    continue
                shares = target_value / p
                cash  -= shares * p
                positions[tkr] = {
                    'entry_price':       p,
                    'shares':            shares,
                    'entry_date':        sim_date,
                    'high_52wk_at_entry': sig['high_52wk'],
                }
                entry_events.append((sim_date, tkr, p))
                last_open_date = sim_date

                if VERBOSE:
                    print(f"  {sim_date}  [ENTRY]  {tkr:6s}  "
                          f"price={p:.2f}  drawdown={sig['drawdown']:.1%}  "
                          f"52wk_high={sig['high_52wk']:.2f}")
                    print(f"    Cond A (dip 20–40%):      {'YES' if sig['cond_A'] else 'NO '}  "
                          f"drawdown={sig['drawdown']:.1%}")
                    print(f"    Cond B (SPY > 200MA):     {'YES' if sig['cond_B'] else 'NO '}  "
                          f"spy_above_200={sig['spy_above_200']}")
                    print(f"    Cond C (idiosync. dip):   {'YES' if sig['cond_C'] else 'NO '}  "
                          f"underperf vs sector={sig['sector_underperf']:.1%}")
                    vix_str = f"{sig['vix']:.1f}" if sig['vix'] is not None else 'N/A'
                    rsi_str = f"{sig['rsi']:.1f}" if sig['rsi'] is not None else 'N/A'
                    print(f"    Panic confirm (VIX/RSI):  {'YES' if sig['panic_confirmed'] else 'NO '}  "
                          f"VIX={vix_str}  RSI={rsi_str}")

        # ── Revalue after entries ─────────────────────────────────────────────
        total_equity = cash
        for tkr, pos in positions.items():
            p = price(tkr)
            if p is not None:
                total_equity += pos['shares'] * p
        portfolio_values[sim_date] = total_equity
        peak_open_count = max(peak_open_count, len(positions))

    # Force-close any remaining open positions at last price
    last_day = trading_days[-1] if trading_days else None
    if last_day:
        ts = pd.Timestamp(last_day)
        for tkr, pos in list(positions.items()):
            p = price(tkr) if ts in prices.index else None
            if p is None:
                continue
            proceeds = pos['shares'] * p
            trade_ret = (p / pos['entry_price']) - 1
            trades.append({
                'ticker':      tkr,
                'entry_date':  pos['entry_date'],
                'exit_date':   last_day,
                'entry_price': pos['entry_price'],
                'exit_price':  p,
                'return':      trade_ret,
                'hold_days':   (last_day - pos['entry_date']).days,
                'exit_reason': 'END_OF_WINDOW',
            })

    portfolio_series = pd.Series(portfolio_values)
    portfolio_series.index = pd.to_datetime(portfolio_series.index)
    return portfolio_series, trades, entry_events, exit_events, peak_open_count


# ──────────────────────────────────────────────────────────────────────────────
# STATISTICS
# ──────────────────────────────────────────────────────────────────────────────
def compute_stats(portfolio_series, trades, prices, start_date, end_date, label='',
                  peak_open_count=0):
    start = _to_date(start_date)
    end   = _to_date(end_date)

    # Total return
    if len(portfolio_series) < 2:
        total_ret = 0.0
    else:
        total_ret = portfolio_series.iloc[-1] / portfolio_series.iloc[0] - 1

    # SPY return
    spy = prices['SPY'].dropna()
    spy = spy[(spy.index >= pd.Timestamp(start)) & (spy.index <= pd.Timestamp(end))]
    spy_ret = float(spy.iloc[-1]) / float(spy.iloc[0]) - 1 if len(spy) >= 2 else float('nan')

    # Trade breakdown — exclude END_OF_WINDOW for win/loss stats
    closed = [t for t in trades if t['exit_reason'] != 'END_OF_WINDOW']
    n_trades = len(closed)

    wins   = [t for t in closed if t['exit_reason'] == 'A_RECOVERY']
    losses = [t for t in closed if t['exit_reason'] == 'B_STOP_LOSS']
    maxhld = [t for t in closed if t['exit_reason'] == 'C_MAX_HOLD']

    win_rate     = len(wins) / n_trades if n_trades else float('nan')
    avg_win_ret  = np.mean([t['return'] for t in wins])   if wins   else float('nan')
    avg_loss_ret = np.mean([t['return'] for t in losses]) if losses else float('nan')
    avg_hold     = np.mean([t['hold_days'] for t in closed]) if closed else float('nan')

    # Maximum simultaneous positions — tracked directly in simulation loop
    max_simul = peak_open_count

    # Longest idle period (calendar days between close of last trade and next entry)
    if len(trades) >= 2:
        events = sorted(
            [(t['entry_date'], 'entry') for t in trades] +
            [(t['exit_date'],  'exit')  for t in trades],
            key=lambda x: x[0]
        )
        longest_idle = 0
        last_activity = start
        in_positions_set = set()
        # Simpler: look at days portfolio_series had zero open positions
        # We'll approximate using entry/exit event gaps
        exit_dates = sorted(t['exit_date'] for t in trades)
        entry_dates = sorted(t['entry_date'] for t in trades)
        for i in range(len(exit_dates)):
            next_entries = [e for e in entry_dates if e > exit_dates[i]]
            if next_entries:
                gap = (next_entries[0] - exit_dates[i]).days
                longest_idle = max(longest_idle, gap)
    else:
        longest_idle = (end - start).days  # never entered

    print(f"\n{'─'*60}")
    print(f"  RESULTS: {label}  ({start} → {end})")
    print(f"{'─'*60}")
    print(f"  Portfolio total return     : {total_ret:+.2%}")
    print(f"  SPY total return           : {spy_ret:+.2%}")
    print(f"  Outperformance vs SPY      : {total_ret - spy_ret:+.2%}")
    print(f"  Trades triggered           : {n_trades}")
    print(f"    → Exit A (recovery)      : {len(wins)}")
    print(f"    → Exit B (stop loss)     : {len(losses)}")
    print(f"    → Exit C (max hold)      : {len(maxhld)}")
    print(f"  Win rate                   : {win_rate:.1%}" if not np.isnan(win_rate) else "  Win rate                   : N/A")
    print(f"  Avg return per win         : {avg_win_ret:+.2%}" if not np.isnan(avg_win_ret) else "  Avg return per win         : N/A")
    print(f"  Avg return per loss        : {avg_loss_ret:+.2%}" if not np.isnan(avg_loss_ret) else "  Avg return per loss        : N/A")
    print(f"  Avg holding period         : {avg_hold:.0f} days" if not np.isnan(avg_hold) else "  Avg holding period         : N/A")
    print(f"  Max simultaneous positions : {max_simul}")
    print(f"  Longest idle period        : {longest_idle} days")

    return {
        'label': label, 'total_ret': total_ret, 'spy_ret': spy_ret,
        'n_trades': n_trades, 'wins': len(wins), 'losses': len(losses),
        'maxhld': len(maxhld), 'win_rate': win_rate,
        'avg_win_ret': avg_win_ret, 'avg_loss_ret': avg_loss_ret,
        'avg_hold': avg_hold, 'max_simul': max_simul, 'longest_idle': longest_idle,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CHARTING
# ──────────────────────────────────────────────────────────────────────────────
def _spy_series(prices, start_date, end_date, base_capital):
    """Return SPY normalized to base_capital over the window."""
    start = pd.Timestamp(start_date)
    end   = pd.Timestamp(end_date)
    spy = prices['SPY'].dropna()
    spy = spy[(spy.index >= start) & (spy.index <= end)]
    if spy.empty:
        return pd.Series(dtype=float)
    return spy / spy.iloc[0] * base_capital


def _plot_window(ax_port, ax_dip, portfolio_series, trades,
                 entry_events, exit_events,
                 prices, start_date, end_date, base_capital, title_prefix):
    """Fill one row (portfolio value + entry/exit overlay charts)."""
    start = pd.Timestamp(start_date)
    end   = pd.Timestamp(end_date)

    # ── Left: Portfolio vs SPY ────────────────────────────────────────────────
    spy_norm = _spy_series(prices, start_date, end_date, base_capital)
    ax_port.plot(portfolio_series.index, portfolio_series.values,
                 color='royalblue', linewidth=1.6, label='Strategy')
    ax_port.plot(spy_norm.index, spy_norm.values,
                 color='darkorange', linewidth=1.2, linestyle='--', label='SPY')
    ax_port.set_title(f'{title_prefix} — Portfolio vs SPY', fontsize=10, fontweight='bold')
    ax_port.set_ylabel('Portfolio Value ($)')
    ax_port.legend(fontsize=8)
    ax_port.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    ax_port.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax_port.tick_params(axis='x', rotation=30, labelsize=7)
    ax_port.grid(alpha=0.25)

    strat_ret = (portfolio_series.iloc[-1] / portfolio_series.iloc[0] - 1
                 if len(portfolio_series) >= 2 else float('nan'))
    spy_ret   = spy_norm.iloc[-1] / spy_norm.iloc[0] - 1 if not spy_norm.empty else float('nan')
    strat_lbl = f"{strat_ret:+.1%}" if not np.isnan(strat_ret) else "N/A"
    spy_lbl   = f"{spy_ret:+.1%}"   if not np.isnan(spy_ret)   else "N/A"
    ax_port.text(
        0.02, 0.05,
        f"Strategy: {strat_lbl}\nSPY: {spy_lbl}",
        transform=ax_port.transAxes, fontsize=8, verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
    )

    # ── Right: Entry/exit overlay on SPY price ───────────────────────────────
    spy_price = prices['SPY'].dropna()
    spy_price = spy_price[(spy_price.index >= start) & (spy_price.index <= end)]
    ax_dip.plot(spy_price.index, spy_price.values,
                color='gray', linewidth=1.0, alpha=0.6, label='SPY Price')

    def _spy_at(ts, fallback):
        cand = spy_price[spy_price.index <= ts]
        return float(cand.iloc[-1]) if not cand.empty else fallback

    if entry_events:
        e_dates  = [pd.Timestamp(e[0]) for e in entry_events]
        e_prices = [_spy_at(pd.Timestamp(ed), ep) for ed, tkr, ep in entry_events]
        ax_dip.scatter(e_dates, e_prices, marker='^', color='green',
                       s=60, zorder=5, label='Entry')

    if exit_events:
        x_dates  = [pd.Timestamp(x[0]) for x in exit_events]
        x_prices = [_spy_at(pd.Timestamp(xd), xp) for xd, tkr, xp, reason in exit_events]
        ax_dip.scatter(x_dates, x_prices, marker='v', color='red',
                       s=60, zorder=5, label='Exit')

    ax_dip.set_title(f'{title_prefix} — Entry/Exit Signals on SPY', fontsize=10, fontweight='bold')
    ax_dip.set_ylabel('SPY Price ($)')
    ax_dip.legend(fontsize=8)
    ax_dip.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    ax_dip.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax_dip.tick_params(axis='x', rotation=30, labelsize=7)
    ax_dip.grid(alpha=0.25)


def generate_charts(
    is_portfolio,  is_trades,  is_entries,  is_exits,
    oos_portfolio, oos_trades, oos_entries, oos_exits,
    prices
):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        'Buy Quality on Unjustified Dips — Backtest Results',
        fontsize=13, fontweight='bold', y=1.01
    )

    _plot_window(
        axes[0, 0], axes[0, 1],
        is_portfolio, is_trades, is_entries, is_exits,
        prices, INSAMPLE_START, INSAMPLE_END, STARTING_CAPITAL,
        'In-Sample (Apr 2024 – Apr 2026)'
    )
    _plot_window(
        axes[1, 0], axes[1, 1],
        oos_portfolio, oos_trades, oos_entries, oos_exits,
        prices, OUTSAMPLE_START, OUTSAMPLE_END, STARTING_CAPITAL,
        'Out-of-Sample (Apr 2022 – Apr 2024)'
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

    print("\nRunning in-sample backtest ...")
    is_portfolio, is_trades, is_entries, is_exits, is_peak = run_backtest(
        prices, INSAMPLE_START, INSAMPLE_END, label='In-Sample'
    )

    print("\nRunning out-of-sample backtest ...")
    oos_portfolio, oos_trades, oos_entries, oos_exits, oos_peak = run_backtest(
        prices, OUTSAMPLE_START, OUTSAMPLE_END, label='Out-of-Sample'
    )

    compute_stats(is_portfolio,  is_trades,  prices, INSAMPLE_START,  INSAMPLE_END,
                  'In-Sample',     peak_open_count=is_peak)
    compute_stats(oos_portfolio, oos_trades, prices, OUTSAMPLE_START, OUTSAMPLE_END,
                  'Out-of-Sample', peak_open_count=oos_peak)

    generate_charts(
        is_portfolio,  is_trades,  is_entries,  is_exits,
        oos_portfolio, oos_trades, oos_entries, oos_exits,
        prices
    )


if __name__ == '__main__':
    main()
