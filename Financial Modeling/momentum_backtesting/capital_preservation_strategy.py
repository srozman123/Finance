# ============================================================================
# capital_preservation_strategy.py — Capital-Preservation Momentum Backtest
# ============================================================================
# Strategy rules:
#   - Momentum universe : 17 growth/tech stocks  (MOMENTUM_UNIVERSE)
#   - Safe haven universe: GLD (80%) / SLV (20%)  (SAFE_HAVEN_UNIVERSE)
#   - Rebalance every 7 trading days (weekly)
#   - Three-regime SPY filter (200-day MA + 10-day slope):
#       BULL:    SPY > 200MA and slope > 0
#                → full momentum scan, score ≥ 4, full position sizing
#       CAUTION: SPY < 200MA but slope > 0
#                → momentum scan, score ≥ 3, 50% position sizing, no safe haven
#       BEAR:    SPY < 200MA and slope ≤ 0  (or SPY > 200MA and slope < -0.5)
#                → rotate to GLD 80% / SLV 20%, hold until CAUTION or BULL
#   - RSI Rate of Change: 3-day MA of 10-day RSI change; +1 if > 1.5, -1 if < -1.0
#   - Fixed 15% stop loss on momentum positions only (GLD/SLV held without stop)
#   - Equal weighting within momentum slots
#
# Original vs Refined comparison:
#   Original: full 17-stock universe, 15% fixed stop (mom only), equal weight
#             score ≥ 4 (BULL) / ≥ 3 (CAUTION), RSI_ROC signal, no vol filter
#   Refined:  identical to Original PLUS [R3] volatility filter — excludes stocks
#             with 90-day annualised vol ≥ 50% from new momentum entries
#
#   [R1] Money-market cash note — see comment block in parameters section
#
# Two-window validation:
#   - In-sample      2024-04-01 → 2026-04-08
#   - Out-of-sample  2022-04-01 → 2024-04-01
# ============================================================================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from datetime import datetime, date, timedelta
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# UNIVERSE DEFINITIONS
# ──────────────────────────────────────────────────────────────────────────────
MOMENTUM_UNIVERSE = [
    'AAPL', 'MSFT', 'GOOG', 'META', 'AVGO', 'AMZN',
    'ADBE', 'PAYC', 'SNPS', 'NVDA', 'AMD', 'NFLX',
    'DDOG', 'NOW', 'CRM', 'PLTR', 'RDDT'
]
SAFE_HAVEN_UNIVERSE = ['GLD', 'SLV']
SAFE_HAVEN_WEIGHTS  = {'GLD': 0.80, 'SLV': 0.20}

# ──────────────────────────────────────────────────────────────────────────────
# PARAMETERS
# ──────────────────────────────────────────────────────────────────────────────

# Toggle: False → revert to 100% cash during SPY-filter blocks (comparison mode)
DEFENSIVE_ROTATION      = True

STARTING_CAPITAL        = 10_000
MAX_POSITIONS           = 5
REBALANCE_FREQ_DAYS     = 7
MIN_MOMENTUM_SCORE      = 4     # BULL regime: score ≥ 4
CAUTION_SCORE_THRESHOLD = 3     # CAUTION regime: score ≥ 3
FIXED_STOP_PCT          = 0.15  # fixed stop — applied to momentum positions only
MAX_VOLATILITY          = 0.50  # [R3] refined only: exclude stocks with 90-day ann. vol ≥ 50%
MAX_HOLD_DAYS           = 90
INSAMPLE_START          = '2024-04-01'
INSAMPLE_END            = '2026-04-08'
OUTSAMPLE_START         = '2022-04-01'
OUTSAMPLE_END           = '2024-04-01'
DATA_DOWNLOAD_START     = '2021-01-01'
RISK_FREE_RATE          = 0.042

# ── [R6 / R2] Stop loss toggle ────────────────────────────────────────────────
USE_TRAILING_STOP = False  # Set True to re-enable trailing stop for comparison
TRAILING_STOP_PCT = None   # Only used when USE_TRAILING_STOP = True
#   When USE_TRAILING_STOP = False (default): FIXED_STOP_PCT (15%) applied to
#   momentum positions only.  Safe haven positions (GLD/SLV) are held without
#   any stop — they exit only when regime shifts to CAUTION or BULL.
#   When USE_TRAILING_STOP = True: track the highest close since entry and exit
#   when price falls more than TRAILING_STOP_PCT below that peak.

# ── [R1] Money-market cash yield note ─────────────────────────────────────────
#   Uninvested cash earns 0% in this backtest — the simplest conservative assumption.
#   In live implementation, idle cash would be swept nightly into a money-market
#   fund (e.g. Fidelity SPAXX, Vanguard VMFXX) currently yielding ~4.0–4.5%
#   annualised (as of 2024–2026).  During extended SPY-filter cash periods, the
#   live strategy therefore earns ~0.35–0.37% per month on full notional,
#   materially narrowing any performance gap versus buy-and-hold during bear markets.
#   Backtested total returns understate the live strategy's total return by roughly:
#     avg_cash_weight × avg_money_market_rate × years_in_window


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def _wilder_rsi(close, period=14):
    delta  = close.diff()
    gains  = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)
    alpha  = 1 / period
    avg_g  = gains.ewm(alpha=alpha, adjust=False).mean()
    avg_l  = losses.ewm(alpha=alpha, adjust=False).mean()
    return 100 - 100 / (1 + avg_g / avg_l)


def _to_date(d):
    if isinstance(d, (datetime, pd.Timestamp)):
        return d.date()
    if isinstance(d, str):
        return datetime.strptime(d, '%Y-%m-%d').date()
    return d


# ── [R3] Annualised vol helper ─────────────────────────────────────────────────
def _annualized_vol(ticker, price_data, sim_date, lookback=90):
    """90-day annualised daily-return vol up to sim_date. Returns None if sparse."""
    if ticker not in price_data.columns:
        return None
    close = price_data[ticker].loc[:sim_date].dropna()
    if len(close) < 20:
        return None
    recent = close.iloc[-lookback:]
    daily_rets = recent.pct_change().dropna()
    if len(daily_rets) < 10:
        return None
    return float(daily_rets.std() * np.sqrt(252))


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — calculate_momentum_score
# ──────────────────────────────────────────────────────────────────────────────
def calculate_momentum_score(ticker, sim_date, price_data, earnings_data,
                             universe_mom_median=None):
    components = {
        'mom_12_1':     None,
        'eps_surprise': None,
        'recent_beat':  False,
        'above_200ma':  False,
        'ma50_above':   False,
        'rsi':          None,
        'rsi_rising':   False,
        'rsi_roc':      None,
    }

    if ticker not in price_data.columns:
        return 0, components

    close = price_data[ticker].loc[:sim_date].dropna()
    if close.empty:
        return 0, components

    # Criterion 1: 12-1 price momentum
    mom_12_1 = None
    if len(close) >= 252:
        p_12m = float(close.iloc[-252])
        p_1m  = float(close.iloc[-21]) if len(close) >= 21 else float(close.iloc[-1])
        if p_12m > 0:
            mom_12_1 = p_1m / p_12m - 1
    components['mom_12_1'] = mom_12_1

    c1 = (universe_mom_median is not None and
          mom_12_1 is not None and
          mom_12_1 >= universe_mom_median)

    # Criteria 2 & 3: earnings surprise and recency
    eh = earnings_data.get(ticker)
    c2, c3, eps_surprise = False, False, None
    if eh is not None and not eh.empty:
        mask = pd.Series(
            [_to_date(idx) <= sim_date for idx in eh.index],
            index=eh.index
        )
        eh_past = eh[mask]
        if not eh_past.empty:
            latest   = eh_past.iloc[-1]
            reported = latest.get('epsActual')
            estimate = latest.get('epsEstimate')
            if (reported is not None and estimate is not None and
                    not pd.isna(reported) and not pd.isna(estimate) and
                    estimate != 0):
                eps_surprise = (float(reported) - float(estimate)) / abs(float(estimate))
                c2 = eps_surprise > 0
                latest_date = _to_date(eh_past.index[-1])
                c3 = c2 and (sim_date - latest_date).days <= 60
    components['eps_surprise'] = eps_surprise
    components['recent_beat']  = c3

    # Criteria 4 & 5: technical momentum; RSI Rate of Change signal
    c4, c5 = False, False
    rsi_roc_adj = 0
    if len(close) >= 210:
        ma50  = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()
        rsi   = _wilder_rsi(close)

        last_close = float(close.iloc[-1])
        last_ma50  = float(ma50.iloc[-1])
        last_ma200 = float(ma200.iloc[-1])
        last_rsi   = float(rsi.iloc[-1])

        c4 = last_close > last_ma200 and last_ma50 > last_ma200
        components['above_200ma'] = last_close > last_ma200
        components['ma50_above']  = last_ma50  > last_ma200
        components['rsi']         = round(last_rsi, 1)

        if len(rsi) >= 6:
            rsi_window = rsi.iloc[-6:]
            rsi_rising = bool((rsi_window.diff().dropna() > 0).all())
            c5 = last_rsi > 50 and rsi_rising
            components['rsi_rising'] = rsi_rising

        # RSI Rate of Change: 3-day MA of (RSI_today − RSI_10_days_ago)
        if len(rsi) >= 13:
            roc_vals = []
            for i in [-1, -2, -3]:
                if len(rsi) >= abs(i) + 10:
                    roc_vals.append(float(rsi.iloc[i]) - float(rsi.iloc[i - 10]))
            if roc_vals:
                rsi_roc = sum(roc_vals) / len(roc_vals)
                components['rsi_roc'] = round(rsi_roc, 2)
                if rsi_roc > 1.5:
                    rsi_roc_adj = 1
                elif rsi_roc < -1.0:
                    rsi_roc_adj = -1

    base_score = sum([c1, c2, c3, c4, c5])
    return base_score + rsi_roc_adj, components


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — spy_regime: three-state classifier using 200MA and 10-day slope
# ──────────────────────────────────────────────────────────────────────────────
def spy_regime(sim_date, price_data):
    """
    Return 'BULL', 'CAUTION', or 'BEAR' based on SPY position relative to
    its 200-day MA and the 10-trading-day slope of that MA.

    BULL:    SPY > 200MA  and  slope > 0
    CAUTION: SPY < 200MA  and  slope > 0
    BEAR:    SPY < 200MA  and  slope ≤ 0
             OR SPY > 200MA  and  slope < -0.5  (strongly falling MA)
    """
    if 'SPY' not in price_data.columns:
        return 'BULL'
    spy = price_data['SPY'].loc[:sim_date].dropna()
    if len(spy) < 210:          # need 200 MA + 10 extra days for slope
        return 'BEAR'
    ma200      = spy.rolling(200).mean()
    last_ma200 = float(ma200.iloc[-1])
    prev_ma200 = float(ma200.iloc[-11])   # 10 trading days ago
    slope      = last_ma200 - prev_ma200
    spy_price  = float(spy.iloc[-1])
    above_200  = spy_price > last_ma200

    if above_200 and slope > 0:
        return 'BULL'
    elif not above_200 and slope > 0:
        return 'CAUTION'
    elif above_200 and slope < -0.5:
        return 'BEAR'
    else:
        # above_200 with slope in [-0.5, 0] → still BULL
        # below_200 with slope ≤ 0           → BEAR
        return 'BULL' if above_200 else 'BEAR'


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — run_backtest
# refined=True  → all refinements active (vol filter)
# refined=False → original strategy (no vol filter)
# ──────────────────────────────────────────────────────────────────────────────
def _make_sell_record(label, tk, pos, px, sim_date, reason):
    held = (sim_date - _to_date(pos['entry_date'])).days
    ret  = (px - pos['entry_price']) / pos['entry_price'] * 100
    return {
        'window':      label,
        'ticker':      tk,
        'action':      'SELL',
        'entry_date':  pos['entry_date'],
        'exit_date':   sim_date,
        'entry_price': pos['entry_price'],
        'exit_price':  px,
        'shares':      pos['shares'],
        'hold_days':   held,
        'return_pct':  round(ret, 2),
        'reason':      reason,
        'pos_type':    pos.get('pos_type', 'momentum'),
    }


def run_backtest(start_date, end_date, price_data, earnings_data, label,
                 refined=False):
    start_date = _to_date(start_date)
    end_date   = _to_date(end_date)

    all_days = price_data.loc[start_date:end_date].index
    if len(all_days) == 0:
        return None

    cash              = float(STARTING_CAPITAL)
    positions         = {}
    daily_values      = []
    daily_modes       = []   # 'momentum' | 'defensive' | 'cash'
    daily_regimes     = []   # 'BULL' | 'CAUTION' | 'BEAR' — one per trading day
    spy_blocked_dates = []   # BEAR rebalance dates → red chart shading
    caution_dates     = []   # CAUTION rebalance dates → yellow chart shading
    trades            = []
    day_counter       = 0
    current_regime    = None  # updated each rebalance

    for day in all_days:
        sim_date = _to_date(day)

        # ── [R6] Update trailing high for momentum positions (when enabled) ─────
        if refined and USE_TRAILING_STOP:
            for tk, pos in positions.items():
                if pos.get('pos_type') != 'momentum':
                    continue
                if tk in price_data.columns:
                    px_today = price_data[tk].get(day)
                    if px_today is not None and not pd.isna(px_today):
                        pos['high_since_entry'] = max(
                            pos.get('high_since_entry', pos['entry_price']),
                            float(px_today)
                        )

        # ── Mark portfolio to market ──────────────────────────────────────────
        mkt_value = 0.0
        for tk, pos in positions.items():
            if tk in price_data.columns:
                px = price_data[tk].get(day)
                if px is not None and not pd.isna(px):
                    mkt_value += pos['shares'] * float(px)
        total_val = cash + mkt_value
        daily_values.append({'date': sim_date, 'value': total_val})

        # ── Rebalance logic ───────────────────────────────────────────────────
        if day_counter % REBALANCE_FREQ_DAYS == 0:
            regime = spy_regime(sim_date, price_data)
            if regime == 'BEAR':
                spy_blocked_dates.append(sim_date)
            elif regime == 'CAUTION':
                caution_dates.append(sim_date)

            # Compute MOMENTUM_UNIVERSE 12-1 median for relative score
            raw_moms = {}
            for tk in MOMENTUM_UNIVERSE:
                if tk not in price_data.columns:
                    continue
                close_tk = price_data[tk].loc[:day].dropna()
                if len(close_tk) >= 252:
                    p_12m = float(close_tk.iloc[-252])
                    p_1m  = float(close_tk.iloc[-21]) if len(close_tk) >= 21 else float(close_tk.iloc[-1])
                    if p_12m > 0:
                        raw_moms[tk] = p_1m / p_12m - 1
            universe_median = float(np.median(list(raw_moms.values()))) if raw_moms else 0.0

            # Score MOMENTUM_UNIVERSE
            scores = {}
            for tk in MOMENTUM_UNIVERSE:
                s, _ = calculate_momentum_score(tk, sim_date, price_data,
                                                earnings_data, universe_median)
                scores[tk] = s

            # ── [R3] Pre-compute vols for MOMENTUM_UNIVERSE (refined only) ───
            ticker_vols = {}
            if refined:
                for tk in MOMENTUM_UNIVERSE:
                    ticker_vols[tk] = _annualized_vol(tk, price_data, sim_date)

            # Pick score threshold for this regime
            score_threshold = (MIN_MOMENTUM_SCORE if regime == 'BULL'
                               else CAUTION_SCORE_THRESHOLD)

            # ── BULL or CAUTION → momentum mode ──────────────────────────────
            if regime in ('BULL', 'CAUTION'):

                # Transition from BEAR: exit all defensive positions
                if current_regime == 'BEAR':
                    for tk in [t for t in list(positions.keys())
                               if positions[t].get('pos_type') == 'defensive']:
                        pos = positions.pop(tk)
                        px  = price_data[tk].get(day) if tk in price_data.columns else None
                        if px is not None and not pd.isna(px):
                            cash += pos['shares'] * float(px)
                            trades.append(_make_sell_record(
                                label, tk, pos, float(px), sim_date,
                                'Regime → CAUTION' if regime == 'CAUTION'
                                else 'Regime → BULL'))

                # Exit conditions for momentum positions
                to_exit = []
                for tk, pos in positions.items():
                    if pos.get('pos_type') != 'momentum':
                        continue
                    px = price_data[tk].get(day) if tk in price_data.columns else None
                    if px is None or pd.isna(px):
                        to_exit.append((tk, 'No price data', pos['entry_price'], 0))
                        continue
                    px        = float(px)
                    days_held = (sim_date - _to_date(pos['entry_date'])).days
                    score_low = scores.get(tk, 0) < score_threshold
                    max_hold  = days_held >= MAX_HOLD_DAYS

                    stop_hit = px < pos['entry_price'] * (1 - FIXED_STOP_PCT)

                    if stop_hit or score_low or max_hold:
                        reason = ('Stop loss' if stop_hit else
                                  f'Score < {score_threshold}' if score_low else
                                  'Max hold')
                        to_exit.append((tk, reason, px, days_held))

                for tk, reason, px, _ in to_exit:
                    pos  = positions.pop(tk)
                    cash += pos['shares'] * px
                    trades.append(_make_sell_record(label, tk, pos, px, sim_date, reason))

                # Entry conditions — MOMENTUM_UNIVERSE only
                candidates = []
                for tk in MOMENTUM_UNIVERSE:
                    if tk in positions:
                        continue
                    if scores.get(tk, 0) < score_threshold:
                        continue
                    if tk not in price_data.columns:
                        continue
                    # [R3] Volatility filter — refined mode only
                    if refined:
                        v = ticker_vols.get(tk)
                        if v is not None and v >= MAX_VOLATILITY:
                            continue
                    candidates.append(tk)
                candidates.sort(key=lambda t: scores[t], reverse=True)

                slots    = MAX_POSITIONS - len(positions)
                entering = candidates[:slots]

                if entering:
                    n       = len(entering)
                    weights = {tk: 1.0 / n for tk in entering}

                    # CAUTION: deploy at 50% of available cash
                    budget = cash * 0.50 if regime == 'CAUTION' else cash

                    for tk in entering:
                        px = price_data[tk].get(day)
                        if px is None or pd.isna(px) or float(px) <= 0:
                            continue
                        px     = float(px)
                        alloc  = budget * weights[tk]
                        shares = int(alloc // px)
                        if shares <= 0:
                            continue
                        cost  = shares * px
                        cash -= cost
                        positions[tk] = {
                            'shares':           shares,
                            'entry_price':      px,
                            'entry_date':       sim_date,
                            'high_since_entry': px,
                            'pos_type':         'momentum',
                        }
                        trades.append({
                            'window':      label,
                            'ticker':      tk,
                            'action':      'BUY',
                            'entry_date':  sim_date,
                            'exit_date':   None,
                            'entry_price': px,
                            'exit_price':  None,
                            'shares':      shares,
                            'hold_days':   None,
                            'return_pct':  None,
                            'reason':      f'Score {scores[tk]} | {regime}',
                            'pos_type':    'momentum',
                        })

            # ── BEAR → defensive rotation ─────────────────────────────────────
            else:  # regime == 'BEAR'
                if DEFENSIVE_ROTATION:
                    if current_regime != 'BEAR':
                        # Transition INTO BEAR: exit all current positions
                        for tk in list(positions.keys()):
                            pos = positions.pop(tk)
                            px  = price_data[tk].get(day) if tk in price_data.columns else None
                            if px is not None and not pd.isna(px):
                                cash += pos['shares'] * float(px)
                                trades.append(_make_sell_record(
                                    label, tk, pos, float(px), sim_date, 'BEAR regime'))

                        # Buy SAFE_HAVEN_UNIVERSE with fixed weights (GLD 80% / SLV 20%)
                        sh_available = [tk for tk in SAFE_HAVEN_UNIVERSE
                                        if tk in price_data.columns]
                        if sh_available and cash > 0:
                            total_w = sum(SAFE_HAVEN_WEIGHTS.get(tk, 1.0 / len(SAFE_HAVEN_UNIVERSE))
                                          for tk in sh_available)
                            for tk in sh_available:
                                px = price_data[tk].get(day)
                                if px is None or pd.isna(px) or float(px) <= 0:
                                    continue
                                px     = float(px)
                                weight = SAFE_HAVEN_WEIGHTS.get(tk, 1.0 / len(SAFE_HAVEN_UNIVERSE)) / total_w
                                shares = int(cash * weight // px)
                                if shares <= 0:
                                    continue
                                cost  = shares * px
                                cash -= cost
                                positions[tk] = {
                                    'shares':           shares,
                                    'entry_price':      px,
                                    'entry_date':       sim_date,
                                    'high_since_entry': px,
                                    'pos_type':         'defensive',
                                }
                                trades.append({
                                    'window':      label,
                                    'ticker':      tk,
                                    'action':      'BUY',
                                    'entry_date':  sim_date,
                                    'exit_date':   None,
                                    'entry_price': px,
                                    'exit_price':  None,
                                    'shares':      shares,
                                    'hold_days':   None,
                                    'return_pct':  None,
                                    'reason':      'Safe haven rotation',
                                    'pos_type':    'defensive',
                                })

                    else:
                        # Already in BEAR — hold GLD/SLV, no stop applied.
                        pass

                else:
                    # DEFENSIVE_ROTATION = False: exit all positions → 100% cash
                    for tk in list(positions.keys()):
                        pos = positions.pop(tk)
                        px  = price_data[tk].get(day) if tk in price_data.columns else None
                        if px is not None and not pd.isna(px):
                            cash += pos['shares'] * float(px)
                            trades.append(_make_sell_record(
                                label, tk, pos, float(px), sim_date, 'SPY filter'))

            current_regime = regime

        # ── Determine daily mode and regime (post-rebalance state) ────────────
        mom_held = any(p.get('pos_type') == 'momentum'  for p in positions.values())
        def_held = any(p.get('pos_type') == 'defensive' for p in positions.values())
        if current_regime == 'BEAR' and def_held:
            daily_modes.append('defensive')
        elif mom_held:
            daily_modes.append('momentum')
        else:
            daily_modes.append('cash')

        daily_regimes.append(current_regime if current_regime else 'BULL')

        day_counter += 1

    # ── Close open positions at last price ────────────────────────────────────
    last_day = all_days[-1]
    for tk, pos in list(positions.items()):
        px = price_data[tk].get(last_day) if tk in price_data.columns else None
        if px is not None and not pd.isna(px):
            px    = float(px)
            cash += pos['shares'] * px
            trades.append(_make_sell_record(label, tk, pos, px, end_date, 'End of window'))
    positions.clear()

    # ── Metrics ───────────────────────────────────────────────────────────────
    pv        = pd.DataFrame(daily_values).set_index('date')['value']
    final_val = float(pv.iloc[-1])
    total_ret = (final_val / STARTING_CAPITAL - 1) * 100
    n_days    = len(pv)
    ann_ret   = ((final_val / STARTING_CAPITAL) ** (252 / n_days) - 1) * 100

    rolling_peak = pv.cummax()
    drawdown     = (pv - rolling_peak) / rolling_peak * 100
    max_dd       = float(drawdown.min())

    daily_rets = pv.pct_change().dropna()
    ann_vol    = float(daily_rets.std() * np.sqrt(252))
    sharpe     = (ann_ret / 100 - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0.0

    sell_trades = [t for t in trades
                   if t['action'] == 'SELL' and t['return_pct'] is not None
                   and t['window'] == label]
    n_trades    = len(sell_trades)
    win_rate    = (sum(1 for t in sell_trades if t['return_pct'] > 0) /
                   n_trades * 100) if n_trades > 0 else 0.0
    avg_hold    = (sum(t['hold_days'] for t in sell_trades if t['hold_days']) /
                   n_trades) if n_trades > 0 else 0.0

    # Time-in-mode percentages
    n_total  = len(daily_modes)
    pct_mom  = sum(1 for m in daily_modes if m == 'momentum')  / n_total * 100
    pct_def  = sum(1 for m in daily_modes if m == 'defensive') / n_total * 100
    pct_cash = sum(1 for m in daily_modes if m == 'cash')      / n_total * 100

    # Time-in-regime percentages
    n_reg      = len(daily_regimes)
    pct_bull   = sum(1 for r in daily_regimes if r == 'BULL')    / n_reg * 100
    pct_caution= sum(1 for r in daily_regimes if r == 'CAUTION') / n_reg * 100
    pct_bear   = sum(1 for r in daily_regimes if r == 'BEAR')    / n_reg * 100

    return {
        'label':             label,
        'total_ret':         round(total_ret, 2),
        'ann_ret':           round(ann_ret, 2),
        'max_drawdown':      round(max_dd, 2),
        'sharpe':            round(sharpe, 2),
        'n_trades':          n_trades,
        'win_rate':          round(win_rate, 1),
        'avg_hold_days':     round(avg_hold, 1),
        'pct_momentum':      round(pct_mom, 1),
        'pct_safe_haven':    round(pct_def, 1),
        'pct_cash':          round(pct_cash, 1),
        'pct_bull':          round(pct_bull, 1),
        'pct_caution':       round(pct_caution, 1),
        'pct_bear':          round(pct_bear, 1),
        'daily_values':      pv,
        'drawdown':          drawdown,
        'spy_blocked_dates': spy_blocked_dates,
        'caution_dates':     caution_dates,
        'trades':            trades,
    }


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 — buy_and_hold benchmark (parameterised universe)
# ──────────────────────────────────────────────────────────────────────────────
def buy_and_hold(start_date, end_date, price_data, label, universe=None):
    if universe is None:
        universe = MOMENTUM_UNIVERSE
    start_date = _to_date(start_date)
    end_date   = _to_date(end_date)
    window     = price_data.loc[start_date:end_date]
    if window.empty:
        return None

    first_row = window.iloc[0]
    available = [t for t in universe
                 if t in price_data.columns and not pd.isna(first_row.get(t))]
    if not available:
        return None

    alloc    = STARTING_CAPITAL / len(available)
    holdings = {}
    cash_rem = 0.0
    for tk in available:
        px = float(first_row[tk])
        shares        = int(alloc // px)
        holdings[tk]  = shares
        cash_rem     += alloc - shares * px

    daily_vals = []
    for day, row in window.iterrows():
        val = cash_rem + sum(
            holdings.get(tk, 0) * float(row[tk])
            for tk in available
            if not pd.isna(row.get(tk))
        )
        daily_vals.append({'date': _to_date(day), 'value': val})

    pv        = pd.DataFrame(daily_vals).set_index('date')['value']
    final_val = float(pv.iloc[-1])
    total_ret = (final_val / STARTING_CAPITAL - 1) * 100
    n_days    = len(pv)
    ann_ret   = ((final_val / STARTING_CAPITAL) ** (252 / n_days) - 1) * 100

    rolling_peak = pv.cummax()
    drawdown     = (pv - rolling_peak) / rolling_peak * 100
    max_dd       = float(drawdown.min())

    daily_rets = pv.pct_change().dropna()
    ann_vol    = float(daily_rets.std() * np.sqrt(252))
    sharpe     = (ann_ret / 100 - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0.0

    return {
        'label':        label,
        'total_ret':    round(total_ret, 2),
        'ann_ret':      round(ann_ret, 2),
        'max_drawdown': round(max_dd, 2),
        'sharpe':       round(sharpe, 2),
        'daily_values': pv,
        'drawdown':     drawdown,
    }


# ──────────────────────────────────────────────────────────────────────────────
# OUTPUT — comparison table
# ──────────────────────────────────────────────────────────────────────────────
def print_comparison_table(is_orig, is_ref, os_orig, os_ref,
                           is_bh_mom, os_bh_mom,
                           is_bh_def, os_bh_def):
    def fv(v, fmt='{:>+.2f}%'):
        if isinstance(v, str):
            return v
        try:
            return fmt.format(v)
        except Exception:
            return str(v)

    def row(label, a, b, c, d, fmt='{:>+.2f}%'):
        print(f"  {label:<34} {fv(a,fmt):>10}  {fv(b,fmt):>10}"
              f"  {fv(c,fmt):>10}  {fv(d,fmt):>10}")

    sep = '─' * 96
    print()
    print(f"  {sep}")
    print(f"  {'':34} {'── IN-SAMPLE ──':>23}  {'── OUT-OF-SAMPLE ──':>23}")
    print(f"  {'METRIC':<34} {'Original':>10}  {'Refined':>10}"
          f"  {'Original':>10}  {'Refined':>10}")
    print(f"  {sep}")

    row('Strategy Total Return',
        is_orig['total_ret'],    is_ref['total_ret'],
        os_orig['total_ret'],    os_ref['total_ret'])
    row('Strategy Annualised Return',
        is_orig['ann_ret'],      is_ref['ann_ret'],
        os_orig['ann_ret'],      os_ref['ann_ret'])
    row('Strategy Max Drawdown',
        is_orig['max_drawdown'], is_ref['max_drawdown'],
        os_orig['max_drawdown'], os_ref['max_drawdown'])
    row('Strategy Sharpe',
        is_orig['sharpe'],       is_ref['sharpe'],
        os_orig['sharpe'],       os_ref['sharpe'],
        fmt='{:>+.2f}')
    row('Number of Trades',
        is_orig['n_trades'],     is_ref['n_trades'],
        os_orig['n_trades'],     os_ref['n_trades'],
        fmt='{:>}')
    row('Win Rate',
        is_orig['win_rate'],     is_ref['win_rate'],
        os_orig['win_rate'],     os_ref['win_rate'],
        fmt='{:>.1f}%')
    row('Avg Holding Days',
        is_orig['avg_hold_days'], is_ref['avg_hold_days'],
        os_orig['avg_hold_days'], os_ref['avg_hold_days'],
        fmt='{:>.1f}')

    print(f"  {'─'*96}")
    print(f"  {'TIME ALLOCATION':34}")
    row('% Time in Momentum Positions',
        f"{is_orig['pct_momentum']:.1f}%", f"{is_ref['pct_momentum']:.1f}%",
        f"{os_orig['pct_momentum']:.1f}%", f"{os_ref['pct_momentum']:.1f}%",
        fmt='{}')
    row('% Time in Safe Haven Positions',
        f"{is_orig['pct_safe_haven']:.1f}%", f"{is_ref['pct_safe_haven']:.1f}%",
        f"{os_orig['pct_safe_haven']:.1f}%", f"{os_ref['pct_safe_haven']:.1f}%",
        fmt='{}')
    row('% Time in Pure Cash',
        f"{is_orig['pct_cash']:.1f}%", f"{is_ref['pct_cash']:.1f}%",
        f"{os_orig['pct_cash']:.1f}%", f"{os_ref['pct_cash']:.1f}%",
        fmt='{}')

    print(f"  {'─'*96}")
    print(f"  {'REGIME DISTRIBUTION (SPY filter)':34}")
    row('% Time in BULL Regime',
        f"{is_orig['pct_bull']:.1f}%",    f"{is_ref['pct_bull']:.1f}%",
        f"{os_orig['pct_bull']:.1f}%",    f"{os_ref['pct_bull']:.1f}%",
        fmt='{}')
    row('% Time in CAUTION Regime',
        f"{is_orig['pct_caution']:.1f}%", f"{is_ref['pct_caution']:.1f}%",
        f"{os_orig['pct_caution']:.1f}%", f"{os_ref['pct_caution']:.1f}%",
        fmt='{}')
    row('% Time in BEAR Regime',
        f"{is_orig['pct_bear']:.1f}%",    f"{is_ref['pct_bear']:.1f}%",
        f"{os_orig['pct_bear']:.1f}%",    f"{os_ref['pct_bear']:.1f}%",
        fmt='{}')

    print(f"  {'─'*96}")
    row('Mom B&H Total Return',
        is_bh_mom['total_ret'],  '—', os_bh_mom['total_ret'],  '—')
    row('Mom B&H Max Drawdown',
        is_bh_mom['max_drawdown'], '—', os_bh_mom['max_drawdown'], '—')
    row('Safe Haven B&H Total Return',
        is_bh_def['total_ret'],  '—', os_bh_def['total_ret'],  '—')
    row('Safe Haven B&H Max Drawdown',
        is_bh_def['max_drawdown'], '—', os_bh_def['max_drawdown'], '—')

    row('Alpha vs Mom B&H (Original)',
        round(is_orig['total_ret'] - is_bh_mom['total_ret'], 2),
        round(is_ref['total_ret']  - is_bh_mom['total_ret'], 2),
        round(os_orig['total_ret'] - os_bh_mom['total_ret'], 2),
        round(os_ref['total_ret']  - os_bh_mom['total_ret'], 2))
    print(f"  {sep}")
    print(f"  In-sample:     {INSAMPLE_START}  →  {INSAMPLE_END}")
    print(f"  Out-of-sample: {OUTSAMPLE_START}  →  {OUTSAMPLE_END}")
    def_mode = 'Safe haven rotation (GLD 80% / SLV 20%)' if DEFENSIVE_ROTATION else 'Cash only'
    print(f"  SPY filter:    3-regime (BULL/CAUTION/BEAR) | 200MA + 10-day slope  |  {def_mode}")
    print(f"  BULL: score ≥ {MIN_MOMENTUM_SCORE}, full sizing  |  CAUTION: score ≥ {CAUTION_SCORE_THRESHOLD}, 50% sizing  |  "
          f"BEAR: GLD/SLV rotation")
    print(f"  RSI_ROC signal: +1 if >1.5 | -1 if <-1.0 (capped ±1)  |  "
          f"Fixed stop {int(FIXED_STOP_PCT*100)}% (mom only)  |  Refined adds: [R3] vol ≤ {int(MAX_VOLATILITY*100)}%")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# OUTPUT — trade log
# ──────────────────────────────────────────────────────────────────────────────
def print_trade_log(is_results, os_results, mode='Refined'):
    all_sells = [t for t in is_results['trades'] + os_results['trades']
                 if t['action'] == 'SELL' and t['return_pct'] is not None]
    all_sells.sort(key=lambda t: (t['window'], t['entry_date']))

    print()
    print(f"  {'─'*108}")
    print(f"  TRADE LOG ({mode})  |  {len(all_sells)} completed trades across both windows")
    print(f"  {'─'*108}")
    print(f"  {'Window':<14} {'Ticker':<7} {'Type':<10} {'Entry':>11} {'Exit':>11} "
          f"{'Entry $':>8} {'Exit $':>8} {'Hold':>5} {'Return':>8}  Reason")
    print(f"  {'─'*108}")
    for t in all_sells:
        ed   = t['entry_date'].strftime('%Y-%m-%d') if hasattr(t['entry_date'], 'strftime') else str(t['entry_date'])
        xd   = t['exit_date'].strftime('%Y-%m-%d')  if hasattr(t['exit_date'],  'strftime') else str(t['exit_date'])
        ret  = t['return_pct']
        ptype = t.get('pos_type', 'momentum')
        print(f"  {t['window']:<14} {t['ticker']:<7} {ptype:<10} {ed:>11} {xd:>11} "
              f"${t['entry_price']:>7.2f} ${t['exit_price']:>7.2f} "
              f"{t['hold_days']:>4}d {ret:>+7.1f}%  "
              f"{'▲' if ret > 0 else '▼'}  {t['reason']}")
    print(f"  {'─'*108}")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 7 — Charts: 2×2 (IS orig | IS refined | OS orig | OS refined)
# Each panel: strategy + momentum B&H + defensive B&H + SPY
#   Red shading   = BEAR regime periods
#   Yellow shading = CAUTION regime periods
# ──────────────────────────────────────────────────────────────────────────────
def _spy_blocked_spans(blocked_dates):
    if not blocked_dates:
        return []
    spans = []
    dates = sorted(blocked_dates)
    s = prev = dates[0]
    for d in dates[1:]:
        if (d - prev).days > 10:
            spans.append((s, prev))
            s = d
        prev = d
    spans.append((s, prev))
    return spans


_REFINEMENTS_TEXT = (
    'Fixed stop 15% (mom only)\n'
    'Equal-weight sizing\n'
    'Score ≥ 4 (BULL) / ≥ 3 (CAUTION)\n'
    'RSI_ROC signal (±1 cap)\n'
    '[R3] Vol ≤ 50% filter (mom)'
)
_ORIGINAL_TEXT = (
    'Fixed stop 15% (mom only)\n'
    'Equal-weight sizing\n'
    'Score ≥ 4 (BULL) / ≥ 3 (CAUTION)\n'
    'RSI_ROC signal (±1 cap)\n'
    'No vol filter'
)


def create_charts(is_orig, is_ref, os_orig, os_ref,
                  is_bh_mom, os_bh_mom,
                  is_bh_def, os_bh_def,
                  price_data):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    fig.suptitle(
        'Momentum Portfolio — Original vs Refined  |  In-Sample & Out-of-Sample\n'
        '(3-Regime SPY filter: BULL/CAUTION/BEAR · GLD 80%/SLV 20% safe haven · RSI_ROC signal)',
        fontsize=12, fontweight='bold', y=0.99
    )

    panels = [
        (axes[0, 0], is_orig,  is_bh_mom, is_bh_def, 'In-Sample — Original',
         INSAMPLE_START,  INSAMPLE_END,  False),
        (axes[0, 1], is_ref,   is_bh_mom, is_bh_def, 'In-Sample — Refined',
         INSAMPLE_START,  INSAMPLE_END,  True),
        (axes[1, 0], os_orig,  os_bh_mom, os_bh_def, 'Out-of-Sample — Original',
         OUTSAMPLE_START, OUTSAMPLE_END, False),
        (axes[1, 1], os_ref,   os_bh_mom, os_bh_def, 'Out-of-Sample — Refined',
         OUTSAMPLE_START, OUTSAMPLE_END, True),
    ]

    for ax, strat, bh_mom, bh_def, title, s_start, s_end, is_refined in panels:
        bear_spans    = _spy_blocked_spans(strat['spy_blocked_dates'])
        caution_spans = _spy_blocked_spans(strat.get('caution_dates', []))
        spy_norm = None
        if 'SPY' in price_data.columns:
            spy_window = price_data['SPY'].loc[_to_date(s_start):_to_date(s_end)].dropna()
            if not spy_window.empty:
                spy_norm = spy_window / float(spy_window.iloc[0]) * STARTING_CAPITAL

        ax.plot(strat['daily_values'].index, strat['daily_values'].values,
                color='steelblue', lw=1.6,
                label=f"Strategy ({strat['total_ret']:+.1f}%)")
        ax.plot(bh_mom['daily_values'].index, bh_mom['daily_values'].values,
                color='gray', lw=1.2, linestyle='--',
                label=f"Mom B&H ({bh_mom['total_ret']:+.1f}%)")
        ax.plot(bh_def['daily_values'].index, bh_def['daily_values'].values,
                color='forestgreen', lw=1.2, linestyle='-.',
                label=f"Safe Haven B&H ({bh_def['total_ret']:+.1f}%)")
        if spy_norm is not None:
            ax.plot(spy_norm.index, spy_norm.values,
                    color='darkorange', lw=1.0, linestyle=':',
                    alpha=0.8, label='SPY (indexed)')
        ax.axhline(STARTING_CAPITAL, color='black', lw=0.7, linestyle=':', alpha=0.5)

        # BEAR shading (red) and CAUTION shading (yellow)
        for span_s, span_e in bear_spans:
            ax.axvspan(pd.Timestamp(span_s), pd.Timestamp(span_e),
                       color='#ffcccc', alpha=0.45, zorder=0)
        for span_s, span_e in caution_spans:
            ax.axvspan(pd.Timestamp(span_s), pd.Timestamp(span_e),
                       color='#fff3cc', alpha=0.55, zorder=0)

        bear_patch    = Patch(color='#ffcccc', alpha=0.6, label='BEAR regime')
        caution_patch = Patch(color='#fff3cc', alpha=0.7, label='CAUTION regime')
        handles, labels_ = ax.get_legend_handles_labels()
        ax.legend(handles + [bear_patch, caution_patch],
                  labels_ + ['BEAR regime', 'CAUTION regime'],
                  fontsize=6.5, loc='upper left')

        # Refinements annotation (bottom-right)
        note_text  = _REFINEMENTS_TEXT if is_refined else _ORIGINAL_TEXT
        face_color = '#e8f4fd' if is_refined else '#f5f5f5'
        edge_color = 'steelblue' if is_refined else 'gray'
        ax.text(0.99, 0.03, note_text,
                transform=ax.transAxes, fontsize=6.5,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=face_color,
                          edgecolor=edge_color, alpha=0.85))

        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('Date', fontsize=8)
        ax.set_ylabel('Portfolio Value (USD)', fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    out_path = 'momentum_backtesting/capital_preservation_charts.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Chart saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print('=' * 72)
    print('  MOMENTUM PORTFOLIO BACKTEST — ORIGINAL vs REFINED')
    print(f'  In-sample:     {INSAMPLE_START}  →  {INSAMPLE_END}')
    print(f'  Out-of-sample: {OUTSAMPLE_START}  →  {OUTSAMPLE_END}')
    def_mode = 'Safe haven rotation (GLD 80% / SLV 20%)' if DEFENSIVE_ROTATION \
               else 'Cash only'
    print(f'  SPY filter:    3-regime (BULL/CAUTION/BEAR) | 200MA + 10-day slope  |  {def_mode}')
    print(f'  RSI_ROC:       +1 if >1.5 | -1 if <-1.0 (capped ±1)')
    print('=' * 72)

    # ── STEP 1: Download price data ───────────────────────────────────────────
    all_tickers = list(dict.fromkeys(
        MOMENTUM_UNIVERSE + SAFE_HAVEN_UNIVERSE + ['SPY']
    ))
    print(f'\n  Downloading price data for {len(all_tickers)} tickers...')
    raw = yf.download(all_tickers, start=DATA_DOWNLOAD_START, end=INSAMPLE_END,
                      auto_adjust=True, progress=False)
    price_data = raw['Close'].copy()
    price_data.index = pd.to_datetime(price_data.index)
    print(f'  Price data: {len(price_data)} trading days '
          f'({price_data.index[0].date()} → {price_data.index[-1].date()})')

    print(f'  Downloading earnings history for {len(MOMENTUM_UNIVERSE)} tickers '
          f'(momentum universe)...')
    earnings_data = {}
    for tk in MOMENTUM_UNIVERSE:
        try:
            eh = yf.Ticker(tk).earnings_history
            if eh is not None and not eh.empty:
                earnings_data[tk] = eh.sort_index()
        except Exception:
            pass
    print(f'  Earnings data loaded for {len(earnings_data)} tickers.')

    # ── Run four backtests ────────────────────────────────────────────────────
    print('\n  Running in-sample backtest (original)...')
    is_orig = run_backtest(INSAMPLE_START, INSAMPLE_END,
                           price_data, earnings_data, 'In-Sample',
                           refined=False)

    print('  Running in-sample backtest (refined)...')
    is_ref  = run_backtest(INSAMPLE_START, INSAMPLE_END,
                           price_data, earnings_data, 'In-Sample',
                           refined=True)

    print('  Running out-of-sample backtest (original)...')
    os_orig = run_backtest(OUTSAMPLE_START, OUTSAMPLE_END,
                           price_data, earnings_data, 'Out-of-Sample',
                           refined=False)

    print('  Running out-of-sample backtest (refined)...')
    os_ref  = run_backtest(OUTSAMPLE_START, OUTSAMPLE_END,
                           price_data, earnings_data, 'Out-of-Sample',
                           refined=True)

    print('  Computing buy-and-hold benchmarks...')
    is_bh_mom  = buy_and_hold(INSAMPLE_START,  INSAMPLE_END,  price_data,
                               'In-Sample',   universe=MOMENTUM_UNIVERSE)
    os_bh_mom  = buy_and_hold(OUTSAMPLE_START, OUTSAMPLE_END, price_data,
                               'Out-of-Sample', universe=MOMENTUM_UNIVERSE)
    is_bh_def  = buy_and_hold(INSAMPLE_START,  INSAMPLE_END,  price_data,
                               'In-Sample',   universe=SAFE_HAVEN_UNIVERSE)
    os_bh_def  = buy_and_hold(OUTSAMPLE_START, OUTSAMPLE_END, price_data,
                               'Out-of-Sample', universe=SAFE_HAVEN_UNIVERSE)

    # ── Comparison table ──────────────────────────────────────────────────────
    print_comparison_table(is_orig, is_ref, os_orig, os_ref,
                           is_bh_mom, os_bh_mom,
                           is_bh_def, os_bh_def)

    # ── Trade log (refined) ───────────────────────────────────────────────────
    print_trade_log(is_ref, os_ref, mode='Refined')

    # ── Charts ────────────────────────────────────────────────────────────────
    create_charts(is_orig, is_ref, os_orig, os_ref,
                  is_bh_mom, os_bh_mom,
                  is_bh_def, os_bh_def,
                  price_data)


if __name__ == '__main__':
    main()
