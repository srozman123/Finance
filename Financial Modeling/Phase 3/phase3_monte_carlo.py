import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf

# ============================================================================
# STEP 1 — Hardcoded inputs (from phase123_msft_report.py, FY2025)
# ============================================================================
# Fundamental inputs (FCF, debt, cash, shares) are hardcoded from FY2025 to
# keep simulations fast and reproducible. Current price is fetched live so
# the market comparison is always up to date.

BASE_FCF      = 67.58e9   # 4-year average FCF ($67.58B)
SHARES        = 7.43e9    # shares outstanding
DEBT          = 60.59e9   # total debt
CASH          = 30.24e9   # cash & equivalents
CURRENT_PRICE = yf.Ticker("MSFT").info["currentPrice"]
print(f"  Live price fetched: ${CURRENT_PRICE:.2f}")
PROJ_YEARS    = 5
N_SIMS        = 10_000

np.random.seed(42)  # reproducibility

# ============================================================================
# STEP 2 — Input distributions
# ============================================================================
# Rather than using single point estimates (which imply false precision), we
# model each key input as a probability distribution. The width of each
# distribution reflects genuine uncertainty about the future.
#
# FCF Growth — Normal(mean=6%, std=3%)
#   MSFT's 4-year FCF CAGR was ~3%, but the Azure/AI buildout is expected to
#   accelerate returns. 6% mean reflects analyst consensus; 3% std captures
#   realistic upside/downside scenarios around that central view.
#
# WACC — Normal(mean=9%, std=1%)
#   MSFT's WACC is relatively low given its AAA credit rating and stable
#   recurring revenue (Azure, Office 365). 9% is consensus; 1% std reflects
#   uncertainty in equity risk premium and beta estimates.
#
# Terminal Growth Rate — Normal(mean=3%, std=0.5%)
#   Anchored near long-run nominal GDP growth (~2.5–3.5%). Tight std because
#   terminal growth is a long-run structural assumption that doesn't vary much
#   across reasonable scenarios — the main risk is mean-reversion, not spikes.

fcf_growth_dist = dict(loc=0.06, scale=0.03)
wacc_dist       = dict(loc=0.09, scale=0.01)
tgr_dist        = dict(loc=0.03, scale=0.005)

# ============================================================================
# STEP 3 — Run 10,000 simulations
# ============================================================================

def run_dcf(base_fcf, fcf_growth, wacc, terminal_growth):
    proj  = [base_fcf * (1 + fcf_growth) ** i for i in range(1, PROJ_YEARS + 1)]
    tv    = proj[-1] * (1 + terminal_growth) / (wacc - terminal_growth)
    pv_cf = sum(cf / (1 + wacc) ** i for i, cf in enumerate(proj, 1))
    pv_tv = tv / (1 + wacc) ** PROJ_YEARS
    eq    = (pv_cf + pv_tv) - DEBT + CASH
    return eq / SHARES

# Draw all random samples at once (vectorised → fast)
fcf_growth_samples = np.random.normal(**fcf_growth_dist, size=N_SIMS)
wacc_samples       = np.random.normal(**wacc_dist,       size=N_SIMS)
tgr_samples        = np.random.normal(**tgr_dist,        size=N_SIMS)

# Clip to prevent numerically degenerate or financially unrealistic scenarios
fcf_growth_samples = np.clip(fcf_growth_samples, -0.10,  0.20)
wacc_samples       = np.clip(wacc_samples,        0.05,   0.15)

# Terminal growth must stay at least 1% below WACC (Gordon Growth Model
# denominator WACC − g must remain positive and meaningful)
tgr_samples = np.minimum(tgr_samples, wacc_samples - 0.01)

implied_prices = np.array([
    run_dcf(BASE_FCF, g, w, tg)
    for g, w, tg in zip(fcf_growth_samples, wacc_samples, tgr_samples)
])

# ============================================================================
# STEP 3b — Normalized FCF simulation
# ============================================================================
# MSFT's reported FCF has been suppressed by a surge in CapEx ($24B → $65B
# in 4 years) as it builds out AI/Azure infrastructure. If that capex spend
# stabilizes — as management has guided — FCF should converge toward operating
# cash flow levels. $96B represents the FY2025 operating cash flow ($136B)
# minus a normalized steady-state CapEx (~$40B), implying capex settles once
# the current buildout is complete. The higher FCF growth mean (10% vs 6%)
# reflects the revenue acceleration expected from the AI investments now
# being made.

BASE_FCF_NORM   = 96e9   # normalized FCF after capex stabilization
norm_growth_dist = dict(loc=0.10, scale=0.03)  # N(mean=10%, std=3%)

norm_growth_samples = np.random.normal(**norm_growth_dist, size=N_SIMS)
norm_growth_samples = np.clip(norm_growth_samples, -0.10, 0.20)

# Reuse same WACC and terminal growth draws for a clean apples-to-apples
# comparison — only the base FCF and growth distribution change
implied_prices_norm = np.array([
    run_dcf(BASE_FCF_NORM, g, w, tg)
    for g, w, tg in zip(norm_growth_samples, wacc_samples, tgr_samples)
])

# ============================================================================
# STEP 4 — Summary statistics (both scenarios)
# ============================================================================

def print_stats(label, prices, base_fcf, growth_dist):
    mean_p   = np.mean(prices)
    median_p = np.median(prices)
    p10_p    = np.percentile(prices, 10)
    p90_p    = np.percentile(prices, 90)
    prob_up  = np.mean(prices > CURRENT_PRICE) * 100
    pct_neg  = np.mean(prices <= 0) * 100
    print(f"\n  [ {label} ]")
    print(f"    Base FCF:  ${base_fcf/1e9:.2f}B   "
          f"FCF Growth ~ N(μ={growth_dist['loc']*100:.0f}%, σ={growth_dist['scale']*100:.0f}%)")
    print(f"    {'Mean':<35} ${mean_p:>8.2f}")
    print(f"    {'Median':<35} ${median_p:>8.2f}")
    print(f"    {'10th Percentile (bear case)':<35} ${p10_p:>8.2f}")
    print(f"    {'90th Percentile (bull case)':<35} ${p90_p:>8.2f}")
    print(f"    {'P(implied > market price)':<35} {prob_up:>7.1f}%")
    print(f"    {'Simulations with price ≤ 0':<35} {pct_neg:>7.2f}%")
    return mean_p, median_p, p10_p, p90_p, prob_up

print("=" * 60)
print(f"  MONTE CARLO DCF — MSFT  ({N_SIMS:,} simulations each)")
print("=" * 60)
print(f"\n  Shared Distributions (both scenarios):")
print(f"    WACC       ~ N(μ={wacc_dist['loc']*100:.0f}%, σ={wacc_dist['scale']*100:.0f}%)  clipped [5%, 15%]")
print(f"    Terminal g ~ N(μ={tgr_dist['loc']*100:.0f}%, σ={tgr_dist['scale']*100:.1f}%)  always ≥ 1% below WACC")
print(f"\n  Market Price (live): ${CURRENT_PRICE:.2f}")

mean1, med1, p10_1, p90_1, prob1 = print_stats(
    "Historical FCF base",  implied_prices,      BASE_FCF,      fcf_growth_dist)
mean2, med2, p10_2, p90_2, prob2 = print_stats(
    "Normalized FCF base",  implied_prices_norm, BASE_FCF_NORM, norm_growth_dist)

# ============================================================================
# STEP 5 — Charts
# ============================================================================

plt.style.use("seaborn-v0_8-whitegrid")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# --- Chart 1: Overlapping histograms ---
bins = np.linspace(
    min(implied_prices.min(), implied_prices_norm.min()),
    max(implied_prices.max(), implied_prices_norm.max()),
    90
)

ax1.hist(implied_prices,      bins=bins, color="steelblue",  edgecolor="white",
         linewidth=0.3, alpha=0.65, density=True, label="Historical FCF base")
ax1.hist(implied_prices_norm, bins=bins, color="darkorange", edgecolor="white",
         linewidth=0.3, alpha=0.65, density=True, label="Normalized FCF base")

ax1.axvline(CURRENT_PRICE, color="crimson", linewidth=2,
            linestyle="-",  label=f"Market price  ${CURRENT_PRICE:.0f}")
ax1.axvline(mean1,  color="steelblue",  linewidth=1.5, linestyle=":",
            label=f"Mean (hist)  ${mean1:.0f}")
ax1.axvline(mean2,  color="darkorange", linewidth=1.5, linestyle=":",
            label=f"Mean (norm)  ${mean2:.0f}")
ax1.axvline(p90_1,  color="steelblue",  linewidth=1,   linestyle="--",
            label=f"90th pct (hist)  ${p90_1:.0f}")
ax1.axvline(p90_2,  color="darkorange", linewidth=1,   linestyle="--",
            label=f"90th pct (norm)  ${p90_2:.0f}")

ax1.set_title("Distribution of Implied Share Prices\nHistorical vs Normalized FCF Base",
              fontsize=11, fontweight="bold")
ax1.set_xlabel("Implied Share Price (USD)", fontsize=9)
ax1.set_ylabel("Density", fontsize=9)
ax1.legend(fontsize=7.5)

# --- Chart 2: Overlapping CDFs ---
sorted_hist = np.sort(implied_prices)
sorted_norm = np.sort(implied_prices_norm)
cdf         = np.arange(1, N_SIMS + 1) / N_SIMS

ax2.plot(sorted_hist, cdf * 100, color="steelblue",  linewidth=2,
         label="Historical FCF base")
ax2.plot(sorted_norm, cdf * 100, color="darkorange", linewidth=2,
         label="Normalized FCF base")
ax2.axvline(CURRENT_PRICE, color="crimson", linewidth=2, linestyle="-",
            label=f"Market price  ${CURRENT_PRICE:.0f}")

prob_below1 = 100 - prob1
prob_below2 = 100 - prob2
ax2.annotate(f"Hist: {prob_below1:.1f}% below",
             xy=(CURRENT_PRICE, prob_below1),
             xytext=(CURRENT_PRICE * 0.55, prob_below1 + 5),
             fontsize=8, color="steelblue",
             arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.1))
ax2.annotate(f"Norm: {prob_below2:.1f}% below",
             xy=(CURRENT_PRICE, prob_below2),
             xytext=(CURRENT_PRICE * 0.55, prob_below2 - 12),
             fontsize=8, color="darkorange",
             arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.1))

ax2.set_title("Cumulative Distribution — Both Scenarios\n(probability of being below a given value)",
              fontsize=11, fontweight="bold")
ax2.set_xlabel("Implied Share Price (USD)", fontsize=9)
ax2.set_ylabel("Cumulative Probability (%)", fontsize=9)
ax2.set_ylim(0, 100)
ax2.legend(fontsize=8)

fig.suptitle("MSFT — Monte Carlo DCF Valuation  (10,000 simulations per scenario)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("Phase 3/msft_monte_carlo.png", dpi=150, bbox_inches="tight")
print(f"\n  Chart saved → Phase 3/msft_monte_carlo.png")
