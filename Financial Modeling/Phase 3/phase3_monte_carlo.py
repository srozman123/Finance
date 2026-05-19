import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf

BASE_FCF      = 67.58e9
SHARES        = 7.43e9
DEBT          = 60.59e9
CASH          = 30.24e9
CURRENT_PRICE = yf.Ticker("MSFT").info["currentPrice"]
print(f"  Live price fetched: ${CURRENT_PRICE:.2f}")
PROJ_YEARS    = 5
N_SIMS        = 10_000

np.random.seed(42)

fcf_growth_dist = dict(loc=0.06, scale=0.03)
wacc_dist       = dict(loc=0.09, scale=0.01)
tgr_dist        = dict(loc=0.03, scale=0.005)

def run_dcf(base_fcf, fcf_growth, wacc, terminal_growth):
    proj  = [base_fcf * (1 + fcf_growth) ** i for i in range(1, PROJ_YEARS + 1)]
    tv    = proj[-1] * (1 + terminal_growth) / (wacc - terminal_growth)
    pv_cf = sum(cf / (1 + wacc) ** i for i, cf in enumerate(proj, 1))
    pv_tv = tv / (1 + wacc) ** PROJ_YEARS
    eq    = (pv_cf + pv_tv) - DEBT + CASH
    return eq / SHARES

fcf_growth_samples = np.random.normal(**fcf_growth_dist, size=N_SIMS)
wacc_samples       = np.random.normal(**wacc_dist,       size=N_SIMS)
tgr_samples        = np.random.normal(**tgr_dist,        size=N_SIMS)

fcf_growth_samples = np.clip(fcf_growth_samples, -0.10,  0.20)
wacc_samples       = np.clip(wacc_samples,        0.05,   0.15)

tgr_samples = np.minimum(tgr_samples, wacc_samples - 0.01)

implied_prices = np.array([
    run_dcf(BASE_FCF, g, w, tg)
    for g, w, tg in zip(fcf_growth_samples, wacc_samples, tgr_samples)
])

BASE_FCF_NORM   = 96e9
norm_growth_dist = dict(loc=0.10, scale=0.03)

norm_growth_samples = np.random.normal(**norm_growth_dist, size=N_SIMS)
norm_growth_samples = np.clip(norm_growth_samples, -0.10, 0.20)

implied_prices_norm = np.array([
    run_dcf(BASE_FCF_NORM, g, w, tg)
    for g, w, tg in zip(norm_growth_samples, wacc_samples, tgr_samples)
])

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

plt.style.use("seaborn-v0_8-whitegrid")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

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
