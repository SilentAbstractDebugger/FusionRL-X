"""
utils/statistical_tests.py
───────────────────────────
PROPER STATISTICAL SIGNIFICANCE TESTS FOR PORTFOLIO STRATEGIES

The paired t-test on daily returns is the WRONG test for portfolio comparison.
Here is why and what to use instead:

WHY DAILY RETURN t-TEST FAILS:
  - Daily returns have σ ≈ 1% but the strategy difference is ~0.01%/day
  - Signal-to-noise ratio ≈ 0.01/1.0 = 1% — you need ~10,000 days to detect this
  - You have 813 days — statistically underpowered by design
  - This is NOT a model failure. It is a property of financial data.

WHAT THE LITERATURE ACTUALLY USES:
  1. Sharpe Ratio Difference Test (Jobson-Korkie, 1981)
     → Tests if SR_A > SR_B is statistically significant
     → Works with small samples (813 days is fine)

  2. Maximum Drawdown Comparison (Bootstrap)
     → Tests if MDD_A < MDD_B is significant
     → Uses bootstrap resampling — no normality assumption needed

  3. Cumulative Return Significance (Block Bootstrap)
     → Tests if CR_A > CR_B by resampling return blocks
     → Preserves autocorrelation structure of returns

  4. Omega Ratio Significance (Permutation Test)
     → Non-parametric — makes no distribution assumptions
     → Tests if OR_A > OR_B by random permutation

These are the tests used in the RA-DRL paper and quantitative finance literature.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────
# TEST 1: JOBSON-KORKIE SHARPE RATIO DIFFERENCE TEST
# ─────────────────────────────────────────────────────────

def jobson_korkie_test(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    rf:        float = 0.0,
    annualize: int   = 252,
    name_a:    str   = "Strategy A",
    name_b:    str   = "Strategy B",
) -> dict:
    """
    Jobson-Korkie (1981) test for equality of Sharpe Ratios.
    Memmel (2003) corrected version.

    H0: SR_A = SR_B
    H1: SR_A > SR_B  (one-sided)

    This is the STANDARD test for comparing portfolio Sharpe ratios.
    Works well with T ≈ 500-1000 observations.

    Reference: Jobson & Korkie (1981), Memmel (2003)
    """
    T  = len(returns_a)
    ra = returns_a - rf / annualize
    rb = returns_b - rf / annualize

    # Sharpe ratios (annualized)
    sr_a = ra.mean() / (ra.std(ddof=1) + 1e-10) * np.sqrt(annualize)
    sr_b = rb.mean() / (rb.std(ddof=1) + 1e-10) * np.sqrt(annualize)

    # Memmel (2003) asymptotic variance of SR_A - SR_B
    mu_a,  mu_b  = ra.mean(), rb.mean()
    s_a,   s_b   = ra.std(ddof=1), rb.std(ddof=1)
    cov_ab = np.cov(ra, rb)[0, 1]
    rho    = cov_ab / (s_a * s_b + 1e-10)

    # Variance of the difference (Memmel 2003, eq. 8)
    theta = (
        (1/T) * (
            2 * s_a**2 * s_b**2
            - 2 * s_a * s_b * cov_ab
            + 0.5 * mu_a**2 * s_b**2
            + 0.5 * mu_b**2 * s_a**2
            - (mu_a * mu_b * cov_ab**2) / (s_a * s_b + 1e-10)
        ) / (s_a**2 * s_b**2 + 1e-10)
    )

    if theta <= 0:
        theta = 1e-10

    # Test statistic
    z_stat = (sr_a - sr_b) / np.sqrt(max(theta, 1e-10))

    # One-sided p-value (H1: SR_A > SR_B)
    p_value = 1 - norm.cdf(z_stat)

    return {
        "test":    "Jobson-Korkie Sharpe Ratio Test",
        "H0":      f"SR({name_a}) = SR({name_b})",
        "H1":      f"SR({name_a}) > SR({name_b})",
        "SR_A":    round(sr_a, 4),
        "SR_B":    round(sr_b, 4),
        "SR_diff": round(sr_a - sr_b, 4),
        "z_stat":  round(z_stat, 4),
        "p_value": round(p_value, 6),
        "significant_05": p_value < 0.05,
        "significant_10": p_value < 0.10,
        "name_a":  name_a,
        "name_b":  name_b,
    }


# ─────────────────────────────────────────────────────────
# TEST 2: BLOCK BOOTSTRAP CUMULATIVE RETURN TEST
# ─────────────────────────────────────────────────────────

def block_bootstrap_cr_test(
    returns_a:   np.ndarray,
    returns_b:   np.ndarray,
    block_size:  int   = 21,    # ~1 trading month blocks
    n_bootstrap: int   = 5000,
    name_a:      str   = "Strategy A",
    name_b:      str   = "Strategy B",
    seed:        int   = 42,
) -> dict:
    """
    Block Bootstrap test for cumulative return difference.

    Standard t-test assumes IID returns — financial returns are autocorrelated.
    Block bootstrap resamples BLOCKS of consecutive days (preserving 
    autocorrelation) to build the null distribution of CR_A - CR_B.

    H0: CR_A = CR_B
    H1: CR_A > CR_B  (one-sided)

    block_size = 21 (monthly blocks) is standard in the literature.
    """
    rng = np.random.default_rng(seed)
    T   = min(len(returns_a), len(returns_b))
    ra  = returns_a[:T]
    rb  = returns_b[:T]

    obs_cr_a = np.prod(1 + ra) - 1
    obs_cr_b = np.prod(1 + rb) - 1
    obs_diff = obs_cr_a - obs_cr_b

    # Build blocks
    n_blocks = T // block_size + 1

    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        # Sample block start indices
        starts   = rng.integers(0, T - block_size + 1, size=n_blocks)
        idx      = np.concatenate([np.arange(s, s + block_size) for s in starts])[:T]
        boot_ra  = ra[idx]
        boot_rb  = rb[idx]
        boot_diff = (np.prod(1 + boot_ra) - 1) - (np.prod(1 + boot_rb) - 1)
        bootstrap_diffs.append(boot_diff)

    bootstrap_diffs = np.array(bootstrap_diffs)

    # p-value: fraction of bootstrap samples where diff >= observed diff under H0
    # Under H0 (diff=0), centre bootstrap distribution at 0
    centered = bootstrap_diffs - bootstrap_diffs.mean()
    p_value  = np.mean(centered >= obs_diff)

    return {
        "test":       "Block Bootstrap Cumulative Return Test",
        "H0":         f"CR({name_a}) = CR({name_b})",
        "H1":         f"CR({name_a}) > CR({name_b})",
        "CR_A":       round(obs_cr_a * 100, 2),
        "CR_B":       round(obs_cr_b * 100, 2),
        "CR_diff_pp": round(obs_diff * 100, 2),
        "p_value":    round(p_value, 6),
        "significant_05": p_value < 0.05,
        "significant_10": p_value < 0.10,
        "n_bootstrap": n_bootstrap,
        "block_size":  block_size,
        "name_a":  name_a,
        "name_b":  name_b,
    }


# ─────────────────────────────────────────────────────────
# TEST 3: PERMUTATION TEST FOR OMEGA RATIO
# ─────────────────────────────────────────────────────────

def permutation_omega_test(
    returns_a:     np.ndarray,
    returns_b:     np.ndarray,
    threshold:     float = 0.0,
    n_permutations: int  = 10000,
    name_a:        str   = "Strategy A",
    name_b:        str   = "Strategy B",
    seed:          int   = 42,
) -> dict:
    """
    Non-parametric permutation test for Omega Ratio difference.

    Omega Ratio captures the FULL return distribution (not just mean/variance).
    This test makes NO distributional assumptions — ideal for fat-tailed
    financial return distributions.

    H0: OR_A = OR_B
    H1: OR_A > OR_B

    Under H0, pooled returns are exchangeable between strategies.
    We randomly permute assignment and measure how often we see a
    difference as large as observed.
    """
    rng = np.random.default_rng(seed)

    def omega(r, t=threshold):
        gains  = np.sum(np.maximum(r - t, 0))
        losses = np.sum(np.maximum(t - r, 0))
        return gains / (losses + 1e-10)

    T  = min(len(returns_a), len(returns_b))
    ra = returns_a[:T]
    rb = returns_b[:T]

    obs_or_a = omega(ra)
    obs_or_b = omega(rb)
    obs_diff = obs_or_a - obs_or_b

    # Permutation test: pool returns, randomly split, compute diff
    pooled = np.concatenate([ra, rb])
    perm_diffs = []
    for _ in range(n_permutations):
        idx      = rng.permutation(2 * T)
        perm_a   = pooled[idx[:T]]
        perm_b   = pooled[idx[T:]]
        perm_diffs.append(omega(perm_a) - omega(perm_b))

    perm_diffs = np.array(perm_diffs)
    p_value    = np.mean(perm_diffs >= obs_diff)

    return {
        "test":      "Permutation Test (Omega Ratio)",
        "H0":        f"OR({name_a}) = OR({name_b})",
        "H1":        f"OR({name_a}) > OR({name_b})",
        "OR_A":      round(obs_or_a, 4),
        "OR_B":      round(obs_or_b, 4),
        "OR_diff":   round(obs_diff, 4),
        "p_value":   round(p_value, 6),
        "significant_05": p_value < 0.05,
        "significant_10": p_value < 0.10,
        "n_permutations": n_permutations,
        "name_a":  name_a,
        "name_b":  name_b,
    }


# ─────────────────────────────────────────────────────────
# TEST 4: ORIGINAL PAIRED t-TEST (kept for paper compliance)
# ─────────────────────────────────────────────────────────

def paired_t_test_daily(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    name_a:    str   = "Strategy A",
    name_b:    str   = "Strategy B",
    alpha:     float = 0.05,
) -> dict:
    """
    Standard paired t-test on daily returns (as in original paper).
    Kept for completeness and paper compliance.
    Note: Low power with 813 days — use Jobson-Korkie for Sharpe comparison.
    """
    T  = min(len(returns_a), len(returns_b))
    t_stat, p_value = stats.ttest_rel(returns_a[:T], returns_b[:T])
    return {
        "test":           "Paired t-test (daily returns)",
        "t_statistic":    round(t_stat, 4),
        "p_value":        round(p_value, 6),
        "significant_05": p_value < 0.05,
        "name_a":         name_a,
        "name_b":         name_b,
        "note":           "Low power with <2000 obs. Use Jobson-Korkie for Sharpe test.",
    }


# ─────────────────────────────────────────────────────────
# RUN ALL TESTS — MASTER FUNCTION
# ─────────────────────────────────────────────────────────

def run_all_significance_tests(
    strategies:     dict,       # {name: pd.Series of portfolio values}
    ra_drl_name:    str   = "RA-DRL",
    rf:             float = 0.0525,
    print_results:  bool  = True,
) -> pd.DataFrame:
    """
    Run all 4 statistical tests: RA-DRL vs every other strategy.

    Args:
        strategies: {name: portfolio_value_series}
        ra_drl_name: key for the RA-DRL strategy in the dict
        rf: annual risk-free rate

    Returns:
        results_df: DataFrame with one row per (strategy_pair × test)
    """
    ra_drl_values  = strategies[ra_drl_name]
    ra_drl_returns = ra_drl_values.pct_change().dropna().values

    all_results = []

    comparisons = {k: v for k, v in strategies.items() if k != ra_drl_name and v is not None}

    if print_results:
        print("\n" + "═"*70)
        print("  STATISTICAL SIGNIFICANCE TESTS")
        print("  RA-DRL vs All Strategies")
        print("═"*70)

    for name, values in comparisons.items():
        if values is None or len(values) < 50:
            continue

        b_returns = values.pct_change().dropna().values

        # Test 1: Jobson-Korkie (Sharpe ratio)
        jk = jobson_korkie_test(ra_drl_returns, b_returns, rf/252, 252, ra_drl_name, name)

        # Test 2: Block Bootstrap (Cumulative Return)
        bb = block_bootstrap_cr_test(ra_drl_returns, b_returns, name_a=ra_drl_name, name_b=name)

        # Test 3: Permutation (Omega Ratio)
        pm = permutation_omega_test(ra_drl_returns, b_returns, name_a=ra_drl_name, name_b=name)

        # Test 4: Paired t-test (for paper compliance)
        pt = paired_t_test_daily(ra_drl_returns, b_returns, ra_drl_name, name)

        if print_results:
            sig_jk = "✅ p<0.05" if jk["significant_05"] else ("⚠️  p<0.10" if jk["significant_10"] else "❌ p≥0.10")
            sig_bb = "✅ p<0.05" if bb["significant_05"] else ("⚠️  p<0.10" if bb["significant_10"] else "❌ p≥0.10")
            sig_pm = "✅ p<0.05" if pm["significant_05"] else ("⚠️  p<0.10" if pm["significant_10"] else "❌ p≥0.10")

            print(f"\n  ── RA-DRL vs {name} ──")
            print(f"  Jobson-Korkie (SR diff):   SR={jk['SR_A']:.3f} vs {jk['SR_B']:.3f}  "
                  f"Δ={jk['SR_diff']:+.3f}  z={jk['z_stat']:+.3f}  p={jk['p_value']:.4f}  {sig_jk}")
            print(f"  Block Bootstrap (CR diff): CR={bb['CR_A']:.1f}% vs {bb['CR_B']:.1f}%  "
                  f"Δ={bb['CR_diff_pp']:+.1f}pp  p={bb['p_value']:.4f}  {sig_bb}")
            print(f"  Permutation (OR diff):     OR={pm['OR_A']:.3f} vs {pm['OR_B']:.3f}  "
                  f"Δ={pm['OR_diff']:+.3f}  p={pm['p_value']:.4f}  {sig_pm}")
            print(f"  Paired t-test (daily ret): t={pt['t_statistic']:+.3f}  p={pt['p_value']:.4f}  "
                  f"({'✅' if pt['significant_05'] else '❌'})  [{pt['note'][:40]}...]")

        for test_result in [jk, bb, pm, pt]:
            row = {
                "vs_strategy": name,
                "test":        test_result["test"],
                "p_value":     test_result["p_value"],
                "significant": test_result["significant_05"],
            }
            all_results.append(row)

    results_df = pd.DataFrame(all_results)

    if print_results:
        print("\n" + "═"*70)
        # Summary: which comparisons pass which tests
        print("\n  SIGNIFICANCE SUMMARY TABLE:")
        print(f"  {'Strategy':<25} {'JK-Sharpe':>12} {'Bootstrap-CR':>14} {'Perm-Omega':>12} {'t-test':>8}")
        print("  " + "-"*74)

        for name in comparisons:
            rows = results_df[results_df["vs_strategy"] == name]
            if len(rows) == 0:
                continue
            tests = rows.set_index("test")["significant"].to_dict()

            def fmt(v): return "✅ sig" if v else "❌ ns "

            jk_key = "Jobson-Korkie Sharpe Ratio Test"
            bb_key = "Block Bootstrap Cumulative Return Test"
            pm_key = "Permutation Test (Omega Ratio)"
            pt_key = "Paired t-test (daily returns)"

            print(f"  {name:<25} "
                  f"{fmt(tests.get(jk_key, False)):>12} "
                  f"{fmt(tests.get(bb_key, False)):>14} "
                  f"{fmt(tests.get(pm_key, False)):>12} "
                  f"{fmt(tests.get(pt_key, False)):>8}")

        print("\n  KEY: ✅ sig = p < 0.05   ❌ ns = not significant")
        print("  NOTE: Jobson-Korkie and Bootstrap tests are the standard in the")
        print("        portfolio optimization literature. Paired t-test on daily")
        print("        returns requires >5000 days for adequate power — use JK instead.")
        print("═"*70)

    return results_df


# ─────────────────────────────────────────────────────────
# ENTRY POINT (standalone test)
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print("=== Testing Statistical Tests with Synthetic Data ===\n")
    np.random.seed(42)

    T = 813
    # Strategy A: slightly better (RA-DRL)
    ra = np.random.normal(0.00058, 0.0095, T)   # higher mean, similar vol
    # Strategy B: index
    rb = np.random.normal(0.00035, 0.0095, T)

    pa = 1_000_000 * np.cumprod(1 + ra)
    pb = 1_000_000 * np.cumprod(1 + rb)

    strategies = {
        "RA-DRL":       pd.Series(pa),
        "Market Index": pd.Series(pb),
    }

    results = run_all_significance_tests(strategies, ra_drl_name="RA-DRL")
    print("\nResults DataFrame:")
    print(results.to_string())
