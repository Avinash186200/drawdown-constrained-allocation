"""
Approach 16 (out-of-the-box, different in kind from 1-15): none of the
prior approaches touched the drawdown CONSTRAINT itself -- they all
worked on the return input (mu) or the allocation algorithm. This one
asks a different question: is the 2% daily constraint, built purely
from OBSERVED historical scenarios (~737 real days), robust to
plausible joint-crash scenarios that simply haven't happened yet in
this ~3-5 year window?

Mechanism: fit a Student-t copula to the daily return panel (rank-
transform each strategy's returns to pseudo-uniforms, then to
Student-t quantiles; estimate the correlation matrix of those
quantiles). A t-copula captures TAIL DEPENDENCE -- the tendency for
extreme moves to co-occur more than a Gaussian assumption would predict
-- unlike the plain covariance matrix used elsewhere in this project,
which only captures average/linear co-movement. Sample many synthetic
joint-return scenarios from the fitted copula, transform back through
each strategy's own EMPIRICAL marginal distribution (so each
strategy's own historical return character is preserved exactly; only
the joint/tail structure is synthetic), and add these as EXTRA
scenarios alongside the real historical days in the same
historical-scenario hard constraint used everywhere else in this
project.

This isn't a return-forecasting test -- there's no Sharpe to "win."
The question is whether the existing risk control is already robust
(synthetic scenarios never bind, cost nothing) or whether there's a
plausible joint-crash pattern the real ~3-5yr sample hasn't yet shown
that would tighten the constraint meaningfully.

IMPORTANT CORRECTION: an earlier version fixed dd/lambda/caps/lookback
at constants picked because they'd recurred across other experiments'
honest-reselection outputs -- hindsight, the same mistake this project
has caught before, and it made the initial "baseline Sharpe <real figure removed> / FWD
4.64" result incomparable to the real project baseline (3.92 FWD).
Fixed by reusing the ACTUAL honest recipe schedule from
cpo_hybrid_monthly's own honest_pipeline for both runs below -- only
the scenario set (real-only vs. real+synthetic) differs, isolating the
copula augmentation itself.
"""
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from cpo_hybrid_monthly import (
    build_extended_monthly_panel, hard_constrained_weights, sharpe_stats,
    STRATEGY_COLS, honest_pipeline as baseline_honest_pipeline,
)
from cpo_audit_fixed import volq_multiplier_fixed, per_strategy_v2_multiplier_fixed

COPULA_DF = 4  # fatter tails than Gaussian; standard choice for daily financial returns
N_SYNTHETIC = 500
RNG = np.random.default_rng(7)


def get_honest_recipe_schedule(monthly_panel, daily_panel):
    _, recipes = baseline_honest_pipeline(monthly_panel, daily_panel)
    return [(pd.Timestamp(date), dd, lam, hcap, ccap, lookback) for date, dd, lam, hcap, ccap, lookback in recipes]


def recipe_for_month(schedule, month_end_date):
    applicable = [s for s in schedule if s[0] <= month_end_date]
    return applicable[-1][1:] if applicable else schedule[0][1:]


def fit_tcopula_and_sample(daily_hist, n_synthetic=N_SYNTHETIC, df=COPULA_DF):
    """daily_hist: DataFrame (rows=days, cols=strategies). Returns an
    array of synthetic daily return vectors, same columns, preserving
    each strategy's own empirical marginal distribution."""
    n, k = daily_hist.shape
    values = daily_hist.values
    ranks = np.apply_along_axis(lambda col: (pd.Series(col).rank().values) / (n + 1), 0, values)
    z = student_t.ppf(ranks, df=df)
    z = np.clip(z, -8, 8)
    corr = np.corrcoef(z, rowvar=False)
    corr = corr + np.eye(k) * 1e-6

    # sample from multivariate t via Gaussian / chi-square mixture
    L = np.linalg.cholesky(corr)
    g = RNG.standard_normal((n_synthetic, k)) @ L.T
    chi2 = RNG.chisquare(df, size=n_synthetic)
    t_sample = g / np.sqrt(chi2 / df)[:, None]
    u_sample = student_t.cdf(t_sample, df=df)

    synthetic = np.zeros_like(u_sample)
    for j in range(k):
        sorted_vals = np.sort(values[:, j])
        synthetic[:, j] = np.quantile(sorted_vals, np.clip(u_sample[:, j], 1e-4, 1 - 1e-4))
    return synthetic


def run(monthly_panel, daily_panel, schedule, augment_with_copula):
    daily_weights = {}
    recipe_notes = []
    monthly_dates = monthly_panel.index
    i = schedule[0][5]
    while i < len(monthly_dates):
        month_end_date = monthly_dates[i]
        dd, lam, hcap, ccap, monthly_lookback = recipe_for_month(schedule, month_end_date)
        if i < monthly_lookback:
            i += 1
            continue
        trailing_m = monthly_panel.iloc[i - monthly_lookback:i]
        mu = trailing_m.mean().values
        sigma = trailing_m.cov().values

        hist_daily = daily_panel[daily_panel.index < month_end_date]
        daily_scen = hist_daily.tail(252).values

        if augment_with_copula and len(hist_daily) >= 60:
            synthetic = fit_tcopula_and_sample(hist_daily.tail(252))
            combined_scen = np.vstack([daily_scen, synthetic]) if len(daily_scen) else synthetic
        else:
            combined_scen = daily_scen

        w, ok = hard_constrained_weights(mu, sigma, combined_scen, dd, lam, {1: hcap, 0: ccap})

        if augment_with_copula and len(hist_daily) >= 60:
            # did any synthetic scenario bind tighter than the worst real one?
            worst_real = (daily_scen @ w).min() if len(daily_scen) else np.inf
            worst_synth = (synthetic @ w).min()
            if worst_synth < worst_real:
                recipe_notes.append((month_end_date.date(), worst_real, worst_synth))

        month_start = (month_end_date - pd.offsets.MonthEnd(1))
        days_in_month = daily_panel[(daily_panel.index > month_start) & (daily_panel.index <= month_end_date)].index
        for d in days_in_month:
            daily_weights[d] = w
        i += 1

    idx = sorted([d for d in daily_weights.keys() if d in daily_panel.index])
    if len(idx) < 10:
        return None, recipe_notes
    W = pd.DataFrame([daily_weights[d] for d in idx], index=idx, columns=STRATEGY_COLS)
    v2_mult = per_strategy_v2_multiplier_fixed(daily_panel, idx, lookback=20, lo=0.7, hi=1.3)
    adj_W = W.mul(v2_mult, axis=0)
    combo_ret = pd.Series((daily_panel.loc[idx].values * adj_W.values).sum(axis=1), index=idx)
    final = combo_ret * volq_multiplier_fixed(combo_ret, lookback=20, lo=0.7, hi=1.3)
    return final, recipe_notes


def report(label, final):
    if final is None:
        print(f"{label}: insufficient data")
        return
    r, v, s, w = sharpe_stats(final)
    n = len(final)
    split_60, split_40 = int(n * 0.6), int(n * 0.4)
    fwd, rev = final.iloc[split_60:], final.iloc[:split_40]
    _, _, s_f, w_f = sharpe_stats(fwd)
    _, _, s_r, w_r = sharpe_stats(rev)
    print(f"{label}: full sharpe={s:.2f} worst_day={w:.3%}  |  FWD sharpe={s_f:.2f} worst_day={w_f:.3%}  |  REV sharpe={s_r:.2f} worst_day={w_r:.3%}")


def main():
    monthly_panel, daily_panel = build_extended_monthly_panel()
    schedule = get_honest_recipe_schedule(monthly_panel, daily_panel)

    final_base, _ = run(monthly_panel, daily_panel, schedule, augment_with_copula=False)
    report("Real baseline recipe schedule, real historical scenarios only (should ~match project headline)", final_base)

    final_copula, notes = run(monthly_panel, daily_panel, schedule, augment_with_copula=True)
    report("SAME recipe schedule + 500 synthetic t-copula stress scenarios per rebalance", final_copula)

    print(f"\nMonths where a synthetic scenario bound tighter than any real historical day: {len(notes)}")
    for d, worst_real, worst_synth in notes[:15]:
        print(f"  {d}: worst_real={worst_real:.3%} worst_synth={worst_synth:.3%}")

    if final_copula is not None:
        final_copula.to_csv("cpo_copula_stress_returns.csv")


if __name__ == "__main__":
    main()
