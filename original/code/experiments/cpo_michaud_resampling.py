"""
Approach 15: Michaud resampled efficiency (Michaud 1998). Unlike HRP
(approach 14), this keeps the return input -- it targets ESTIMATION
ERROR specifically, not return-blindness. Markowitz mean-variance is
famously an "error maximizer": small sampling noise in mu/sigma gets
amplified by the optimizer into large, overconfident position bets,
especially with only ~60-95 months of data across 4 assets.

Mechanism: at each rebalance point, bootstrap-resample the trailing
monthly return panel (sampling MONTHS with replacement, jointly across
all 4 strategies to preserve realistic cross-sectional dependence) many
times. Solve the SAME constrained mean-variance optimizer on each
resample, then average the resulting WEIGHT VECTORS (not the inputs).
This is bagging applied to portfolio optimization: averaging many noisy
point estimates should be more stable than trusting any single one.

IMPORTANT CORRECTION: an earlier version of this script fixed
dd_limit/lambda/caps at constants I picked because I'd SEEN them
recur across other experiments' honest-reselection outputs in this
project -- that's hindsight, the same test-set-contamination mistake
this project has caught and fixed multiple times before, and it made
the initial "Sharpe <real figure removed> / FWD 4.72" result incomparable to the real
baseline. Fixed by reusing the ACTUAL already-validated recipe
schedule from cpo_hybrid_monthly's own honest_pipeline (dd/lambda/caps/
lookback chosen using only trailing data, with zero knowledge of
Michaud resampling) for every rebalance point -- only the weight-
computation method (resampled-average vs. point-estimate) differs
between the two runs below, isolating the technique itself.
"""
import numpy as np
import pandas as pd

from cpo_hybrid_monthly import (
    build_extended_monthly_panel, hard_constrained_weights, sharpe_stats,
    STRATEGY_COLS, honest_pipeline as baseline_honest_pipeline,
)
from cpo_audit_fixed import volq_multiplier_fixed, per_strategy_v2_multiplier_fixed

N_RESAMPLES = 200
RNG = np.random.default_rng(42)


def michaud_weights(trailing_m, daily_scen, dd, lam, hcap, ccap, n_resamples=N_RESAMPLES):
    if n_resamples <= 1:
        mu = trailing_m.values.mean(axis=0)
        sigma = np.cov(trailing_m.values, rowvar=False)
        w, ok = hard_constrained_weights(mu, sigma, daily_scen, dd, lam, {1: hcap, 0: ccap})
        return w

    n_months = len(trailing_m)
    values = trailing_m.values
    weight_sum = np.zeros(values.shape[1])
    n_ok = 0
    for _ in range(n_resamples):
        sample_idx = RNG.integers(0, n_months, size=n_months)
        resample = values[sample_idx]
        mu_r = resample.mean(axis=0)
        sigma_r = np.cov(resample, rowvar=False)
        w_r, ok = hard_constrained_weights(mu_r, sigma_r, daily_scen, dd, lam, {1: hcap, 0: ccap})
        weight_sum += w_r
        n_ok += 1
    return weight_sum / n_ok


def get_honest_recipe_schedule(monthly_panel, daily_panel):
    """Re-derives the exact (date -> dd,lam,hcap,ccap,lookback) schedule
    the real baseline pipeline uses, so this test can apply the SAME
    honest schedule instead of inventing fixed constants."""
    _, recipes = baseline_honest_pipeline(monthly_panel, daily_panel)
    schedule = []  # list of (month_end_date, dd, lam, hcap, ccap, lookback)
    for date, dd, lam, hcap, ccap, lookback in recipes:
        schedule.append((pd.Timestamp(date), dd, lam, hcap, ccap, lookback))
    return schedule


def recipe_for_month(schedule, month_end_date):
    applicable = [s for s in schedule if s[0] <= month_end_date]
    return applicable[-1][1:] if applicable else schedule[0][1:]


def run(monthly_panel, daily_panel, schedule, n_resamples):
    monthly_dates = monthly_panel.index
    daily_weights = {}
    i = schedule[0][5]  # first lookback value, matches baseline's start index
    while i < len(monthly_dates):
        month_end_date = monthly_dates[i]
        dd, lam, hcap, ccap, monthly_lookback = recipe_for_month(schedule, month_end_date)

        if i < monthly_lookback:
            i += 1
            continue
        trailing_m = monthly_panel.iloc[i - monthly_lookback:i]
        daily_scen = daily_panel[daily_panel.index < month_end_date].tail(252).values
        w = michaud_weights(trailing_m, daily_scen, dd, lam, hcap, ccap, n_resamples=n_resamples)

        month_start = (month_end_date - pd.offsets.MonthEnd(1))
        days_in_month = daily_panel[(daily_panel.index > month_start) & (daily_panel.index <= month_end_date)].index
        for d in days_in_month:
            daily_weights[d] = w
        i += 1

    idx = [d for d in daily_weights.keys() if d in daily_panel.index]
    if len(idx) < 10:
        return None
    idx = sorted(idx)
    W = pd.DataFrame([daily_weights[d] for d in idx], index=idx, columns=STRATEGY_COLS)
    v2_mult = per_strategy_v2_multiplier_fixed(daily_panel, idx, lookback=20, lo=0.7, hi=1.3)
    adj_W = W.mul(v2_mult, axis=0)
    combo_ret = pd.Series((daily_panel.loc[idx].values * adj_W.values).sum(axis=1), index=idx)
    final = combo_ret * volq_multiplier_fixed(combo_ret, lookback=20, lo=0.7, hi=1.3)
    return final


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
    print("Reusing the ACTUAL honest recipe schedule from the real baseline pipeline:")
    for date, dd, lam, hcap, ccap, lookback in schedule:
        print(f"  {date.date()}  dd={dd:.1%} lam={lam} hcap={hcap:.0%} ccap={ccap:.0%} lookback={lookback}mo")
    print()

    final_control = run(monthly_panel, daily_panel, schedule, n_resamples=1)
    report("Control (point-estimate MV, same honest recipe schedule -- should ~match real baseline)", final_control)

    final_michaud = run(monthly_panel, daily_panel, schedule, n_resamples=N_RESAMPLES)
    report(f"Michaud resampled (n_resamples={N_RESAMPLES}, SAME honest recipe schedule)", final_michaud)

    if final_michaud is not None:
        final_michaud.to_csv("cpo_michaud_resampling_returns.csv")


if __name__ == "__main__":
    main()
