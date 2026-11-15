"""
Two-stage architecture #2, per user's explicit ask: "figure out a way
where we can possibly use both" mean-variance AND Bayesian shrinkage,
rather than treating them as competing alternatives (which is how the
prior Bayesian test, approach 9, was framed -- Bayesian-shrunk mu fed
into the SAME optimizer, replacing the plain trailing mean entirely).

This is architecturally different from approach 9: instead of
replacing the optimizer's return input, it computes TWO FULL
ALLOCATIONS independently -- the proven mean-variance baseline, and a
Bayesian-shrunk-mu allocation from the same optimizer -- and blends
the two ALLOCATION VECTORS at the portfolio level (model-averaging,
the same idea as an ML ensemble averaging predictions, applied to
whole portfolios instead of one signal). A fixed 50/50 blend is also
reported as a robustness check immune to any fitting bias, alongside
the honestly-reselected blend ratio.

Bayesian formula is the exact precision-weighted update from the
Client project's own bayesian.py: prior = long-run trailing mean/var,
view = short-run trailing mean/var, posterior precision = sum of
precisions, posterior mean = precision-weighted average.
"""
import numpy as np
import pandas as pd

from cpo_hybrid_monthly import (
    build_extended_monthly_panel, hard_constrained_weights, sharpe_stats,
    MONTHLY_LOOKBACK_CANDIDATES, STRATEGY_COLS,
)
from cpo_audit_fixed import volq_multiplier_fixed, per_strategy_v2_multiplier_fixed

RETUNE_EVERY_MONTHS = 6
BAYES_WINDOW_CANDIDATES = [3, 6, 12]
BLEND_RATIO_CANDIDATES = [0.0, 0.25, 0.5, 0.75, 1.0]  # 0 = pure MV, 1 = pure Bayesian


def bayesian_mu(trailing_m, bayes_window):
    """Precision-weighted posterior mean, per strategy, per Client's
    bayesian.py: prior = full trailing_m history, view = last
    bayes_window months of it. Both in monthly-return units already,
    no rescaling needed since neither side comes from a prob-space
    ensemble here."""
    n_long = len(trailing_m)
    mu_prior = trailing_m.mean().values
    var_prior = trailing_m.var().values
    sigma2_prior = np.maximum(var_prior / max(n_long, 1), 1e-10)

    view_slice = trailing_m.iloc[-bayes_window:]
    n_short = len(view_slice)
    mu_view = view_slice.mean().values
    var_view = view_slice.var().values
    sigma2_view = np.maximum(var_view / max(n_short, 1), 1e-10)

    prec_prior = 1.0 / sigma2_prior
    prec_view = 1.0 / sigma2_view
    prec_post = prec_prior + prec_view
    mu_post = (mu_prior * prec_prior + mu_view * prec_view) / prec_post
    return mu_post


def select_recipe(monthly_slice, daily_slice_upto):
    min_lookback = min(MONTHLY_LOOKBACK_CANDIDATES)
    if len(monthly_slice) < min_lookback + 6:
        return 0.008, 10, 1.0, 1.0, min_lookback, 6, 0.0

    def backtest_segment(dd_limit, lam, hcap, ccap, monthly_lookback, bayes_window, blend_ratio):
        if len(monthly_slice) < monthly_lookback + 3 or len(monthly_slice) < bayes_window + 3:
            return pd.Series(dtype=float)
        oos = []
        for i in range(monthly_lookback, len(monthly_slice)):
            trailing_m = monthly_slice.iloc[i - monthly_lookback:i]
            sigma = trailing_m.cov().values
            month_end_date = monthly_slice.index[i]
            daily_scen = daily_slice_upto[daily_slice_upto.index < month_end_date].tail(252).values

            mu_mv = trailing_m.mean().values
            w_mv, _ = hard_constrained_weights(mu_mv, sigma, daily_scen, dd_limit, lam, {1: hcap, 0: ccap})

            if blend_ratio > 0.0:
                mu_bayes = bayesian_mu(trailing_m, min(bayes_window, len(trailing_m)))
                w_bayes, _ = hard_constrained_weights(mu_bayes, sigma, daily_scen, dd_limit, lam, {1: hcap, 0: ccap})
                w = blend_ratio * w_bayes + (1 - blend_ratio) * w_mv
            else:
                w = w_mv

            actual = monthly_slice.iloc[i].values
            oos.append(w @ actual)
        return pd.Series(oos)

    best = None
    for dd_limit in [0.010, 0.006, 0.005]:
        for lam in [5, 10]:
            ret = backtest_segment(dd_limit, lam, 1.0, 1.0, 36, 6, 0.0)
            _, _, s, _ = sharpe_stats(ret, freq=12)
            if best is None or (not np.isnan(s) and s > best[0]):
                best = (s, dd_limit, lam)
    _, sel_dd, sel_lam = best

    best_hcap = None
    for cap in [1.0, 0.25, 0.10, 0.05]:
        ret = backtest_segment(sel_dd, sel_lam, cap, 1.0, 36, 6, 0.0)
        _, _, s, _ = sharpe_stats(ret, freq=12)
        if best_hcap is None or (not np.isnan(s) and s > best_hcap[0]):
            best_hcap = (s, cap)
    sel_hcap = best_hcap[1]

    best_ccap = None
    for cap in [1.0, 0.6, 0.4, 0.25]:
        ret = backtest_segment(sel_dd, sel_lam, sel_hcap, cap, 36, 6, 0.0)
        _, _, s, _ = sharpe_stats(ret, freq=12)
        if best_ccap is None or (not np.isnan(s) and s > best_ccap[0]):
            best_ccap = (s, cap)
    sel_ccap = best_ccap[1]

    best_lb = None
    for lb in MONTHLY_LOOKBACK_CANDIDATES:
        ret = backtest_segment(sel_dd, sel_lam, sel_hcap, sel_ccap, lb, 6, 0.0)
        _, _, s, _ = sharpe_stats(ret, freq=12)
        if best_lb is None or (not np.isnan(s) and s > best_lb[0]):
            best_lb = (s, lb)
    sel_lb = best_lb[1]

    best_bw = None
    for bw in BAYES_WINDOW_CANDIDATES:
        ret = backtest_segment(sel_dd, sel_lam, sel_hcap, sel_ccap, sel_lb, bw, 1.0)
        _, _, s, _ = sharpe_stats(ret, freq=12)
        if best_bw is None or (not np.isnan(s) and s > best_bw[0]):
            best_bw = (s, bw)
    sel_bw = best_bw[1]

    best_blend = None
    for br in BLEND_RATIO_CANDIDATES:
        ret = backtest_segment(sel_dd, sel_lam, sel_hcap, sel_ccap, sel_lb, sel_bw, br)
        _, _, s, _ = sharpe_stats(ret, freq=12)
        if best_blend is None or (not np.isnan(s) and s > best_blend[0]):
            best_blend = (s, br)
    sel_blend = best_blend[1]

    return sel_dd, sel_lam, sel_hcap, sel_ccap, sel_lb, sel_bw, sel_blend


def honest_pipeline(monthly_panel, daily_panel, fixed_blend_ratio=None):
    """If fixed_blend_ratio is given, skip honest reselection of the
    blend ratio and use this fixed value throughout (a robustness
    check immune to any fitting bias, since it's chosen a priori)."""
    monthly_dates = monthly_panel.index
    daily_weights = {}
    recipes = []

    i = min(MONTHLY_LOOKBACK_CANDIDATES)
    dd_limit = lam = hcap = ccap = monthly_lookback = bayes_window = blend_ratio = None
    months_since_retune = 0
    while i < len(monthly_dates):
        month_end_date = monthly_dates[i]
        if dd_limit is None or months_since_retune >= RETUNE_EVERY_MONTHS:
            avail_monthly = monthly_panel.iloc[:i]
            avail_daily = daily_panel[daily_panel.index < month_end_date]
            dd_limit, lam, hcap, ccap, monthly_lookback, bayes_window, blend_ratio = select_recipe(
                avail_monthly, avail_daily)
            if fixed_blend_ratio is not None:
                blend_ratio = fixed_blend_ratio
            recipes.append((month_end_date.date(), dd_limit, lam, hcap, ccap, monthly_lookback, bayes_window, blend_ratio))
            months_since_retune = 0
        months_since_retune += 1

        if i < monthly_lookback:
            i += 1
            continue
        trailing_m = monthly_panel.iloc[i - monthly_lookback:i]
        sigma = trailing_m.cov().values
        daily_scen = daily_panel[daily_panel.index < month_end_date].tail(252).values

        mu_mv = trailing_m.mean().values
        w_mv, _ = hard_constrained_weights(mu_mv, sigma, daily_scen, dd_limit, lam, {1: hcap, 0: ccap})
        if blend_ratio > 0.0:
            mu_bayes = bayesian_mu(trailing_m, min(bayes_window, len(trailing_m)))
            w_bayes, _ = hard_constrained_weights(mu_bayes, sigma, daily_scen, dd_limit, lam, {1: hcap, 0: ccap})
            w = blend_ratio * w_bayes + (1 - blend_ratio) * w_mv
        else:
            w = w_mv

        month_start = (month_end_date - pd.offsets.MonthEnd(1))
        days_in_month = daily_panel[(daily_panel.index > month_start) & (daily_panel.index <= month_end_date)].index
        for d in days_in_month:
            daily_weights[d] = w
        i += 1

    idx = [d for d in daily_weights.keys() if d in daily_panel.index]
    if len(idx) < 10:
        return None, recipes
    idx = sorted(idx)
    W = pd.DataFrame([daily_weights[d] for d in idx], index=idx, columns=STRATEGY_COLS)
    v2_mult = per_strategy_v2_multiplier_fixed(daily_panel, idx, lookback=20, lo=0.7, hi=1.3)
    adj_W = W.mul(v2_mult, axis=0)
    combo_ret = pd.Series((daily_panel.loc[idx].values * adj_W.values).sum(axis=1), index=idx)
    final = combo_ret * volq_multiplier_fixed(combo_ret, lookback=20, lo=0.7, hi=1.3)
    return final, recipes


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

    final_honest, recipes = honest_pipeline(monthly_panel, daily_panel, fixed_blend_ratio=None)
    report("Honestly-reselected blend ratio (incl. 0=pure-MV as a candidate)", final_honest)
    print("Recipe history (bw=bayes window, br=blend ratio, 0=pure MV, 1=pure Bayesian):")
    for d, dd, lam, hcap, ccap, lb, bw, br in recipes:
        print(f"  {d}  dd={dd:.1%} lam={lam} hcap={hcap:.0%} ccap={ccap:.0%} lookback={lb}mo bw={bw}mo br={br:.0%}")

    print()
    for fixed in [0.25, 0.5, 0.75, 1.0]:
        final_fixed, _ = honest_pipeline(monthly_panel, daily_panel, fixed_blend_ratio=fixed)
        report(f"Fixed blend ratio = {fixed:.0%} Bayesian / {1-fixed:.0%} MV (no fitting, robustness check)", final_fixed)

    if final_honest is not None:
        final_honest.to_csv("cpo_two_stage_blend_returns.csv")


if __name__ == "__main__":
    main()
