"""
Approach 13: regime-conditional Bayesian prior -- combines regime
detection (approach 8's 2-state HMM) with Bayesian shrinkage (approach
9's precision-weighted update) in a way neither prior test did.

Approach 8 used the HMM only as a single book-wide exposure scalar
(on/off), blind to which strategy was doing what. Approach 9's
Bayesian prior was unconditional -- the full trailing history,
regardless of what regime it occurred in. This combines them
properly: at each rebalance point, decode which regime the book is
CURRENTLY in (from a 2-state Gaussian HMM fit on trailing book
returns), then build each strategy's long-run PRIOR only from past
months that were in that SAME regime (not just all history) -- i.e.
"how has this strategy performed historically, specifically during
periods like the one we're in now" -- and combine that regime-matched
prior with the usual short-run view via the same precision-weighted
formula as approach 9.

Falls back to the unconditional (all-history) prior when the current
regime doesn't have enough matched historical months (min_regime_obs)
-- a stability safeguard, not a tuned escape hatch, since a
regime-matched prior estimated from 3 data points is not credible.

Tested two ways, same as the approach 9/11 split:
  (a) regime-Bayes mu fed directly into the SAME optimizer (parallel
      to approach 9's design)
  (b) the resulting allocation blended with the plain MV baseline
      allocation at the portfolio level, both honestly-reselected and
      fixed (non-fitted) ratios as a robustness check (parallel to
      approach 11's design)
"""
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

USE_HMM = False  # see module docstring update: HMM collapses to a
                 # single state at this sample size (36-48 months), so
                 # a deterministic median-split regime label is used
                 # instead -- it can't collapse by construction.

from cpo_hybrid_monthly import (
    build_extended_monthly_panel, hard_constrained_weights, sharpe_stats,
    MONTHLY_LOOKBACK_CANDIDATES, STRATEGY_COLS,
)
from cpo_audit_fixed import volq_multiplier_fixed, per_strategy_v2_multiplier_fixed

RETUNE_EVERY_MONTHS = 6
BAYES_WINDOW_CANDIDATES = [3, 6, 12]
BLEND_RATIO_CANDIDATES = [0.0, 0.25, 0.5, 0.75, 1.0]
N_REGIME_STATES = 2
MIN_REGIME_OBS = 8


def decode_regimes_hmm(trailing_m):
    """2-state Gaussian HMM on the book-level (equal-weight) trailing
    monthly return. NOTE: empirically collapses to a single state at
    this sample size (36-48 months) -- verified by direct diagnostic,
    every recipe actually used in the backtest produced a single-state
    decode, making this identical (and useless) versus an unconditional
    prior. Kept for reference; not used (see USE_HMM / decode_regimes)."""
    book_ret = trailing_m.mean(axis=1).values.reshape(-1, 1)
    if len(book_ret) < 2 * N_REGIME_STATES * MIN_REGIME_OBS:
        return np.zeros(len(book_ret), dtype=int), 0
    try:
        model = GaussianHMM(n_components=N_REGIME_STATES, covariance_type="diag",
                             n_iter=50, random_state=42)
        model.fit(book_ret)
        states = model.predict(book_ret)
        return states, states[-1]
    except Exception:
        return np.zeros(len(book_ret), dtype=int), 0


def decode_regimes_median_split(trailing_m):
    """Deterministic regime label: each month's OWN trailing 3-month
    book realized volatility, classified 'high-vol'/'low-vol' by the
    median of those trailing-vol values within the window. Can't
    collapse to one state by construction (median split always yields
    two non-trivial groups), unlike a from-scratch HMM fit on 36-48
    monthly points -- trades a stochastic regime model for a robust,
    deterministic one at this sample size."""
    book_ret = trailing_m.mean(axis=1)
    trail_vol = book_ret.rolling(3, min_periods=1).std().bfill()
    med = trail_vol.median()
    states = (trail_vol > med).astype(int).values
    return states, states[-1]


def decode_regimes(trailing_m):
    return decode_regimes_hmm(trailing_m) if USE_HMM else decode_regimes_median_split(trailing_m)


def regime_bayesian_mu(trailing_m, bayes_window):
    states, current_regime = decode_regimes(trailing_m)
    regime_mask = states == current_regime
    n_regime = regime_mask.sum()

    if n_regime >= MIN_REGIME_OBS:
        regime_hist = trailing_m.iloc[regime_mask]
        mu_prior = regime_hist.mean().values
        var_prior = regime_hist.var().values
        n_prior = n_regime
    else:
        mu_prior = trailing_m.mean().values
        var_prior = trailing_m.var().values
        n_prior = len(trailing_m)
    sigma2_prior = np.maximum(var_prior / max(n_prior, 1), 1e-10)

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
                mu_rb = regime_bayesian_mu(trailing_m, min(bayes_window, len(trailing_m)))
                w_rb, _ = hard_constrained_weights(mu_rb, sigma, daily_scen, dd_limit, lam, {1: hcap, 0: ccap})
                w = blend_ratio * w_rb + (1 - blend_ratio) * w_mv
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
            mu_rb = regime_bayesian_mu(trailing_m, min(bayes_window, len(trailing_m)))
            w_rb, _ = hard_constrained_weights(mu_rb, sigma, daily_scen, dd_limit, lam, {1: hcap, 0: ccap})
            w = blend_ratio * w_rb + (1 - blend_ratio) * w_mv
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
    report("Honestly-reselected regime-Bayes blend ratio (incl. 0=pure-MV)", final_honest)
    print("Recipe history (bw=bayes window, br=regime-Bayes blend ratio):")
    for d, dd, lam, hcap, ccap, lb, bw, br in recipes:
        print(f"  {d}  dd={dd:.1%} lam={lam} hcap={hcap:.0%} ccap={ccap:.0%} lookback={lb}mo bw={bw}mo br={br:.0%}")

    print()
    for fixed in [0.25, 0.5, 0.75, 1.0]:
        final_fixed, _ = honest_pipeline(monthly_panel, daily_panel, fixed_blend_ratio=fixed)
        report(f"Fixed regime-Bayes ratio = {fixed:.0%} / {1-fixed:.0%} MV (no fitting, robustness check)", final_fixed)

    if final_honest is not None:
        final_honest.to_csv("cpo_regime_bayesian_returns.csv")


if __name__ == "__main__":
    main()
