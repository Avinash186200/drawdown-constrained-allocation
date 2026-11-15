"""
Approach 17, per user's specific idea: every approach so far (1-16)
estimated mu/sigma from a FLAT, equally-weighted trailing window --
a month from 3 years ago counts exactly the same as last month. This
tests the classic alternative (RiskMetrics/EWMA): exponentially decay
the weight given to older observations, so recent performance and
recent volatility matter more than distant history, directly
implementing "give more importance to recent days."

Two variants tested:
  (a) EWMA on the MONTHLY panel (same frequency/window as the existing
      pipeline, only the WEIGHTING within the window changes from flat
      to exponential-decay -- cleanest, most directly comparable test)
  (b) EWMA on the DAILY panel with short halflives (10/20 days, as the
      user specifically suggested) for the volatility/covariance term
      only, mapped to the same monthly-equivalent scale used elsewhere
      -- tests whether SHORT-HORIZON daily vol responsiveness (not just
      recency-weighted monthly returns) adds anything.

Halflife (including an effectively-flat "very long halflife" option,
so the honest process can reject exponential weighting outright) is
folded into the same periodic trailing-data-only reselection as every
other parameter in this project.
"""
import numpy as np
import pandas as pd

from cpo_hybrid_monthly import (
    build_extended_monthly_panel, hard_constrained_weights, sharpe_stats,
    MONTHLY_LOOKBACK_CANDIDATES, STRATEGY_COLS,
)
from cpo_audit_fixed import volq_multiplier_fixed, per_strategy_v2_multiplier_fixed

RETUNE_EVERY_MONTHS = 6
HALFLIFE_CANDIDATES_MONTHS = [3, 6, 12, 9999]  # 9999 ~= flat/equal-weighted (no decay)
DAILY_VOL_HALFLIFE_CANDIDATES = [10, 20, 9999]  # days, per user's specific suggestion


def ewma_mean_cov(trailing_m, halflife):
    n = len(trailing_m)
    values = trailing_m.values
    if halflife >= n * 5:  # effectively flat
        return values.mean(axis=0), np.cov(values, rowvar=False)
    ages = np.arange(n - 1, -1, -1)  # 0 = most recent
    w = 0.5 ** (ages / halflife)
    w = w / w.sum()
    mu = (values * w[:, None]).sum(axis=0)
    centered = values - mu
    sigma = (centered.T * w) @ centered
    return mu, sigma


def ewma_daily_vol(daily_hist, halflife_days):
    """Per-strategy EWMA volatility from daily data, short halflife
    (10/20 days per user's request), annualized then rescaled to
    monthly-variance units so it's comparable to the monthly-panel
    covariance diagonal used elsewhere."""
    if halflife_days >= 9000 or len(daily_hist) < 15:
        return None
    n = len(daily_hist)
    values = daily_hist.values
    ages = np.arange(n - 1, -1, -1)
    w = 0.5 ** (ages / halflife_days)
    w = w / w.sum()
    mu_d = (values * w[:, None]).sum(axis=0)
    centered = values - mu_d
    daily_var = (centered ** 2 * w[:, None]).sum(axis=0)
    return daily_var * 21  # daily variance -> ~monthly-equivalent variance (21 trading days/month)


def apply_daily_vol_to_sigma(sigma_monthly, daily_var_monthly):
    """Rescales the monthly covariance matrix so its diagonal matches
    the more-responsive daily-EWMA variance, preserving the monthly
    correlation structure (only the marginal vol estimate is swapped)."""
    if daily_var_monthly is None:
        return sigma_monthly
    std_monthly = np.sqrt(np.clip(np.diag(sigma_monthly), 1e-12, None))
    std_daily_ewma = np.sqrt(np.clip(daily_var_monthly, 1e-12, None))
    scale = std_daily_ewma / std_monthly
    D = np.diag(scale)
    return D @ sigma_monthly @ D


def select_recipe(monthly_slice, daily_slice_upto):
    min_lookback = min(MONTHLY_LOOKBACK_CANDIDATES)
    if len(monthly_slice) < min_lookback + 6:
        return 0.008, 10, 1.0, 1.0, min_lookback, 9999, 9999

    def backtest_segment(dd_limit, lam, hcap, ccap, monthly_lookback, mo_halflife, daily_halflife):
        if len(monthly_slice) < monthly_lookback + 3:
            return pd.Series(dtype=float)
        oos = []
        for i in range(monthly_lookback, len(monthly_slice)):
            trailing_m = monthly_slice.iloc[i - monthly_lookback:i]
            mu, sigma = ewma_mean_cov(trailing_m, mo_halflife)
            month_end_date = monthly_slice.index[i]
            daily_scen = daily_slice_upto[daily_slice_upto.index < month_end_date].tail(252).values

            if daily_halflife < 9000:
                hist_daily = daily_slice_upto[daily_slice_upto.index < month_end_date].tail(252)
                daily_var = ewma_daily_vol(hist_daily, daily_halflife)
                sigma = apply_daily_vol_to_sigma(sigma, daily_var)

            w, ok = hard_constrained_weights(mu, sigma, daily_scen, dd_limit, lam, {1: hcap, 0: ccap})
            oos.append(w @ monthly_slice.iloc[i].values)
        return pd.Series(oos)

    best = None
    for dd_limit in [0.010, 0.006, 0.005]:
        for lam in [5, 10]:
            ret = backtest_segment(dd_limit, lam, 1.0, 1.0, 36, 9999, 9999)
            _, _, s, _ = sharpe_stats(ret, freq=12)
            if best is None or (not np.isnan(s) and s > best[0]):
                best = (s, dd_limit, lam)
    _, sel_dd, sel_lam = best

    best_hcap = None
    for cap in [1.0, 0.25, 0.10, 0.05]:
        ret = backtest_segment(sel_dd, sel_lam, cap, 1.0, 36, 9999, 9999)
        _, _, s, _ = sharpe_stats(ret, freq=12)
        if best_hcap is None or (not np.isnan(s) and s > best_hcap[0]):
            best_hcap = (s, cap)
    sel_hcap = best_hcap[1]

    best_ccap = None
    for cap in [1.0, 0.6, 0.4, 0.25]:
        ret = backtest_segment(sel_dd, sel_lam, sel_hcap, cap, 36, 9999, 9999)
        _, _, s, _ = sharpe_stats(ret, freq=12)
        if best_ccap is None or (not np.isnan(s) and s > best_ccap[0]):
            best_ccap = (s, cap)
    sel_ccap = best_ccap[1]

    best_lb = None
    for lb in MONTHLY_LOOKBACK_CANDIDATES:
        ret = backtest_segment(sel_dd, sel_lam, sel_hcap, sel_ccap, lb, 9999, 9999)
        _, _, s, _ = sharpe_stats(ret, freq=12)
        if best_lb is None or (not np.isnan(s) and s > best_lb[0]):
            best_lb = (s, lb)
    sel_lb = best_lb[1]

    best_mohl = None
    for hl in HALFLIFE_CANDIDATES_MONTHS:
        ret = backtest_segment(sel_dd, sel_lam, sel_hcap, sel_ccap, sel_lb, hl, 9999)
        _, _, s, _ = sharpe_stats(ret, freq=12)
        if best_mohl is None or (not np.isnan(s) and s > best_mohl[0]):
            best_mohl = (s, hl)
    sel_mohl = best_mohl[1]

    best_dhl = None
    for dhl in DAILY_VOL_HALFLIFE_CANDIDATES:
        ret = backtest_segment(sel_dd, sel_lam, sel_hcap, sel_ccap, sel_lb, sel_mohl, dhl)
        _, _, s, _ = sharpe_stats(ret, freq=12)
        if best_dhl is None or (not np.isnan(s) and s > best_dhl[0]):
            best_dhl = (s, dhl)
    sel_dhl = best_dhl[1]

    return sel_dd, sel_lam, sel_hcap, sel_ccap, sel_lb, sel_mohl, sel_dhl


def honest_pipeline(monthly_panel, daily_panel):
    monthly_dates = monthly_panel.index
    daily_weights = {}
    recipes = []

    i = min(MONTHLY_LOOKBACK_CANDIDATES)
    dd_limit = lam = hcap = ccap = monthly_lookback = mo_halflife = daily_halflife = None
    months_since_retune = 0
    while i < len(monthly_dates):
        month_end_date = monthly_dates[i]
        if dd_limit is None or months_since_retune >= RETUNE_EVERY_MONTHS:
            avail_monthly = monthly_panel.iloc[:i]
            avail_daily = daily_panel[daily_panel.index < month_end_date]
            dd_limit, lam, hcap, ccap, monthly_lookback, mo_halflife, daily_halflife = select_recipe(
                avail_monthly, avail_daily)
            recipes.append((month_end_date.date(), dd_limit, lam, hcap, ccap, monthly_lookback, mo_halflife, daily_halflife))
            months_since_retune = 0
        months_since_retune += 1

        if i < monthly_lookback:
            i += 1
            continue
        trailing_m = monthly_panel.iloc[i - monthly_lookback:i]
        mu, sigma = ewma_mean_cov(trailing_m, mo_halflife)
        daily_scen = daily_panel[daily_panel.index < month_end_date].tail(252).values

        if daily_halflife < 9000:
            hist_daily = daily_panel[daily_panel.index < month_end_date].tail(252)
            daily_var = ewma_daily_vol(hist_daily, daily_halflife)
            sigma = apply_daily_vol_to_sigma(sigma, daily_var)

        w, ok = hard_constrained_weights(mu, sigma, daily_scen, dd_limit, lam, {1: hcap, 0: ccap})

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


def main():
    monthly_panel, daily_panel = build_extended_monthly_panel()
    final, recipes = honest_pipeline(monthly_panel, daily_panel)
    if final is None:
        print("insufficient data")
        return
    r, v, s, w = sharpe_stats(final)
    n = len(final)
    print(f"EWMA-weighted mu/sigma (monthly halflife + optional daily-vol halflife, honestly selected): "
          f"n={n} sharpe={s:.2f} ann_ret={r:.2%} worst_day={w:.3%}")
    split_60, split_40 = int(n * 0.6), int(n * 0.4)
    fwd, rev = final.iloc[split_60:], final.iloc[:split_40]
    _, _, s_f, w_f = sharpe_stats(fwd)
    _, _, s_r, w_r = sharpe_stats(rev)
    print(f"FWD: sharpe={s_f:.2f} worst_day={w_f:.3%}   REV: sharpe={s_r:.2f} worst_day={w_r:.3%}")
    print("\nRecipe over time (mohl=monthly halflife in months, dhl=daily-vol halflife in days, 9999=flat/off):")
    for d, dd, lam, hcap, ccap, lb, mohl, dhl in recipes:
        print(f"  {d}  dd={dd:.1%} lam={lam} hcap={hcap:.0%} ccap={ccap:.0%} lookback={lb}mo mohl={mohl} dhl={dhl}")
    final.to_csv("cpo_ewma_features_returns.csv")


if __name__ == "__main__":
    main()
