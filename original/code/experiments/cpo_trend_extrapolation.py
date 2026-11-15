"""
Approach 18: directly uses the finding from BASE_STRATEGIES.md that all
four CPO strategies show a measurable multi-year DECLINE in returns
(Strategy C 65.8%->2.4%, StrategyB's first negative year, Cross
Border Arb's rolling Sharpe <real figure removed>). Every prior approach (1-17),
including the recency-weighted EWMA test, estimated mu as some flavor
of weighted AVERAGE of the trailing window -- which can only approach
the most recent value, never project past it. This is the first
approach to explicitly fit a trend and extrapolate it FORWARD one step,
directly encoding "if the decline continues, what does next month look
like" rather than "what has the recent past looked like."

Mechanism: OLS trend line (return vs. time index) fit over the trailing
monthly window per strategy, extrapolated one step beyond the most
recent observation. Blended with the plain trailing mean via a
trend_weight (0 = pure flat mean, 1 = pure trend projection), folded
into the same honest periodic reselection as every other parameter --
"don't use the trend" is always a candidate, so the honest process can
reject this if the trend doesn't generalize.
"""
import numpy as np
import pandas as pd

from cpo_hybrid_monthly import (
    build_extended_monthly_panel, hard_constrained_weights, sharpe_stats,
    MONTHLY_LOOKBACK_CANDIDATES, STRATEGY_COLS,
)
from cpo_audit_fixed import volq_multiplier_fixed, per_strategy_v2_multiplier_fixed

RETUNE_EVERY_MONTHS = 6
TREND_WEIGHT_CANDIDATES = [0.0, 0.25, 0.5, 0.75, 1.0]


def trend_extrapolated_mu(trailing_m, trend_weight):
    flat_mu = trailing_m.values.mean(axis=0)
    if trend_weight <= 0:
        return flat_mu
    n = len(trailing_m)
    t = np.arange(n)
    trend_mu = np.zeros(trailing_m.shape[1])
    for j, col in enumerate(trailing_m.columns):
        y = trailing_m[col].values
        slope, intercept = np.polyfit(t, y, 1)
        trend_mu[j] = intercept + slope * n  # project one step beyond the window
    return (1 - trend_weight) * flat_mu + trend_weight * trend_mu


def select_recipe(monthly_slice, daily_slice_upto):
    min_lookback = min(MONTHLY_LOOKBACK_CANDIDATES)
    if len(monthly_slice) < min_lookback + 6:
        return 0.008, 10, 1.0, 1.0, min_lookback, 0.0

    def backtest_segment(dd_limit, lam, hcap, ccap, monthly_lookback, trend_weight):
        if len(monthly_slice) < monthly_lookback + 3:
            return pd.Series(dtype=float)
        oos = []
        for i in range(monthly_lookback, len(monthly_slice)):
            trailing_m = monthly_slice.iloc[i - monthly_lookback:i]
            sigma = trailing_m.cov().values
            mu = trend_extrapolated_mu(trailing_m, trend_weight)
            month_end_date = monthly_slice.index[i]
            daily_scen = daily_slice_upto[daily_slice_upto.index < month_end_date].tail(252).values
            w, ok = hard_constrained_weights(mu, sigma, daily_scen, dd_limit, lam, {1: hcap, 0: ccap})
            oos.append(w @ monthly_slice.iloc[i].values)
        return pd.Series(oos)

    best = None
    for dd_limit in [0.010, 0.006, 0.005]:
        for lam in [5, 10]:
            ret = backtest_segment(dd_limit, lam, 1.0, 1.0, 36, 0.0)
            _, _, s, _ = sharpe_stats(ret, freq=12)
            if best is None or (not np.isnan(s) and s > best[0]):
                best = (s, dd_limit, lam)
    _, sel_dd, sel_lam = best

    best_hcap = None
    for cap in [1.0, 0.25, 0.10, 0.05]:
        ret = backtest_segment(sel_dd, sel_lam, cap, 1.0, 36, 0.0)
        _, _, s, _ = sharpe_stats(ret, freq=12)
        if best_hcap is None or (not np.isnan(s) and s > best_hcap[0]):
            best_hcap = (s, cap)
    sel_hcap = best_hcap[1]

    best_ccap = None
    for cap in [1.0, 0.6, 0.4, 0.25]:
        ret = backtest_segment(sel_dd, sel_lam, sel_hcap, cap, 36, 0.0)
        _, _, s, _ = sharpe_stats(ret, freq=12)
        if best_ccap is None or (not np.isnan(s) and s > best_ccap[0]):
            best_ccap = (s, cap)
    sel_ccap = best_ccap[1]

    best_lb = None
    for lb in MONTHLY_LOOKBACK_CANDIDATES:
        ret = backtest_segment(sel_dd, sel_lam, sel_hcap, sel_ccap, lb, 0.0)
        _, _, s, _ = sharpe_stats(ret, freq=12)
        if best_lb is None or (not np.isnan(s) and s > best_lb[0]):
            best_lb = (s, lb)
    sel_lb = best_lb[1]

    best_tw = None
    for tw in TREND_WEIGHT_CANDIDATES:
        ret = backtest_segment(sel_dd, sel_lam, sel_hcap, sel_ccap, sel_lb, tw)
        _, _, s, _ = sharpe_stats(ret, freq=12)
        if best_tw is None or (not np.isnan(s) and s > best_tw[0]):
            best_tw = (s, tw)
    sel_tw = best_tw[1]

    return sel_dd, sel_lam, sel_hcap, sel_ccap, sel_lb, sel_tw


def honest_pipeline(monthly_panel, daily_panel):
    monthly_dates = monthly_panel.index
    daily_weights = {}
    recipes = []

    i = min(MONTHLY_LOOKBACK_CANDIDATES)
    dd_limit = lam = hcap = ccap = monthly_lookback = trend_weight = None
    months_since_retune = 0
    while i < len(monthly_dates):
        month_end_date = monthly_dates[i]
        if dd_limit is None or months_since_retune >= RETUNE_EVERY_MONTHS:
            avail_monthly = monthly_panel.iloc[:i]
            avail_daily = daily_panel[daily_panel.index < month_end_date]
            dd_limit, lam, hcap, ccap, monthly_lookback, trend_weight = select_recipe(avail_monthly, avail_daily)
            recipes.append((month_end_date.date(), dd_limit, lam, hcap, ccap, monthly_lookback, trend_weight))
            months_since_retune = 0
        months_since_retune += 1

        if i < monthly_lookback:
            i += 1
            continue
        trailing_m = monthly_panel.iloc[i - monthly_lookback:i]
        sigma = trailing_m.cov().values
        mu = trend_extrapolated_mu(trailing_m, trend_weight)
        daily_scen = daily_panel[daily_panel.index < month_end_date].tail(252).values
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
    print(f"Trend-extrapolated mu (OLS trend projected 1 step forward, honestly blended with flat mean): "
          f"n={n} sharpe={s:.2f} ann_ret={r:.2%} worst_day={w:.3%}")
    split_60, split_40 = int(n * 0.6), int(n * 0.4)
    fwd, rev = final.iloc[split_60:], final.iloc[:split_40]
    _, _, s_f, w_f = sharpe_stats(fwd)
    _, _, s_r, w_r = sharpe_stats(rev)
    print(f"FWD: sharpe={s_f:.2f} worst_day={w_f:.3%}   REV: sharpe={s_r:.2f} worst_day={w_r:.3%}")
    print("\nRecipe over time (tw=trend_weight, 0=pure flat mean, 1=pure trend projection):")
    for d, dd, lam, hcap, ccap, lb, tw in recipes:
        print(f"  {d}  dd={dd:.1%} lam={lam} hcap={hcap:.0%} ccap={ccap:.0%} lookback={lb}mo tw={tw:.0%}")
    final.to_csv("cpo_trend_extrapolation_returns.csv")


if __name__ == "__main__":
    main()
