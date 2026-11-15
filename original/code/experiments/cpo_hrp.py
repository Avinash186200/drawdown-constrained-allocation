"""
Approach 14 (out-of-the-box, not another forecasting variant):
Hierarchical Risk Parity (HRP, Lopez de Prado 2016). Every prior
approach (1-13) tried to improve or combine forecasts of expected
return, fed into the same mean-variance-with-matrix-inversion
machinery. HRP replaces the algorithm itself: no expected-return input
at all, no covariance matrix inversion (the single biggest source of
estimation-error amplification in Markowitz optimization -- small
errors in a noisy covariance matrix get magnified by inversion,
exactly the failure mode you'd expect with only 4 assets and ~60-95
months of data). Instead: hierarchical clustering on the correlation
structure, then recursive bisection allocating inverse-variance weight
within each cluster split. Specifically designed for small/noisy
samples like this one.

Structural caps (StrategyB/SA) and the daily VolQ/per-strategy overlay are
kept exactly as in the rest of this project -- those are proven
mechanisms, not under test here. Only the core allocation algorithm
changes. The daily hard-drawdown constraint isn't directly enforceable
on HRP (it's a deterministic recursive procedure, not a constrained
optimization) -- realized worst-day is reported honestly either way.

Only real free parameter is the covariance lookback window, honestly
reselected the same way as everything else in this project.
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage

from cpo_hybrid_monthly import build_extended_monthly_panel, sharpe_stats, MONTHLY_LOOKBACK_CANDIDATES, STRATEGY_COLS
from cpo_audit_fixed import volq_multiplier_fixed, per_strategy_v2_multiplier_fixed, apply_caps

RETUNE_EVERY_MONTHS = 6
HCAP_CANDIDATES = [1.0, 0.25, 0.10, 0.05]
CCAP_CANDIDATES = [1.0, 0.6, 0.4, 0.25]


def get_ivp(cov):
    ivp = 1.0 / np.diag(cov)
    return ivp / ivp.sum()


def get_cluster_var(cov, items):
    cov_slice = cov[np.ix_(items, items)]
    w = get_ivp(cov_slice).reshape(-1, 1)
    return float(w.T @ cov_slice @ w)


def get_quasi_diag(link, n):
    link = link.astype(int)
    sort_ix = [link[-1, 0], link[-1, 1]]
    num_items = link[-1, 3]
    while max(sort_ix) >= num_items:
        new_sort_ix = []
        for v in sort_ix:
            if v >= num_items:
                j = v - num_items
                new_sort_ix.append(link[j, 0])
                new_sort_ix.append(link[j, 1])
            else:
                new_sort_ix.append(v)
        sort_ix = new_sort_ix
    return sort_ix


def get_rec_bipart(cov, sort_ix):
    w = np.ones(len(sort_ix))
    idx_map = {name: i for i, name in enumerate(sort_ix)}
    c_items = [sort_ix]
    while len(c_items) > 0:
        c_items = [i[j:k] for i in c_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
        for i in range(0, len(c_items), 2):
            c0, c1 = c_items[i], c_items[i + 1]
            v0 = get_cluster_var(cov, c0)
            v1 = get_cluster_var(cov, c1)
            alpha = 1 - v0 / (v0 + v1)
            for name in c0:
                w[idx_map[name]] *= alpha
            for name in c1:
                w[idx_map[name]] *= (1 - alpha)
    return w


def hrp_weights(trailing_m):
    """trailing_m: DataFrame of monthly returns, columns = STRATEGY_COLS order."""
    corr = trailing_m.corr().values
    cov = trailing_m.cov().values
    n = corr.shape[0]
    dist = np.sqrt(np.clip(0.5 * (1 - corr), 0, None))
    condensed = []
    for i in range(n):
        for j in range(i + 1, n):
            condensed.append(dist[i, j])
    link = linkage(condensed, method="single")
    sort_ix = get_quasi_diag(link, n)
    w_sorted = get_rec_bipart(cov, sort_ix)
    w = np.zeros(n)
    for pos, asset_idx in enumerate(sort_ix):
        w[asset_idx] = w_sorted[pos]
    return w / w.sum()


def select_recipe(monthly_slice):
    min_lookback = min(MONTHLY_LOOKBACK_CANDIDATES)
    if len(monthly_slice) < min_lookback + 6:
        return min_lookback, 1.0, 1.0

    def backtest_segment(monthly_lookback, hcap, ccap):
        if len(monthly_slice) < monthly_lookback + 3:
            return pd.Series(dtype=float)
        oos = []
        for i in range(monthly_lookback, len(monthly_slice)):
            trailing_m = monthly_slice.iloc[i - monthly_lookback:i]
            w = hrp_weights(trailing_m)
            w = apply_caps(w, {1: hcap, 0: ccap})
            oos.append(w @ monthly_slice.iloc[i].values)
        return pd.Series(oos)

    best_lb = None
    for lb in MONTHLY_LOOKBACK_CANDIDATES:
        ret = backtest_segment(lb, 1.0, 1.0)
        _, _, s, _ = sharpe_stats(ret, freq=12)
        if best_lb is None or (not np.isnan(s) and s > best_lb[0]):
            best_lb = (s, lb)
    sel_lb = best_lb[1]

    best_hcap = None
    for cap in HCAP_CANDIDATES:
        ret = backtest_segment(sel_lb, cap, 1.0)
        _, _, s, _ = sharpe_stats(ret, freq=12)
        if best_hcap is None or (not np.isnan(s) and s > best_hcap[0]):
            best_hcap = (s, cap)
    sel_hcap = best_hcap[1]

    best_ccap = None
    for cap in CCAP_CANDIDATES:
        ret = backtest_segment(sel_lb, sel_hcap, cap)
        _, _, s, _ = sharpe_stats(ret, freq=12)
        if best_ccap is None or (not np.isnan(s) and s > best_ccap[0]):
            best_ccap = (s, cap)
    sel_ccap = best_ccap[1]

    return sel_lb, sel_hcap, sel_ccap


def honest_pipeline(monthly_panel, daily_panel):
    monthly_dates = monthly_panel.index
    daily_weights = {}
    recipes = []

    i = min(MONTHLY_LOOKBACK_CANDIDATES)
    monthly_lookback = hcap = ccap = None
    months_since_retune = 0
    while i < len(monthly_dates):
        month_end_date = monthly_dates[i]
        if monthly_lookback is None or months_since_retune >= RETUNE_EVERY_MONTHS:
            avail_monthly = monthly_panel.iloc[:i]
            monthly_lookback, hcap, ccap = select_recipe(avail_monthly)
            recipes.append((month_end_date.date(), monthly_lookback, hcap, ccap))
            months_since_retune = 0
        months_since_retune += 1

        if i < monthly_lookback:
            i += 1
            continue
        trailing_m = monthly_panel.iloc[i - monthly_lookback:i]
        w = hrp_weights(trailing_m)
        w = apply_caps(w, {1: hcap, 0: ccap})

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
    print(f"HRP (hierarchical risk parity, structural caps + VolQ overlay unchanged): "
          f"n={n} sharpe={s:.2f} ann_ret={r:.2%} worst_day={w:.3%}")
    split_60, split_40 = int(n * 0.6), int(n * 0.4)
    fwd, rev = final.iloc[split_60:], final.iloc[:split_40]
    _, _, s_f, w_f = sharpe_stats(fwd)
    _, _, s_r, w_r = sharpe_stats(rev)
    print(f"FWD: sharpe={s_f:.2f} worst_day={w_f:.3%}   REV: sharpe={s_r:.2f} worst_day={w_r:.3%}")
    print("\nRecipe over time:")
    for d, lb, hcap, ccap in recipes:
        print(f"  {d}  lookback={lb}mo hcap={hcap:.0%} ccap={ccap:.0%}")
    final.to_csv("cpo_hrp_returns.csv")


if __name__ == "__main__":
    main()
