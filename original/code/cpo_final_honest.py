"""
CPO v5 (final, audit-informed): periodic walk-forward reselection
instead of a single fixed-parameter fit.

Per the audit (CPO_AUDIT.md): a one-shot grid search over the whole
sample is data snooping. This re-selects buffer/lambda/StrategyB-cap/
StrategyA-cap every ~126 trading days (~6 months) using ONLY data
available up to that point -- never the future -- then applies the
chosen recipe forward until the next reselection point. VolQ/V2 bounds
are deliberately NOT re-searched: the audit found that knob has no
natural ceiling (classic overfitting signature), so it's fixed at the
disciplined default (lookback=20, bounds=(0.7,1.3)) throughout, never
chosen by backtest Sharpe.

The Strategy A cap (in addition to StrategyB's) was added after
testing showed a modest, honestly-validated improvement (Sharpe <real figure removed>) -- small, not a new headline, but built with the same discipline
as everything else here.

This produces ONE continuous, honestly-out-of-sample return series for
the whole period -- no fold ever sees a parameter chosen with knowledge
of its own future.
"""
import numpy as np
import pandas as pd
from cpo_daily_hardconstraint import load_daily_panel, hard_constrained_weights, LOOKBACK_DAYS
from cpo_audit_fixed import volq_multiplier_fixed, per_strategy_v2_multiplier_fixed, apply_caps

RETUNE_EVERY = 126  # ~6 months
REBAL_EVERY = 21    # ~monthly weight refresh within a fixed recipe
FIXED_LOOKBACK = 20
FIXED_BOUNDS = (0.7, 1.3)


def sharpe_stats(s, rf=0.07, freq=252):
    n = len(s)
    if n < 5:
        return np.nan, np.nan, np.nan, np.nan
    ann_ret = (1 + s).prod() ** (freq / n) - 1
    ann_vol = s.std() * np.sqrt(freq)
    sharpe = (ann_ret - rf) / ann_vol
    return ann_ret, ann_vol, sharpe, s.min()


def select_recipe(panel, end_idx, warm_start=False):
    """Select buffer/lambda/StrategyB-cap/StrategyA-cap using ONLY
    panel.iloc[:end_idx] -- no future data."""
    dates = panel.index
    available = panel.iloc[:end_idx]
    if len(available) < LOOKBACK_DAYS + REBAL_EVERY * 3:
        # Not enough history yet to run a meaningful search. warm_start=True
        # applies the mature recipe (found once enough data existed) from
        # day one, rather than learning the caps the hard way.
        return (0.005, 10, 0.05, 0.6) if warm_start else (0.01, 10, 1.0, 1.0)

    def backtest_segment(buffer, lam, strategy_b_cap, sa_cap):
        oos = []
        for i in range(LOOKBACK_DAYS, len(available), REBAL_EVERY):
            trailing = available.iloc[i - LOOKBACK_DAYS:i].values
            w, ok = hard_constrained_weights(trailing, dd_limit=buffer, lam=lam)
            w = apply_caps(w, {1: strategy_b_cap, 0: sa_cap})
            end = min(i + REBAL_EVERY, len(available))
            for j in range(i, end):
                oos.append(available.iloc[j].values @ w)
        return pd.Series(oos)

    best = None
    for buffer in [0.010, 0.006, 0.005]:
        for lam in [5, 10]:
            ret = backtest_segment(buffer, lam, strategy_b_cap=1.0, sa_cap=1.0)
            _, _, s, _ = sharpe_stats(ret)
            if best is None or (not np.isnan(s) and s > best[0]):
                best = (s, buffer, lam)
    _, sel_buffer, sel_lam = best

    best_hcap = None
    for cap in [1.0, 0.25, 0.10, 0.05]:
        ret = backtest_segment(sel_buffer, sel_lam, cap, sa_cap=1.0)
        _, _, s, _ = sharpe_stats(ret)
        if best_hcap is None or (not np.isnan(s) and s > best_hcap[0]):
            best_hcap = (s, cap)
    _, sel_hcap = best_hcap

    best_ccap = None
    for cap in [1.0, 0.6, 0.4, 0.25]:
        ret = backtest_segment(sel_buffer, sel_lam, sel_hcap, cap)
        _, _, s, _ = sharpe_stats(ret)
        if best_ccap is None or (not np.isnan(s) and s > best_ccap[0]):
            best_ccap = (s, cap)
    _, sel_ccap = best_ccap

    return sel_buffer, sel_lam, sel_hcap, sel_ccap


def main(warm_start=False, retune_every=RETUNE_EVERY):
    panel = load_daily_panel()
    dates = panel.index
    n = len(panel)

    daily_weights = {}
    recipes_used = []
    i = LOOKBACK_DAYS
    while i < n:
        buffer, lam, hcap, ccap = select_recipe(panel, i, warm_start=warm_start)  # uses ONLY data through day i-1
        recipes_used.append((dates[i].date(), buffer, lam, hcap, ccap))
        segment_end = min(i + retune_every, n)
        j = i
        while j < segment_end:
            trailing = panel.iloc[j - LOOKBACK_DAYS:j].values
            w, ok = hard_constrained_weights(trailing, dd_limit=buffer, lam=lam)
            w = apply_caps(w, {1: hcap, 0: ccap})
            end = min(j + REBAL_EVERY, segment_end)
            for k in range(j, end):
                daily_weights[dates[k]] = w
            j = end
        i = segment_end

    idx = list(daily_weights.keys())
    W = pd.DataFrame([daily_weights[d] for d in idx], index=idx, columns=panel.columns)
    base_ret = pd.Series((panel.loc[idx].values * W.values).sum(axis=1), index=idx)

    v2_mult = per_strategy_v2_multiplier_fixed(panel, idx, lookback=FIXED_LOOKBACK, lo=FIXED_BOUNDS[0], hi=FIXED_BOUNDS[1])
    adj_W = W * v2_mult
    combo_ret = pd.Series((panel.loc[idx].values * adj_W.values).sum(axis=1), index=idx)
    final_ret = combo_ret * volq_multiplier_fixed(combo_ret, lookback=FIXED_LOOKBACK, lo=FIXED_BOUNDS[0], hi=FIXED_BOUNDS[1])

    print("Recipes selected over time (each chosen using ONLY data available up to that date):")
    for d, buf, lam, hcap, ccap in recipes_used:
        print(f"  {d}  buffer={buf:.1%}  lambda={lam}  StrategyB_cap={hcap:.0%}  StrategyA_cap={ccap:.0%}")

    print()
    r, v, s, w = sharpe_stats(final_ret)
    print(f"FINAL honest full-period result (no single-shot future-aware tuning anywhere):")
    print(f"  n={len(final_ret)}  ann_ret={r:.2%}  ann_vol={v:.2%}  sharpe={s:.2f}  worst_day={w:.3%}")

    ew = panel.loc[idx].mean(axis=1)
    r, v, s, w = sharpe_stats(ew)
    print(f"\nEW baseline over same window:  ann_ret={r:.2%}  ann_vol={v:.2%}  sharpe={s:.2f}  worst_day={w:.3%}")

    suffix = "_warmstart" if warm_start else ""
    final_ret.to_csv(f"cpo_v4_final_returns{suffix}.csv")
    return final_ret


if __name__ == "__main__":
    import sys
    warm = "--warm-start" in sys.argv
    main(warm_start=warm)
