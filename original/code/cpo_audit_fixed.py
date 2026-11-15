"""
Audit + fix for CPO v3: the VolQ and per-strategy V2 overlays in
cpo_volq_overlay.py compute "today's" trailing vol using rolling(window)
.std(), which is right-aligned and INCLUDES today's own return. Since
that value is then used to size exposure applied to today's return,
this is look-ahead bias -- today's exposure decision was allowed to see
a sliver of today's own outcome.

Fix: shift the vol series by 1 day before using it to size exposure, so
day t's multiplier only ever depends on information available through
day t-1.
"""
import numpy as np
import pandas as pd
from cpo_daily_hardconstraint import load_daily_panel, hard_constrained_weights, LOOKBACK_DAYS, REBAL_EVERY


def sharpe_stats(s, rf=0.07, freq=252):
    n = len(s)
    ann_ret = (1 + s).prod() ** (freq / n) - 1
    ann_vol = s.std() * np.sqrt(freq)
    sharpe = (ann_ret - rf) / ann_vol
    return ann_ret, ann_vol, sharpe, s.min()


def volq_multiplier_fixed(port_ret, lookback=20, lo=0.7, hi=1.3, min_hist=60):
    """Same as volq_multiplier but vol is shifted 1 day -- day t's multiplier
    only sees vol computed through day t-1."""
    trailing_vol = port_ret.rolling(lookback).std().shift(1)  # <-- the fix
    mult = pd.Series(1.0, index=port_ret.index)
    for i in range(len(port_ret)):
        if i < min_hist:
            continue
        hist = trailing_vol.iloc[:i].dropna()  # already excludes today via the shift + slice
        cur = trailing_vol.iloc[i]
        if len(hist) < 20 or pd.isna(cur):
            continue
        q = (hist < cur).mean()
        mult.iloc[i] = hi - q * (hi - lo)
    return mult


def per_strategy_v2_multiplier_fixed(returns_panel, idx, lookback=20, lo=0.7, hi=1.3, min_hist=60):
    mult = pd.DataFrame(1.0, index=idx, columns=returns_panel.columns)
    for col in returns_panel.columns:
        r = returns_panel[col].loc[:idx[-1]]
        trailing_vol = r.rolling(lookback).std().shift(1)  # <-- the fix
        for i, d in enumerate(idx):
            if i < min_hist:
                continue
            hist = trailing_vol.loc[:d].iloc[:-1].dropna()
            cur = trailing_vol.loc[d]
            if len(hist) < 20 or pd.isna(cur):
                continue
            q = (hist < cur).mean()
            mult.loc[d, col] = hi - q * (hi - lo)
    return mult


def apply_strategy_b_cap(w, cap=0.05):
    return apply_caps(w, {1: cap})


def apply_caps(w, cap_indices):
    """cap_indices: dict {column_index: cap_fraction}. Applies each cap in
    turn with proportional redistribution among the other strategies,
    repeating a few passes since redistribution can push another strategy
    over its own cap."""
    w = w.copy()
    n = len(w)
    for _ in range(3):
        moved = False
        for idx_, cap in cap_indices.items():
            if w[idx_] > cap + 1e-9:
                excess = w[idx_] - cap
                w[idx_] = cap
                others = [k for k in range(n) if k != idx_]
                denom = sum(w[k] for k in others)
                if denom > 0:
                    for k in others:
                        w[k] += excess * (w[k] / denom)
                else:
                    for k in others:
                        w[k] += excess / len(others)
                moved = True
        if not moved:
            break
    return w


def main():
    panel = load_daily_panel()
    dates = panel.index

    print("=== Re-running the full stack with leakage FIXED ===\n")

    # Step 1: base + StrategyB cap (this part had no leakage -- weights already
    # fit on strictly-prior data in cpo_daily_hardconstraint.py)
    daily_weights = {}
    for i in range(LOOKBACK_DAYS, len(dates), REBAL_EVERY):
        trailing = panel.iloc[i - LOOKBACK_DAYS:i].values
        w, ok = hard_constrained_weights(trailing, dd_limit=0.005, lam=10)
        w = apply_strategy_b_cap(w)
        end = min(i + REBAL_EVERY, len(dates))
        for j in range(i, end):
            daily_weights[dates[j]] = w
    idx = list(daily_weights.keys())
    W = pd.DataFrame([daily_weights[d] for d in idx], index=idx, columns=panel.columns)
    capped_ret = pd.Series((panel.loc[idx].values * W.values).sum(axis=1), index=idx)
    r, v, s, w_ = sharpe_stats(capped_ret)
    print(f"1. StrategyB cap=5% only (no leakage here)         sharpe={s:.2f}  ann_ret={r:.2%}  worst_day={w_:.3%}")

    # Step 2: book-level VolQ, FIXED (shifted)
    volq_ret = capped_ret * volq_multiplier_fixed(capped_ret)
    r, v, s, w_ = sharpe_stats(volq_ret)
    print(f"2. + book-level VolQ [FIXED, no leakage]      sharpe={s:.2f}  ann_ret={r:.2%}  worst_day={w_:.3%}")

    # Step 3: + per-strategy V2, FIXED
    v2_mult = per_strategy_v2_multiplier_fixed(panel, idx)
    adj_W = W * v2_mult
    combo_ret = pd.Series((panel.loc[idx].values * adj_W.values).sum(axis=1), index=idx)
    final_ret = combo_ret * volq_multiplier_fixed(combo_ret)
    r, v, s, w_ = sharpe_stats(final_ret)
    print(f"3. + per-strategy V2 [FIXED, full stack]      sharpe={s:.2f}  ann_ret={r:.2%}  worst_day={w_:.3%}")

    final_ret.to_csv("cpo_v3_fixed_returns.csv")
    return final_ret


if __name__ == "__main__":
    main()
