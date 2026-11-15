"""
Hybrid CPO: extended monthly history informs mu/covariance estimation
(more data points -> more robust estimates), while the actual 2% daily
drawdown constraint stays enforced using real daily data -- since
monthly returns average away single-day risk entirely and can't verify
that constraint, only daily data can.

Per Client's data-availability confirmation (16/07/2026): daily data
doesn't exist before Apr 2023 for Strategy A, Strategy C
Intraday, or Strategy D. Monthly data goes back further, but the common
window across all 4 strategies is capped by StrategyB (Jan 2021 -- no
monthly tab was ever provided for StrategyB, only its own daily history
starting then). Strategy A's extended monthly series (Fund Class
A, since 2015) is a DIFFERENT SHARE CLASS than the daily series used
elsewhere in this project (Fund Class D) -- flagged, not silently
substituted.

Design: at each rebalance point (still ~monthly cadence), mu and
covariance are estimated from a trailing window of the EXTENDED
monthly panel (real monthly data 2021-Mar2023, spliced with the daily
data aggregated to monthly from Apr2023 onward). The hard drawdown
constraint is still built from real daily scenarios -- using whatever
daily history exists in the trailing window (thin at first, growing
over time, same cold-start dynamic already documented for the
daily-only baseline). Backtest performance is reported only over
Apr2023+ (the only period with real daily data to verify against).
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from cpo_daily_hardconstraint import load_daily_panel
from cpo_audit_fixed import volq_multiplier_fixed, per_strategy_v2_multiplier_fixed, apply_caps

STRATEGY_COLS = ["StrategyA", "StrategyB", "StrategyC", "StrategyD"]
REBAL_EVERY = 21
RETUNE_EVERY = 126
DD_LIMIT_DEFAULT = 0.005


def sharpe_stats(s, rf=0.07, freq=252):
    n = len(s)
    if n < 5:
        return np.nan, np.nan, np.nan, np.nan
    ann_ret = (1 + s).prod() ** (freq / n) - 1
    ann_vol = s.std() * np.sqrt(freq)
    sharpe = (ann_ret - rf) / ann_vol
    return ann_ret, ann_vol, sharpe, s.min()


def month_end(dates, values):
    s = pd.Series(np.asarray(values), index=pd.to_datetime(np.asarray(dates))).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s.index = s.index + pd.offsets.MonthEnd(0)
    return s


def load_simple(xlsx, sheet, skiprows=0, cols=None):
    df = pd.read_excel(xlsx, sheet_name=sheet, skiprows=skiprows)
    df = df[cols] if cols else df.iloc[:, [0, 1]]
    df = df.dropna()
    df.columns = ["date", "ret"]
    return month_end(df["date"], df["ret"])


def build_extended_monthly_panel(xlsx="raw_data/Client_ReturnSeries.xlsx"):
    """Real monthly data pre-Apr2023 (per strategy's own earliest
    reliable monthly source), spliced with the real daily data
    aggregated to monthly from Apr2023 onward -- NOT the same series
    used twice, spliced cleanly at the point where daily data begins."""
    sa_monthly_early = load_simple(xlsx, "Fund Class A ($)")  # share-class caveat, see module docstring
    strategy_b_daily = load_simple(xlsx, "StrategyB Strategy_Daily INR")
    strategy_b_monthly_full = (1 + strategy_b_daily).resample("ME").prod() - 1
    strategy_c_monthly_early = load_simple(xlsx, "Strategy C_Monthly INR")
    strategy_d_monthly_early = load_simple(xlsx, "Strategy D_Monthly INR")  # confirmed correct by Client, not Long History

    daily_panel = load_daily_panel()
    daily_to_monthly = (1 + daily_panel).resample("ME").prod() - 1
    splice_date = daily_to_monthly.index[0]  # first month with real daily-derived data

    def splice(early_monthly, col):
        early = early_monthly[early_monthly.index < splice_date]
        late = daily_to_monthly[col]
        return pd.concat([early, late]).sort_index()

    panel = pd.DataFrame({
        "StrategyA": splice(sa_monthly_early, "StrategyA"),
        "StrategyB": splice(strategy_b_monthly_full, "StrategyB"),
        "StrategyC": splice(strategy_c_monthly_early, "StrategyC"),
        "StrategyD": splice(strategy_d_monthly_early, "StrategyD"),
    })
    common_start = panel.dropna().index[0]  # StrategyB is the binding constraint (Jan 2021)
    panel = panel[panel.index >= common_start]
    # Known source-data issue: both "Strategy C_Monthly INR" and
    # "Strategy D_Monthly INR" have a duplicated 2021-06-01 date entry
    # where an 2021-08-01 entry should be (same row, same error, in
    # both tabs) -- flagged for Client, dropped here rather than guessed.
    panel = panel.dropna()
    return panel, daily_panel


def hard_constrained_weights(mu, sigma, daily_scenarios, dd_limit, lam, cap_indices=None):
    """Same historical-scenario constraint as the rest of this project,
    but scenarios come from whatever real daily data is available in
    the trailing window (may be thin or empty early on)."""
    n = len(mu)

    def neg_obj(w):
        return -(w @ mu - 0.5 * lam * (w @ sigma @ w))

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    for t in range(len(daily_scenarios)):
        r_t = daily_scenarios[t]
        cons.append({"type": "ineq", "fun": (lambda w, r_t=r_t: w @ r_t + dd_limit)})
    bounds = [(0.0, 1.0)] * n
    res = minimize(neg_obj, np.full(n, 1.0 / n), method="SLSQP", bounds=bounds,
                    constraints=cons, options={"maxiter": 300, "ftol": 1e-9})
    w = res.x.copy()
    if cap_indices:
        w = apply_caps(w, cap_indices)
    return w, res.success


MONTHLY_LOOKBACK_CANDIDATES = [18, 24, 36, 48]


def select_recipe(monthly_slice, daily_slice_upto):
    """Selects dd_limit/lambda/StrategyB-cap/SA-cap/monthly_lookback using
    ONLY monthly_slice and daily_slice_upto -- both already restricted
    to trailing data by the caller. monthly_lookback is chosen here,
    not swept externally and picked by eye, specifically to avoid the
    test-set-contamination mistake made (and fixed) three times
    already this project with other parameters."""
    min_lookback = min(MONTHLY_LOOKBACK_CANDIDATES)
    if len(monthly_slice) < min_lookback + 6:
        return 0.008, 10, 1.0, 1.0, min_lookback

    def backtest_segment(dd_limit, lam, hcap, ccap, monthly_lookback):
        if len(monthly_slice) < monthly_lookback + 3:
            return pd.Series(dtype=float)
        oos = []
        for i in range(monthly_lookback, len(monthly_slice)):
            trailing_m = monthly_slice.iloc[i - monthly_lookback:i]
            mu = trailing_m.mean().values
            sigma = trailing_m.cov().values
            month_end_date = monthly_slice.index[i]
            daily_scen = daily_slice_upto[daily_slice_upto.index < month_end_date].tail(252).values
            w, ok = hard_constrained_weights(mu, sigma, daily_scen, dd_limit, lam, {1: hcap, 0: ccap})
            actual = monthly_slice.iloc[i].values
            oos.append(w @ actual)
        return pd.Series(oos)

    best = None
    for dd_limit in [0.010, 0.006, 0.005]:
        for lam in [5, 10]:
            ret = backtest_segment(dd_limit, lam, 1.0, 1.0, 36)
            _, _, s, _ = sharpe_stats(ret, freq=12)
            if best is None or (not np.isnan(s) and s > best[0]):
                best = (s, dd_limit, lam)
    _, sel_dd, sel_lam = best

    best_hcap = None
    for cap in [1.0, 0.25, 0.10, 0.05]:
        ret = backtest_segment(sel_dd, sel_lam, cap, 1.0, 36)
        _, _, s, _ = sharpe_stats(ret, freq=12)
        if best_hcap is None or (not np.isnan(s) and s > best_hcap[0]):
            best_hcap = (s, cap)
    sel_hcap = best_hcap[1]

    best_ccap = None
    for cap in [1.0, 0.6, 0.4, 0.25]:
        ret = backtest_segment(sel_dd, sel_lam, sel_hcap, cap, 36)
        _, _, s, _ = sharpe_stats(ret, freq=12)
        if best_ccap is None or (not np.isnan(s) and s > best_ccap[0]):
            best_ccap = (s, cap)
    sel_ccap = best_ccap[1]

    best_lookback = None
    for monthly_lookback in MONTHLY_LOOKBACK_CANDIDATES:
        ret = backtest_segment(sel_dd, sel_lam, sel_hcap, sel_ccap, monthly_lookback)
        _, _, s, _ = sharpe_stats(ret, freq=12)
        if best_lookback is None or (not np.isnan(s) and s > best_lookback[0]):
            best_lookback = (s, monthly_lookback)
    sel_lookback = best_lookback[1]

    return sel_dd, sel_lam, sel_hcap, sel_ccap, sel_lookback


def honest_pipeline(monthly_panel, daily_panel):
    """Rebalance monthly (matching the monthly estimation cadence).
    Report performance ONLY over the window where real daily data
    exists (Apr2023+) -- that's the only period we can verify the
    daily constraint or realized worst-day against. monthly_lookback
    is now selected per-period by select_recipe, not fixed here."""
    monthly_dates = monthly_panel.index
    daily_weights = {}
    recipes = []
    RETUNE_EVERY_MONTHS = 6  # matches the ~6-month cadence used everywhere else in this project

    i = min(MONTHLY_LOOKBACK_CANDIDATES)
    dd_limit = lam = hcap = ccap = monthly_lookback = None
    months_since_retune = 0
    while i < len(monthly_dates):
        month_end_date = monthly_dates[i]
        if dd_limit is None or months_since_retune >= RETUNE_EVERY_MONTHS:
            avail_monthly = monthly_panel.iloc[:i]
            avail_daily = daily_panel[daily_panel.index < month_end_date]
            dd_limit, lam, hcap, ccap, monthly_lookback = select_recipe(avail_monthly, avail_daily)
            recipes.append((month_end_date.date(), dd_limit, lam, hcap, ccap, monthly_lookback))
            months_since_retune = 0
        months_since_retune += 1

        if i < monthly_lookback:
            i += 1
            continue
        trailing_m = monthly_panel.iloc[i - monthly_lookback:i]
        mu = trailing_m.mean().values
        sigma = trailing_m.cov().values
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
    base_ret = pd.Series((daily_panel.loc[idx].values * W.values).sum(axis=1), index=idx)

    v2_mult = per_strategy_v2_multiplier_fixed(daily_panel, idx, lookback=20, lo=0.7, hi=1.3)
    adj_W = W.mul(v2_mult, axis=0)
    combo_ret = pd.Series((daily_panel.loc[idx].values * adj_W.values).sum(axis=1), index=idx)
    final = combo_ret * volq_multiplier_fixed(combo_ret, lookback=20, lo=0.7, hi=1.3)
    return final, recipes


def main():
    monthly_panel, daily_panel = build_extended_monthly_panel()
    print(f"Extended monthly panel: {monthly_panel.index[0].date()} -> {monthly_panel.index[-1].date()}, "
          f"{len(monthly_panel)} months")
    print(f"Daily panel (for constraint + backtest): {daily_panel.index[0].date()} -> {daily_panel.index[-1].date()}, "
          f"{len(daily_panel)} days\n")

    final, recipes = honest_pipeline(monthly_panel, daily_panel)
    if final is None:
        print("Pipeline returned None -- insufficient data.")
        return
    r, v, s, w = sharpe_stats(final)
    print(f"HYBRID (monthly-estimated mu/cov, daily-verified constraint, monthly_lookback honestly selected):")
    print(f"  n={len(final)} days  ann_ret={r:.2%}  ann_vol={v:.2%}  sharpe={s:.2f}  worst_day={w:.3%}")
    print(f"  ({'WITHIN' if w >= -0.02 else 'BREACHED'} the 2% daily limit)")

    n = len(final)
    split_60, split_40 = int(n * 0.6), int(n * 0.4)
    fwd, rev = final.iloc[split_60:], final.iloc[:split_40]
    _, _, s_f, w_f = sharpe_stats(fwd)
    _, _, s_r, w_r = sharpe_stats(rev)
    print(f"\n  FWD (harder regime, n={len(fwd)}): sharpe={s_f:.2f} worst_day={w_f:.3%}")
    print(f"  REV (easier regime, n={len(rev)}): sharpe={s_r:.2f} worst_day={w_r:.3%}")

    print("\nRecipes over time:")
    for d, dd, lam, hcap, ccap, lb in recipes:
        print(f"  {d}  dd_limit={dd:.1%} lambda={lam} StrategyB_cap={hcap:.0%} SA_cap={ccap:.0%} monthly_lookback={lb}mo")

    final.to_csv("cpo_hybrid_monthly_returns.csv")


if __name__ == "__main__":
    main()
