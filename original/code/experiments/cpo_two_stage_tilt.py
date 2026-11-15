"""
Two-stage architecture, per user request: keep the mean-variance
optimizer as the BASE allocation (unchanged, at its validated
monthly/6-month cadence), then apply a small, CAPPED, ML-ensemble-driven
TILT to the individual per-strategy weights at a faster (weekly)
cadence, using real daily data.

This is architecturally different from every one of the prior
approaches: those either fed ML into the mean-variance mu input
(replacing/blending it) or applied a single scalar to the WHOLE book.
This applies a bounded per-strategy adjustment ON TOP of an
already-computed, already-validated base allocation, and can never
override it by more than the cap allows.

Same full discipline: tilt magnitude cap, refresh cadence, and whether
to use a tilt at all are folded into the same periodic, trailing-data-
only reselection as everything else -- "no tilt" is always one of the
candidates, so the honest process can reject this architecture
entirely if it doesn't help, same as every overlay tested so far.

RESULT (17/07/2026): full-period Sharpe <real figure removed> (== baseline 4.22, no net
gain), FWD (held-out) Sharpe <real figure removed> vs baseline 3.92 (slightly worse).
Honest reselection chose to use the tilt in 5/8 periods (cap 20-35%)
-- looked worth using on trailing data, still didn't survive the
held-out split. Same failure signature as the Bayesian test.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from cpo_hybrid_monthly import build_extended_monthly_panel, MONTHLY_LOOKBACK_CANDIDATES
from cpo_daily_hardconstraint import load_daily_panel
from cpo_audit_fixed import volq_multiplier_fixed, per_strategy_v2_multiplier_fixed, apply_caps

STRATEGY_COLS = ["StrategyA", "StrategyB", "StrategyC", "StrategyD"]
RETUNE_EVERY_MONTHS = 6
TILT_HORIZON_DAYS = 5  # ~1 week forward target
TILT_REFRESH_DAYS = 5  # weekly refresh
TILT_CAP_CANDIDATES = [0.0, 0.10, 0.20, 0.35]  # 0.0 = no tilt allowed


def sharpe_stats(s, rf=0.07, freq=252):
    n = len(s)
    if n < 5:
        return np.nan, np.nan, np.nan, np.nan
    ann_ret = (1 + s).prod() ** (freq / n) - 1
    ann_vol = s.std() * np.sqrt(freq)
    sharpe = (ann_ret - rf) / ann_vol
    return ann_ret, ann_vol, sharpe, s.min()


def hard_constrained_weights(mu, sigma, daily_scenarios, dd_limit, lam, cap_indices=None):
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


def build_daily_tilt_features(daily_panel):
    """Per-strategy short-horizon features from real daily data, all
    shift(1)'d so day t's features only use data through t-1."""
    feats_by_col = {}
    for col in STRATEGY_COLS:
        r = daily_panel[col]
        f = pd.DataFrame(index=daily_panel.index)
        f["mom5"] = r.rolling(5).sum().shift(1)
        f["mom10"] = r.rolling(10).sum().shift(1)
        f["vol10"] = r.rolling(10).std().shift(1)
        f["cs_mom5"] = daily_panel.rolling(5).sum().shift(1).rank(axis=1, pct=True)[col]
        f["cs_vol10"] = 1.0 - daily_panel.rolling(10).std().shift(1).rank(axis=1, pct=True)[col]
        feats_by_col[col] = f
    return feats_by_col


def make_classifiers():
    return [
        lgb.LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, num_leaves=8,
                            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1),
        Pipeline([("scaler", StandardScaler()),
                  ("clf", LogisticRegression(penalty="l2", C=5, solver="liblinear", max_iter=1000, random_state=42))]),
        Pipeline([("scaler", StandardScaler()),
                  ("clf", ExtraTreesClassifier(n_estimators=100, max_depth=4, min_samples_leaf=15,
                                                max_features="sqrt", random_state=42, n_jobs=1))]),
        Pipeline([("scaler", StandardScaler()),
                  ("clf", RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=15,
                                                  max_features="sqrt", random_state=42, n_jobs=1))]),
    ]


def tilt_signal(daily_panel_upto, features_by_col, as_of_idx, horizon=TILT_HORIZON_DAYS, min_train=100):
    """Per-strategy tilt score in [-0.5, 0.5] (0 = no tilt), from an
    ensemble of classifiers predicting P(next `horizon`-day return > 0)."""
    signal = np.zeros(len(STRATEGY_COLS))
    for k, col in enumerate(STRATEGY_COLS):
        x_today = features_by_col[col].iloc[[as_of_idx]]
        if x_today.isna().any(axis=1).values[0]:
            continue
        r = daily_panel_upto[col]
        fwd_ret = r.shift(-1).rolling(horizon).sum().shift(-(horizon - 1))
        usable_end = as_of_idx - horizon
        X_all = features_by_col[col].iloc[:usable_end]
        y_all = (fwd_ret.iloc[:usable_end] > 0).astype(int)
        valid = X_all.notna().all(axis=1) & fwd_ret.iloc[:usable_end].notna()
        X_train, y_train = X_all[valid], y_all[valid]
        if len(X_train) < min_train or len(np.unique(y_train)) < 2:
            continue
        scaler = StandardScaler().fit(X_train)
        probs = []
        for clf in make_classifiers():
            try:
                clf.fit(scaler.transform(X_train), y_train)
                probs.append(clf.predict_proba(scaler.transform(x_today))[0, 1])
            except Exception:
                pass
        if len(probs) >= 2:
            signal[k] = np.mean(probs) - 0.5
    return signal


def apply_tilt(base_w, signal, tilt_cap):
    if tilt_cap <= 0.0:
        return base_w
    # multiplicative tilt, bounded, then renormalize
    adj = base_w * (1.0 + np.clip(signal, -0.5, 0.5) * 2 * tilt_cap)
    adj = np.clip(adj, 0.0, 1.0)
    if adj.sum() > 0:
        adj = adj / adj.sum()
    return adj


def select_recipe(monthly_slice, daily_slice_upto, daily_features_upto):
    min_lookback = min(MONTHLY_LOOKBACK_CANDIDATES)
    if len(monthly_slice) < min_lookback + 6:
        return 0.008, 10, 1.0, 1.0, min_lookback, 0.0

    def base_weights_series(dd_limit, lam, hcap, ccap, monthly_lookback):
        """Returns {month_end_date: (base_w, daily_scen_used)} for months
        within monthly_slice -- reused across tilt-cap candidates."""
        out = {}
        if len(monthly_slice) < monthly_lookback + 3:
            return out
        for i in range(monthly_lookback, len(monthly_slice)):
            trailing_m = monthly_slice.iloc[i - monthly_lookback:i]
            mu = trailing_m.mean().values
            sigma = trailing_m.cov().values
            month_end_date = monthly_slice.index[i]
            daily_scen = daily_slice_upto[daily_slice_upto.index < month_end_date].tail(252).values
            w, ok = hard_constrained_weights(mu, sigma, daily_scen, dd_limit, lam, {1: hcap, 0: ccap})
            out[month_end_date] = w
        return out

    def backtest_segment(dd_limit, lam, hcap, ccap, monthly_lookback, tilt_cap):
        base_by_month = base_weights_series(dd_limit, lam, hcap, ccap, monthly_lookback)
        if not base_by_month:
            return pd.Series(dtype=float)
        oos = []
        for month_end_date, base_w in base_by_month.items():
            i_pos = monthly_slice.index.get_loc(month_end_date)
            if i_pos + 1 >= len(monthly_slice.index):
                continue
            actual = monthly_slice.iloc[i_pos].values
            if tilt_cap > 0.0:
                # approximate: apply ONE tilt signal (as of month start) for
                # the whole month in this fast recipe-selection pass --
                # the full honest_pipeline refreshes weekly properly.
                daily_idx = daily_slice_upto.index[daily_slice_upto.index < month_end_date]
                if len(daily_idx) > 100:
                    as_of_idx = daily_slice_upto.index.get_loc(daily_idx[-1])
                    sig = tilt_signal(daily_slice_upto, daily_features_upto, as_of_idx)
                    w = apply_tilt(base_w, sig, tilt_cap)
                else:
                    w = base_w
            else:
                w = base_w
            oos.append(w @ actual)
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

    best_tilt = None
    for tilt_cap in TILT_CAP_CANDIDATES:
        ret = backtest_segment(sel_dd, sel_lam, sel_hcap, sel_ccap, sel_lb, tilt_cap)
        _, _, s, _ = sharpe_stats(ret, freq=12)
        if best_tilt is None or (not np.isnan(s) and s > best_tilt[0]):
            best_tilt = (s, tilt_cap)
    sel_tilt = best_tilt[1]

    return sel_dd, sel_lam, sel_hcap, sel_ccap, sel_lb, sel_tilt


def honest_pipeline(monthly_panel, daily_panel, daily_features):
    monthly_dates = monthly_panel.index
    daily_weights = {}
    recipes = []

    i = min(MONTHLY_LOOKBACK_CANDIDATES)
    dd_limit = lam = hcap = ccap = monthly_lookback = tilt_cap = None
    months_since_retune = 0
    while i < len(monthly_dates):
        month_end_date = monthly_dates[i]
        if dd_limit is None or months_since_retune >= RETUNE_EVERY_MONTHS:
            avail_monthly = monthly_panel.iloc[:i]
            avail_daily = daily_panel[daily_panel.index < month_end_date]
            avail_feats = {c: daily_features[c][daily_features[c].index < month_end_date] for c in STRATEGY_COLS}
            dd_limit, lam, hcap, ccap, monthly_lookback, tilt_cap = select_recipe(
                avail_monthly, avail_daily, avail_feats)
            recipes.append((month_end_date.date(), dd_limit, lam, hcap, ccap, monthly_lookback, tilt_cap))
            months_since_retune = 0
        months_since_retune += 1

        if i < monthly_lookback:
            i += 1
            continue
        trailing_m = monthly_panel.iloc[i - monthly_lookback:i]
        mu = trailing_m.mean().values
        sigma = trailing_m.cov().values
        daily_scen = daily_panel[daily_panel.index < month_end_date].tail(252).values
        base_w, ok = hard_constrained_weights(mu, sigma, daily_scen, dd_limit, lam, {1: hcap, 0: ccap})

        month_start = (month_end_date - pd.offsets.MonthEnd(1))
        days_in_month = daily_panel[(daily_panel.index > month_start) & (daily_panel.index <= month_end_date)].index

        # weekly tilt refresh within the month, using only data through
        # each week's own start (no look-ahead)
        last_tilt_w = base_w
        for j, d in enumerate(days_in_month):
            if tilt_cap > 0.0 and j % TILT_REFRESH_DAYS == 0:
                hist_upto_d = daily_panel[daily_panel.index < d]
                if len(hist_upto_d) > 100:
                    as_of_idx = daily_panel.index.get_loc(hist_upto_d.index[-1])
                    feats_upto = {c: daily_features[c][daily_features[c].index <= hist_upto_d.index[-1]]
                                  for c in STRATEGY_COLS}
                    sig = tilt_signal(daily_panel[daily_panel.index <= d], feats_upto, as_of_idx)
                    last_tilt_w = apply_tilt(base_w, sig, tilt_cap)
            daily_weights[d] = last_tilt_w if tilt_cap > 0.0 else base_w
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
    daily_features = build_daily_tilt_features(daily_panel)

    final, recipes = honest_pipeline(monthly_panel, daily_panel, daily_features)
    if final is None:
        print("insufficient data")
        return
    r, v, s, w = sharpe_stats(final)
    n = len(final)
    print(f"Two-stage (MV base + weekly ML tilt): n={n} sharpe={s:.2f} ann_ret={r:.2%} worst_day={w:.3%}")
    split_60, split_40 = int(n * 0.6), int(n * 0.4)
    fwd, rev = final.iloc[split_60:], final.iloc[:split_40]
    _, _, s_f, w_f = sharpe_stats(fwd)
    _, _, s_r, w_r = sharpe_stats(rev)
    print(f"FWD: sharpe={s_f:.2f} worst_day={w_f:.3%}   REV: sharpe={s_r:.2f} worst_day={w_r:.3%}")
    print("\nRecipe over time (tilt_cap=0.0 means no tilt used):")
    for d, dd, lam, hcap, ccap, lb, tc in recipes:
        print(f"  {d}  dd={dd:.1%} lam={lam} hcap={hcap:.0%} ccap={ccap:.0%} lookback={lb}mo tilt_cap={tc:.0%}")
    final.to_csv("cpo_two_stage_tilt_returns.csv")


if __name__ == "__main__":
    main()
