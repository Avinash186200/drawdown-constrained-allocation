"""
Approach 19: feature conditioning / interaction effects. Every prior
ML feature set (approaches 1, 6, 7, 10, 12) fed momentum, volatility,
and cross-sectional-rank features into a model as separate, additive
inputs. This explicitly engineers INTERACTION terms -- "momentum
matters, but especially when the book is in an elevated-volatility
regime" or "especially when cross-strategy correlation has spiked" --
rather than leaving it to chance whether the tree-based models in the
ensemble stumble onto that combination with very little data.

Same architecture as approach 10 (capped weekly tilt on top of the
honest MV base allocation), same classifier ensemble, so this isolates
ONE thing: does adding explicit interaction features to the tilt
signal change the result versus approach 10's main-effects-only
version (FWD 3.82 vs. 3.92 baseline, tilt used 5/8 periods)?

Two feature sets tested head-to-head, same everything else:
  (a) main effects only (momentum, vol, cross-sectional rank) --
      reproduces approach 10's design as the control
  (b) main effects + explicit interaction terms (momentum x book-level
      vol regime, momentum x cross-strategy correlation dispersion,
      cross-sectional momentum rank x vol regime)
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from cpo_hybrid_monthly import build_extended_monthly_panel, MONTHLY_LOOKBACK_CANDIDATES
from cpo_audit_fixed import volq_multiplier_fixed, per_strategy_v2_multiplier_fixed, apply_caps

STRATEGY_COLS = ["StrategyA", "StrategyB", "StrategyC", "StrategyD"]
RETUNE_EVERY_MONTHS = 6
TILT_HORIZON_DAYS = 5
TILT_REFRESH_DAYS = 5
TILT_CAP_CANDIDATES = [0.0, 0.10, 0.20, 0.35]


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


def build_features(daily_panel, use_interactions):
    """Main-effect features for every strategy, plus (if
    use_interactions) explicit interaction terms combining a
    strategy's own momentum with book-level regime state. All
    shift(1)'d -- day t's features only use data through t-1."""
    book_ret = daily_panel[STRATEGY_COLS].mean(axis=1)
    book_vol_20 = book_ret.rolling(20).std().shift(1)
    book_vol_ref = book_vol_20.rolling(120, min_periods=40).median()
    vol_regime = (book_vol_20 / book_vol_ref).fillna(1.0)  # >1 = elevated-vol regime

    pairwise_corrs = []
    cols = STRATEGY_COLS
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairwise_corrs.append(
                daily_panel[cols[i]].rolling(60).corr(daily_panel[cols[j]]).shift(1))
    corr_dispersion = pd.concat(pairwise_corrs, axis=1).mean(axis=1).fillna(0.0)

    feats_by_col = {}
    for col in STRATEGY_COLS:
        r = daily_panel[col]
        f = pd.DataFrame(index=daily_panel.index)
        f["mom5"] = r.rolling(5).sum().shift(1)
        f["mom10"] = r.rolling(10).sum().shift(1)
        f["vol10"] = r.rolling(10).std().shift(1)
        f["cs_mom5"] = daily_panel[STRATEGY_COLS].rolling(5).sum().shift(1).rank(axis=1, pct=True)[col]
        f["cs_vol10"] = 1.0 - daily_panel[STRATEGY_COLS].rolling(10).std().shift(1).rank(axis=1, pct=True)[col]
        if use_interactions:
            f["mom5_x_volregime"] = f["mom5"] * vol_regime
            f["mom10_x_corrdisp"] = f["mom10"] * corr_dispersion
            f["csmom5_x_volregime"] = f["cs_mom5"] * vol_regime
        feats_by_col[col] = f
    return feats_by_col


def make_classifiers():
    return [
        lgb.LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, num_leaves=8,
                            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
                            deterministic=True, force_row_wise=True, num_threads=1),
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
            actual = monthly_slice.iloc[i_pos].values
            if tilt_cap > 0.0:
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
    v2_mult = per_strategy_v2_multiplier_fixed(daily_panel, idx, lookback=20, lo=0.7, hi=1.3)
    adj_W = W.mul(v2_mult, axis=0)
    combo_ret = pd.Series((daily_panel.loc[idx].values * adj_W.values).sum(axis=1), index=idx)
    final = combo_ret * volq_multiplier_fixed(combo_ret, lookback=20, lo=0.7, hi=1.3)
    return final, recipes


def report(label, final):
    if final is None:
        print(f"{label}: insufficient data")
        return None
    r, v, s, w = sharpe_stats(final)
    n = len(final)
    split_60, split_40 = int(n * 0.6), int(n * 0.4)
    fwd, rev = final.iloc[split_60:], final.iloc[:split_40]
    _, _, s_f, w_f = sharpe_stats(fwd)
    _, _, s_r, w_r = sharpe_stats(rev)
    print(f"{label}: full sharpe={s:.2f} worst_day={w:.3%}  |  FWD sharpe={s_f:.2f} worst_day={w_f:.3%}  |  REV sharpe={s_r:.2f} worst_day={w_r:.3%}")
    return s_f


def main():
    monthly_panel, daily_panel = build_extended_monthly_panel()

    print("=== Main effects only (reproduces approach 10's design as control) ===")
    feats_main = build_features(daily_panel, use_interactions=False)
    final_main, recipes_main = honest_pipeline(monthly_panel, daily_panel, feats_main)
    report("Main-effects-only tilt", final_main)
    for d, dd, lam, hcap, ccap, lb, tc in recipes_main:
        print(f"  {d}  dd={dd:.1%} lam={lam} hcap={hcap:.0%} ccap={ccap:.0%} lookback={lb}mo tilt_cap={tc:.0%}")

    print("\n=== Main effects + explicit interaction terms (feature conditioning) ===")
    feats_cond = build_features(daily_panel, use_interactions=True)
    final_cond, recipes_cond = honest_pipeline(monthly_panel, daily_panel, feats_cond)
    report("Feature-conditioned tilt", final_cond)
    for d, dd, lam, hcap, ccap, lb, tc in recipes_cond:
        print(f"  {d}  dd={dd:.1%} lam={lam} hcap={hcap:.0%} ccap={ccap:.0%} lookback={lb}mo tilt_cap={tc:.0%}")

    if final_cond is not None:
        final_cond.to_csv("cpo_feature_conditioning_returns.csv")


if __name__ == "__main__":
    main()
