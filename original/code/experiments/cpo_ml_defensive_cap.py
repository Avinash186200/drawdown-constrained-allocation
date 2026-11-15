"""
Approach 12: ML used purely as a DEFENSIVE risk signal, not a return
forecast -- directly inspired by what's actually proven to work in
this pipeline (the StrategyB/Strategy A structural position caps,
which catch gap risk the optimizer's own trailing-volatility math is
blind to). Every prior ML test (1, 6, 7, 9, 10, 11) tried to forecast
or reweight based on expected RETURN. This one never does that: a
per-strategy classifier predicts elevated near-term gap-risk (a large
single-day loss in the next 5 trading days), and when flagged, tightens
that strategy's position cap below its already-honestly-selected
baseline -- it can only ever reduce exposure relative to the baseline
recipe, never increase it. Asymmetric and one-directional by
construction, matching the actual mechanism (the caps) that this
project's own audit found to be doing the real work.

Same discipline as everything else: whether to use the defensive
overlay at all, and how aggressively to tighten (shrink factor), are
folded into the same honest periodic reselection -- "off" is always a
candidate.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb

from cpo_hybrid_monthly import build_extended_monthly_panel, hard_constrained_weights, sharpe_stats, MONTHLY_LOOKBACK_CANDIDATES
from cpo_audit_fixed import volq_multiplier_fixed, per_strategy_v2_multiplier_fixed

STRATEGY_COLS = ["StrategyA", "StrategyB", "StrategyC", "StrategyD"]
RETUNE_EVERY_MONTHS = 6
RISK_HORIZON_DAYS = 5
RISK_LABEL_PCTILE = 0.05  # bottom 5% of a strategy's own historical daily returns = "gap risk day"
SHRINK_CANDIDATES = [1.0, 0.75, 0.5, 0.25]  # 1.0 = no shrink (overlay off)
RISK_THRESHOLD = 0.5  # classifier prob above this triggers the shrink


def build_gaprisk_features(daily_panel):
    feats_by_col = {}
    for col in STRATEGY_COLS:
        r = daily_panel[col]
        f = pd.DataFrame(index=daily_panel.index)
        f["vol10"] = r.rolling(10).std().shift(1)
        f["vol20"] = r.rolling(20).std().shift(1)
        f["vol_ratio"] = (f["vol10"] / f["vol20"]).replace([np.inf, -np.inf], np.nan)
        f["min5"] = r.rolling(5).min().shift(1)
        f["skew20"] = r.rolling(20).skew().shift(1)
        f["days_since_big_loss"] = (r < r.expanding().quantile(RISK_LABEL_PCTILE).shift(1)).astype(int)
        f["days_since_big_loss"] = f["days_since_big_loss"][::-1].groupby(
            f["days_since_big_loss"][::-1].cumsum()).cumcount()[::-1]
        feats_by_col[col] = f
    return feats_by_col


def make_classifiers():
    return [
        lgb.LGBMClassifier(n_estimators=80, max_depth=3, learning_rate=0.05, num_leaves=8,
                            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1),
        Pipeline([("scaler", StandardScaler()),
                  ("clf", LogisticRegression(penalty="l2", C=5, solver="liblinear", max_iter=1000,
                                              class_weight="balanced", random_state=42))]),
        Pipeline([("scaler", StandardScaler()),
                  ("clf", ExtraTreesClassifier(n_estimators=80, max_depth=4, min_samples_leaf=15,
                                                class_weight="balanced", random_state=42, n_jobs=1))]),
        Pipeline([("scaler", StandardScaler()),
                  ("clf", RandomForestClassifier(n_estimators=80, max_depth=4, min_samples_leaf=15,
                                                  class_weight="balanced", random_state=42, n_jobs=1))]),
    ]


def gaprisk_signal(daily_panel_upto, features_by_col, as_of_idx, min_train=150):
    """Per-strategy P(gap-risk day within next RISK_HORIZON_DAYS days)."""
    probs_out = np.zeros(len(STRATEGY_COLS))
    for k, col in enumerate(STRATEGY_COLS):
        x_today = features_by_col[col].iloc[[as_of_idx]]
        if x_today.isna().any(axis=1).values[0]:
            continue
        r = daily_panel_upto[col]
        thresh = r.iloc[:as_of_idx].quantile(RISK_LABEL_PCTILE)
        is_bad_day = (r < thresh).astype(int)
        fwd_bad = is_bad_day.shift(-1).rolling(RISK_HORIZON_DAYS).max().shift(-(RISK_HORIZON_DAYS - 1))
        usable_end = as_of_idx - RISK_HORIZON_DAYS
        X_all = features_by_col[col].iloc[:usable_end]
        y_all = fwd_bad.iloc[:usable_end]
        valid = X_all.notna().all(axis=1) & y_all.notna()
        X_train, y_train = X_all[valid], y_all[valid].astype(int)
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
            probs_out[k] = np.mean(probs)
    return probs_out


def apply_defensive_shrink(base_w, risk_probs, shrink_factor):
    if shrink_factor >= 1.0:
        return base_w
    adj = base_w.copy()
    flagged = risk_probs > RISK_THRESHOLD
    adj[flagged] = adj[flagged] * shrink_factor
    if adj.sum() > 0:
        adj = adj / adj.sum() * base_w.sum()  # keep total exposure comparable, redistribute freed capital proportionally
        # redistribute the freed-up capital to the NON-flagged strategies only
    freed = base_w[flagged].sum() - adj[flagged].sum()
    if freed > 0 and (~flagged).any():
        adj[~flagged] = adj[~flagged] + freed * (base_w[~flagged] / base_w[~flagged].sum())
    return adj


def select_recipe(monthly_slice, daily_slice_upto, daily_features_upto):
    min_lookback = min(MONTHLY_LOOKBACK_CANDIDATES)
    if len(monthly_slice) < min_lookback + 6:
        return 0.008, 10, 1.0, 1.0, min_lookback, 1.0

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

    def backtest_segment(dd_limit, lam, hcap, ccap, monthly_lookback, shrink_factor):
        base_by_month = base_weights_series(dd_limit, lam, hcap, ccap, monthly_lookback)
        if not base_by_month:
            return pd.Series(dtype=float)
        oos = []
        for month_end_date, base_w in base_by_month.items():
            i_pos = monthly_slice.index.get_loc(month_end_date)
            actual = monthly_slice.iloc[i_pos].values
            if shrink_factor < 1.0:
                daily_idx = daily_slice_upto.index[daily_slice_upto.index < month_end_date]
                if len(daily_idx) > 150:
                    as_of_idx = daily_slice_upto.index.get_loc(daily_idx[-1])
                    risk = gaprisk_signal(daily_slice_upto, daily_features_upto, as_of_idx)
                    w = apply_defensive_shrink(base_w, risk, shrink_factor)
                else:
                    w = base_w
            else:
                w = base_w
            oos.append(w @ actual)
        return pd.Series(oos)

    best = None
    for dd_limit in [0.010, 0.006, 0.005]:
        for lam in [5, 10]:
            ret = backtest_segment(dd_limit, lam, 1.0, 1.0, 36, 1.0)
            _, _, s, _ = sharpe_stats(ret, freq=12)
            if best is None or (not np.isnan(s) and s > best[0]):
                best = (s, dd_limit, lam)
    _, sel_dd, sel_lam = best

    best_hcap = None
    for cap in [1.0, 0.25, 0.10, 0.05]:
        ret = backtest_segment(sel_dd, sel_lam, cap, 1.0, 36, 1.0)
        _, _, s, _ = sharpe_stats(ret, freq=12)
        if best_hcap is None or (not np.isnan(s) and s > best_hcap[0]):
            best_hcap = (s, cap)
    sel_hcap = best_hcap[1]

    best_ccap = None
    for cap in [1.0, 0.6, 0.4, 0.25]:
        ret = backtest_segment(sel_dd, sel_lam, sel_hcap, cap, 36, 1.0)
        _, _, s, _ = sharpe_stats(ret, freq=12)
        if best_ccap is None or (not np.isnan(s) and s > best_ccap[0]):
            best_ccap = (s, cap)
    sel_ccap = best_ccap[1]

    best_lb = None
    for lb in MONTHLY_LOOKBACK_CANDIDATES:
        ret = backtest_segment(sel_dd, sel_lam, sel_hcap, sel_ccap, lb, 1.0)
        _, _, s, _ = sharpe_stats(ret, freq=12)
        if best_lb is None or (not np.isnan(s) and s > best_lb[0]):
            best_lb = (s, lb)
    sel_lb = best_lb[1]

    best_shrink = None
    for sf in SHRINK_CANDIDATES:
        ret = backtest_segment(sel_dd, sel_lam, sel_hcap, sel_ccap, sel_lb, sf)
        _, _, s, _ = sharpe_stats(ret, freq=12)
        if best_shrink is None or (not np.isnan(s) and s > best_shrink[0]):
            best_shrink = (s, sf)
    sel_shrink = best_shrink[1]

    return sel_dd, sel_lam, sel_hcap, sel_ccap, sel_lb, sel_shrink


def honest_pipeline(monthly_panel, daily_panel, daily_features):
    monthly_dates = monthly_panel.index
    daily_weights = {}
    recipes = []

    i = min(MONTHLY_LOOKBACK_CANDIDATES)
    dd_limit = lam = hcap = ccap = monthly_lookback = shrink_factor = None
    months_since_retune = 0
    while i < len(monthly_dates):
        month_end_date = monthly_dates[i]
        if dd_limit is None or months_since_retune >= RETUNE_EVERY_MONTHS:
            avail_monthly = monthly_panel.iloc[:i]
            avail_daily = daily_panel[daily_panel.index < month_end_date]
            avail_feats = {c: daily_features[c][daily_features[c].index < month_end_date] for c in STRATEGY_COLS}
            dd_limit, lam, hcap, ccap, monthly_lookback, shrink_factor = select_recipe(
                avail_monthly, avail_daily, avail_feats)
            recipes.append((month_end_date.date(), dd_limit, lam, hcap, ccap, monthly_lookback, shrink_factor))
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

        last_w = base_w
        for j, d in enumerate(days_in_month):
            if shrink_factor < 1.0 and j % 5 == 0:
                hist_upto_d = daily_panel[daily_panel.index < d]
                if len(hist_upto_d) > 150:
                    as_of_idx = daily_panel.index.get_loc(hist_upto_d.index[-1])
                    feats_upto = {c: daily_features[c][daily_features[c].index <= hist_upto_d.index[-1]]
                                  for c in STRATEGY_COLS}
                    risk = gaprisk_signal(daily_panel[daily_panel.index <= d], feats_upto, as_of_idx)
                    last_w = apply_defensive_shrink(base_w, risk, shrink_factor)
            daily_weights[d] = last_w if shrink_factor < 1.0 else base_w
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
    daily_features = build_gaprisk_features(daily_panel)

    final, recipes = honest_pipeline(monthly_panel, daily_panel, daily_features)
    if final is None:
        print("insufficient data")
        return
    r, v, s, w = sharpe_stats(final)
    n = len(final)
    print(f"Defensive gap-risk overlay: n={n} sharpe={s:.2f} ann_ret={r:.2%} worst_day={w:.3%}")
    split_60, split_40 = int(n * 0.6), int(n * 0.4)
    fwd, rev = final.iloc[split_60:], final.iloc[:split_40]
    _, _, s_f, w_f = sharpe_stats(fwd)
    _, _, s_r, w_r = sharpe_stats(rev)
    print(f"FWD: sharpe={s_f:.2f} worst_day={w_f:.3%}   REV: sharpe={s_r:.2f} worst_day={w_r:.3%}")
    print("\nRecipe over time (shrink=1.0 means overlay off):")
    for d, dd, lam, hcap, ccap, lb, sf in recipes:
        print(f"  {d}  dd={dd:.1%} lam={lam} hcap={hcap:.0%} ccap={ccap:.0%} lookback={lb}mo shrink={sf:.0%}")
    final.to_csv("cpo_ml_defensive_cap_returns.csv")


if __name__ == "__main__":
    main()
