"""
CPO v2: daily-frequency allocator with a HARD 2% max-daily-loss
constraint, per the review direction (14/07/2026): "focus on daily" for
Strategy A (use Fund Class D, not the monthly Class A series) and
"whatever we do should limit to 2% max daily dd" (hard constraint, not
the soft variance-penalty proxy used in v1).

Common daily window across all 4 strategies: Apr 2023 - Jun 2026
(~3 years) -- shorter than the v1 monthly panel (which went back to
2015/2009 for two of the four), but necessary since Strategy A's
only true daily series (Class D) and Strategy C / Strategy D
daily data all start Apr 2023.

Constraint formulation: worst-case historical-scenario constraint --
for every day t in the trailing estimation window, w'r_t >= -0.02. This
is a hard, non-parametric constraint (no distributional assumption),
solved with SLSQP (linear constraints, small N so scales fine even with
~250 daily scenario constraints).
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

XLSX = "raw_data/Strategy_ReturnSeries.xlsx"
RF_ANNUAL = 0.07
DD_LIMIT = 0.02
LOOKBACK_DAYS = 252  # ~1y trailing window for both mu estimation and the hard constraint
REBAL_EVERY = 21     # ~monthly rebalance


def load_daily(sheet, val_col=1, skiprows=0):
    df = pd.read_excel(XLSX, sheet_name=sheet, skiprows=skiprows).dropna(how="all")
    df = df.iloc[:, [0, val_col]].dropna()
    df.columns = ["date", "ret"]
    s = pd.Series(df["ret"].values, index=pd.to_datetime(df["date"].values)).sort_index()
    return s[~s.index.duplicated(keep="last")]


def load_daily_panel():
    strategy_a = load_daily("Fund Share Class D ($)", val_col=3, skiprows=1)  # "Total" col
    strategy_b = load_daily("StrategyB Strategy_Daily INR")
    strategy_c = load_daily("Strategy C_Daily INR")
    strategy_d = load_daily("Strategy D_Daily INR")
    panel = pd.DataFrame({
        "StrategyA": strategy_a,
        "StrategyB": strategy_b,
        "StrategyC": strategy_c,
        "StrategyD": strategy_d,
    }).dropna()
    return panel


def hard_constrained_weights(returns_window, dd_limit=DD_LIMIT, lam=5.0):
    """
    returns_window: (T, N) trailing daily returns.
    maximize w'mu - (lam/2) w'Sigma w   (small variance term for stability/tie-break)
    s.t. sum(w) = 1, w >= 0, w'r_t >= -dd_limit for every historical day t
    """
    n = returns_window.shape[1]
    mu = returns_window.mean(axis=0)
    sigma = np.cov(returns_window.T)

    def neg_obj(w):
        return -(w @ mu - 0.5 * lam * (w @ sigma @ w))

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    # one inequality per historical day: w'r_t + dd_limit >= 0
    for t in range(returns_window.shape[0]):
        r_t = returns_window[t]
        cons.append({"type": "ineq", "fun": (lambda w, r_t=r_t: w @ r_t + dd_limit)})
    bounds = [(0.0, 1.0)] * n
    w0 = np.full(n, 1.0 / n)

    # equal-weight is not guaranteed feasible; check and fall back if needed
    res = minimize(neg_obj, w0, method="SLSQP", bounds=bounds, constraints=cons,
                    options={"maxiter": 300, "ftol": 1e-9})
    if not res.success:
        # relax: search along the EW-to-single-asset boundary isn't robust; just
        # return best found and flag
        pass
    return res.x, res.success


def stats(s, label, freq=252):
    n = len(s)
    ann_ret = (1 + s).prod() ** (freq / n) - 1
    ann_vol = s.std() * np.sqrt(freq)
    sharpe = (ann_ret - RF_ANNUAL) / ann_vol
    print(f"  {label:24s} n={n:4d}  ann_ret={ann_ret:7.2%}  ann_vol={ann_vol:6.2%}  "
          f"sharpe={sharpe:5.2f}  worst_day={s.min():7.3%}")


def backtest(panel, lam=5.0):
    dates = panel.index
    oos_rets, weights_hist, fail_count = [], [], 0
    start_idx = LOOKBACK_DAYS
    for i in range(start_idx, len(dates), REBAL_EVERY):
        trailing = panel.iloc[i - LOOKBACK_DAYS:i].values
        w, ok = hard_constrained_weights(trailing, lam=lam)
        if not ok:
            fail_count += 1
        end = min(i + REBAL_EVERY, len(dates))
        for j in range(i, end):
            oos_rets.append({"date": dates[j], "ret": panel.iloc[j].values @ w})
        weights_hist.append({"date": dates[i], "w": w})
    oos = pd.DataFrame(oos_rets).set_index("date")["ret"]
    return oos, weights_hist, fail_count


def main():
    panel = load_daily_panel()
    print(f"Daily panel: {panel.index[0].date()} -> {panel.index[-1].date()}, {len(panel)} days\n")

    print("=== EW baseline ===")
    ew = panel.mean(axis=1)
    stats(ew, "Equal weight")

    print("\n=== Hard-constrained (w'r_t >= -2% every historical day in trailing window) ===")
    for lam in [1, 5, 20]:
        oos, weights_hist, fails = backtest(panel, lam=lam)
        print(f"\nlambda={lam}  (rebalances={len(weights_hist)}, solver_failures={fails})")
        stats(oos, f"Hard-constrained OOS")
        avg_w = pd.DataFrame([w["w"] for w in weights_hist], columns=panel.columns).mean()
        print("  avg weights:", avg_w.round(3).to_dict())
        realized_worst = oos.min()
        print(f"  realized OOS worst day = {realized_worst:.3%}  "
              f"({'WITHIN' if realized_worst >= -DD_LIMIT else 'BREACHED'} the {DD_LIMIT:.0%} limit)")


if __name__ == "__main__":
    main()
