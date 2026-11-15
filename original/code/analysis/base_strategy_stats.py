"""
Standalone descriptive analysis of the 4 CPO base strategies
(Strategy A, StrategyB, Strategy C, Strategy D), independent
of any allocation/optimization layer. After 17 rounds of testing ways
to combine these strategies, this steps back to understand what's
actually being combined: standalone risk/return character, drawdowns,
distribution shape, correlation structure, and stability over time.

Uses the daily panel (Apr2023+, real data) for anything requiring daily
granularity (drawdowns, distribution, rolling stats) and the extended
monthly panel (Jan2021+) for the longer-history return/vol picture.
"""
import numpy as np
import pandas as pd

from cpo_hybrid_monthly import build_extended_monthly_panel, STRATEGY_COLS
from cpo_daily_hardconstraint import load_daily_panel


def ann_stats(daily_ret, rf=0.07):
    n = len(daily_ret)
    ann_ret = (1 + daily_ret).prod() ** (252 / n) - 1
    ann_vol = daily_ret.std() * np.sqrt(252)
    sharpe = (ann_ret - rf) / ann_vol
    return ann_ret, ann_vol, sharpe


def max_drawdown(daily_ret):
    cum = (1 + daily_ret).cumprod()
    running_max = cum.cummax()
    dd = cum / running_max - 1
    trough_idx = dd.idxmin()
    max_dd = dd.min()
    peak_idx = cum.loc[:trough_idx].idxmax()
    recovery = cum.loc[trough_idx:]
    recovered = recovery[recovery >= cum.loc[peak_idx]]
    recovery_idx = recovered.index[0] if len(recovered) else None
    duration_to_trough = (trough_idx - peak_idx).days
    duration_to_recover = (recovery_idx - peak_idx).days if recovery_idx is not None else None
    return max_dd, peak_idx.date(), trough_idx.date(), duration_to_trough, duration_to_recover


def describe_daily(daily_panel):
    print("=" * 100)
    print(f"DAILY PANEL: {daily_panel.index[0].date()} -> {daily_panel.index[-1].date()}  ({len(daily_panel)} days)")
    print("=" * 100)
    for col in STRATEGY_COLS:
        r = daily_panel[col]
        ann_ret, ann_vol, sharpe = ann_stats(r)
        max_dd, peak_d, trough_d, dur_trough, dur_recover = max_drawdown(r)
        win_rate = (r > 0).mean()
        skew, kurt = r.skew(), r.kurtosis()
        best, worst = r.max(), r.min()
        recover_str = f"{dur_recover}d" if dur_recover is not None else "NOT YET RECOVERED"
        print(f"\n{col}:")
        print(f"  ann_return={ann_ret:.2%}  ann_vol={ann_vol:.2%}  sharpe={sharpe:.2f}")
        print(f"  max_drawdown={max_dd:.2%}  (peak {peak_d} -> trough {trough_d}, {dur_trough}d down, recovery: {recover_str})")
        print(f"  best_day={best:.2%}  worst_day={worst:.2%}  win_rate={win_rate:.1%}  skew={skew:.2f}  excess_kurtosis={kurt:.2f}")

    print(f"\n{'-'*100}")
    print("CORRELATION MATRIX (daily returns, full period):")
    print(daily_panel[STRATEGY_COLS].corr().round(2).to_string())

    print(f"\n{'-'*100}")
    print("ROLLING 6-MONTH CORRELATION STABILITY (min/max pairwise corr across rolling 126d windows):")
    roll_corr = daily_panel[STRATEGY_COLS].rolling(126).corr()
    pairs = [(a, b) for i, a in enumerate(STRATEGY_COLS) for b in STRATEGY_COLS[i + 1:]]
    for a, b in pairs:
        series = roll_corr.xs(a, level=1)[b].dropna()
        if len(series):
            print(f"  {a} vs {b}: min={series.min():.2f}  max={series.max():.2f}  current={series.iloc[-1]:.2f}")

    print(f"\n{'-'*100}")
    print("ROLLING 6-MONTH SHARPE PER STRATEGY (is the edge stable, growing, or decaying?):")
    for col in STRATEGY_COLS:
        r = daily_panel[col]
        roll = r.rolling(126).apply(lambda x: (x.mean() * 252 - 0.07) / (x.std() * np.sqrt(252)) if x.std() > 0 else np.nan)
        roll = roll.dropna()
        if len(roll):
            checkpoints = roll.iloc[::max(1, len(roll) // 6)]
            vals = "  ".join(f"{d.date()}:{v:.1f}" for d, v in checkpoints.items())
            print(f"  {col}: {vals}")


def describe_monthly(monthly_panel):
    print(f"\n{'='*100}")
    print(f"EXTENDED MONTHLY PANEL: {monthly_panel.index[0].date()} -> {monthly_panel.index[-1].date()}  "
          f"({len(monthly_panel)} months)")
    print("=" * 100)
    for col in STRATEGY_COLS:
        r = monthly_panel[col]
        ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
        ann_vol = r.std() * np.sqrt(12)
        sharpe = (ann_ret - 0.07) / ann_vol
        win_rate = (r > 0).mean()
        print(f"  {col}: ann_return={ann_ret:.2%}  ann_vol={ann_vol:.2%}  sharpe={sharpe:.2f}  "
              f"monthly_win_rate={win_rate:.1%}  n={len(r)}")

    print(f"\n{'-'*100}")
    print("YEAR-BY-YEAR ANNUAL RETURN (monthly-compounded, shows consistency over time):")
    yearly = (1 + monthly_panel[STRATEGY_COLS]).resample("YE").prod() - 1
    print(yearly.round(3).to_string())


def load_external(name):
    df = pd.read_csv(f"raw_data/external_{name}.csv", index_col=0, parse_dates=True)
    return df[df.columns[0]]


def external_vol_relationship(daily_panel):
    try:
        india_vix = load_external("INDIAVIX")
        vix = india_vix.reindex(daily_panel.index).ffill()
    except Exception as e:
        print(f"\n(skipping external-vol relationship: {e})")
        return
    print(f"\n{'-'*100}")
    print("STRATEGY RETURNS IN HIGH vs LOW INDIA VIX REGIME (above/below trailing median):")
    vix_median = vix.expanding(min_periods=60).median()
    high_regime = vix > vix_median
    for col in STRATEGY_COLS:
        r = daily_panel[col]
        valid = vix_median.notna()
        r_high = r[valid & high_regime]
        r_low = r[valid & ~high_regime]
        if len(r_high) > 20 and len(r_low) > 20:
            sh_high = (r_high.mean() * 252 - 0.07) / (r_high.std() * np.sqrt(252))
            sh_low = (r_low.mean() * 252 - 0.07) / (r_low.std() * np.sqrt(252))
            print(f"  {col}: high-VIX Sharpe={sh_high:.2f} (n={len(r_high)})   low-VIX Sharpe={sh_low:.2f} (n={len(r_low)})")


def external_correlations(daily_panel):
    """Direct correlations against India VIX, US VIX, and equity markets --
    both LEVEL/CHANGE correlation with strategy RETURNS, and correlation
    between each strategy's own rolling realized volatility and market vol
    (does this strategy's risk track market risk, an exploitable signal if
    so -- already tested as an overlay and ruled out, but worth seeing the
    raw numbers directly)."""
    series = {}
    for name, label in [("INDIAVIX", "IndiaVIX"), ("VIX", "US_VIX"),
                         ("NIFTY500", "Nifty500"), ("BSESN", "Sensex"),
                         ("USDINR", "USDINR"), ("GOLD", "Gold"), ("CRUDE", "Crude")]:
        try:
            s = load_external(name).reindex(daily_panel.index).ffill()
            series[label] = s
        except Exception:
            continue
    if not series:
        print("\n(no external series available)")
        return

    print(f"\n{'='*100}")
    print("CORRELATION WITH EXTERNAL MARKET DATA")
    print("=" * 100)

    print("\nStrategy DAILY RETURN vs external series' DAILY CHANGE (does the strategy move with market moves?):")
    header = "  " + " ".join(f"{lbl:>10}" for lbl in series)
    print(header)
    for col in STRATEGY_COLS:
        r = daily_panel[col]
        row = []
        for lbl, s in series.items():
            chg = s.pct_change() if lbl not in ("IndiaVIX", "US_VIX") else s.diff()
            c = r.corr(chg)
            row.append(f"{c:>10.2f}")
        print(f"  {col:<22}" + " ".join(row))

    print("\nStrategy's OWN 20-DAY REALIZED VOL vs external series LEVEL (does this strategy's risk track market vol/regime?):")
    print(header)
    for col in STRATEGY_COLS:
        own_vol = daily_panel[col].rolling(20).std()
        row = []
        for lbl, s in series.items():
            c = own_vol.corr(s)
            row.append(f"{c:>10.2f}")
        print(f"  {col:<22}" + " ".join(row))

    print("\nBOOK-LEVEL (equal-weight combo) 20-day realized vol vs external series LEVEL:")
    combo_ret = daily_panel[STRATEGY_COLS].mean(axis=1)
    combo_vol = combo_ret.rolling(20).std()
    for lbl, s in series.items():
        print(f"  vs {lbl}: {combo_vol.corr(s):.2f}")

    print(f"\n{'-'*100}")
    print("SAME, BUT DE-TRENDED (correlation of 20-day CHANGES, not levels -- with only ~3yr of daily")
    print("data, two series that both drift over calendar time can show a strong LEVEL correlation that")
    print("is spurious, not a genuine day-to-day co-movement. This filters that out.):")
    print(header)
    for col in STRATEGY_COLS:
        own_vol_chg = daily_panel[col].rolling(20).std().diff(20)
        row = []
        for lbl, s in series.items():
            s_chg = s.diff(20)
            c = own_vol_chg.corr(s_chg)
            row.append(f"{c:>10.2f}")
        print(f"  {col:<22}" + " ".join(row))
    combo_vol_chg = combo_vol.diff(20)
    print(f"\n  De-trended book-level vol change vs external series change:")
    for lbl, s in series.items():
        s_chg = s.diff(20)
        print(f"    vs {lbl}: {combo_vol_chg.corr(s_chg):.2f}")


def main():
    monthly_panel, daily_panel = build_extended_monthly_panel()
    describe_daily(daily_panel)
    describe_monthly(monthly_panel)
    external_vol_relationship(daily_panel)
    external_correlations(daily_panel)


if __name__ == "__main__":
    main()
