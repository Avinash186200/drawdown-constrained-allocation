"""Run the walk-forward allocator on synthetic data or on your own CSVs.

    python scripts/run_backtest.py
    python scripts/run_backtest.py --monthly m.csv --daily d.csv --loss-cap 0.02
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ddalloc import Config, equal_weight, load_csv, make_synthetic, run, split_report  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--monthly", help="monthly returns CSV (date, one column per strategy)")
    p.add_argument("--daily", help="daily returns CSV (same columns)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--noise", action="store_true", help="synthetic data with no signal")
    p.add_argument("--cost-bps", type=float, default=2.0)
    p.add_argument("--rf", type=float, default=0.0, help="annual risk-free rate for Sharpe")
    p.add_argument("--loss-cap", type=float, default=0.02, help="mandate: max single-day book loss")
    p.add_argument("--no-overlay", action="store_true")
    p.add_argument("--out", help="directory for daily returns and weights CSVs")
    a = p.parse_args()

    if a.monthly and a.daily:
        panels = load_csv(a.monthly, a.daily)
    else:
        panels = make_synthetic(seed=a.seed, signal=not a.noise)
    cfg = Config(cost_bps=a.cost_bps, overlay=not a.no_overlay)
    res = run(panels.monthly, panels.daily, cfg)
    ew = equal_weight(panels.daily, res.returns.index)

    print(f"scored days: {len(res.returns)}  ({res.returns.index[0].date()} -> {res.returns.index[-1].date()})")
    print(f"{'book':<12}{'period':<10}{'ann_ret':>9}{'ann_vol':>9}{'sharpe':>8}{'worst':>9}{'max_dd':>9}")
    for name, series in (("allocator", res.returns), ("equal_wt", ew)):
        for period, s in split_report(series, a.rf).items():
            print(f"{name:<12}{period:<10}{s['ann_ret']:>9.2%}{s['ann_vol']:>9.2%}{s['sharpe']:>8.2f}"
                  f"{s['worst_day']:>9.2%}{s['max_dd']:>9.2%}")
    worst = res.returns.min()
    print(f"worst day {worst:.2%}: {'within' if worst >= -a.loss_cap else 'BREACHES'} the {a.loss_cap:.0%} cap")
    print("avg effective weights (incl. overlay):", res.weights.mean().round(3).to_dict())
    for d, r in res.recipes:
        print(f"  retune {d.date()}: loss_limit={r.loss_limit:.1%} lam={r.lam:g} "
              f"gap_cap={r.gap_cap:.0%} second_cap={r.second_cap:.0%} lookback={r.lookback}m")
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        res.returns.to_csv(out / "returns.csv", header=["ret"])
        res.weights.to_csv(out / "weights.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
