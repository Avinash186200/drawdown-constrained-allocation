"""
CPCV robustness check for trend-extrapolated mu (cpo_trend_extrapolation.py),
same purpose and structure as the other cpo_cpcv_*.py scripts.
"""
import itertools
import numpy as np
import pandas as pd

from cpo_hybrid_monthly import build_extended_monthly_panel, hard_constrained_weights, sharpe_stats
from cpo_trend_extrapolation import trend_extrapolated_mu

N_BLOCKS = 8
K_TEST = 2
EMBARGO_MONTHS = 1
FIXED_LOOKBACK = 36
FIXED_DD, FIXED_LAM, FIXED_HCAP, FIXED_CCAP = 0.006, 5, 0.10, 1.0
TREND_WEIGHT = 0.5  # a priori midpoint, not read off the single-split recipe output


def make_blocks(monthly_panel):
    n = len(monthly_panel)
    edges = np.linspace(0, n, N_BLOCKS + 1).astype(int)
    return [monthly_panel.index[edges[i]:edges[i + 1]] for i in range(N_BLOCKS)]


def purge_embargo(monthly_panel, blocks, test_block_ids):
    test_months, embargo_months = set(), set()
    for bid in test_block_ids:
        block_idx = blocks[bid]
        test_months.update(block_idx)
        pos0 = monthly_panel.index.get_loc(block_idx[0])
        pos1 = monthly_panel.index.get_loc(block_idx[-1])
        for p in range(max(0, pos0 - EMBARGO_MONTHS), pos0):
            embargo_months.add(monthly_panel.index[p])
        for p in range(pos1 + 1, min(len(monthly_panel), pos1 + 1 + EMBARGO_MONTHS)):
            embargo_months.add(monthly_panel.index[p])
    excluded = test_months | embargo_months
    train_months = [d for d in monthly_panel.index if d not in excluded]
    return sorted(test_months), train_months


def eval_combo(monthly_panel, daily_panel, blocks, test_block_ids):
    test_months, train_months = purge_embargo(monthly_panel, blocks, test_block_ids)
    training_panel = monthly_panel.loc[train_months]

    flat_rets, trend_rets = [], []
    for tm in test_months:
        trailing = training_panel[training_panel.index < tm].tail(FIXED_LOOKBACK)
        if len(trailing) < FIXED_LOOKBACK // 2:
            continue
        sigma = trailing.cov().values
        daily_scen = daily_panel[daily_panel.index < tm].tail(252).values
        actual = monthly_panel.loc[tm].values

        mu_flat = trailing.mean().values
        w_flat, _ = hard_constrained_weights(mu_flat, sigma, daily_scen, FIXED_DD, FIXED_LAM,
                                              {1: FIXED_HCAP, 0: FIXED_CCAP})
        flat_rets.append(w_flat @ actual)

        mu_trend = trend_extrapolated_mu(trailing, TREND_WEIGHT)
        w_trend, _ = hard_constrained_weights(mu_trend, sigma, daily_scen, FIXED_DD, FIXED_LAM,
                                               {1: FIXED_HCAP, 0: FIXED_CCAP})
        trend_rets.append(w_trend @ actual)

    return pd.Series(flat_rets), pd.Series(trend_rets)


def main():
    monthly_panel, daily_panel = build_extended_monthly_panel()
    blocks = make_blocks(monthly_panel)
    combos = list(itertools.combinations(range(N_BLOCKS), K_TEST))
    print(f"CPCV (trend-extrapolation check): {len(monthly_panel)} months, {N_BLOCKS} blocks, k={K_TEST}, "
          f"{len(combos)} combinations, embargo={EMBARGO_MONTHS}mo, trend_weight={TREND_WEIGHT}\n")

    flat_sharpes, trend_sharpes = [], []
    beat_count = 0
    for combo in combos:
        flat_ret, trend_ret = eval_combo(monthly_panel, daily_panel, blocks, combo)
        _, _, s_flat, _ = sharpe_stats(flat_ret, freq=12)
        _, _, s_trend, _ = sharpe_stats(trend_ret, freq=12)
        if not (np.isnan(s_flat) or np.isnan(s_trend)):
            flat_sharpes.append(s_flat)
            trend_sharpes.append(s_trend)
            if s_trend > s_flat:
                beat_count += 1

    flat_arr, trend_arr = np.array(flat_sharpes), np.array(trend_sharpes)
    diffs = trend_arr - flat_arr
    print(f"Valid combinations: {len(flat_arr)} / {len(combos)}\n")
    print(f"Flat mean Sharpe across {len(flat_arr)} CPCV paths:")
    print(f"  mean={flat_arr.mean():.2f}  median={np.median(flat_arr):.2f}  std={flat_arr.std():.2f}  "
          f"min={flat_arr.min():.2f}  max={flat_arr.max():.2f}")
    print(f"\nTrend-extrapolated Sharpe across {len(trend_arr)} CPCV paths:")
    print(f"  mean={trend_arr.mean():.2f}  median={np.median(trend_arr):.2f}  std={trend_arr.std():.2f}  "
          f"min={trend_arr.min():.2f}  max={trend_arr.max():.2f}")
    print(f"\nTrend beat flat in {beat_count}/{len(flat_arr)} paths ({beat_count/len(flat_arr):.0%})")
    print(f"Paired difference (trend - flat): mean={diffs.mean():+.2f}  median={np.median(diffs):+.2f}  "
          f"std={diffs.std():.2f}")


if __name__ == "__main__":
    main()
