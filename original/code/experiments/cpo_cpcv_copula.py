"""
CPCV robustness check for the copula-augmented drawdown constraint
(cpo_copula_stress.py), same purpose as cpo_cpcv_validation.py did for
Bayesian shrinkage: the single FWD/REV split showed a small
improvement from adding synthetic t-copula stress scenarios to the
hard drawdown constraint (FWD 4.01 vs 3.92, worst day -1.00% vs
-1.16%) -- before trusting that, check it across the same 22
combinatorial out-of-sample paths used to validate/invalidate every
other approach in this project, not just the one split.
"""
import itertools
import numpy as np
import pandas as pd

from cpo_hybrid_monthly import build_extended_monthly_panel, hard_constrained_weights, sharpe_stats
from cpo_copula_stress import fit_tcopula_and_sample

N_BLOCKS = 8
K_TEST = 2
EMBARGO_MONTHS = 1
FIXED_LOOKBACK = 36
FIXED_DD, FIXED_LAM, FIXED_HCAP, FIXED_CCAP = 0.006, 5, 0.10, 1.0


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

    base_rets, copula_rets = [], []
    for tm in test_months:
        trailing = training_panel[training_panel.index < tm].tail(FIXED_LOOKBACK)
        if len(trailing) < FIXED_LOOKBACK // 2:
            continue
        mu = trailing.mean().values
        sigma = trailing.cov().values
        hist_daily = daily_panel[daily_panel.index < tm]
        daily_scen = hist_daily.tail(252).values
        actual = monthly_panel.loc[tm].values

        w_base, _ = hard_constrained_weights(mu, sigma, daily_scen, FIXED_DD, FIXED_LAM,
                                              {1: FIXED_HCAP, 0: FIXED_CCAP})
        base_rets.append(w_base @ actual)

        if len(hist_daily) >= 60:
            synthetic = fit_tcopula_and_sample(hist_daily.tail(252))
            combined = np.vstack([daily_scen, synthetic]) if len(daily_scen) else synthetic
        else:
            combined = daily_scen
        w_cop, _ = hard_constrained_weights(mu, sigma, combined, FIXED_DD, FIXED_LAM,
                                             {1: FIXED_HCAP, 0: FIXED_CCAP})
        copula_rets.append(w_cop @ actual)

    return pd.Series(base_rets), pd.Series(copula_rets)


def main():
    monthly_panel, daily_panel = build_extended_monthly_panel()
    blocks = make_blocks(monthly_panel)
    combos = list(itertools.combinations(range(N_BLOCKS), K_TEST))
    print(f"CPCV (copula check): {len(monthly_panel)} months, {N_BLOCKS} blocks, k={K_TEST}, "
          f"{len(combos)} combinations, embargo={EMBARGO_MONTHS}mo\n")

    base_sharpes, copula_sharpes = [], []
    beat_count = 0
    for combo in combos:
        base_ret, copula_ret = eval_combo(monthly_panel, daily_panel, blocks, combo)
        _, _, s_base, _ = sharpe_stats(base_ret, freq=12)
        _, _, s_cop, _ = sharpe_stats(copula_ret, freq=12)
        if not (np.isnan(s_base) or np.isnan(s_cop)):
            base_sharpes.append(s_base)
            copula_sharpes.append(s_cop)
            if s_cop > s_base:
                beat_count += 1

    base_arr, cop_arr = np.array(base_sharpes), np.array(copula_sharpes)
    diffs = cop_arr - base_arr
    print(f"Valid combinations: {len(base_arr)} / {len(combos)}\n")
    print(f"Real-scenarios-only Sharpe across {len(base_arr)} CPCV paths:")
    print(f"  mean={base_arr.mean():.2f}  median={np.median(base_arr):.2f}  std={base_arr.std():.2f}  "
          f"min={base_arr.min():.2f}  max={base_arr.max():.2f}")
    print(f"\nCopula-augmented Sharpe across {len(cop_arr)} CPCV paths:")
    print(f"  mean={cop_arr.mean():.2f}  median={np.median(cop_arr):.2f}  std={cop_arr.std():.2f}  "
          f"min={cop_arr.min():.2f}  max={cop_arr.max():.2f}")
    print(f"\nCopula-augmented beat real-only in {beat_count}/{len(base_arr)} paths ({beat_count/len(base_arr):.0%})")
    print(f"Paired difference (copula - base): mean={diffs.mean():+.2f}  median={np.median(diffs):+.2f}  "
          f"std={diffs.std():.2f}")


if __name__ == "__main__":
    main()
