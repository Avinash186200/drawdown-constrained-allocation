"""
CPCV robustness check for EWMA-weighted mu/sigma (cpo_ewma_features.py),
same purpose and structure as the other cpo_cpcv_*.py scripts: don't
trust a single FWD/REV split, even when it shows a plausible-looking
mixed result (full/REV up, FWD down), without checking across the same
22 combinatorial out-of-sample paths used for everything else.

Fixed dd/lambda/caps match the other CPCV scripts exactly (applied
identically to BOTH variants within each combo, so the comparison is
paired/fair regardless of whether that recipe is itself optimal) --
this is a controlled comparison, not the same hindsight-recipe issue
caught earlier tonight, since both sides of the comparison share the
identical recipe. Monthly halflife (6mo) and daily-vol halflife (20d)
are principled a priori choices -- 6mo matches this project's own
retune cadence, 20d matches the daily lookback already used by the
existing VolQ/per-strategy overlay elsewhere in this pipeline -- NOT
values read off this test's own single-split recipe output.
"""
import itertools
import numpy as np
import pandas as pd

from cpo_hybrid_monthly import build_extended_monthly_panel, hard_constrained_weights, sharpe_stats
from cpo_ewma_features import ewma_mean_cov, ewma_daily_vol, apply_daily_vol_to_sigma

N_BLOCKS = 8
K_TEST = 2
EMBARGO_MONTHS = 1
FIXED_LOOKBACK = 36
FIXED_DD, FIXED_LAM, FIXED_HCAP, FIXED_CCAP = 0.006, 5, 0.10, 1.0
MONTHLY_HALFLIFE = 6
DAILY_HALFLIFE = 20


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

    flat_rets, ewma_rets = [], []
    for tm in test_months:
        trailing = training_panel[training_panel.index < tm].tail(FIXED_LOOKBACK)
        if len(trailing) < FIXED_LOOKBACK // 2:
            continue
        daily_scen = daily_panel[daily_panel.index < tm].tail(252).values
        actual = monthly_panel.loc[tm].values

        mu_flat = trailing.mean().values
        sigma_flat = trailing.cov().values
        w_flat, _ = hard_constrained_weights(mu_flat, sigma_flat, daily_scen, FIXED_DD, FIXED_LAM,
                                              {1: FIXED_HCAP, 0: FIXED_CCAP})
        flat_rets.append(w_flat @ actual)

        mu_e, sigma_e = ewma_mean_cov(trailing, MONTHLY_HALFLIFE)
        hist_daily = daily_panel[daily_panel.index < tm].tail(252)
        daily_var = ewma_daily_vol(hist_daily, DAILY_HALFLIFE)
        sigma_e = apply_daily_vol_to_sigma(sigma_e, daily_var)
        w_e, _ = hard_constrained_weights(mu_e, sigma_e, daily_scen, FIXED_DD, FIXED_LAM,
                                           {1: FIXED_HCAP, 0: FIXED_CCAP})
        ewma_rets.append(w_e @ actual)

    return pd.Series(flat_rets), pd.Series(ewma_rets)


def main():
    monthly_panel, daily_panel = build_extended_monthly_panel()
    blocks = make_blocks(monthly_panel)
    combos = list(itertools.combinations(range(N_BLOCKS), K_TEST))
    print(f"CPCV (EWMA check): {len(monthly_panel)} months, {N_BLOCKS} blocks, k={K_TEST}, "
          f"{len(combos)} combinations, embargo={EMBARGO_MONTHS}mo, "
          f"monthly_halflife={MONTHLY_HALFLIFE}mo, daily_vol_halflife={DAILY_HALFLIFE}d\n")

    flat_sharpes, ewma_sharpes = [], []
    beat_count = 0
    for combo in combos:
        flat_ret, ewma_ret = eval_combo(monthly_panel, daily_panel, blocks, combo)
        _, _, s_flat, _ = sharpe_stats(flat_ret, freq=12)
        _, _, s_ewma, _ = sharpe_stats(ewma_ret, freq=12)
        if not (np.isnan(s_flat) or np.isnan(s_ewma)):
            flat_sharpes.append(s_flat)
            ewma_sharpes.append(s_ewma)
            if s_ewma > s_flat:
                beat_count += 1

    flat_arr, ewma_arr = np.array(flat_sharpes), np.array(ewma_sharpes)
    diffs = ewma_arr - flat_arr
    print(f"Valid combinations: {len(flat_arr)} / {len(combos)}\n")
    print(f"Flat (equal-weighted) MV Sharpe across {len(flat_arr)} CPCV paths:")
    print(f"  mean={flat_arr.mean():.2f}  median={np.median(flat_arr):.2f}  std={flat_arr.std():.2f}  "
          f"min={flat_arr.min():.2f}  max={flat_arr.max():.2f}")
    print(f"\nEWMA (recency-weighted) Sharpe across {len(ewma_arr)} CPCV paths:")
    print(f"  mean={ewma_arr.mean():.2f}  median={np.median(ewma_arr):.2f}  std={ewma_arr.std():.2f}  "
          f"min={ewma_arr.min():.2f}  max={ewma_arr.max():.2f}")
    print(f"\nEWMA beat flat in {beat_count}/{len(flat_arr)} paths ({beat_count/len(flat_arr):.0%})")
    print(f"Paired difference (EWMA - flat): mean={diffs.mean():+.2f}  median={np.median(diffs):+.2f}  "
          f"std={diffs.std():.2f}")


if __name__ == "__main__":
    main()
