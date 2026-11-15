"""
CPCV robustness check for Michaud resampling (cpo_michaud_resampling.py),
same purpose and structure as cpo_cpcv_copula.py: don't trust a single
FWD/REV split result, even a corrected one, without checking it across
many combinatorial out-of-sample paths first -- the copula test just
showed a single-split "improvement" can be a lucky draw that reverses
under CPCV.
"""
import itertools
import numpy as np
import pandas as pd

from cpo_hybrid_monthly import build_extended_monthly_panel, hard_constrained_weights, sharpe_stats

N_BLOCKS = 8
K_TEST = 2
EMBARGO_MONTHS = 1
FIXED_LOOKBACK = 36
FIXED_DD, FIXED_LAM, FIXED_HCAP, FIXED_CCAP = 0.006, 5, 0.10, 1.0
N_RESAMPLES = 100
RNG = np.random.default_rng(42)


def michaud_weights(trailing_m, daily_scen, n_resamples=N_RESAMPLES):
    n_months = len(trailing_m)
    values = trailing_m.values
    weight_sum = np.zeros(values.shape[1])
    n_ok = 0
    for _ in range(n_resamples):
        sample_idx = RNG.integers(0, n_months, size=n_months)
        resample = values[sample_idx]
        mu_r = resample.mean(axis=0)
        sigma_r = np.cov(resample, rowvar=False)
        w_r, ok = hard_constrained_weights(mu_r, sigma_r, daily_scen, FIXED_DD, FIXED_LAM,
                                            {1: FIXED_HCAP, 0: FIXED_CCAP})
        weight_sum += w_r
        n_ok += 1
    return weight_sum / n_ok


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

    mv_rets, michaud_rets = [], []
    for tm in test_months:
        trailing = training_panel[training_panel.index < tm].tail(FIXED_LOOKBACK)
        if len(trailing) < FIXED_LOOKBACK // 2:
            continue
        sigma = trailing.cov().values
        daily_scen = daily_panel[daily_panel.index < tm].tail(252).values
        actual = monthly_panel.loc[tm].values

        mu_mv = trailing.mean().values
        w_mv, _ = hard_constrained_weights(mu_mv, sigma, daily_scen, FIXED_DD, FIXED_LAM,
                                            {1: FIXED_HCAP, 0: FIXED_CCAP})
        mv_rets.append(w_mv @ actual)

        w_michaud = michaud_weights(trailing, daily_scen)
        michaud_rets.append(w_michaud @ actual)

    return pd.Series(mv_rets), pd.Series(michaud_rets)


def main():
    monthly_panel, daily_panel = build_extended_monthly_panel()
    blocks = make_blocks(monthly_panel)
    combos = list(itertools.combinations(range(N_BLOCKS), K_TEST))
    print(f"CPCV (Michaud check): {len(monthly_panel)} months, {N_BLOCKS} blocks, k={K_TEST}, "
          f"{len(combos)} combinations, embargo={EMBARGO_MONTHS}mo, n_resamples={N_RESAMPLES}\n")

    mv_sharpes, michaud_sharpes = [], []
    beat_count = 0
    for combo in combos:
        mv_ret, michaud_ret = eval_combo(monthly_panel, daily_panel, blocks, combo)
        _, _, s_mv, _ = sharpe_stats(mv_ret, freq=12)
        _, _, s_mic, _ = sharpe_stats(michaud_ret, freq=12)
        if not (np.isnan(s_mv) or np.isnan(s_mic)):
            mv_sharpes.append(s_mv)
            michaud_sharpes.append(s_mic)
            if s_mic > s_mv:
                beat_count += 1

    mv_arr, mic_arr = np.array(mv_sharpes), np.array(michaud_sharpes)
    diffs = mic_arr - mv_arr
    print(f"Valid combinations: {len(mv_arr)} / {len(combos)}\n")
    print(f"Plain point-estimate MV Sharpe across {len(mv_arr)} CPCV paths:")
    print(f"  mean={mv_arr.mean():.2f}  median={np.median(mv_arr):.2f}  std={mv_arr.std():.2f}  "
          f"min={mv_arr.min():.2f}  max={mv_arr.max():.2f}")
    print(f"\nMichaud resampled Sharpe across {len(mic_arr)} CPCV paths:")
    print(f"  mean={mic_arr.mean():.2f}  median={np.median(mic_arr):.2f}  std={mic_arr.std():.2f}  "
          f"min={mic_arr.min():.2f}  max={mic_arr.max():.2f}")
    print(f"\nMichaud beat plain MV in {beat_count}/{len(mv_arr)} paths ({beat_count/len(mv_arr):.0%})")
    print(f"Paired difference (Michaud - MV): mean={diffs.mean():+.2f}  median={np.median(diffs):+.2f}  "
          f"std={diffs.std():.2f}")


if __name__ == "__main__":
    main()
