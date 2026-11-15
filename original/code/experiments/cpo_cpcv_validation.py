"""
Combinatorial Purged Cross-Validation (CPCV, Lopez de Prado), applied
to this project's recurring pain point: every headline Sharpe so far
has come from exactly ONE train/test split (FWD vs REV), which is
really just 2 data points -- we've seen the number swing a lot
depending on which period gets held out (the very first overnight
search found REV-direction Sharpe ranging <real figure removed>-7.08 depending purely
on lookback choice, excluded from the final number for exactly this
reason).

CPCV fixes this by splitting the monthly panel into N contiguous
blocks and generating every combination of k blocks as a test set
(purged: test months excluded from training; embargoed: 1 month on
either side of each test block also excluded, guarding against
boundary leakage from any backward-looking rolling feature). Each
combination gives one full out-of-sample Sharpe -- with N=8, k=2 that's
C(8,2) = 28 independent-ish out-of-sample estimates instead of 1,
letting us report a genuine DISTRIBUTION (median, spread) rather than
a single point, and directly compare the MV baseline against the
Bayesian-shrunk allocation (the one prior ML approach that looked most
attractive on trailing data) across all 28 paths paired, not just the
single FWD split.

Simplifications, documented rather than hidden:
  - dd_limit/lambda/StrategyB-cap/SA-cap/lookback are held FIXED at
    representative values already shown elsewhere in this project to
    be robust (not fragile) within reasonable ranges -- re-deriving
    them per combination (28x a full grid search) is not where the
    open question is; the open question is whether Bayesian beats MV,
    which this isolates directly.
  - The daily drawdown-constraint scenarios use the trailing 252 daily
    rows immediately before each test month from the ORIGINAL daily
    panel (not re-purged) -- a minor, harmless simplification since
    these scenarios only shape the constraint boundary, not the return
    driver being compared, and a trailing window ending before a test
    month structurally can't include that same test month's own data.
"""
import itertools
import numpy as np
import pandas as pd

from cpo_hybrid_monthly import build_extended_monthly_panel, hard_constrained_weights, sharpe_stats, STRATEGY_COLS
from cpo_two_stage_blend import bayesian_mu

N_BLOCKS = 8
K_TEST = 2
EMBARGO_MONTHS = 1
FIXED_LOOKBACK = 36
FIXED_DD, FIXED_LAM, FIXED_HCAP, FIXED_CCAP = 0.006, 5, 0.10, 1.0
BAYES_WINDOW = 6


def make_blocks(monthly_panel):
    n = len(monthly_panel)
    edges = np.linspace(0, n, N_BLOCKS + 1).astype(int)
    return [monthly_panel.index[edges[i]:edges[i + 1]] for i in range(N_BLOCKS)]


def purge_embargo(monthly_panel, blocks, test_block_ids):
    test_months = set()
    embargo_months = set()
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

    mv_rets, bayes_rets = [], []
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

        mu_bayes = bayesian_mu(trailing, min(BAYES_WINDOW, len(trailing)))
        w_bayes, _ = hard_constrained_weights(mu_bayes, sigma, daily_scen, FIXED_DD, FIXED_LAM,
                                               {1: FIXED_HCAP, 0: FIXED_CCAP})
        bayes_rets.append(w_bayes @ actual)

    return pd.Series(mv_rets), pd.Series(bayes_rets)


def main():
    monthly_panel, daily_panel = build_extended_monthly_panel()
    blocks = make_blocks(monthly_panel)
    combos = list(itertools.combinations(range(N_BLOCKS), K_TEST))
    print(f"CPCV: {len(monthly_panel)} months, {N_BLOCKS} blocks, k={K_TEST} test blocks/combo, "
          f"{len(combos)} combinations, embargo={EMBARGO_MONTHS}mo\n")

    mv_sharpes, bayes_sharpes = [], []
    beat_count = 0
    for combo in combos:
        mv_ret, bayes_ret = eval_combo(monthly_panel, daily_panel, blocks, combo)
        _, _, s_mv, _ = sharpe_stats(mv_ret, freq=12)
        _, _, s_bayes, _ = sharpe_stats(bayes_ret, freq=12)
        if not (np.isnan(s_mv) or np.isnan(s_bayes)):
            mv_sharpes.append(s_mv)
            bayes_sharpes.append(s_bayes)
            if s_bayes > s_mv:
                beat_count += 1

    mv_arr, bayes_arr = np.array(mv_sharpes), np.array(bayes_sharpes)
    diffs = bayes_arr - mv_arr
    print(f"Valid combinations: {len(mv_arr)} / {len(combos)}\n")
    print(f"MV baseline Sharpe across {len(mv_arr)} CPCV paths:")
    print(f"  mean={mv_arr.mean():.2f}  median={np.median(mv_arr):.2f}  std={mv_arr.std():.2f}  "
          f"min={mv_arr.min():.2f}  max={mv_arr.max():.2f}")
    print(f"\nBayesian-mu Sharpe across {len(bayes_arr)} CPCV paths:")
    print(f"  mean={bayes_arr.mean():.2f}  median={np.median(bayes_arr):.2f}  std={bayes_arr.std():.2f}  "
          f"min={bayes_arr.min():.2f}  max={bayes_arr.max():.2f}")
    print(f"\nBayesian beat MV in {beat_count}/{len(mv_arr)} paths ({beat_count/len(mv_arr):.0%})")
    print(f"Paired difference (Bayesian - MV): mean={diffs.mean():+.2f}  median={np.median(diffs):+.2f}  "
          f"std={diffs.std():.2f}")


if __name__ == "__main__":
    main()
