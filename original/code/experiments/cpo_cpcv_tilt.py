"""
CPCV robustness check for the corrected (deterministic) tilt design
from approach 10/19. Seed-robustness alone showed FWD Sharpe <real figure removed>
across 5 different random seeds with zero variance -- a much stronger
starting signal than anything else in this project -- but seed
-stability only rules out model-initialization randomness, not whether
the signal actually generalizes to different data. Given how many
promising single-split results have not survived CPCV before (copula:
looked like a real win, reversed to 32% win rate), this gets the same
proper check before anything is trusted.

Design: same 8-block/22-path combinatorial structure as the other
cpo_cpcv_*.py scripts. For each combo, the ML tilt classifier trains
ONLY on daily data whose calendar month falls in the (purged,
embargoed) training months -- which may be non-contiguous, since CPCV
test blocks can sit anywhere in the timeline. Forward-return labels
are computed from the full daily panel (features are already causal/
shift(1)'d) but any training row whose forward-return window would
extend past its own contiguous stretch's end is excluded, preventing
leakage into a removed test/embargo block. Fixed dd/lambda/caps match
the other CPCV scripts (applied identically to both variants, so the
comparison is paired/fair); tilt_cap is fixed at a principled a priori
value (20%, the most common non-zero value the honest process
actually chose across the earlier walk-forward runs of this exact
design) rather than swept per combo, to keep this tractable.
"""
import itertools
import numpy as np
import pandas as pd

from cpo_hybrid_monthly import build_extended_monthly_panel, hard_constrained_weights, sharpe_stats, STRATEGY_COLS
from cpo_feature_conditioning import build_features, apply_tilt, make_classifiers, TILT_HORIZON_DAYS

N_BLOCKS = 8
K_TEST = 2
EMBARGO_MONTHS = 1
FIXED_LOOKBACK = 36
FIXED_DD, FIXED_LAM, FIXED_HCAP, FIXED_CCAP = 0.006, 5, 0.10, 1.0
FIXED_TILT_CAP = 0.20
MIN_TRAIN = 100


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


def contiguous_stretches(daily_panel, train_months_set):
    # matches cpo_hybrid_monthly.month_end()'s own convention exactly
    day_month_end = daily_panel.index + pd.offsets.MonthEnd(0)
    mask = day_month_end.isin(train_months_set)
    positions = np.where(mask)[0]
    if len(positions) == 0:
        return []
    stretches = []
    start = prev = positions[0]
    for p in positions[1:]:
        if p != prev + 1:
            stretches.append((start, prev))
            start = p
        prev = p
    stretches.append((start, prev))
    return stretches


def tilt_signal_cpcv(daily_panel, features_by_col, stretches, x_today_idx, horizon=TILT_HORIZON_DAYS, min_train=MIN_TRAIN):
    signal = np.zeros(len(STRATEGY_COLS))
    for k, col in enumerate(STRATEGY_COLS):
        x_today = features_by_col[col].iloc[[x_today_idx]]
        if x_today.isna().any(axis=1).values[0]:
            continue
        r = daily_panel[col]
        fwd_ret = r.shift(-1).rolling(horizon).sum().shift(-(horizon - 1))

        X_parts, y_parts = [], []
        for (s, e) in stretches:
            e_trim = e - horizon  # exclude rows whose fwd window would cross the stretch boundary
            if e_trim <= s:
                continue
            X_seg = features_by_col[col].iloc[s:e_trim + 1]
            y_seg = (fwd_ret.iloc[s:e_trim + 1] > 0).astype(int)
            valid = X_seg.notna().all(axis=1) & fwd_ret.iloc[s:e_trim + 1].notna()
            X_parts.append(X_seg[valid])
            y_parts.append(y_seg[valid])
        if not X_parts:
            continue
        X_train = pd.concat(X_parts)
        y_train = pd.concat(y_parts)
        if len(X_train) < min_train or len(np.unique(y_train)) < 2:
            continue

        from sklearn.preprocessing import StandardScaler
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


def eval_combo(monthly_panel, daily_panel, features, blocks, test_block_ids):
    test_months, train_months = purge_embargo(monthly_panel, blocks, test_block_ids)
    training_panel = monthly_panel.loc[train_months]
    train_months_set = set(train_months)
    stretches = contiguous_stretches(daily_panel, train_months_set)

    mv_rets, tilt_rets = [], []
    for tm in test_months:
        trailing = training_panel[training_panel.index < tm].tail(FIXED_LOOKBACK)
        if len(trailing) < FIXED_LOOKBACK // 2:
            continue
        mu = trailing.mean().values
        sigma = trailing.cov().values
        daily_scen = daily_panel[daily_panel.index < tm].tail(252).values
        actual = monthly_panel.loc[tm].values

        w_mv, _ = hard_constrained_weights(mu, sigma, daily_scen, FIXED_DD, FIXED_LAM,
                                            {1: FIXED_HCAP, 0: FIXED_CCAP})
        mv_rets.append(w_mv @ actual)

        hist_before_tm = daily_panel[daily_panel.index < tm]
        if len(hist_before_tm) < 100:
            tilt_rets.append(w_mv @ actual)
            continue
        as_of_idx = daily_panel.index.get_loc(hist_before_tm.index[-1])
        sig = tilt_signal_cpcv(daily_panel, features, stretches, as_of_idx)
        w_tilt = apply_tilt(w_mv, sig, FIXED_TILT_CAP)
        tilt_rets.append(w_tilt @ actual)

    return pd.Series(mv_rets), pd.Series(tilt_rets)


def main():
    monthly_panel, daily_panel = build_extended_monthly_panel()
    features = build_features(daily_panel, use_interactions=False)
    blocks = make_blocks(monthly_panel)
    combos = list(itertools.combinations(range(N_BLOCKS), K_TEST))
    print(f"CPCV (tilt check): {len(monthly_panel)} months, {N_BLOCKS} blocks, k={K_TEST}, "
          f"{len(combos)} combinations, embargo={EMBARGO_MONTHS}mo, tilt_cap={FIXED_TILT_CAP:.0%}\n")

    mv_sharpes, tilt_sharpes = [], []
    beat_count = 0
    for ci, combo in enumerate(combos):
        mv_ret, tilt_ret = eval_combo(monthly_panel, daily_panel, features, blocks, combo)
        _, _, s_mv, _ = sharpe_stats(mv_ret, freq=12)
        _, _, s_tilt, _ = sharpe_stats(tilt_ret, freq=12)
        if not (np.isnan(s_mv) or np.isnan(s_tilt)):
            mv_sharpes.append(s_mv)
            tilt_sharpes.append(s_tilt)
            if s_tilt > s_mv:
                beat_count += 1
        print(f"  combo {ci+1}/{len(combos)} (blocks {combo}): MV={s_mv:.2f} tilt={s_tilt:.2f}")

    mv_arr, tilt_arr = np.array(mv_sharpes), np.array(tilt_sharpes)
    diffs = tilt_arr - mv_arr
    print(f"\nValid combinations: {len(mv_arr)} / {len(combos)}\n")
    print(f"Plain MV Sharpe across {len(mv_arr)} CPCV paths:")
    print(f"  mean={mv_arr.mean():.2f}  median={np.median(mv_arr):.2f}  std={mv_arr.std():.2f}  "
          f"min={mv_arr.min():.2f}  max={mv_arr.max():.2f}")
    print(f"\nTilt Sharpe across {len(tilt_arr)} CPCV paths:")
    print(f"  mean={tilt_arr.mean():.2f}  median={np.median(tilt_arr):.2f}  std={tilt_arr.std():.2f}  "
          f"min={tilt_arr.min():.2f}  max={tilt_arr.max():.2f}")
    print(f"\nTilt beat plain MV in {beat_count}/{len(mv_arr)} paths ({beat_count/len(mv_arr):.0%})")
    print(f"Paired difference (tilt - MV): mean={diffs.mean():+.2f}  median={np.median(diffs):+.2f}  "
          f"std={diffs.std():.2f}")


if __name__ == "__main__":
    main()
