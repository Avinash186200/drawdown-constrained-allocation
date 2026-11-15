"""
Follow-up to approach 19's reproducibility catch: the deterministic
rerun of approach 10's tilt design showed FWD Sharpe <real figure removed>, above the
3.92 baseline -- genuinely promising, but found by accident while
testing something else, never stress-tested on its own merits.

Before investing in a full CPCV check (expensive -- retrains
classifiers per test month across 22 combinatorial paths), this does a
cheap first check: does 4.14 hold up across different classifier
random seeds, or is it itself just one lucky draw from model-
initialization randomness? If the FWD Sharpe swings a lot across
seeds, that's a strong signal the "improvement" isn't real before
spending the time on the expensive check.
"""
import numpy as np

from cpo_feature_conditioning import build_features, honest_pipeline, sharpe_stats, make_classifiers
from cpo_hybrid_monthly import build_extended_monthly_panel
import cpo_feature_conditioning as fc

SEEDS = [42, 7, 123, 2024, 99]


def make_classifiers_seeded(seed):
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    return [
        lgb.LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, num_leaves=8,
                            subsample=0.8, colsample_bytree=0.8, random_state=seed, verbose=-1,
                            deterministic=True, force_row_wise=True, num_threads=1),
        Pipeline([("scaler", StandardScaler()),
                  ("clf", LogisticRegression(penalty="l2", C=5, solver="liblinear", max_iter=1000, random_state=seed))]),
        Pipeline([("scaler", StandardScaler()),
                  ("clf", ExtraTreesClassifier(n_estimators=100, max_depth=4, min_samples_leaf=15,
                                                max_features="sqrt", random_state=seed, n_jobs=1))]),
        Pipeline([("scaler", StandardScaler()),
                  ("clf", RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=15,
                                                  max_features="sqrt", random_state=seed, n_jobs=1))]),
    ]


def main():
    monthly_panel, daily_panel = build_extended_monthly_panel()
    feats = build_features(daily_panel, use_interactions=False)

    results = []
    for seed in SEEDS:
        fc.make_classifiers = lambda s=seed: make_classifiers_seeded(s)
        final, recipes = honest_pipeline(monthly_panel, daily_panel, feats)
        if final is None:
            print(f"seed={seed}: insufficient data")
            continue
        n = len(final)
        split_60, split_40 = int(n * 0.6), int(n * 0.4)
        fwd, rev = final.iloc[split_60:], final.iloc[:split_40]
        r, v, s, w = sharpe_stats(final)
        _, _, s_f, w_f = sharpe_stats(fwd)
        _, _, s_r, w_r = sharpe_stats(rev)
        print(f"seed={seed}: full={s:.2f} FWD={s_f:.2f} REV={s_r:.2f} worst_day={w:.3%}")
        results.append((seed, s, s_f, s_r))

    fwd_vals = np.array([r[2] for r in results])
    print(f"\nFWD Sharpe across {len(fwd_vals)} seeds: mean={fwd_vals.mean():.2f} std={fwd_vals.std():.2f} "
          f"min={fwd_vals.min():.2f} max={fwd_vals.max():.2f}")
    print(f"(baseline FWD Sharpe for comparison: <real figure removed>)")


if __name__ == "__main__":
    main()
