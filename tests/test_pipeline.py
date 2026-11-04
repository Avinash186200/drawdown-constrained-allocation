from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ddalloc import (  # noqa: E402
    Config, Grid, Recipe, apply_caps, constrained_weights, equal_weight, make_synthetic, run,
    stats, to_monthly, vol_quantile_multiplier,
)
from ddalloc.walkforward import weights_at  # noqa: E402

SMALL = Grid(loss_limits=(0.01, 0.005), lams=(10.0,), gap_caps=(1.0, 0.1), second_caps=(1.0,), lookbacks=(18, 24))


@pytest.fixture(scope="module")
def panels():
    return make_synthetic(seed=3, monthly_only_months=24, daily_days=320)


def test_caps_respected_and_sum_to_one():
    w = apply_caps(np.array([0.7, 0.2, 0.1, 0.0]), {0: 0.4})
    assert w[0] == pytest.approx(0.4)
    assert w.sum() == pytest.approx(1.0)


def test_loss_limit_holds_on_scenarios():
    rng = np.random.default_rng(0)
    scen = rng.normal(0.0005, 0.004, (250, 4))
    scen[10, 1] = -0.08
    m = scen.mean(0) * 21
    w = constrained_weights(m, np.cov(scen.T) * 21, scen, 0.01, 5.0)
    assert (scen @ w).min() >= -0.01 - 1e-6
    assert w.sum() == pytest.approx(1.0) and (w >= -1e-9).all()


def test_infeasible_limit_warns():
    scen = np.full((5, 3), -0.03)
    with pytest.warns(RuntimeWarning):
        constrained_weights(np.zeros(3), np.eye(3), scen, 0.01, 1.0)


def test_overlay_is_causal():
    r = pd.Series(np.random.default_rng(1).normal(0, 0.01, 200))
    base = vol_quantile_multiplier(r)
    shocked = r.copy()
    shocked.iloc[150] = -0.2
    after = vol_quantile_multiplier(shocked)
    assert after.iloc[:151].equals(base.iloc[:151])  # day 150's own return cannot move day 150's exposure


def test_weights_ignore_the_month_being_traded(panels):
    """Leakage check: corrupting month M_i (daily rows and its monthly row) must not change M_i's weights."""
    cfg = Config(grid=SMALL)
    me = panels.monthly.index
    decision = me[30]
    recipe = Recipe(0.006, 10.0, 0.25, 1.0, 18)
    w0 = weights_at(panels.monthly, panels.daily, decision, recipe, cfg)
    d2 = panels.daily.copy()
    d2.loc[d2.index > decision] = -0.05
    m2 = panels.monthly.copy()
    m2.loc[m2.index > decision] = 0.5
    w1 = weights_at(m2, d2, decision, recipe, cfg)
    np.testing.assert_allclose(w0, w1)


def test_run_is_point_in_time(panels):
    """Corrupt the future after a cut: every return up to the cut must be identical."""
    cfg = Config(grid=SMALL, overlay=True)
    base = run(panels.monthly, panels.daily, cfg)
    cut = base.returns.index[150]
    cut_me = cut + pd.offsets.MonthEnd(0)
    d2 = panels.daily.copy()
    d2.loc[d2.index > cut_me] *= -3.0
    m2 = panels.monthly.copy()
    m2.loc[m2.index > cut_me] = 0.5
    alt = run(m2, d2, cfg)
    common = base.returns.index[base.returns.index <= cut_me]
    pd.testing.assert_series_equal(base.returns.loc[common], alt.returns.loc[common])


def test_to_monthly_compounds():
    d = pd.DataFrame({"a": [0.1, 0.1]}, index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    assert to_monthly(d).iloc[0, 0] == pytest.approx(0.21)


def test_noise_has_no_edge():
    """On zero-mean noise the allocator must not manufacture a Sharpe advantage over equal weight."""
    diffs = []
    for seed in (11, 12):
        p = make_synthetic(seed=seed, monthly_only_months=24, daily_days=320, signal=False)
        res = run(p.monthly, p.daily, Config(grid=SMALL))
        diffs.append(stats(res.returns)["sharpe"] - stats(equal_weight(p.daily, res.returns.index))["sharpe"])
    assert np.mean(diffs) < 1.0
    assert abs(stats(res.returns)["sharpe"]) < 1.5


def test_signal_beats_noise():
    p = make_synthetic(seed=5, monthly_only_months=24, daily_days=320, signal=True, decay=0.0)
    res = run(p.monthly, p.daily, Config(grid=SMALL))
    assert stats(res.returns)["sharpe"] > 1.0
