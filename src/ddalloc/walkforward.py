"""Hybrid monthly/daily walk-forward allocator.

Timing contract (the core leakage control):

* The decision for calendar month ``M_i`` is made at the close of ``M_{i-1}``.
* Means and covariances use monthly rows dated ``<= M_{i-1}``.
* Loss-limit scenarios use daily rows dated ``<= M_{i-1}``.
* The weights are applied to daily returns in ``(M_{i-1}, M_i]``.
* Hyper-parameters are re-selected every ``retune_every`` months using only
  data available at that decision date, from a grid fixed in ``Grid``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .metrics import stats
from .optimizer import constrained_weights
from .overlays import vol_quantile_multiplier


@dataclass(frozen=True)
class Grid:
    """Candidate values. Fixed before any run; never edited after looking at results."""

    loss_limits: tuple[float, ...] = (0.010, 0.006, 0.005)
    lams: tuple[float, ...] = (5.0, 10.0)
    gap_caps: tuple[float, ...] = (1.0, 0.25, 0.10, 0.05)  # cap on the gap-risk strategy
    second_caps: tuple[float, ...] = (1.0, 0.6, 0.4, 0.25)  # cap on the largest-weight strategy
    lookbacks: tuple[int, ...] = (18, 24, 36)


@dataclass(frozen=True)
class Recipe:
    """One set of hyper-parameters, chosen at a retune and held until the next one."""

    loss_limit: float
    lam: float
    gap_cap: float
    second_cap: float
    lookback: int


@dataclass
class Config:
    """Run settings. ``gap_idx``/``second_idx`` are column positions of the two capped strategies."""

    grid: Grid = field(default_factory=Grid)
    gap_idx: int = 1
    second_idx: int = 0
    retune_every: int = 6
    scenario_days: int = 252
    selection_lookback: int = 24
    cost_bps: float = 2.0
    overlay: bool = True


def weights_at(
    monthly: pd.DataFrame, daily: pd.DataFrame, decision_date: pd.Timestamp, r: Recipe, cfg: Config
) -> np.ndarray:
    """Weights decided at ``decision_date`` using only rows dated <= decision_date."""
    m = monthly[monthly.index <= decision_date].tail(r.lookback)
    scen = daily[daily.index <= decision_date].tail(cfg.scenario_days).to_numpy()
    mu = m.mean().to_numpy()
    sigma = m.cov().to_numpy()
    caps = {cfg.gap_idx: r.gap_cap, cfg.second_idx: r.second_cap}
    return constrained_weights(mu, sigma, scen, r.loss_limit, r.lam, caps)


def _inner_sharpe(monthly: pd.DataFrame, daily: pd.DataFrame, r: Recipe, cfg: Config) -> float:
    """Monthly walk-forward Sharpe of recipe ``r`` inside the data the caller passed in."""
    rets = []
    for j in range(r.lookback, len(monthly)):
        w = weights_at(monthly, daily, monthly.index[j - 1], r, cfg)
        rets.append(float(w @ monthly.iloc[j].to_numpy()))
    if len(rets) < 3:
        return float("nan")
    s = pd.Series(rets)
    return float(s.mean() / s.std() * np.sqrt(12)) if s.std() > 0 else float("nan")


def select_recipe(monthly: pd.DataFrame, daily: pd.DataFrame, cfg: Config) -> Recipe:
    """Coordinate-wise grid search on trailing data only (the caller has already truncated it)."""
    g = cfg.grid
    default = Recipe(0.008, 10.0, 1.0, 1.0, min(g.lookbacks))
    if len(monthly) < cfg.selection_lookback + 6:
        return default

    def best(cands: list[Recipe]) -> Recipe:
        scored = [(_inner_sharpe(monthly, daily, c, cfg), k, c) for k, c in enumerate(cands)]
        scored = [s for s in scored if not np.isnan(s[0])]
        return max(scored, key=lambda s: (s[0], -s[1]))[2] if scored else cands[0]

    lb = cfg.selection_lookback
    r = best([Recipe(d, l, 1.0, 1.0, lb) for d in g.loss_limits for l in g.lams])
    r = best([Recipe(r.loss_limit, r.lam, c, 1.0, lb) for c in g.gap_caps])
    r = best([Recipe(r.loss_limit, r.lam, r.gap_cap, c, lb) for c in g.second_caps])
    r = best([Recipe(r.loss_limit, r.lam, r.gap_cap, r.second_cap, k) for k in g.lookbacks])
    return r


@dataclass
class Result:
    """Net daily returns, effective (overlay-scaled) daily weights, and retune history."""

    returns: pd.Series
    weights: pd.DataFrame
    recipes: list[tuple[pd.Timestamp, Recipe]]


def run(monthly: pd.DataFrame, daily: pd.DataFrame, cfg: Config | None = None) -> Result:
    """Walk forward month by month over the daily panel and return net daily book returns."""
    cfg = cfg or Config()
    month_ends = monthly.index
    first_daily = daily.index[0]
    recipe: Recipe | None = None
    since = 0
    recipes: list[tuple[pd.Timestamp, Recipe]] = []
    rows: dict[pd.Timestamp, np.ndarray] = {}
    for i in range(1, len(month_ends)):
        decision, end = month_ends[i - 1], month_ends[i]
        if end < first_daily:
            continue
        avail_m = monthly[monthly.index <= decision]
        if len(avail_m) < min(cfg.grid.lookbacks):
            continue
        avail_d = daily[daily.index <= decision]
        if recipe is None or since >= cfg.retune_every:
            recipe = select_recipe(avail_m, avail_d, cfg)
            recipes.append((decision, recipe))
            since = 0
        since += 1
        w = weights_at(avail_m, avail_d, decision, recipe, cfg)
        for d in daily.index[(daily.index > decision) & (daily.index <= end)]:
            rows[d] = w
    W = pd.DataFrame.from_dict(rows, orient="index", columns=daily.columns).sort_index()
    gross = (daily.loc[W.index] * W).sum(axis=1)
    if cfg.overlay:
        mult = vol_quantile_multiplier(gross)
        W = W.mul(mult, axis=0)
        gross = gross * mult
    turnover = W.diff().abs().sum(axis=1).fillna(W.iloc[0].abs().sum() if len(W) else 0.0)
    net = gross - turnover * cfg.cost_bps / 1e4
    return Result(returns=net, weights=W, recipes=recipes)


def equal_weight(daily: pd.DataFrame, index: pd.Index) -> pd.Series:
    """Daily-rebalanced equal-weight book over the same dates (no costs: an easy-to-beat-on-cost baseline)."""
    return daily.loc[index].mean(axis=1)


def split_report(r: pd.Series, rf: float = 0.0) -> dict[str, dict[str, float]]:
    """Full period, first 40% ("early") and last 40% ("late") of the scored days."""
    n = len(r)
    return {
        "full": stats(r, rf),
        "early_40": stats(r.iloc[: int(n * 0.4)], rf),
        "late_40": stats(r.iloc[int(n * 0.6):], rf),
    }
