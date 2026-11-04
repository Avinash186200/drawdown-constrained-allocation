"""Synthetic data and loaders.

The real return series are proprietary. The generator below produces four
strategy return streams with the same shape as the original problem:

* a long monthly-only history before daily data starts, and
* a shorter daily panel used for the daily loss constraint and for scoring.

Strategy profiles (all market-neutral in spirit, so no market factor):

* ``ArbA``   - steady arbitrage, low vol, occasional mild drawdown
* ``OptGap`` - option selling: high mean, rare large single-day gap losses
* ``Intra``  - intraday systematic: high Sharpe, fat right tail
* ``Carry``  - near-cash, very low vol
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

STRATEGIES: list[str] = ["ArbA", "OptGap", "Intra", "Carry"]

# annualised mean, annualised vol (daily, before jumps)
_PROFILE = {
    "ArbA": (0.14, 0.05),
    "OptGap": (0.30, 0.10),
    "Intra": (0.20, 0.06),
    "Carry": (0.075, 0.01),
}


@dataclass(frozen=True)
class Panels:
    """Monthly estimation panel and daily scoring panel."""

    monthly: pd.DataFrame
    daily: pd.DataFrame


def _daily_returns(
    n_days: int, rng: np.random.Generator, signal: bool, decay: float
) -> np.ndarray:
    """Draw (n_days, 4) daily returns. ``signal=False`` gives zero-mean noise."""
    out = np.empty((n_days, len(STRATEGIES)))
    trend = np.linspace(1.0, 1.0 - decay, n_days)  # optional slow decay of edge
    corr = np.array(
        [
            [1.0, 0.05, 0.5, 0.1],
            [0.05, 1.0, 0.05, 0.0],
            [0.5, 0.05, 1.0, 0.1],
            [0.1, 0.0, 0.1, 1.0],
        ]
    )
    chol = np.linalg.cholesky(corr)
    z = rng.standard_normal((n_days, len(STRATEGIES))) @ chol.T
    for k, name in enumerate(STRATEGIES):
        mu_a, vol_a = _PROFILE[name]
        mu = (mu_a / 252.0) * trend if signal else 0.0
        out[:, k] = mu + z[:, k] * vol_a / np.sqrt(252.0)
    # gap risk on the option strategy: rare, large, unannounced losses
    gaps = rng.random(n_days) < 0.004
    out[gaps, 1] -= rng.uniform(0.02, 0.06, gaps.sum())
    if signal:
        out[~gaps, 1] += 0.004 * 0.04 / (1 - 0.004)  # keep the mean roughly as profiled
    return out


def make_synthetic(
    seed: int = 7,
    monthly_only_months: int = 27,
    daily_days: int = 760,
    signal: bool = True,
    decay: float = 0.5,
) -> Panels:
    """Build a monthly panel (history + daily-aggregated) and a daily panel.

    Args:
        seed: RNG seed.
        monthly_only_months: months of history that exist only at monthly frequency.
        daily_days: business days in the daily panel.
        signal: if False, every strategy is zero-mean noise (used by the noise test).
        decay: fraction by which each strategy's mean decays over the sample.
    """
    rng = np.random.default_rng(seed)
    total_days = monthly_only_months * 21 + daily_days
    start = pd.Timestamp("2021-01-01")
    idx = pd.bdate_range(start, periods=total_days)
    all_daily = pd.DataFrame(_daily_returns(total_days, rng, signal, decay), index=idx, columns=STRATEGIES)
    # daily data is only "available" from the first month boundary after the monthly-only stretch
    cut = idx[monthly_only_months * 21]
    cut = (cut + pd.offsets.MonthBegin(0)).normalize()
    daily = all_daily[all_daily.index >= cut]
    monthly = to_monthly(all_daily)
    return Panels(monthly=monthly, daily=daily)


def to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    """Compound daily returns into calendar month-end returns."""
    m = (1.0 + daily).resample("ME").prod() - 1.0
    counts = daily.resample("ME").size()
    return m[counts > 0]


def load_csv(monthly_path: str, daily_path: str) -> Panels:
    """Load user data. Both CSVs: first column a date, one column per strategy, simple returns."""
    monthly = pd.read_csv(monthly_path, index_col=0, parse_dates=True).sort_index()
    daily = pd.read_csv(daily_path, index_col=0, parse_dates=True).sort_index()
    monthly.index = monthly.index + pd.offsets.MonthEnd(0)
    monthly = monthly[~monthly.index.duplicated(keep="last")].dropna()
    daily = daily[~daily.index.duplicated(keep="last")].dropna()
    if list(monthly.columns) != list(daily.columns):
        raise ValueError("monthly and daily files must have the same strategy columns")
    return Panels(monthly=monthly, daily=daily)
