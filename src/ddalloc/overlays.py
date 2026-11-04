"""Volatility-quantile exposure scaler, strictly causal."""
from __future__ import annotations

import numpy as np
import pandas as pd


def vol_quantile_multiplier(
    r: pd.Series, lookback: int = 20, lo: float = 0.7, hi: float = 1.3, min_hist: int = 60
) -> pd.Series:
    """Exposure multiplier in [lo, hi] for each day, lower when recent vol is high.

    Day t's multiplier uses trailing vol through day t-1 only (``shift(1)``),
    ranked against the history of that series before t. The original research
    code omitted the shift, so the multiplier saw the day's own return.
    """
    vol = r.rolling(lookback).std().shift(1).to_numpy()
    mult = np.ones(len(r))
    for i in range(min_hist, len(r)):
        cur = vol[i]
        hist = vol[:i]
        hist = hist[~np.isnan(hist)]
        if np.isnan(cur) or len(hist) < 20:
            continue
        q = float((hist < cur).mean())
        mult[i] = hi - q * (hi - lo)
    return pd.Series(mult, index=r.index)
