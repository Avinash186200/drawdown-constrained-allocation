"""Performance statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def stats(r: pd.Series, rf: float = 0.0, freq: int = 252) -> dict[str, float]:
    """Annualised return, vol, Sharpe (excess of ``rf``), worst day, max drawdown."""
    r = r.dropna()
    n = len(r)
    if n < 5:
        return {k: float("nan") for k in ("ann_ret", "ann_vol", "sharpe", "worst_day", "max_dd", "n")}
    ann_ret = float((1 + r).prod() ** (freq / n) - 1)
    ann_vol = float(r.std() * np.sqrt(freq))
    eq = (1 + r).cumprod()
    max_dd = float((eq / eq.cummax() - 1).min())
    return {
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": (ann_ret - rf) / ann_vol if ann_vol > 0 else float("nan"),
        "worst_day": float(r.min()),
        "max_dd": max_dd,
        "n": float(n),
    }
