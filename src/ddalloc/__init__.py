"""Drawdown-constrained portfolio allocation across a small book of strategies."""
from .data import Panels, load_csv, make_synthetic, to_monthly
from .metrics import stats
from .optimizer import apply_caps, constrained_weights
from .overlays import vol_quantile_multiplier
from .walkforward import Config, Grid, Recipe, equal_weight, run, select_recipe, split_report

__all__ = [
    "Panels", "load_csv", "make_synthetic", "to_monthly", "stats", "apply_caps",
    "constrained_weights", "vol_quantile_multiplier", "Config", "Grid", "Recipe",
    "equal_weight", "run", "select_recipe", "split_report",
]
