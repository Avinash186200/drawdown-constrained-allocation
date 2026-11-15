# Walkthrough

This follows one run of `scripts/run_backtest.py` on the default synthetic panels (seed 7), function by function.
Every number here comes from running the code in this repo. To follow along in a Python shell started at the repo
root, first run `import sys; sys.path.insert(0, "src")` and `from ddalloc import *`.

## 1. Two panels

```python
p = make_synthetic(seed=7)
```

Four strategies: `ArbA` (steady arbitrage), `OptGap` (option selling with rare gap losses), `Intra` (intraday,
high Sharpe) and `Carry` (near cash). `p.monthly` has 62 month-end rows from January 2021. `p.daily` has 741
business days from April 2023. Before daily data starts the monthly rows are drawn directly; after that they are
the daily returns compounded to month-end. The worst single day for `OptGap` is -6.1%, which is the gap risk the
cap is there for. Their edge decays by half over the sample. `scripts/make_synthetic_data.py` writes both panels
to `data/`.

## 2. One decision

```python
cfg = Config()
w = weights_at(p.monthly, p.daily, decision_date, recipe, cfg)
```

At each month-end `weights_at` keeps only rows dated on or before the decision date. It takes the mean and
covariance of the last `L` monthly rows, and the last 252 daily rows as loss scenarios. `constrained_weights`
then solves

    max  w'mu - lam/2 * w'Sigma w
    s.t. sum(w) = 1,  w >= 0,  w'r_t >= -limit for every scenario day t

with SciPy's SLSQP, and `apply_caps` trims the capped strategies and spreads the excess pro rata.

## 3. Choosing the recipe

`select_recipe` runs every 6 months. It tries `limit` and `lam` together, then the gap cap, then the second cap,
then `L`, keeping the best inner walk-forward Sharpe at each step, on data up to the decision date only. The first
retune (31 March 2023) has too little history, so it uses the cold-start recipe: 0.8% limit, `lam` 10, no caps,
18-month lookback. There are no daily rows yet at that point, so the first month is 100% `Intra`. In this run
there are 6 retunes. The gap cap first switches on in September 2024 (5%).

## 4. The overlay

`vol_quantile_multiplier` takes the allocator's gross daily returns, computes 20-day vol up to the day before, and
ranks it against its own history. Low vol gives up to 1.3x, high vol down to 0.7x. Because of this the effective
weights sum to between 0.7 and 1.3, with a mean of 1.11.

## 5. Costs and comparison

`run` charges 2 bp per unit of turnover, including turnover from the overlay, and returns net daily returns,
weights and the recipe history. `split_report` prints the full period, the first 40% and the last 40%.
On the full 741 days the allocator has a Sharpe of 2.69 and a worst day of -1.08%. Equal weight has a Sharpe of
3.08 and a worst day of -1.61%. So the allocator keeps the daily loss well inside the 2% cap, but it does not beat
equal weight on Sharpe here.

## Things to try

- `--no-overlay`: how much of the result is the overlay?
- `--noise`: the allocator should show no edge.
- Change `Grid` in `walkforward.py`, and note that doing so after seeing results is exactly the residual bias the
  audit table describes.
