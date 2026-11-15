# Drawdown-Constrained Allocation

[![CI](https://github.com/Avinash186200/drawdown-constrained-allocation/actions/workflows/ci.yml/badge.svg)](https://github.com/Avinash186200/drawdown-constrained-allocation/actions/workflows/ci.yml) ![Python](https://img.shields.io/badge/python-3.11-blue) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A walk-forward allocator that splits capital across a small book of strategies (four, in my case) under a **hard cap on single-day book loss**. Means and covariances come from monthly history, which goes back further. The daily loss limit is enforced on real daily returns, because monthly data averages single-day risk away and cannot check it.

I built this for a small book of market-neutral strategies, one of which carries gap risk. The return series are proprietary and not in this repo. Everything here runs on a **synthetic panel with the same shape** (a monthly-only stretch, then about three years of daily data), so you can clone the repo and reproduce every number below.

## What problem this solves

A fund runs a few strategies and has to decide how much capital each one gets. The mandate says the whole book must never lose more than a fixed share of capital in a single day. Monthly history is long but hides bad days, and daily history is short, so this repo uses both: monthly data for the averages, daily data to check the loss limit.

## At a glance

| | |
|---|---|
| **Input** | A monthly returns panel and a daily returns panel, one column per strategy |
| **Method** | Mean-variance with a historical-scenario constraint (`w'r_t >= -limit` for every past day), per-strategy caps, and a causal volatility-quantile exposure scaler. Hyper-parameters are re-selected every 6 months from trailing data only |
| **Output** | Daily net book returns, daily weights, the hyper-parameters chosen at each retune, and a comparison with equal weight |
| **Leakage controls** | The decision for month M uses data dated at or before the end of M-1. The overlay is shifted by one day. The grid is fixed in code. Costs are charged on turnover. Point-in-time and noise tests run in CI |
| **Status** | Research complete. On the real book the allocator looked strong, but the figure is not validated on a final hold-out, so I don't publish it. ML forecasting layers did not add value |

## How it works

```mermaid
flowchart TD
    M[monthly panel<br/>data.py] --> R
    D[daily panel<br/>data.py] --> R
    R[month-end M-1: re-select recipe every 6 months<br/>trailing data only, walkforward.py] --> E
    E[mu and Sigma from last L months<br/>last 252 daily rows as loss scenarios] --> O
    O[max w'mu - lam/2 w'Sigma w<br/>s.t. w'r_t >= -limit, then caps<br/>optimizer.py] --> H
    H[hold w through month M] --> V
    V[vol-quantile scaler 0.7x-1.3x<br/>uses vol up to t-1, overlays.py] --> C
    C[net of turnover costs<br/>compare with equal weight, metrics.py]
```

1. **Estimate** each strategy's mean and covariance from the trailing `L` months of the monthly panel.
2. **Constrain** the portfolio so that it would have lost no more than `limit` on any of the last 252 days. The constraint is non-parametric and makes no distribution assumption.
3. **Cap** the gap-risk strategy and the largest strategy, then redistribute the excess pro rata.
4. **Re-select** `limit`, risk aversion `lam`, the two caps and `L` every 6 months. Each candidate runs as an inner walk-forward on data available at that date only, and the one with the best inner Sharpe wins.
5. **Scale** daily exposure between 0.7x and 1.3x by where yesterday's trailing 20-day book vol ranks in its own history.
6. **Charge costs** on turnover, and compare with an equal-weight book on the same days.

A step-by-step code walkthrough with real intermediate output is in [docs/walkthrough.md](docs/walkthrough.md).

## Key concepts

- **Strategy book**: a set of trading strategies, each with its own return stream. Here the allocator picks weights across them, not across assets.
- **Mean-variance optimisation**: choose weights that maximise expected return minus `lam/2` times variance. `lam` is the risk aversion.
- **Historical-scenario constraint**: the book, with today's weights, must not have lost more than `limit` on any of the last 252 days. It needs no distribution assumption.
- **Per-strategy cap**: a maximum weight on one strategy, with the excess spread over the others.
- **Hyper-parameter (recipe)**: the settings `limit`, `lam`, the two caps and the lookback `L`.
- **Trailing reselection**: pick the recipe again every 6 months using only data that existed at that date.
- **Walk-forward**: decide at the end of month M-1, hold through month M, move one month on.
- **Point-in-time**: nothing used for a decision is dated after the decision. A test corrupts future data and checks that past results do not change.
- **Vol-quantile overlay**: scale the whole book down when its recent volatility is high relative to its own past, and up when it is low.
- **Turnover cost**: a charge in basis points for each unit of weight that changes.
- **Equal weight**: 1/N in every strategy. A simple baseline that is hard to beat when means are noisy.
- **Sharpe ratio**: annualised mean return divided by annualised volatility. **Max drawdown**: the worst fall from a previous peak.

## What you will learn

- How to combine a long monthly history with a short daily history in one allocator.
- How to write a daily loss limit as a linear constraint and solve it with SciPy.
- How to choose hyper-parameters honestly, with a nested walk-forward on trailing data.
- How to write tests that prove a backtest is point-in-time.
- Why an honest result can be "this does not beat equal weight", and how to report it.
- Why more ML did not help on a small sample (see the list of negative results).

## Results on synthetic data

All numbers in this section come from the **synthetic** panel. They show how the method behaves, not what it earns on real money.

`python scripts/run_backtest.py` (seed 7, 741 scored days, 2 bps per unit of turnover):

| Book | Period | Ann. return | Ann. vol | Sharpe | Worst day | Max DD |
|---|---|---|---|---|---|---|
| Allocator | full | 13.06% | 4.85% | 2.69 | -1.08% | -3.13% |
| Allocator | first 40% | 16.78% | 5.68% | 2.95 | -1.08% | -2.47% |
| Allocator | last 40% | 5.95% | 3.95% | 1.51 | -0.67% | -3.13% |
| Equal weight | full | 11.82% | 3.84% | 3.08 | -1.61% | -2.35% |
| Equal weight | first 40% | 15.42% | 3.56% | 4.33 | -0.69% | -1.35% |
| Equal weight | last 40% | 10.13% | 3.80% | 2.66 | -0.67% | -1.90% |

On this synthetic book, **the allocator does not beat equal weight on Sharpe**. It does keep the worst day inside the 2% cap with room to spare (-1.08% against -1.61% for equal weight). The generator plants four low-correlation strategies whose edge decays by half over the sample. That is a setting where estimated means are noisy and naive diversification is hard to beat. I left the result as it is and did not tune the generator until the allocator won. On pure noise (`--noise`) the allocator shows no edge (Sharpe 0.03), and CI checks this.

Two details to read the table correctly. The allocator's figures are net of 2 bps turnover costs and include the 0.7x-1.3x overlay, so its gross exposure is not always 1 (average weights printed by the script sum to about 1.1). Equal weight is rebalanced daily at no cost, which flatters it slightly. The first retune has too little history for the grid, so it uses a fixed cold-start recipe (0.8% limit, no caps, 18-month lookback); every later retune is chosen from the grid.

> **About the real-data result.** On the real book the method looked better than equal weight. I don't publish the figure, for three reasons. (1) The source code had a timing leak that I fixed here (see below), and I have not re-run the corrected code on that data. (2) It comes from about three years of daily data covering two regimes, and the "held-out" split was looked at many times while I tried about 19 alternative designs. (3) The strategies were decaying over the sample, so any number from it is regime-dependent. Treat this repo as a method, not a performance claim.

## What did not work: ML and other overlays

I tested whether ML or a more sophisticated estimator improves the allocation. Each test used the same trailing-only reselection and a forward/reverse split, and the later ones also used combinatorial purged cross-validation (CPCV, 22 paths). **None of them beat the plain mean-variance baseline out of sample:**

| # | Idea | Outcome |
|---|---|---|
| 1 | Forecast each strategy's return (ridge, gradient boosting) and feed it to the optimizer | Lost to the no-ML baseline |
| 2 | ML volatility forecast in place of trailing vol | Never selected over the simple signal |
| 3 | Classifier ensemble that flags bad-regime months | No consistent edge |
| 4 | One pooled regression across strategies (4x the rows) | Worse than per-strategy |
| 5 | Daily retraining with cross-asset features | Broke the 2% daily cap |
| 6 | A multi-model ensemble that had worked on a larger book | Lost to the baseline in 16 configurations |
| 7 | Approach 1 on twice the calendar history | Within noise |
| 8 | Unsupervised 2-state HMM regimes | Within noise. The HMM collapsed to one state at this sample size |
| 9 | Bayesian precision-weighted shrinkage of the means | Chosen on trailing data in 7 of 8 periods, still worse out of sample |

Later, non-forecasting ideas also failed. Hierarchical risk parity was clearly worse. Michaud resampling, EWMA weighting and trend extrapolation came out near-neutral on CPCV. A Student-t copula stress set looked like a win on one split, but CPCV reversed it (7 of 22 paths). External market data (volatility indices, FX, commodities) never activated under honest selection. My reading: with about 6 retune decisions in the sample, the binding limit is the data, not the technique.

## Leakage and bias audit

| Issue | Effect | Fix |
|---|---|---|
| Loss-limit scenarios and recipe selection took daily rows dated `< month_end` of the month being traded, which includes that month's own days | The constraint "knew" the coming month's worst days | Everything uses rows dated `<= end of M-1`. A test corrupts month M and checks that M's weights do not change |
| Vol overlay used `rolling().std()` without a shift (an earlier version) | Each day's exposure saw that day's return | `shift(1)`. A test checks that a shock on day t leaves the multiplier on day t unchanged |
| Every hyper-parameter was chosen by scanning full-sample Sharpe (an earlier version) | Headline Sharpe inflated well above the held-out figure | Reselection every 6 months on trailing data only |
| Lookback window swept and the best reported | Selection bias | The lookback sits inside the trailing reselection |
| Grid values (caps, limits) were designed after seeing the full sample, e.g. a gap-risk cap prompted by one known loss day | Remaining selection bias that no walk-forward removes | Grid frozen in `Grid`. Flagged here as residual bias |
| No transaction costs | Optimistic | Turnover is charged (`--cost-bps`, default 2) |
| Monthly and daily histories did not always splice cleanly, and some dates were duplicated | A splice mismatch | The loader drops duplicate dates; splice quality is documented as a data limitation |

## Repo layout

```
src/ddalloc/   data.py · optimizer.py · overlays.py · walkforward.py · metrics.py
scripts/       run_backtest.py · make_synthetic_data.py (writes data/monthly.csv, data/daily.csv)
tests/         caps, loss limit, infeasibility, overlay causality, month-M leakage, point-in-time run, noise
docs/          README.md (index) · methodology.md · walkthrough.md
original/      the original research code (read-only)
```

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q
python scripts/run_backtest.py                       # synthetic
python scripts/run_backtest.py --noise               # no-signal check
python scripts/make_synthetic_data.py                # writes data/monthly.csv and data/daily.csv
python scripts/run_backtest.py --monthly data/monthly.csv --daily data/daily.csv --cost-bps 5 --rf 0.05
```

The last command shows how to run your own data. Both CSVs need a date column first, then one column per strategy holding simple returns. The column order matters: by default, index 1 is the gap-risk strategy that gets the gap cap, and index 0 gets the second cap (see `Config`).

## Original research code

[`original/`](original/README.md) holds the code as I first wrote it for this project. It is kept so you can
read the first version alongside the maintained one; some of it contains the issues in the audit table above,
and comparing the two shows each fix.

## License

MIT

See [CONTRIBUTING.md](CONTRIBUTING.md) and the [CHANGELOG](CHANGELOG.md).
