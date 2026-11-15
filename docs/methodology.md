# Methodology and bias audit

## Problem

A fund runs a few strategies and wants one book. The mandate is to maximise return on capital, with a hard limit on the worst single-day loss of the book (a fixed percentage of capital). Per-strategy limits are soft.

The data problem is that daily history is short (about three years for most strategies), while monthly history goes back further. A daily loss limit cannot be checked on monthly data, because a month's return hides its worst day.

## Design

- **Hybrid estimation.** The means and covariance in the objective come from the trailing `L` months of the monthly panel. Before daily data starts, those rows are true monthly returns. After it starts, they are daily returns compounded to month-end.
- **Historical-scenario constraint.** `w'r_t >= -limit` must hold for each of the last 252 daily rows. It is linear in `w` and solved with SLSQP, with an analytic Jacobian. With no daily rows (cold start), the constraint is absent.
- **Safety margin.** In my research, a constraint fitted at exactly 2% on trailing days broke out of sample: one future day reached -3.9%, because future days can be worse than any day in the window. So `limit` is drawn from {1.0%, 0.6%, 0.5%}, well inside the mandate.
- **Caps.** One cap applies to the strategy with gap risk and another to the largest allocation, with the excess redistributed pro rata. The caps are applied *after* the solve, so in principle redistribution can push a past-scenario loss above `limit`. The overlay can also scale exposure up to 1.3x, which scales a past-scenario loss by the same factor. The realised worst day is reported every run for both reasons.
- **Overlay.** A book-level exposure multiplier in [0.7, 1.3] is set by the percentile rank of yesterday's 20-day vol. In research, wider bounds gave monotonically higher Sharpe with no ceiling, which is a classic overfitting signature, so the bounds are fixed.
- **Cold start.** Until there are `selection_lookback + 6` months of history, a fixed default recipe is used (0.8% limit, `lam` 10, no caps, the shortest lookback).
- **Reselection.** Every 6 months the coordinate-wise grid search picks (`limit`, `lam`), then the gap cap, then the second cap, then `L`. Each candidate is scored by the Sharpe of an inner monthly walk-forward over data dated at or before the decision date.
- **Costs.** Each day is charged `cost_bps` per unit of absolute change in effective weights, including changes driven by the overlay.

## Timing contract

| Quantity for month M | Data allowed |
|---|---|
| mu, Sigma | monthly rows dated <= end of M-1 |
| loss scenarios | daily rows dated <= end of M-1 |
| recipe | both panels truncated at end of M-1 |
| overlay multiplier for day t | book returns up to t-1 |

## Bias found in the original research code, and fixes

1. **Same-month scenarios.** The original filtered daily scenarios with `index < month_end_date`, where `month_end_date` was the end of the month being traded. That window includes nearly all of the month's own days, so the loss constraint could see the coming month's bad days. The recipe search had the same filter. Fixed by using rows dated at or before the end of the previous month. `test_weights_ignore_the_month_being_traded` and `test_run_is_point_in_time` guard it.
2. **Overlay without a shift.** In an earlier version, the trailing vol included the day's own return. This was fixed in the research and kept here, and `test_overlay_is_causal` guards it.
3. **Hyper-parameters picked on the reporting sample.** An early headline figure came from choosing each knob by full-sample Sharpe, so it was inflated. A genuine development/validation split gave very different numbers on each side, which showed the sample cannot pin down a single number. Fixed by trailing reselection.
4. **Repeated looks at the "held-out" split.** The forward split (the last 40%) was used to judge about 19 alternative designs. It is therefore not a clean hold-out, so no real-data number from it is published.
5. **Grid designed with hindsight.** The gap cap exists because one large gap-loss day was seen in the full sample. Walk-forward cannot remove that. It is disclosed here, not fixed.
6. **Costs missing.** These are now charged on turnover. Turnover was low in research (under about 10% a month), so the effect is small but no longer zero.
7. **Data splices.** In the real data, monthly and daily histories did not always splice cleanly and some dates were duplicated. Duplicates are dropped; splice quality remains a limitation of any real-data result.

## Honest verdict

- On the real book, the method allocated sensibly, kept the worst day well inside 2%, and looked better than equal weight. That figure is preliminary: there was no untouched hold-out, it rests on a two-regime sample, and the timing leak above was fixed only after the fact.
- On the synthetic book in this repo, the method does **not** beat equal weight on Sharpe. It does have a better worst day.
- ML did not help. Nine forecasting or regime approaches failed, as did several non-forecasting ones (see the README). External market data never helped either. The constraint is sample size.
- A separate analysis found that returns of the real strategies declined over the sample. No allocator can fix that, and it explains why the later period is consistently weaker.

## Tests

- caps hold and weights sum to 1
- the loss limit holds on the scenarios, and an infeasible limit raises a warning
- the overlay is causal
- corrupting month M leaves M's weights unchanged
- corrupting everything after a cut leaves every earlier return identical
- the noise test: no edge over equal weight on zero-mean data
- a planted signal is recovered

## Dropped from this repo

I dropped the per-strategy ML exploratory scripts, the CPCV harnesses, the copula, Michaud, EWMA, HRP and trend experiments, the external-data pulls, a separate per-strategy ML improvement workstream (no edge on monthly-only data), presentation documents and all real data. Their conclusions are summarised above.
