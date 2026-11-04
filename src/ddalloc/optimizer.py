"""Mean-variance allocation with a hard historical-scenario daily loss limit."""
from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize


def apply_caps(w: np.ndarray, caps: dict[int, float], passes: int = 5) -> np.ndarray:
    """Clip weights at per-asset caps and redistribute the excess pro rata to the others."""
    w = np.asarray(w, dtype=float).copy()
    n = len(w)
    for _ in range(passes):
        moved = False
        for i, cap in caps.items():
            if w[i] > cap + 1e-9:
                excess = w[i] - cap
                w[i] = cap
                others = [k for k in range(n) if k != i and not (k in caps and w[k] >= caps[k] - 1e-9)]
                if not others:
                    others = [k for k in range(n) if k != i]
                denom = w[others].sum()
                if denom > 0:
                    w[others] += excess * w[others] / denom
                else:
                    w[others] += excess / len(others)
                moved = True
        if not moved:
            break
    return w


def constrained_weights(
    mu: np.ndarray,
    sigma: np.ndarray,
    scenarios: np.ndarray,
    loss_limit: float,
    lam: float,
    caps: dict[int, float] | None = None,
) -> np.ndarray:
    """Solve max w'mu - lam/2 w'Sigma w  s.t. sum(w)=1, w>=0, w'r_t >= -loss_limit for all scenarios.

    ``scenarios`` is a (T, N) array of past daily returns. It must contain only
    data observed before the decision date; the caller is responsible for that.
    If T == 0 the loss constraint is simply absent (cold start).
    """
    n = len(mu)
    scen = np.asarray(scenarios, dtype=float).reshape(-1, n)
    cons: list[dict] = [{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)}]
    if len(scen):
        cons.append({"type": "ineq", "fun": lambda w: scen @ w + loss_limit, "jac": lambda w: scen})

    def obj(w: np.ndarray) -> float:
        return -(w @ mu - 0.5 * lam * w @ sigma @ w)

    def grad(w: np.ndarray) -> np.ndarray:
        return -(mu - lam * sigma @ w)

    res = minimize(
        obj, np.full(n, 1.0 / n), jac=grad, method="SLSQP",
        bounds=[(0.0, 1.0)] * n, constraints=cons, options={"maxiter": 300, "ftol": 1e-10},
    )
    if not res.success or (len(scen) and (scen @ res.x).min() < -loss_limit - 1e-6):
        warnings.warn(f"loss-limit problem not solved cleanly: {res.message}", RuntimeWarning, stacklevel=2)
    w = np.clip(res.x, 0.0, None)
    w = w / w.sum() if w.sum() > 0 else np.full(n, 1.0 / n)
    if caps:
        w = apply_caps(w, caps)
    return w
