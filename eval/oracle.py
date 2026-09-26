"""Oracle router bound.

Before building a learned router it is worth knowing whether per-gap allocation
can help *at all*. The oracle knows each gap's true error at every budget and
allocates optimally under a total-NFE constraint. It is an upper bound on any
router, learned or otherwise.

If the oracle cannot beat spending the same total budget uniformly, per-gap
triage is not a real idea and we report that instead of building on it.

The allocation problem is a multiple-choice knapsack: each gap picks exactly one
budget, costs are NFE, and we minimise total absolute error. We solve the
Lagrangian relaxation by bisection on the cost multiplier, which is exact enough
for a frontier and is O(n log n) rather than pseudo-polynomial.
"""
from __future__ import annotations

import numpy as np


def allocate_lagrangian(errors: np.ndarray, costs: np.ndarray, budget: float,
                        weights: np.ndarray | None = None, tol: float = 1e-9,
                        max_iter: int = 200):
    """Choose one budget index per gap to minimise sum(weight*error) s.t. sum(cost) <= budget.

    errors: (n_gaps, n_budgets) -- error if that gap gets that budget
    costs:  (n_budgets,)        -- NFE of each budget
    Returns (choice_idx, total_cost, total_weighted_error).
    """
    n, B = errors.shape
    w = np.ones(n, dtype=np.float64) if weights is None else weights.astype(np.float64)
    we = errors.astype(np.float64) * w[:, None]

    def pick(lmbda):
        # minimise weighted error + lambda * cost, independently per gap
        idx = np.argmin(we + lmbda * costs[None, :], axis=1)
        return idx, costs[idx].sum()

    # lambda = 0 -> cheapest error (most expensive budget); large lambda -> cheapest cost
    lo, hi = 0.0, 1.0
    idx, cost = pick(hi)
    while cost > budget and hi < 1e12:
        hi *= 4.0
        idx, cost = pick(hi)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        idx, cost = pick(mid)
        if cost > budget:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * max(1.0, hi):
            break
    idx, cost = pick(hi)
    return idx, float(cost), float(we[np.arange(n), idx].sum())


def uniform_choice(errors: np.ndarray, costs: np.ndarray, budget: float):
    """Best single budget applied to every gap, subject to the same constraint.

    This is the global knob -- what CSDI's ODE-solver suggestion and DDIM give
    you. It is the thing the router has to beat.
    """
    n = errors.shape[0]
    feasible = [b for b in range(len(costs)) if costs[b] * n <= budget]
    if not feasible:
        b = int(np.argmin(costs))
        return b, float(costs[b] * n), float(errors[:, b].sum())
    b = min(feasible, key=lambda b: errors[:, b].mean())
    return b, float(costs[b] * n), float(errors[:, b].sum())


def oracle_frontier(errors: np.ndarray, costs: np.ndarray, n_points: int = 14,
                    weights: np.ndarray | None = None):
    """Sweep the budget and compare oracle allocation against the uniform knob."""
    n = errors.shape[0]
    lo, hi = costs.min() * n, costs.max() * n
    out = []
    for budget in np.linspace(lo, hi, n_points):
        o_idx, o_cost, _ = allocate_lagrangian(errors, costs, budget, weights)
        o_mae = float(errors[np.arange(n), o_idx].mean())
        u_b, u_cost, u_sum = uniform_choice(errors, costs, budget)
        out.append({
            "budget_nfe": float(budget),
            "budget_per_gap": float(budget / n),
            "oracle_cost": o_cost,
            "oracle_MAE": o_mae,
            "uniform_budget_idx": int(u_b),
            "uniform_cost": u_cost,
            "uniform_MAE": float(u_sum / n),
            "oracle_gain_pct": float(100.0 * (u_sum / n - o_mae) / max(u_sum / n, 1e-12)),
            "escalation_rate": float(np.mean(o_idx > 0)),
        })
    return out
