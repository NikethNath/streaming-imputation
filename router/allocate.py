"""Budget allocation across gaps, with an equity reservation.

Two allocators:

  * `allocate_by_gain` -- pure expected-gain knapsack. Spend where the predicted
    error reduction per NFE is greatest.

  * `allocate_with_equity` -- the same, but a fraction of the budget is reserved
    for the stations currently being served worst.

The second exists because the first is *regressive*. A chronically noisy,
chronically gappy station tends to have low predicted gain -- more diffusion
does not rescue it -- so a pure expected-gain rule starves it forever, and the
stations that are already worst get worse. Reserving budget for the worst-served
stations is what converts the cost saving into an equity gain, and it is the
reason the triage and the fairness work belong in one system.
"""
from __future__ import annotations

import numpy as np

from eval.oracle import allocate_lagrangian


def allocate_by_gain(pred_error: np.ndarray, costs: np.ndarray, budget: float):
    """pred_error: (n_gaps, n_budgets) predicted error. Returns budget index per gap."""
    idx, cost, _ = allocate_lagrangian(pred_error, costs, budget)
    return idx, cost


def allocate_with_equity(pred_error: np.ndarray, costs: np.ndarray, budget: float,
                         station_of_gap: np.ndarray,
                         station_deficit: dict | None = None,
                         reserve_frac: float = 0.3,
                         worst_frac: float = 0.3):
    """Reserve `reserve_frac` of the budget for gaps on the worst-served stations.

    station_deficit maps station index -> recent mean error (higher = worse
    served). Stations in the worst `worst_frac` get their gaps allocated first,
    out of the reserved budget; everything else competes for the remainder.
    """
    n = pred_error.shape[0]
    if not station_deficit or reserve_frac <= 0:
        idx, cost = allocate_by_gain(pred_error, costs, budget)
        return idx, cost, np.zeros(n, dtype=bool)

    deficits = np.array([station_deficit.get(int(s), 0.0) for s in station_of_gap])
    if not np.any(deficits > 0):
        idx, cost = allocate_by_gain(pred_error, costs, budget)
        return idx, cost, np.zeros(n, dtype=bool)

    thresh = np.quantile(deficits[deficits > 0], 1.0 - worst_frac)
    priority = deficits >= thresh
    if not priority.any():
        idx, cost = allocate_by_gain(pred_error, costs, budget)
        return idx, cost, np.zeros(n, dtype=bool)

    choice = np.zeros(n, dtype=int)

    # Worst-served gaps draw on the reserved share first, weighted by deficit so
    # the very worst stations get the most help.
    res_budget = budget * reserve_frac
    p_idx, p_cost, _ = allocate_lagrangian(
        pred_error[priority], costs, res_budget,
        weights=deficits[priority] / max(deficits[priority].mean(), 1e-12))
    choice[priority] = p_idx

    # Everyone else shares what is left.
    rest = ~priority
    if rest.any():
        r_budget = max(budget - p_cost, costs.min() * rest.sum())
        r_idx, r_cost = allocate_by_gain(pred_error[rest], costs, r_budget)
        choice[rest] = r_idx
    total = float(costs[choice].sum())
    return choice, total, priority


class StationDeficitTracker:
    """Running per-station error, so the reservation adapts as the stream runs."""

    def __init__(self, n_stations: int, decay: float = 0.9):
        self.err = np.zeros(n_stations)
        self.seen = np.zeros(n_stations)
        self.decay = decay

    def update(self, station_errors: dict):
        self.err *= self.decay
        self.seen *= self.decay
        for s, e in station_errors.items():
            if np.isfinite(e):
                self.err[int(s)] += e
                self.seen[int(s)] += 1.0

    def deficits(self) -> dict:
        out = {}
        for s in range(len(self.err)):
            if self.seen[s] > 0:
                out[s] = float(self.err[s] / self.seen[s])
        return out
