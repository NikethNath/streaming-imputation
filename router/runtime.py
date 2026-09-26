"""Serving-time budget allocation inside the streaming path.

Two modes, and the difference between them is the main experimental finding of
this project.

  mode="gain"    per-gap triage. Predict each gap's error reduction from extra
                 compute and knapsack the budget across gaps. **This does not
                 work on AQI-36** and is kept so the negative result is
                 reproducible rather than asserted. Per-gap gain has
                 test-retest reliability r ~= 0.015 (scripts/gain_reliability.py):
                 which gap benefits from more compute is essentially resampling
                 noise, so there is no signal for any router to learn.

  mode="equity"  station-level allocation, and the one we ship. Per-gap *error*
                 is highly reliable (r = 0.76 .. 0.91) and concentrates by
                 station, so "which station is served worst" is a real and
                 persistent property. Worst-served stations get the expensive
                 budget, the rest get the cheap one, at a fraction of the cost
                 of giving everyone the expensive budget.

Both report NFE = steps x samples so cost is comparable across modes.
"""
from __future__ import annotations

import pickle

import numpy as np

from models import sampler
from router import features as feat
from router.allocate import allocate_with_equity, StationDeficitTracker


class GapRouter:
    def __init__(self, router_path: str | None, adj: np.ndarray,
                 mode: str = "equity",
                 cheap: tuple[int, int] = (10, 4),
                 rich: tuple[int, int] = (20, 16),
                 worst_frac: float = 0.3,
                 reserve_frac: float = 0.3):
        self.mode = mode
        self.adj = adj
        self.cheap = cheap
        self.rich = rich
        self.worst_frac = worst_frac
        self.reserve_frac = reserve_frac
        self.tracker = StationDeficitTracker(adj.shape[0])
        self.models = self.costs = self.budgets = self.feature_names = None
        if router_path:
            with open(router_path, "rb") as f:
                b = pickle.load(f)
            self.models = b["models"]
            self.costs = np.asarray(b["costs"], dtype=float)
            self.budgets = np.asarray(b["budgets"], dtype=int)
            self.feature_names = b["feature_names"]

    # ------------------------------------------------------------------
    def _predict_gain(self, Xmat):
        g = np.zeros((Xmat.shape[0], len(self.models)))
        for b in range(1, len(self.models)):
            if self.models[b] is not None:
                g[:, b] = self.models[b].predict(Xmat)
        g = np.maximum(g, 0.0)
        return np.maximum.accumulate(g, axis=1)

    def _nfe(self, b):
        return int(b[0]) * int(b[1])

    # ------------------------------------------------------------------
    def impute_window(self, csdi, X: np.ndarray, nfe_budget: float = None,
                      use_equity: bool = True):
        """X: (L, K) standardised window with NaN. Returns (imputed, info)."""
        if self.mode == "equity" or self.models is None:
            return self._impute_station_tiered(csdi, X, use_equity)
        return self._impute_gap_triage(csdi, X, nfe_budget)

    # -- shipped path ---------------------------------------------------
    def _impute_station_tiered(self, csdi, X, use_equity=True):
        L, K = X.shape
        deficits = self.tracker.deficits()
        worst = set()
        if use_equity and deficits:
            ranked = sorted(deficits, key=lambda s: -deficits[s])
            k = max(1, int(round(self.worst_frac * K)))
            worst = set(ranked[:k])

        smp_c, nfe_c = sampler.sample(csdi, X[None, ...], n_steps=self.cheap[0],
                                      n_samples=self.cheap[1], mode="ddim")
        out = sampler.point_estimate(smp_c)[0]
        nfe_used = nfe_c
        if worst:
            smp_r, nfe_r = sampler.sample(csdi, X[None, ...], n_steps=self.rich[0],
                                          n_samples=self.rich[1], mode="ddim")
            rich = sampler.point_estimate(smp_r)[0]
            cols = np.array(sorted(worst), dtype=int)
            out[:, cols] = rich[:, cols]
            # only the promoted stations pay the rich rate
            nfe_used = int(nfe_c + nfe_r * len(cols) / K)

        observed = ~np.isnan(X)
        out = np.where(observed, np.nan_to_num(X), out)
        return out, {
            "mode": "equity",
            "n_promoted_stations": int(len(worst)),
            "nfe_used": int(nfe_used),
            "escalation_rate": float(len(worst) / K),
        }

    # -- reproducible negative result -----------------------------------
    def _impute_gap_triage(self, csdi, X, nfe_budget):
        feats, keys = feat.features_for_window(X, self.adj)
        out = np.array(X, dtype=np.float64, copy=True)
        if not feats:
            return out, {"mode": "gain", "n_gaps": 0, "nfe_used": 0,
                         "escalation_rate": 0.0}
        Xmat = feat.to_matrix(feats, self.feature_names)
        pred_cost = -self._predict_gain(Xmat)      # allocator minimises
        station_of_gap = np.array([k[0] for k in keys])
        choice, cost, priority = allocate_with_equity(
            pred_cost, self.costs, nfe_budget or float(self.costs.max() * len(keys)),
            station_of_gap, self.tracker.deficits(),
            reserve_frac=self.reserve_frac, worst_frac=self.worst_frac)

        nfe_used = 0
        for b in np.unique(choice):
            steps, samples = int(self.budgets[b][0]), int(self.budgets[b][1])
            smp, nfe = sampler.sample(csdi, X[None, ...], n_steps=steps,
                                      n_samples=samples, mode="ddim")
            point = sampler.point_estimate(smp)[0]
            nfe_used += nfe
            for gi in np.where(choice == b)[0]:
                st, start, length = keys[gi]
                out[start:start + length, st] = point[start:start + length, st]
        observed = ~np.isnan(X)
        out = np.where(observed, np.nan_to_num(X), out)
        return out, {"mode": "gain", "n_gaps": len(keys), "nfe_used": int(nfe_used),
                     "escalation_rate": float(np.mean(choice > 0)),
                     "n_priority_gaps": int(priority.sum())}

    def observe_errors(self, station_errors: dict):
        self.tracker.update(station_errors)
