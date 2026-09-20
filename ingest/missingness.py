"""Missingness processes layered on top of the archive's genuine gaps.

Protocols follow the CSDI / PriSTI conventions so results stay comparable:
point (MCAR), block (bursty per-station outage), sensor failure, and blackout.
`reliability_skewed` is ours -- it draws per-station failure probabilities from a
heavy-tailed distribution so a minority of stations carry most of the
missingness, which is the realistic case and the one where a dataset-average
metric hides the damage.
"""
from __future__ import annotations

import numpy as np


class MissingnessSimulator:
    def __init__(self, n_stations: int, seed: int = 27,
                 point_rate: float = 0.10,
                 block_rate: float = 0.02,
                 block_len_mean: float = 6.0,
                 mode: str = "mixed",
                 skew_alpha: float = 2.0):
        self.n = n_stations
        self.rng = np.random.default_rng(seed)
        self.point_rate = point_rate
        self.block_rate = block_rate
        self.block_len_mean = block_len_mean
        self.mode = mode
        # Remaining outage length per station; >0 means the station is dark.
        self._outage = np.zeros(n_stations, dtype=int)
        if mode == "reliability_skewed":
            # Pareto-ish weights: a few stations are far less reliable.
            w = self.rng.pareto(skew_alpha, n_stations) + 1.0
            self._station_weight = w / w.mean()
        else:
            self._station_weight = np.ones(n_stations)

    def step(self) -> np.ndarray:
        """Return a boolean mask for one timestep. True = we hide this value."""
        hide = np.zeros(self.n, dtype=bool)

        # Continue any outage already in progress.
        ongoing = self._outage > 0
        hide[ongoing] = True
        self._outage[ongoing] -= 1

        if self.mode in ("block", "mixed", "reliability_skewed"):
            start = (self.rng.random(self.n) < self.block_rate * self._station_weight)
            start &= ~ongoing
            n_new = int(start.sum())
            if n_new:
                lengths = self.rng.geometric(1.0 / self.block_len_mean, n_new)
                self._outage[start] = lengths
                hide[start] = True

        if self.mode in ("point", "mixed", "reliability_skewed"):
            p = np.clip(self.point_rate * self._station_weight, 0, 1)
            hide |= (self.rng.random(self.n) < p) & ~hide

        return hide

    def blackout(self, length: int):
        """Force every station dark for `length` timesteps."""
        self._outage[:] = length
