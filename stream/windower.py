"""Sliding-window state for the streaming imputer.

Holds the last `window_len` timesteps across all stations. When a window fills,
it is handed to the imputer; the window then advances by `stride`. This is the
stateful piece that a Spark Structured Streaming job or a plain Kafka consumer
both sit on top of.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np


class SlidingWindow:
    def __init__(self, stations: list[str], window_len: int, stride: int = 1):
        self.stations = list(stations)
        self.index = {s: i for i, s in enumerate(self.stations)}
        self.window_len = window_len
        self.stride = stride
        self._rows: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._truth: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._mask: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._since_emit = 0

    def _blank(self):
        return np.full(len(self.stations), np.nan, dtype=np.float32)

    def add(self, ts: str, station: str, value, truth, masked: bool):
        """Add one reading. Returns a window dict when one is ready, else None."""
        if station not in self.index:
            return None
        if ts not in self._rows:
            self._rows[ts] = self._blank()
            self._truth[ts] = self._blank()
            self._mask[ts] = np.zeros(len(self.stations), dtype=bool)
        j = self.index[station]
        if value is not None:
            self._rows[ts][j] = value
        if truth is not None:
            self._truth[ts][j] = truth
        self._mask[ts][j] = bool(masked)

        # Trim to the window length, dropping the oldest timestamps.
        while len(self._rows) > self.window_len:
            old = next(iter(self._rows))
            self._rows.pop(old); self._truth.pop(old); self._mask.pop(old)
            self._since_emit += 1

        if len(self._rows) == self.window_len and self._since_emit >= self.stride:
            self._since_emit = 0
            return self.snapshot()
        return None

    def snapshot(self) -> dict:
        ts = list(self._rows.keys())
        return {
            "timestamps": ts,
            "X": np.stack([self._rows[t] for t in ts]),          # (win, n_stations)
            "truth": np.stack([self._truth[t] for t in ts]),
            "eval_mask": np.stack([self._mask[t] for t in ts]),  # artificially hidden
        }

    def ready(self) -> bool:
        return len(self._rows) == self.window_len
