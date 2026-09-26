"""Features for deciding how much diffusion a gap deserves.

Every feature here must be computable *before* running the expensive model --
otherwise the router cannot save anything. So these are all cheap statistics of
the observed context plus the geography, with no diffusion forward passes.

A "gap" is one contiguous run of missing timesteps in one station within one
window. That is the unit the router allocates budget to.
"""
from __future__ import annotations

import numpy as np


def find_gaps(window_mask: np.ndarray):
    """window_mask: (L, K) True where MISSING. Yields (station, start, length)."""
    L, K = window_mask.shape
    for k in range(K):
        col = window_mask[:, k]
        t = 0
        while t < L:
            if col[t]:
                s = t
                while t < L and col[t]:
                    t += 1
                yield k, s, t - s
            else:
                t += 1


def gap_features(X: np.ndarray, adj: np.ndarray, station: int, start: int,
                 length: int) -> dict:
    """Cheap descriptors of one gap.

    X: (L, K) window with NaN for missing, in standardised units.
    adj: (K, K) geographic weight matrix.
    """
    L, K = X.shape
    col = X[:, station]
    observed_col = col[~np.isnan(col)]

    # Context immediately around the gap.
    pre = col[max(0, start - 3):start]
    post = col[start + length:start + length + 3]
    pre = pre[~np.isnan(pre)]
    post = post[~np.isnan(post)]

    # Geographic neighbours that are reporting during the gap.
    nb = np.argsort(-adj[station])[:5]
    nb = [j for j in nb if adj[station, j] > 0]
    seg = slice(start, start + length)
    nb_vals, nb_avail = [], []
    for j in nb:
        v = X[seg, j]
        nb_avail.append(float(np.mean(~np.isnan(v))))
        if np.any(~np.isnan(v)):
            nb_vals.append(float(np.nanmean(v)))

    station_missing = float(np.mean(np.isnan(col)))
    window_missing = float(np.mean(np.isnan(X)))

    return {
        "gap_length": float(length),
        "gap_rel_pos": float(start / max(L - 1, 1)),
        "gap_at_edge": float(start == 0 or start + length >= L),
        "station_missing_rate": station_missing,
        "window_missing_rate": window_missing,
        "station_observed_std": float(observed_col.std()) if observed_col.size > 1 else 0.0,
        "station_observed_mean": float(observed_col.mean()) if observed_col.size else 0.0,
        "pre_mean": float(pre.mean()) if pre.size else 0.0,
        "post_mean": float(post.mean()) if post.size else 0.0,
        "pre_post_jump": float(abs(post.mean() - pre.mean())) if (pre.size and post.size) else 0.0,
        "has_pre": float(pre.size > 0),
        "has_post": float(post.size > 0),
        "n_neighbours": float(len(nb)),
        "neighbour_availability": float(np.mean(nb_avail)) if nb_avail else 0.0,
        "neighbour_mean": float(np.mean(nb_vals)) if nb_vals else 0.0,
        "neighbour_disagreement": float(np.std(nb_vals)) if len(nb_vals) > 1 else 0.0,
        "neighbour_vs_station": (float(abs(np.mean(nb_vals) - pre.mean()))
                                 if (nb_vals and pre.size) else 0.0),
    }


FEATURE_NAMES = None  # populated on first call to features_for_window


def features_for_window(X: np.ndarray, adj: np.ndarray):
    """Return (list_of_feature_dicts, list_of_(station,start,length))."""
    global FEATURE_NAMES
    mask = np.isnan(X)
    feats, keys = [], []
    for station, start, length in find_gaps(mask):
        f = gap_features(X, adj, station, start, length)
        if FEATURE_NAMES is None:
            FEATURE_NAMES = sorted(f)
        feats.append(f)
        keys.append((station, start, length))
    return feats, keys


def to_matrix(feature_dicts, names=None) -> np.ndarray:
    names = names or FEATURE_NAMES
    if not feature_dicts:
        return np.empty((0, len(names)), dtype=np.float32)
    return np.array([[d[n] for n in names] for d in feature_dicts], dtype=np.float32)
