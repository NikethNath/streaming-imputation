"""Imputation metrics, reported the way CSDI and PriSTI report them.

Everything here is evaluated *only* on artificially masked cells, i.e. cells we
hid from the model but whose true value we know. Cells that were already missing
in the archive have no ground truth and are excluded.
"""
from __future__ import annotations

import numpy as np


def _masked(pred, truth, mask):
    m = mask.astype(bool)
    return pred[m], truth[m]


def mae(pred, truth, mask) -> float:
    p, t = _masked(pred, truth, mask)
    return float(np.abs(p - t).mean()) if p.size else float("nan")


def rmse(pred, truth, mask) -> float:
    p, t = _masked(pred, truth, mask)
    return float(np.sqrt(((p - t) ** 2).mean())) if p.size else float("nan")


def mre(pred, truth, mask) -> float:
    """Mean relative error, as used by SAITS."""
    p, t = _masked(pred, truth, mask)
    denom = np.abs(t).sum()
    return float(np.abs(p - t).sum() / denom) if denom else float("nan")


def crps(samples, truth, mask, n_quantiles: int = 19) -> float:
    """Continuous ranked probability score via the quantile-loss approximation.

    This is the estimator CSDI and PriSTI use: average the pinball loss over an
    evenly spaced quantile grid, normalised by the mean absolute ground truth.

    `samples` has shape (n_samples, *truth.shape).
    """
    m = mask.astype(bool)
    if not m.any():
        return float("nan")
    t = truth[m]
    s = samples[:, m]
    qs = np.arange(1, n_quantiles + 1) / (n_quantiles + 1)
    total = 0.0
    for q in qs:
        pred_q = np.quantile(s, q, axis=0)
        diff = t - pred_q
        # pinball loss
        total += 2 * np.sum(np.where(diff >= 0, q * diff, (q - 1) * diff))
    denom = np.abs(t).sum()
    return float(total / (len(qs) * denom)) if denom else float("nan")


def picp(samples, truth, mask, level: float = 0.9) -> float:
    """Prediction interval coverage probability at the given nominal level.

    A fast sampler that is no longer calibrated has given up the only thing
    diffusion had over a deterministic imputer, and MAE will not show it.
    """
    m = mask.astype(bool)
    if not m.any():
        return float("nan")
    t = truth[m]
    s = samples[:, m]
    lo = np.quantile(s, (1 - level) / 2, axis=0)
    hi = np.quantile(s, 1 - (1 - level) / 2, axis=0)
    return float(((t >= lo) & (t <= hi)).mean())


def interval_width(samples, mask, level: float = 0.9) -> float:
    m = mask.astype(bool)
    if not m.any():
        return float("nan")
    s = samples[:, m]
    lo = np.quantile(s, (1 - level) / 2, axis=0)
    hi = np.quantile(s, 1 - (1 - level) / 2, axis=0)
    return float((hi - lo).mean())


def per_station(pred, truth, mask, station_ids) -> dict:
    """MAE for each station separately.

    This is the number none of the three base papers reports, and the one the
    equity audit is built on.
    """
    out = {}
    for j, sid in enumerate(station_ids):
        m = mask[..., j].astype(bool)
        if m.any():
            out[sid] = float(np.abs(pred[..., j][m] - truth[..., j][m]).mean())
        else:
            out[sid] = float("nan")
    return out


def summarise(pred, truth, mask, samples=None, station_ids=None) -> dict:
    res = {
        "MAE": mae(pred, truth, mask),
        "RMSE": rmse(pred, truth, mask),
        "MRE": mre(pred, truth, mask),
        "n_evaluated": int(mask.astype(bool).sum()),
    }
    if samples is not None:
        res["CRPS"] = crps(samples, truth, mask)
        res["PICP@90"] = picp(samples, truth, mask)
        res["interval_width@90"] = interval_width(samples, mask)
    if station_ids is not None:
        res["per_station_MAE"] = per_station(pred, truth, mask, station_ids)
    return res
