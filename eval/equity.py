"""Per-sensor equity audit.

CSDI, PriSTI and SAITS all report a single dataset-averaged error. That average
hides the thing that matters in a public sensing network: imputation error
concentrates on the stations with the most missing data. This module measures
that concentration so it can be reported and, in Phase 2, optimised against.
"""
from __future__ import annotations

import numpy as np

# Indian CPCB / US EPA style PM2.5 breakpoints (ug/m3, 24h).
# A 5 ug/m3 error matters enormously at a boundary and not at all mid-band,
# which is why band misclassification is a better public-health proxy than MAE.
AQI_BANDS = [
    (0.0, 12.0, "Good"),
    (12.0, 35.4, "Moderate"),
    (35.4, 55.4, "Unhealthy for Sensitive Groups"),
    (55.4, 150.4, "Unhealthy"),
    (150.4, 250.4, "Very Unhealthy"),
    (250.4, float("inf"), "Hazardous"),
]


def band_of(values):
    v = np.asarray(values, dtype=float)
    out = np.full(v.shape, -1, dtype=int)
    for i, (lo, hi, _) in enumerate(AQI_BANDS):
        out[(v >= lo) & (v < hi)] = i
    return out


def gini(values) -> float:
    """Gini coefficient of a non-negative error vector. 0 = perfectly equal."""
    x = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if x.size == 0 or x.sum() == 0:
        return float("nan")
    x = np.sort(x)
    n = x.size
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum()) / (n * x.sum()) - (n + 1) / n)


def worst_decile_mae(per_station_mae: dict) -> float:
    vals = np.array([v for v in per_station_mae.values() if np.isfinite(v)])
    if vals.size == 0:
        return float("nan")
    k = max(1, int(np.ceil(0.1 * vals.size)))
    return float(np.sort(vals)[-k:].mean())


def error_vs_missingness_slope(per_station_mae: dict, missing_rate: dict):
    """OLS slope of per-station error against that station's missing rate.

    A positive slope is the precise statement "the model is worst where data is
    scarcest". Returns (slope, pearson_r).
    """
    keys = [k for k in per_station_mae
            if np.isfinite(per_station_mae[k]) and k in missing_rate]
    if len(keys) < 3:
        return float("nan"), float("nan")
    x = np.array([missing_rate[k] for k in keys])
    y = np.array([per_station_mae[k] for k in keys])
    if x.std() == 0:
        return float("nan"), float("nan")
    slope = np.polyfit(x, y, 1)[0]
    r = np.corrcoef(x, y)[0, 1]
    return float(slope), float(r)


def band_misclassification(pred, truth, mask, station_ids=None):
    """Fraction of imputed values that land in the wrong public-health band.

    Returned overall and, if station ids are given, per station.
    """
    m = mask.astype(bool)
    if not m.any():
        return {"overall": float("nan"), "per_station": {}}
    overall = float((band_of(pred[m]) != band_of(truth[m])).mean())
    per = {}
    if station_ids is not None:
        for j, sid in enumerate(station_ids):
            mj = mask[..., j].astype(bool)
            if mj.any():
                per[sid] = float(
                    (band_of(pred[..., j][mj]) != band_of(truth[..., j][mj])).mean()
                )
            else:
                per[sid] = float("nan")
    return {"overall": overall, "per_station": per}


def audit(per_station_mae: dict, missing_rate: dict = None, band_result: dict = None) -> dict:
    vals = np.array([v for v in per_station_mae.values() if np.isfinite(v)])
    out = {
        "mean_station_MAE": float(vals.mean()) if vals.size else float("nan"),
        "median_station_MAE": float(np.median(vals)) if vals.size else float("nan"),
        "worst_decile_MAE": worst_decile_mae(per_station_mae),
        "best_station_MAE": float(vals.min()) if vals.size else float("nan"),
        "worst_station_MAE": float(vals.max()) if vals.size else float("nan"),
        "max_min_gap": float(vals.max() - vals.min()) if vals.size else float("nan"),
        "error_gini": gini(list(per_station_mae.values())),
    }
    if vals.size:
        out["worst_over_mean_ratio"] = float(vals.max() / vals.mean())
    if missing_rate:
        slope, r = error_vs_missingness_slope(per_station_mae, missing_rate)
        out["error_vs_missingness_slope"] = slope
        out["error_vs_missingness_r"] = r
    if band_result:
        out["band_misclassification"] = band_result["overall"]
    return out
